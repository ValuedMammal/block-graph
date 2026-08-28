//! [`BlockGraph`] - A directed acyclic graph (DAG) structure for managing blockchain data.
//!
//! `BlockGraph` maintains a complete history of blocks and their relationships, supporting:
//! - Multiple competing chains (forks)
//! - Efficient chain reorganizations
//! - Fast lookups using skip pointers via [`CheckPoint`]
//! - A canonical chain tip that follows the longest chain rule
//!
//! The structure uses [`CheckPoint`] for the canonical chain and maintains parent-child
//! relationships between blocks to efficiently handle updates and reorgs.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt::{self, Debug, Display};
use core::ops::RangeBounds;

use bitcoin::{hashes::Hash, BlockHash};

use crate::checkpoint::{Block, BlockId, CheckPoint};
use crate::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Block graph.
#[derive(Debug, Clone)]
pub struct BlockGraph<T> {
    /// Nodes of `(Height, T)` in the block graph keyed by block hash.
    blocks: HashMap<BlockHash, (u32, T)>,
    /// Map of block hash to set of parent IDs.
    ///
    /// `parents` is a set because a child can point to its own parent and/or
    /// a more distant ancestor.
    parents: HashMap<BlockHash, BTreeSet<BlockId>>,
    /// `next_hashes` maps a block hash to the set of hashes extending from it.
    next_hashes: HashMap<BlockHash, HashSet<BlockHash>>,
    /// The root hash, aka genesis.
    root: BlockHash,
    /// The canonical chain tip.
    tip: CheckPoint<T>,
}

impl<T: Block + PartialEq + Debug + Clone> BlockGraph<T> {
    /// From genesis `value`.
    pub fn from_genesis(value: T) -> Self {
        let genesis_height = 0;
        let genesis_hash = value.to_blockhash();

        let mut blocks = HashMap::new();
        blocks.insert(genesis_hash, (genesis_height, value.clone()));
        let mut next_hashes = HashMap::new();
        next_hashes.insert(BlockHash::all_zeros(), [genesis_hash].into());
        let root = genesis_hash;
        let tip = CheckPoint::new(genesis_height, value);

        Self {
            blocks,
            parents: Default::default(),
            next_hashes,
            root,
            tip,
        }
    }

    /// Get the chain tip block id.
    pub fn tip(&self) -> CheckPoint<T> {
        self.tip.clone()
    }

    /// Get the value of a node in the best chain by `height`.
    pub fn get(&self, height: u32) -> Option<CheckPoint<T>> {
        self.tip.get(height)
    }

    /// Iterate items of the canonical chain.
    pub fn iter(&self) -> impl Iterator<Item = CheckPoint<T>> {
        self.tip.iter()
    }

    /// Iterate items of the canonical chain within a specified `range` of heights.
    pub fn range(&self, range: impl RangeBounds<u32>) -> impl Iterator<Item = CheckPoint<T>> {
        self.tip.range(range)
    }

    /// Return the genesis block data.
    pub fn genesis_block(&self) -> T {
        self.blocks
            .get(&self.root)
            .cloned()
            .map(|(_, b)| b)
            .expect("graph must contain root")
    }

    /// Retrieve the block id of a given `hash` if it exists.
    pub fn block_id(&self, hash: &BlockHash) -> Option<BlockId> {
        self.blocks.get(hash).map(|(height, _)| BlockId {
            height: *height,
            hash: *hash,
        })
    }

    /// Construct a [`BlockGraph`] from a [`ChangeSet`]. Returns `None` if `changeset` is empty.
    ///
    /// This method rebuilds the block graph from a changeset by:
    ///
    /// 1. Finding the genesis block (height 0)
    /// 2. Building the graph structure with parent-child relationships
    /// 3. Determining the canonical chain tip
    /// 4. Constructing the canonical chain by traversing back from the tip
    ///
    /// # Errors
    ///
    /// Returns [`FromChangeSetError::MissingGenesis`] if no block at height 0 exists in the
    /// changeset, [`FromChangeSetError::MultipleGenesisBlocks`] if more than one distinct block
    /// is declared at height 0, [`FromChangeSetError::InvalidEdgeHeight`] if an edge's
    /// declared parent (when present in `changeset.blocks`) is not at a strictly lower height
    /// than its child, or [`FromChangeSetError::InconsistentPrevBlockhash`] if an adjacent pair
    /// in the reconstructed chain declares a `prev_blockhash` that doesn't match its parent.
    pub fn from_changeset(changeset: ChangeSet<T>) -> Result<Option<Self>, FromChangeSetError> {
        if changeset.blocks.is_empty() {
            return Ok(None);
        }
        let mut genesis_blocks = changeset.blocks.iter().filter(|(_, (height, _))| *height == 0);
        let (&genesis_hash, (_, genesis_value)) =
            genesis_blocks.next().ok_or(FromChangeSetError::MissingGenesis)?;
        if let Some((&other_hash, _)) = genesis_blocks.next() {
            return Err(FromChangeSetError::MultipleGenesisBlocks {
                first: genesis_hash,
                second: other_hash,
            });
        }

        check_changeset_edge_heights(&changeset)?;

        let mut graph = Self::from_genesis(genesis_value.clone());

        // Populate blocks from changeset blocks
        for (&hash, (height, value)) in &changeset.blocks {
            graph.blocks.insert(hash, (*height, value.clone()));
        }

        // Populate next_hashes and parents from changeset edges. An edge's parent may
        // legitimately be a hash with no block data (e.g. the sentinel predecessor of genesis,
        // or a gapped connection's declared-but-absent parent), but a child with no block data
        // describes no real relationship, so such edges are dropped entirely rather than left
        // dangling in `next_hashes`/`parents`.
        for &(parent_hash, child_hash) in &changeset.edges {
            if !graph.blocks.contains_key(&child_hash) {
                continue;
            }
            graph.next_hashes.entry(parent_hash).or_default().insert(child_hash);
            if let Some(parent_id) = graph.block_id(&parent_hash) {
                graph.parents.entry(child_hash).or_default().insert(parent_id);
            }
        }

        let items = graph.canonicalize(graph.root);
        graph.tip = graph.tip.extend(items).map_err(|err| {
            FromChangeSetError::InconsistentPrevBlockhash {
                hash: err.value.to_blockhash(),
            }
        })?;

        Ok(Some(graph))
    }

    /// Canonicalize the [`BlockGraph`] starting from the given `root` and return a
    /// collection of `(height, value)` tuples in ascending height order.
    ///
    /// Note: The caller must ensure that `root` exists in the current active chain.
    fn canonicalize(&self, root: BlockHash) -> Vec<(u32, T)> {
        // Find the possible tips by exploring `.next_hashes` starting from the root.
        //
        // `visited` guards against re-queuing a hash whose children were already expanded.
        // Without it, diamond-shaped DAGs (a common shape once skip-chain edges exist) are
        // re-traversed once per incoming path, and a cyclic edge (e.g. corrupt or adversarial
        // data ingested via `from_changeset`) sends this into an infinite loop rather than just
        // being slow.
        let mut visited = HashSet::<BlockHash>::new();
        let mut tips = HashSet::<BlockHash>::new();
        let mut queue = vec![];
        queue.push(root);

        while let Some(hash) = queue.pop() {
            if !visited.insert(hash) {
                continue;
            }
            match self.next_hashes.get(&hash) {
                Some(next_hashes) => {
                    queue.extend(next_hashes);
                }
                // This must be a candidate tip.
                None => {
                    tips.insert(hash);
                }
            }
        }

        // Find the longest chain. If there's a tie, use the smaller of the two
        // block hashes, as it implies more work.
        let best_block = tips
            .iter()
            .filter_map(|hash| self.block_id(hash))
            .max_by_key(|id| (id.height, core::cmp::Reverse(id.hash)))
            .expect("failed to find best tip");

        // We have a new tip. Populate the canonical block data by traversing
        // back to the root and collecting block data along the way.
        let mut canonical_values: Vec<(u32, T)> = self
            .iter_block_graph(best_block.hash)
            .take_while(|(_, hash, _)| *hash != root)
            .map(|(height, _, value)| (height, value))
            .collect();

        canonical_values.reverse();
        canonical_values
    }

    /// Create an initial changeset representing the entire block graph.
    ///
    /// This changeset represents the difference between `self` and an empty [`BlockGraph`],
    /// containing all blocks and their parent relationships needed to reconstruct the graph.
    pub fn initial_changeset(&self) -> ChangeSet<T> {
        let mut changeset = ChangeSet::default();

        for (&hash, (height, value)) in &self.blocks {
            changeset.blocks.insert(hash, (*height, value.clone()));
        }

        for (&parent_hash, child_hashes) in &self.next_hashes {
            for &block_hash in child_hashes {
                changeset.edges.insert((parent_hash, block_hash));
            }
        }

        changeset
    }

    /// Find the best parent [`BlockId`] of the given `hash` if it exists in `self.parents`.
    ///
    /// The "best" parent is the one with the highest order [`BlockId`].
    //
    // Open design question: it's not established that "highest order" (by `(height, hash)`) is
    // always the *correct* choice for reconstructing actual ancestry when a block has more than
    // one recorded parent, versus e.g. "whichever parent was most recently recorded". A block
    // with two legitimate, same-height parents at a non-root height isn't currently caught by
    // any validation, so this could in principle pick the "wrong" one without violating
    // `check_invariants`/`check_best_tip`.
    fn parent(&self, hash: &BlockHash) -> Option<BlockId> {
        self.parents.get(hash)?.iter().last().copied()
    }

    /// Iterate over `(height, blockhash, value)` tuples in the [`BlockGraph`] starting from the given [`BlockHash`].
    fn iter_block_graph(&self, hash: BlockHash) -> impl Iterator<Item = (u32, BlockHash, T)> + '_ {
        let mut current_hash = Some(hash);

        core::iter::from_fn(move || {
            let hash = current_hash?;
            let (height, value) = self.blocks.get(&hash).cloned()?;
            current_hash = self.parent(&hash).map(|id| id.hash);
            Some((height, hash, value))
        })
    }

    /// Connects a value of `T` at `height` which is connected to `prev_hash`.
    ///
    /// Note: This method only adds block data to the [`BlockGraph`] without updating the
    /// canonical chain tip. To do that use [`apply_update`](Self::apply_update).
    ///
    /// # Errors
    ///
    /// - If the parent with `prev_hash` doesn't exist at a strictly lower height than
    ///   the `height` being connected. The graph is left unchanged in that case.
    /// - If `height` is directly adjacent to the parent's height and `value` declares a
    ///   `prev_blockhash` that doesn't match `prev_hash`. Gapped connections are unaffected,
    ///   since intermediate blocks are legitimately absent from the graph; likewise, values
    ///   that don't declare a `prev_blockhash` (see [`Block`]) are never rejected
    ///   on this basis.
    pub fn connect_block(
        &mut self,
        height: u32,
        value: T,
        prev_hash: BlockHash,
    ) -> Result<ChangeSet<T>, ConnectBlockError> {
        if !self.is_connected(height, &value, &prev_hash) {
            let hash = value.to_blockhash();
            check_not_second_genesis(self.root, hash, height)?;
            let existing_height = self.blocks.get(&hash).map(|(h, _)| *h);
            check_height_unchanged(hash, height, existing_height)?;
            if existing_height.is_none() {
                self.check_no_child_height_violation(hash, height)?;
            }
            let parent_height = self_parent_height(
                hash,
                prev_hash,
                height,
                self.blocks.get(&prev_hash).map(|(h, _)| *h),
            );
            check_parent_height(height, parent_height)?;
            check_prev_blockhash(height, &value, &prev_hash, parent_height)?;
        }

        Ok(self.connect_block_unchecked(height, value, prev_hash))
    }

    /// Whether `value` at `height` extending from `prev_hash` is already recorded.
    fn is_connected(&self, height: u32, value: &T, prev_hash: &BlockHash) -> bool {
        let hash = value.to_blockhash();

        self.blocks
            .get(&hash)
            .is_some_and(|(existing_height, existing_value)| {
                existing_height == &height && existing_value == value
            })
            // The same parent-child dependency exists
            && self.next_hashes.get(prev_hash).is_some_and(|set| set.contains(&hash))
    }

    /// `hash` may already be recorded as some other block's declared parent via `next_hashes`
    /// (a gapped connection can reference a parent hash before that parent is ever connected as
    /// an actual block). If so, connecting `hash` for the first time at `height` must not
    /// retroactively put it at or above one of those already-connected children — otherwise the
    /// parent/child height invariant would be violated the moment `hash` gets a committed
    /// height. Only relevant the first time `hash` is connected, since its height can never
    /// change afterward (see [`check_height_unchanged`]).
    fn check_no_child_height_violation(
        &self,
        hash: BlockHash,
        height: u32,
    ) -> Result<(), ConnectBlockError> {
        if let Some(children) = self.next_hashes.get(&hash) {
            for &child_hash in children {
                if let Some((child_height, _)) = self.blocks.get(&child_hash) {
                    if height >= *child_height {
                        return Err(ConnectBlockError::ChildHeightNotGreater {
                            hash,
                            height,
                            child_hash,
                            child_height: *child_height,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Connects a value of `T` at `height` to `prev_hash` without validating the parent height.
    ///
    /// The caller must have already validated the connection with [`check_parent_height`].
    fn connect_block_unchecked(
        &mut self,
        height: u32,
        value: T,
        prev_hash: BlockHash,
    ) -> ChangeSet<T> {
        if self.is_connected(height, &value, &prev_hash) {
            return ChangeSet::default();
        }

        let hash = value.to_blockhash();
        let is_new_hash = !self.blocks.contains_key(&hash);
        let parent_opt = self.block_id(&prev_hash);

        let mut changeset = ChangeSet::default();

        // Add block to graph
        self.blocks.insert(hash, (height, value.clone()));
        // Record that this block extends from its parent
        self.next_hashes.entry(prev_hash).or_default().insert(hash);
        // Record prev as a parent of this block
        if let Some(parent_id) = parent_opt {
            self.parents.entry(hash).or_default().insert(parent_id);
        }

        // `hash` may already be recorded as some other block's declared parent from an earlier
        // gapped connection (made before `hash` itself was known). Now that `hash` has a
        // committed height, backfill `parents` for each of those children.
        if is_new_hash {
            if let Some(children) = self.next_hashes.get(&hash).cloned() {
                let this_id = BlockId { height, hash };
                for child in children {
                    self.parents.entry(child).or_default().insert(this_id);
                }
            }
        }

        changeset.blocks.insert(hash, (height, value));
        changeset.edges.insert((prev_hash, hash));

        changeset
    }

    /// Validate that every item of `items` can be connected, assuming the items are
    /// connected in the given order. Adjacent items are additionally validated against
    /// their declared `prev_blockhash` whenever the value provides one.
    fn validate_connections(
        &self,
        items: &[(BlockId, T, BlockHash)],
    ) -> Result<(), ConnectBlockError> {
        // Track each item's staged height so a later item that declares an earlier one as its
        // parent sees its about-to-be-connected height rather than any stale height already in
        // the graph, and so a hash connected twice within the same batch at conflicting heights
        // is caught here rather than silently overwriting the first connection.
        let mut staged = HashMap::<BlockHash, u32>::new();

        for (block_id, value, prev_hash) in items {
            let hash = value.to_blockhash();
            if !self.is_connected(block_id.height, value, prev_hash) {
                check_not_second_genesis(self.root, hash, block_id.height)?;
                let existing_height = staged
                    .get(&hash)
                    .copied()
                    .or_else(|| self.blocks.get(&hash).map(|(h, _)| *h));
                check_height_unchanged(hash, block_id.height, existing_height)?;
                if existing_height.is_none() {
                    self.check_no_child_height_violation(hash, block_id.height)?;
                }
                let parent_height = staged
                    .get(prev_hash)
                    .copied()
                    .or_else(|| self.blocks.get(prev_hash).map(|(height, _)| *height));
                let parent_height =
                    self_parent_height(hash, *prev_hash, block_id.height, parent_height);
                check_parent_height(block_id.height, parent_height)?;
                check_prev_blockhash(block_id.height, value, prev_hash, parent_height)?;
            }
            staged.insert(hash, block_id.height);
        }

        Ok(())
    }

    /// Applies a [`CheckPoint`] update to the [`BlockGraph`] and returns the resulting [`ChangeSet`].
    ///
    /// If the update results in a new best tip, or if a fork is detected, the BlockGraph reconciles
    /// the single canonical chain tip internally. Adjacent blocks are validated against their
    /// declared `prev_blockhash` whenever `T` provides one; gapped connections are unaffected,
    /// since intermediate blocks are legitimately absent from the graph.
    ///
    /// Errors if a checkpoint item doesn't declare its parent and isn't already recorded in
    /// this `BlockGraph` at the same height, indicating no common ancestor was found between
    /// `checkpoint` and this `BlockGraph`.
    /// Or if an item can't be connected due to an invalid parent-child dependency.
    /// All items are validated before any of them are applied, so the graph is left
    /// unchanged if an error is returned.
    pub fn apply_update(
        &mut self,
        checkpoint: CheckPoint<T>,
    ) -> Result<ChangeSet<T>, ApplyUpdateError> {
        let mut items = self.merge_chains(checkpoint);
        items.reverse();

        // Every item must declare its parent, otherwise we don't know where the update connects.
        //
        // The one item that can legitimately lack a parent is the update's *root* checkpoint,
        // which is an anchor rather than something to connect. `merge_chains` diffs the update
        // against the canonical tip chain only, so it surfaces that root whenever the tip chain
        // no longer covers it — but the graph knows about far more blocks than that chain, and
        // if the root is already recorded here at the very same height there is nothing to
        // connect and no edge to add. Drop it instead of erroring; otherwise replaying an update
        // whose base was moved off the tip chain by reconciliation would fail (see
        // `apply_update_is_idempotent_when_reconciliation_moves_the_base_off_the_tip_chain`).
        let items = items
            .into_iter()
            .filter_map(|(block_id, value, prev)| match prev {
                Some(prev_hash) => Some(Ok((block_id, value, prev_hash))),
                None if self.blocks.get(&block_id.hash).map(|(height, _)| *height)
                    == Some(block_id.height) =>
                {
                    None
                }
                None => Some(Err(ApplyUpdateError::MissingParent)),
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.validate_connections(&items)?;

        // Connect each item in ascending height order.
        let mut changeset = ChangeSet::default();
        for (block_id, value, prev_hash) in items {
            changeset.merge(self.connect_block_unchecked(block_id.height, value, prev_hash));
        }

        // Re-canonicalize from the root, not from a narrower "fork point" relative only to this
        // update's own new items: the graph can already contain other, entirely unrelated
        // branches (e.g. left by an earlier `connect_block` call, or a gapped/forward-referenced
        // connection) that such a fork point would never discover, even when they're already the
        // true best chain. This also means a no-op update (contributing nothing new) still
        // surfaces an already-connected better chain that was never promoted.
        //
        // Reuse as much of the current tip chain as still matches the recomputed best chain,
        // rebuilding only from the point where the two actually diverge (or where the best chain
        // extends further) — this both preserves the `eq_ptr` structure-sharing callers may rely
        // on and avoids reallocating the whole chain on every call.
        let items = self.canonicalize(self.root);
        let mut new_tip = self.tip.get(0).expect("tip chain must include the root");
        let mut i = 0;
        while i < items.len() {
            let (height, value) = &items[i];
            match self.tip.get(*height) {
                Some(cp) if cp.hash() == value.to_blockhash() => {
                    new_tip = cp;
                    i += 1;
                }
                _ => break,
            }
        }
        for (height, value) in &items[i..] {
            new_tip = new_tip.insert(*height, value.clone());
        }
        self.tip = new_tip;
        Ok(changeset)
    }

    /// TODO: We should be able to combine two BlockGraphs into one
    #[allow(unused)]
    pub fn apply_graph_update(&mut self, other: Self) {
        todo!();
    }
}

/// Invariant checks used by tests and the fuzz harness.
#[cfg(any(test, fuzzing))]
impl<T: Block + PartialEq + Debug + Clone> BlockGraph<T> {
    /// Check structural invariants that must hold at all times.
    #[doc(hidden)]
    pub fn check_invariants(&self) -> Result<(), alloc::string::String> {
        // 1. `root` exists in `blocks` at height 0.
        match self.blocks.get(&self.root) {
            Some((0, _)) => {}
            Some((height, _)) => {
                return Err(format!(
                    "root {} is recorded at height {height}, expected 0",
                    self.root
                ))
            }
            None => return Err(format!("root {} is missing from blocks", self.root)),
        }

        // 2. every block's value hashes to its own key.
        for (hash, (_, value)) in &self.blocks {
            if value.to_blockhash() != *hash {
                return Err(format!(
                    "block {hash} has a value that hashes to {}",
                    value.to_blockhash()
                ));
            }
        }

        // 3. every edge's child exists in `blocks`; if the parent is known, its height
        // must be strictly less than the child's. The sentinel `BlockHash::all_zeros()`
        // parent of genesis is not itself a block, so it's exempt from the height check.
        for (parent_hash, children) in &self.next_hashes {
            for child_hash in children {
                let (child_height, _) = self.blocks.get(child_hash).ok_or_else(|| {
                    format!("edge ({parent_hash}, {child_hash}): child is missing from blocks")
                })?;
                if let Some((parent_height, _)) = self.blocks.get(parent_hash) {
                    if *parent_height >= *child_height {
                        return Err(format!(
                            "edge ({parent_hash}, {child_hash}): parent height {parent_height} \
                             is not less than child height {child_height}"
                        ));
                    }
                }
            }
        }

        // 4. `parents` is exactly the inverse of `next_hashes` restricted to known parents.
        for (child, ids) in &self.parents {
            for id in ids {
                match self.blocks.get(&id.hash) {
                    Some((height, _)) if *height == id.height => {}
                    Some((height, _)) => {
                        return Err(format!(
                            "parents[{child}] contains {id:?}, but block {} is at height {height}",
                            id.hash
                        ))
                    }
                    None => {
                        return Err(format!(
                            "parents[{child}] contains {id:?}, but that block is missing"
                        ))
                    }
                }
                if !self.next_hashes.get(&id.hash).is_some_and(|s| s.contains(child)) {
                    return Err(format!(
                        "parents[{child}] contains {id:?}, but next_hashes[{}] doesn't contain {child}",
                        id.hash
                    ));
                }
            }
        }
        for (parent_hash, children) in &self.next_hashes {
            if let Some((height, _)) = self.blocks.get(parent_hash) {
                let parent_id = BlockId {
                    height: *height,
                    hash: *parent_hash,
                };
                for child in children {
                    if !self.parents.get(child).is_some_and(|s| s.contains(&parent_id)) {
                        return Err(format!(
                            "edge ({parent_hash}, {child}) exists, but parents[{child}] doesn't contain {parent_id:?}"
                        ));
                    }
                }
            }
        }

        // 5. tip chain sanity: heights strictly decrease to genesis at height 0, and every
        // checkpoint agrees with `blocks`.
        let mut prev_height: Option<u32> = None;
        let mut last = None;
        for cp in self.tip.iter() {
            if let Some(prev_height) = prev_height {
                if cp.height() >= prev_height {
                    return Err(format!(
                        "tip chain heights are not strictly decreasing at height {}",
                        cp.height()
                    ));
                }
            }
            prev_height = Some(cp.height());
            if cp.hash() != cp.value().to_blockhash() {
                return Err(format!(
                    "tip checkpoint at height {} has hash {} but its value hashes to {}",
                    cp.height(),
                    cp.hash(),
                    cp.value().to_blockhash()
                ));
            }
            if !matches!(self.blocks.get(&cp.hash()), Some((height, _)) if *height == cp.height()) {
                return Err(format!("tip checkpoint {:?} doesn't match blocks", cp.block_id()));
            }
            last = Some(cp);
        }
        match last {
            Some(cp) if cp.height() == 0 && cp.hash() == self.root => {}
            Some(cp) => {
                return Err(format!(
                    "tip chain's genesis is {:?}, expected height 0 hash {}",
                    cp.block_id(),
                    self.root
                ))
            }
            None => return Err("tip chain is empty".into()),
        }

        Ok(())
    }

    /// Check the longest-chain rule: only valid after [`apply_update`](Self::apply_update) or
    /// [`from_changeset`](Self::from_changeset), since [`connect_block`](Self::connect_block)
    /// deliberately leaves the tip alone.
    #[doc(hidden)]
    pub fn check_best_tip(&self) -> Result<(), alloc::string::String> {
        // BFS every block reachable from root via `next_hashes`.
        let mut visited = HashSet::<BlockHash>::new();
        let mut queue = vec![self.root];
        let mut best: Option<BlockId> = None;

        while let Some(hash) = queue.pop() {
            if !visited.insert(hash) {
                continue;
            }
            if let Some(id) = self.block_id(&hash) {
                let key = (id.height, core::cmp::Reverse(id.hash));
                if best.is_none_or(|b| (b.height, core::cmp::Reverse(b.hash)) < key) {
                    best = Some(id);
                }
            }
            if let Some(children) = self.next_hashes.get(&hash) {
                queue.extend(children);
            }
        }

        let best = best.ok_or_else(|| format!("no blocks reachable from root {}", self.root))?;
        let tip_id = self.tip.block_id();
        if tip_id != best {
            return Err(format!("tip is {tip_id:?}, but the best reachable block is {best:?}"));
        }
        Ok(())
    }
}

impl<T: Debug + Clone + PartialEq> PartialEq for BlockGraph<T> {
    fn eq(&self, other: &Self) -> bool {
        self.blocks == other.blocks
            && self.parents == other.parents
            && self.next_hashes == other.next_hashes
            && self.root == other.root
            && self.tip == other.tip
    }
}

impl<T: Block + PartialEq + Debug + Clone> BlockGraph<T> {
    /// Get chain tip
    pub fn get_chain_tip(&self) -> BlockId {
        self.tip().block_id()
    }

    /// Is block in chain
    pub fn is_block_in_chain(&self, block: BlockId, chain_tip: BlockId) -> Option<bool> {
        // `block` height must be within that of `chain_tip`.
        if block.height > chain_tip.height {
            return None;
        }
        // `chain_tip` must exist in chain.
        if self
            .tip
            .get(chain_tip.height)
            .is_none_or(|cp| cp.value().to_blockhash() != chain_tip.hash)
        {
            return None;
        }
        // A block of given height must exist in this chain, and the hashes must match.
        self.tip
            .get(block.height)
            .map(|cp| cp.value().to_blockhash() == block.hash)
    }
}

/// A changeset representing modifications to a [`BlockGraph`].
///
/// Contains the set of blocks to be added to the graph, along with their parent relationships.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: for<'d> serde::Deserialize<'d>"
    ))
)]
pub struct ChangeSet<T> {
    /// Map from block hash to `(height, value)`.
    pub blocks: BTreeMap<BlockHash, (u32, T)>,
    /// Set of `(parent_hash, child_hash)` edges.
    pub edges: BTreeSet<(BlockHash, BlockHash)>,
}

impl<T> Default for ChangeSet<T> {
    fn default() -> Self {
        Self {
            blocks: Default::default(),
            edges: Default::default(),
        }
    }
}

impl<T> ChangeSet<T> {
    /// Merge
    pub fn merge(&mut self, other: Self) {
        self.blocks.extend(other.blocks);
        self.edges.extend(other.edges);
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty() && self.edges.is_empty()
    }
}

impl<T> BlockGraph<T>
where
    T: Block + Debug + Clone,
{
    /// This method iterates self and update in tandem backwards from the tip,
    /// and returns the new "items to connect".
    fn merge_chains(&self, update: CheckPoint<T>) -> Vec<(BlockId, T, Option<BlockHash>)> {
        let mut original_iter = self.tip.iter().peekable();
        let mut update_iter = update.iter().peekable();

        let mut items_to_connect = vec![];

        // While there are updates to process, add each item to `items_to_connect` if
        // it doesn't already exist in the original chain.
        loop {
            match (original_iter.peek(), update_iter.peek()) {
                // Compare heights
                (Some(original), Some(update)) => {
                    match update.height().cmp(&original.height()) {
                        // Update is greater
                        Ordering::Greater => {
                            items_to_connect.push((
                                update.block_id(),
                                update.value().clone(),
                                update.prev().as_ref().map(CheckPoint::hash),
                            ));
                            update_iter.next();
                        }
                        // Original is greater
                        Ordering::Less => {
                            original_iter.next();
                        }
                        Ordering::Equal => {
                            // Found an agreement height
                            if original.hash() == update.hash() {
                                // We may be adding a new parent-child dependency if this block already exists in graph
                                let prev_hash = update.prev().as_ref().map(CheckPoint::hash);
                                if let Some(prev_hash) = prev_hash {
                                    if !self.blocks.contains_key(&prev_hash) {
                                        items_to_connect.push((
                                            update.block_id(),
                                            update.value().clone(),
                                            Some(prev_hash),
                                        ));
                                    }
                                }
                                // If the update shares the same Arc pointer we can stop here
                                if original.eq_ptr(update) {
                                    break;
                                }
                            } else {
                                items_to_connect.push((
                                    update.block_id(),
                                    update.value().clone(),
                                    update.prev().as_ref().map(CheckPoint::hash),
                                ));
                            }
                            original_iter.next();
                            update_iter.next();
                        }
                    }
                }
                (_, None) => break,
                (None, Some(..)) => unreachable!("Original can't be exhausted before update"),
            }
        }

        items_to_connect
    }
}

/// Error returned by [`BlockGraph::from_changeset`].
#[derive(Debug, PartialEq)]
pub enum FromChangeSetError {
    /// The changeset contains no block at height 0.
    MissingGenesis,
    /// More than one distinct block is declared at height 0. A graph has exactly one root, so
    /// this is ambiguous: canonicalizing could otherwise walk back through a block's `parents`
    /// into a tree rooted at the block that lost the ambiguity, never reaching the chosen root.
    MultipleGenesisBlocks {
        /// The first height-0 block found (by hash order).
        first: BlockHash,
        /// A second, distinct height-0 block found in the same changeset.
        second: BlockHash,
    },
    /// An edge's declared parent is not at a strictly lower height than its child, which would
    /// otherwise allow cyclic or backwards-height data into the graph.
    InvalidEdgeHeight {
        /// The edge's declared parent.
        parent: BlockId,
        /// The edge's declared child.
        child: BlockId,
    },
    /// An adjacent pair of blocks in the reconstructed canonical chain declares a
    /// `prev_blockhash` (see [`Block`]) that doesn't match its recorded parent. Unlike
    /// [`connect_block`](BlockGraph::connect_block)/[`apply_update`](BlockGraph::apply_update),
    /// `from_changeset` populates the graph directly from possibly-adversarial data, so this
    /// inconsistency isn't caught until the tip chain is rebuilt.
    InconsistentPrevBlockhash {
        /// The block whose declared `prev_blockhash` doesn't match its recorded parent.
        hash: BlockHash,
    },
}

impl Display for FromChangeSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingGenesis => write!(f, "changeset contains no genesis block (height 0)"),
            Self::MultipleGenesisBlocks { first, second } => write!(
                f,
                "changeset declares more than one block at height 0: {first} and {second}"
            ),
            Self::InvalidEdgeHeight { parent, child } => write!(
                f,
                "edge parent {parent:?} is not at a strictly lower height than child {child:?}"
            ),
            Self::InconsistentPrevBlockhash { hash } => write!(
                f,
                "block {hash} declares a prev_blockhash that doesn't match its recorded parent"
            ),
        }
    }
}

impl core::error::Error for FromChangeSetError {}

/// Check that every edge in `changeset` whose parent and child both appear in
/// `changeset.blocks` has a parent height strictly less than the child's height.
///
/// This is the same rule [`check_parent_height`] enforces on [`connect_block`](BlockGraph::connect_block)
/// and [`apply_update`](BlockGraph::apply_update), applied to raw changeset data before it's trusted
/// to build a [`BlockGraph`]. Rejecting these edges up front also rules out cycles (since any
/// cycle must contain at least one edge whose parent height doesn't decrease).
fn check_changeset_edge_heights<T>(changeset: &ChangeSet<T>) -> Result<(), FromChangeSetError> {
    for &(parent_hash, child_hash) in &changeset.edges {
        let parent_height = changeset.blocks.get(&parent_hash).map(|(height, _)| *height);
        let child_height = changeset.blocks.get(&child_hash).map(|(height, _)| *height);
        if let (Some(parent_height), Some(child_height)) = (parent_height, child_height) {
            if parent_height >= child_height {
                return Err(FromChangeSetError::InvalidEdgeHeight {
                    parent: BlockId {
                        height: parent_height,
                        hash: parent_hash,
                    },
                    child: BlockId {
                        height: child_height,
                        hash: child_hash,
                    },
                });
            }
        }
    }
    Ok(())
}

/// The parent of a block must exist at a strictly lower height than the block itself.
fn check_parent_height(height: u32, parent_height: Option<u32>) -> Result<(), ConnectBlockError> {
    match parent_height {
        Some(parent_height) if parent_height >= height => {
            Err(ConnectBlockError::ParentHeightNotSmaller)
        }
        _ => Ok(()),
    }
}

/// Height 0 is reserved for the graph's single root, established by `from_genesis`. An unknown
/// parent never blocks a connection (that's how gapped connections work), so without this check
/// any other hash could also be connected at height 0, creating a second "genesis" that
/// `canonicalize`'s backward walk and `from_changeset`'s `MultipleGenesisBlocks` check both
/// assume can't happen.
fn check_not_second_genesis(
    root: BlockHash,
    hash: BlockHash,
    height: u32,
) -> Result<(), ConnectBlockError> {
    if height == 0 && hash != root {
        return Err(ConnectBlockError::HeightZeroReservedForRoot { hash });
    }
    Ok(())
}

/// A block can't be its own parent. A self-referential `prev_hash` would otherwise slip past
/// [`check_parent_height`], since the block isn't recorded in `self.blocks` yet at validation
/// time (so an unknown parent's height looks like `None`, which always passes) — this treats a
/// self-reference as if the parent were already recorded at `height`, which always fails the
/// strict inequality.
fn self_parent_height(
    hash: BlockHash,
    prev_hash: BlockHash,
    height: u32,
    parent_height: Option<u32>,
) -> Option<u32> {
    if prev_hash == hash {
        Some(height)
    } else {
        parent_height
    }
}

/// Blocks are monotone/append-only: once a hash is recorded, it must always be connected at the
/// same height.
///
/// Without this, reconnecting an already-known hash at a different height would overwrite its
/// recorded height in place.
fn check_height_unchanged(
    hash: BlockHash,
    height: u32,
    existing_height: Option<u32>,
) -> Result<(), ConnectBlockError> {
    match existing_height {
        Some(existing_height) if existing_height != height => {
            Err(ConnectBlockError::HeightConflict {
                hash,
                existing_height,
                new_height: height,
            })
        }
        _ => Ok(()),
    }
}

/// For adjacent blocks (`height == parent_height + 1`), `value`'s declared `prev_blockhash` (if
/// any) must match `prev_hash`. Gapped connections, and values that don't declare a
/// `prev_blockhash`, skip this check.
fn check_prev_blockhash<T: Block>(
    height: u32,
    value: &T,
    prev_hash: &BlockHash,
    parent_height: Option<u32>,
) -> Result<(), ConnectBlockError> {
    if parent_height.map(|h| h + 1) == Some(height)
        && value.prev_blockhash().is_some_and(|hash| hash != *prev_hash)
    {
        return Err(ConnectBlockError::PrevBlockhashMismatch);
    }
    Ok(())
}

/// Error returned by [`BlockGraph::connect_block`].
#[derive(Debug, PartialEq)]
pub enum ConnectBlockError {
    /// The declared parent's height is not strictly less than the new block's height.
    ParentHeightNotSmaller,
    /// An adjacent block's declared `prev_blockhash` doesn't match its parent's hash.
    PrevBlockhashMismatch,
    /// The block's hash is already recorded in the graph at a different height. Blocks are
    /// monotone/append-only, so a hash's height must never change once recorded.
    HeightConflict {
        /// The block's hash.
        hash: BlockHash,
        /// The height already recorded for `hash`.
        existing_height: u32,
        /// The height this connection attempted to record instead.
        new_height: u32,
    },
    /// This block is already recorded as an already-connected child's parent, but the height
    /// declared for it here is not strictly less than that child's height.
    ChildHeightNotGreater {
        /// The block's hash.
        hash: BlockHash,
        /// The height this connection attempted to record.
        height: u32,
        /// The already-connected child that declares `hash` as its parent.
        child_hash: BlockHash,
        /// The child's recorded height.
        child_height: u32,
    },
    /// Height 0 is reserved for the graph's single root; this hash is not the root.
    HeightZeroReservedForRoot {
        /// The block's hash.
        hash: BlockHash,
    },
}

impl Display for ConnectBlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParentHeightNotSmaller => write!(
                f,
                "parent block height must be strictly less than the new block's height",
            ),
            Self::PrevBlockhashMismatch => {
                write!(f, "adjacent block's prev_blockhash does not match its parent's hash",)
            }
            Self::HeightConflict {
                hash,
                existing_height,
                new_height,
            } => write!(
                f,
                "block {hash} is already recorded at height {existing_height}, cannot reconnect at height {new_height}",
            ),
            Self::ChildHeightNotGreater {
                hash,
                height,
                child_hash,
                child_height,
            } => write!(
                f,
                "block {hash} at height {height} is not strictly less than its already-connected child {child_hash} at height {child_height}",
            ),
            Self::HeightZeroReservedForRoot { hash } => write!(
                f,
                "height 0 is reserved for the graph's root; {hash} is not the root",
            ),
        }
    }
}

impl core::error::Error for ConnectBlockError {}

/// Error returned by [`BlockGraph::apply_update`].
#[derive(Debug, PartialEq)]
pub enum ApplyUpdateError {
    /// A block in the update has no declared parent hash and isn't already known to the graph
    /// at that height, so its connection point in the graph cannot be determined.
    MissingParent,
    /// A block could not be connected due to an invalid parent-child height relationship.
    ConnectBlock(ConnectBlockError),
}

impl Display for ApplyUpdateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingParent => write!(
                f,
                "a block in the update has no declared parent; cannot determine connection point",
            ),
            Self::ConnectBlock(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for ApplyUpdateError {}

impl From<ConnectBlockError> for ApplyUpdateError {
    fn from(e: ConnectBlockError) -> Self {
        Self::ConnectBlock(e)
    }
}

#[cfg(test)]
mod test {
    use bitcoin::block::Header;
    use bitcoin::hashes::Hash;
    use bitcoin::{constants, pow, Network, TxMerkleNode};

    use super::*;

    fn checkpoint<T>(blocks: impl IntoIterator<Item = (u32, T)>) -> CheckPoint<T>
    where
        T: Block + Clone + Debug,
    {
        CheckPoint::from_entries(blocks).expect("failed to create CheckPoint")
    }

    fn genesis_header() -> Header {
        constants::genesis_block(Network::Regtest).header
    }

    /// Build a header extending `prev_blockhash`. `nonce` acts as a fork discriminator: headers
    /// with the same `prev_blockhash` but different `nonce` produce distinct hashes.
    fn header(prev_blockhash: BlockHash, nonce: Option<u32>) -> Header {
        Header {
            version: bitcoin::block::Version::default(),
            merkle_root: TxMerkleNode::all_zeros(),
            time: 1234567,
            bits: pow::Target::MAX_ATTAINABLE_REGTEST.to_compact_lossy(),
            nonce: nonce.unwrap_or_default(),
            prev_blockhash,
        }
    }

    /// Push `count` headers onto `cp`, linked in sequence starting at `nonce_start`.
    /// Returns the extended checkpoint along with the pushed headers in ascending height order.
    fn extend_with_headers(
        cp: CheckPoint<Header>,
        count: u32,
        nonce_start: u32,
    ) -> (CheckPoint<Header>, Vec<Header>) {
        let mut pushed = Vec::with_capacity(count as usize);
        let mut cur = cp;
        let start_height = cur.height() + 1;
        for i in 0..count {
            let h = header(cur.hash(), Some(nonce_start + i));
            cur = cur.push(start_height + i, h).unwrap();
            pushed.push(h);
        }
        (cur, pushed)
    }

    #[test]
    fn test_from_genesis() {
        let graph = BlockGraph::from_genesis(genesis_header());
        assert_eq!(graph.blocks.len(), 1);
        assert_eq!(graph.next_hashes.len(), 1);
        assert_eq!(graph.tip.iter().count(), 1);
    }

    #[test]
    fn test_apply_update() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let cp = CheckPoint::new(0, genesis);
        let (cp, headers) = extend_with_headers(cp, 3, 1);
        let [h1, h2, h3] = headers[..] else {
            unreachable!()
        };

        let changeset = graph.apply_update(cp).unwrap();

        // Verify the changeset contains the expected blocks
        assert_eq!(changeset.blocks.len(), 3);
        assert_eq!(
            changeset.blocks,
            [
                (h1.block_hash(), (1, h1)),
                (h2.block_hash(), (2, h2)),
                (h3.block_hash(), (3, h3)),
            ]
            .into()
        );
        assert_eq!(
            changeset.edges,
            [
                (genesis.block_hash(), h1.block_hash()),
                (h1.block_hash(), h2.block_hash()),
                (h2.block_hash(), h3.block_hash()),
            ]
            .into()
        );
    }

    // Test that we can iterate blocks of the main chain, and
    // the blocks are correct.
    #[test]
    fn iter_timechain() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let mut blocks: Vec<BlockId> = vec![BlockId {
            height: 0,
            hash: genesis.block_hash(),
        }];

        let cp = CheckPoint::new(0, genesis);
        let (cp, headers) = extend_with_headers(cp, 3, 1);
        for (height, h) in (1u32..=3).zip(headers) {
            blocks.push(BlockId {
                height,
                hash: h.block_hash(),
            });
        }
        let _ = graph.apply_update(cp).unwrap();

        blocks.reverse();
        let tip_blocks = graph.iter().map(|cp| cp.block_id()).collect::<Vec<_>>();
        assert_eq!(tip_blocks, blocks);
    }

    #[test]
    fn test_initial_changeset() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let cp = CheckPoint::new(0, genesis);
        let (cp, _headers) = extend_with_headers(cp, 3, 1);

        let _ = graph.apply_update(cp).unwrap();

        // Collect the initial changeset
        let init_cs = graph.initial_changeset();
        assert_eq!(init_cs.blocks.len(), 4);

        // Now recover from changeset
        let recovered = BlockGraph::from_changeset(init_cs).unwrap().unwrap();
        assert_eq!(recovered, graph);
    }

    #[test]
    fn test_merge_chains_connect() {
        // case: connect 1
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);
        let (tip, headers) = extend_with_headers(graph.tip(), 1, 1);
        let block_1 = BlockId {
            height: 1,
            hash: headers[0].block_hash(),
        };
        let changeset = graph.apply_update(tip).unwrap();

        assert_eq!(changeset.blocks.len(), 1);
        assert_eq!(changeset.blocks, [(block_1.hash, (block_1.height, headers[0]))].into());
        assert_eq!(changeset.edges, [(genesis.block_hash(), block_1.hash)].into());
    }

    #[test]
    fn test_merge_chains_connect_two() {
        // case: connect 2
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);
        let (tip, headers) = extend_with_headers(graph.tip(), 2, 1);
        let block_1 = BlockId {
            height: 1,
            hash: headers[0].block_hash(),
        };
        let block_2 = BlockId {
            height: 2,
            hash: headers[1].block_hash(),
        };
        let changeset = graph.apply_update(tip).unwrap();

        assert_eq!(changeset.blocks.len(), 2);
        assert_eq!(
            changeset.blocks,
            [
                (block_1.hash, (block_1.height, headers[0])),
                (block_2.hash, (block_2.height, headers[1])),
            ]
            .into(),
        );
        assert_eq!(
            changeset.edges,
            [(genesis.block_hash(), block_1.hash), (block_1.hash, block_2.hash)].into(),
        );
    }

    #[test]
    fn test_is_block_in_chain() {
        let genesis = genesis_header();
        let h1 = header(genesis.block_hash(), Some(1));
        let h2 = header(h1.block_hash(), Some(2));
        let genesis_block = BlockId {
            height: 0,
            hash: genesis.block_hash(),
        };
        let block_1 = BlockId {
            height: 1,
            hash: h1.block_hash(),
        };
        let block_2 = BlockId {
            height: 2,
            hash: h2.block_hash(),
        };
        let changeset = ChangeSet {
            blocks: [
                (genesis_block.hash, (genesis_block.height, genesis)),
                (block_1.hash, (block_1.height, h1)),
                (block_2.hash, (block_2.height, h2)),
            ]
            .into(),
            edges: [
                (BlockHash::all_zeros(), genesis_block.hash),
                (genesis_block.hash, block_1.hash),
                (block_1.hash, block_2.hash),
            ]
            .into(),
        };
        let graph = BlockGraph::from_changeset(changeset).unwrap().unwrap();
        let chain_tip = graph.tip().block_id();
        assert_eq!(chain_tip, block_2);
        for block in [genesis_block, block_1, block_2] {
            assert!(matches!(graph.is_block_in_chain(block, chain_tip), Some(true)))
        }
        // A different header at height 2 (wrong hash) can't be in chain.
        let h2_wrong = header(h1.block_hash(), Some(99));
        assert!(
            matches!(
                graph.is_block_in_chain(
                    BlockId {
                        height: 2,
                        hash: h2_wrong.block_hash()
                    },
                    chain_tip
                ),
                Some(false)
            ),
            "block of wrong hash cannot be in chain"
        );
        let h3 = header(h2.block_hash(), Some(3));
        assert!(
            graph
                .is_block_in_chain(
                    BlockId {
                        height: 3,
                        hash: h3.block_hash()
                    },
                    chain_tip
                )
                .is_none(),
            "block height past tip cannot be in chain"
        );
    }

    #[test]
    fn insert_older_block_should_be_canonical() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::<Header>::from_genesis(genesis);

        // Build the full chain of headers 1..=5, properly linked.
        let mut full_chain = vec![genesis];
        let mut prev_hash = genesis.block_hash();
        for height in 1u32..=5 {
            let h = header(prev_hash, Some(height));
            prev_hash = h.block_hash();
            full_chain.push(h);
        }

        let mut cp = CheckPoint::new(0, genesis);

        // Add blocks to graph, leaving a gap at height = 1
        for (height, &h) in (2u32..=5).zip(&full_chain[2..=5]) {
            cp = cp.insert(height, h);
        }
        let _ = graph.apply_update(cp.clone()).unwrap();

        // Now insert block 1
        let h1 = full_chain[1];
        let block_1 = BlockId {
            height: 1,
            hash: h1.block_hash(),
        };
        cp = cp.insert(1, h1);
        let block_2 = cp.get(2).map(|cp| cp.block_id()).expect("block_2 should exist in CP");
        let changeset = graph.apply_update(cp).unwrap();

        // Verify changeset contains the expected block
        assert_eq!(
            changeset.blocks,
            [
                (block_1.hash, (block_1.height, h1)),
                (block_2.hash, (block_2.height, full_chain[2])),
            ]
            .into(),
            "Expected changeset to contain blocks 1 and 2"
        );
        assert_eq!(
            changeset.edges,
            [(genesis.block_hash(), block_1.hash), (block_1.hash, block_2.hash)].into(),
            "Expected changeset to contain edges for blocks 1 and 2"
        );

        // Canonical chain should remain unchanged
        let test_blocks = graph.iter().map(|cp| cp.block_id()).collect::<Vec<_>>();
        assert_eq!(test_blocks.len(), 6);
        for i in 0..=5 {
            let expected_height = i as u32;
            let expected_hash = full_chain[i].block_hash();
            let test_block = test_blocks[test_blocks.len() - 1 - i];
            assert_eq!(test_block.height, expected_height);
            assert_eq!(test_block.hash, expected_hash);
        }
    }

    #[test]
    fn test_apply_update_single_block() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let (cp, headers) = extend_with_headers(graph.tip(), 1, 1);
        let h1 = headers[0];

        let changeset = graph.apply_update(cp).unwrap();

        assert_eq!(graph.tip().height(), 1);
        assert_eq!(graph.tip().hash(), h1.block_hash());
        assert_eq!(graph.blocks.len(), 2);

        // Verify changeset
        assert_eq!(changeset.blocks.len(), 1);
        assert_eq!(changeset.blocks, [(h1.block_hash(), (1u32, h1))].into());
        assert_eq!(changeset.edges, [(genesis.block_hash(), h1.block_hash())].into());
    }

    #[test]
    fn test_apply_update_multiple_blocks() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let (cp, headers) = extend_with_headers(graph.tip(), 3, 1);
        let [h1, h2, h3] = headers[..] else {
            unreachable!()
        };

        let changeset = graph.apply_update(cp).unwrap();

        assert_eq!(graph.tip().height(), 3);
        assert_eq!(graph.blocks.len(), 4);

        // Verify changeset contains all 3 blocks
        assert_eq!(changeset.blocks.len(), 3);
        let expected_blocks = [
            (h1.block_hash(), (1u32, h1)),
            (h2.block_hash(), (2u32, h2)),
            (h3.block_hash(), (3u32, h3)),
        ];
        assert_eq!(changeset.blocks, expected_blocks.into());
        let expected_edges = [
            (genesis.block_hash(), h1.block_hash()),
            (h1.block_hash(), h2.block_hash()),
            (h2.block_hash(), h3.block_hash()),
        ];
        assert_eq!(changeset.edges, expected_edges.into());
    }

    #[test]
    fn test_apply_update_maintains_chain_integrity() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let (cp, headers) = extend_with_headers(graph.tip(), 2, 1);
        let [h1, h2] = headers[..] else {
            unreachable!()
        };

        graph.apply_update(cp).unwrap();

        // Verify parent-child relationships
        assert!(graph
            .next_hashes
            .get(&genesis.block_hash())
            .unwrap()
            .contains(&h1.block_hash()));
        assert!(graph
            .next_hashes
            .get(&h1.block_hash())
            .unwrap()
            .contains(&h2.block_hash()));
    }

    #[test]
    fn test_apply_update_updates_tip() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let (cp, headers) = extend_with_headers(graph.tip(), 1, 1);

        graph.apply_update(cp).unwrap();

        let tip_id = graph.tip().block_id();
        assert_eq!(tip_id.height, 1);
        assert_eq!(tip_id.hash, headers[0].block_hash());
    }

    #[test]
    fn test_apply_update_extended_chain() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        // First update
        let (cp1, headers1) = extend_with_headers(graph.tip(), 2, 1);
        let changeset1 = graph.apply_update(cp1).unwrap();

        assert_eq!(changeset1.blocks.len(), 2);

        // Second update extending from previous tip
        let (cp2, headers2) = extend_with_headers(graph.tip(), 1, 3);
        let h3 = headers2[0];
        let changeset2 = graph.apply_update(cp2).unwrap();

        assert_eq!(graph.tip().height(), 3);
        assert_eq!(graph.blocks.len(), 4);

        // Verify second changeset
        assert_eq!(changeset2.blocks.len(), 1);
        assert_eq!(changeset2.blocks, [(h3.block_hash(), (3u32, h3))].into());
        assert_eq!(changeset2.edges, [(headers1[1].block_hash(), h3.block_hash())].into());
    }

    #[test]
    fn test_connect_block() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let h1 = header(genesis.block_hash(), Some(1));
        let h2 = header(h1.block_hash(), Some(2));

        // Connect block 1
        let cs = graph.connect_block(1, h1, genesis.block_hash()).unwrap();
        assert_eq!(cs.blocks, [(h1.block_hash(), (1u32, h1))].into());
        assert_eq!(cs.edges, [(genesis.block_hash(), h1.block_hash())].into());
        assert!(
            graph
                .blocks
                .get(&h1.block_hash())
                .is_some_and(|(height, value)| height == &1 && value == &h1),
            "Block 1 should exist"
        );
        assert!(
            graph
                .next_hashes
                .get(&genesis.block_hash())
                .unwrap()
                .contains(&h1.block_hash()),
            "Hash 1 should extend from hash 0"
        );
        assert_eq!(
            graph.parent(&h1.block_hash()).unwrap(),
            (0, genesis.block_hash()).into(),
            "Block 0 should be parent of block 1"
        );

        // Connect block 2
        let cs = graph.connect_block(2, h2, h1.block_hash()).unwrap();
        assert_eq!(cs.blocks, [(h2.block_hash(), (2u32, h2))].into());
        assert_eq!(cs.edges, [(h1.block_hash(), h2.block_hash())].into());
        assert!(
            graph
                .blocks
                .get(&h2.block_hash())
                .is_some_and(|(height, value)| height == &2 && value == &h2),
            "Block 2 should exist"
        );
        assert!(
            graph
                .next_hashes
                .get(&h1.block_hash())
                .unwrap()
                .contains(&h2.block_hash()),
            "Hash 2 should extend from hash 1"
        );
        assert_eq!(
            graph.parent(&h2.block_hash()).unwrap(),
            (1, h1.block_hash()).into(),
            "Block 1 should be parent of block 2"
        );
    }

    #[test]
    fn test_canonicalize_selects_longest_chain() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        // Add two competing chains
        let h1a = header(genesis.block_hash(), Some(1));
        let h2a = header(h1a.block_hash(), Some(2));
        let h1b = header(genesis.block_hash(), Some(101));

        let _ = graph
            .apply_update(checkpoint([(0, genesis), (1, h1a), (2, h2a)]))
            .unwrap();

        // Add shorter competing chain
        let _ = graph.apply_update(checkpoint([(0, genesis), (1, h1b)])).unwrap();

        // Longer chain should be canonical
        assert_eq!(graph.tip().block_id(), (2, h2a.block_hash()).into());
    }

    #[test]
    fn test_merge_chains_with_gap() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let h1 = header(genesis.block_hash(), Some(1));
        let h2 = header(h1.block_hash(), Some(2));
        let h3 = header(h2.block_hash(), Some(3));
        let cp = graph.tip().insert(3, h3);

        let changeset = graph.apply_update(cp).unwrap();

        // Should create a single block at height 3 connecting to genesis
        assert_eq!(changeset.blocks.len(), 1);
        assert_eq!(changeset.blocks, [(h3.block_hash(), (3u32, h3))].into());
        assert_eq!(changeset.edges, [(genesis.block_hash(), h3.block_hash())].into());
    }

    #[test]
    fn test_connect_same_block_twice_ok() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let h1 = header(genesis.block_hash(), Some(1));
        graph.connect_block(1, h1, genesis.block_hash()).unwrap();

        assert_eq!(graph.blocks.len(), 2);

        // Connecting same block again should be ok
        let cs = graph.connect_block(1, h1, genesis.block_hash()).unwrap();
        assert!(
            cs.is_empty(),
            "Same block connection should return an empty change set"
        );
    }

    #[test]
    fn test_block_id_retrieval() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let h1 = header(genesis.block_hash(), Some(1));
        graph.connect_block(1, h1, genesis.block_hash()).unwrap();

        let block_id = graph.block_id(&h1.block_hash()).unwrap();
        assert_eq!(block_id.height, 1);
        assert_eq!(block_id.hash, h1.block_hash());
    }

    #[test]
    fn test_block_id_nonexistent() {
        let genesis = genesis_header();
        let graph = BlockGraph::from_genesis(genesis);

        let nonexistent = header(genesis.block_hash(), Some(1));
        assert!(graph.block_id(&nonexistent.block_hash()).is_none());
    }

    #[test]
    fn test_genesis_block_retrieval() {
        let genesis = genesis_header();
        let graph = BlockGraph::from_genesis(genesis);

        assert_eq!(graph.genesis_block(), genesis);
    }

    #[test]
    fn test_range_blocks() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let (cp, _headers) = extend_with_headers(graph.tip(), 5, 1);

        let _ = graph.apply_update(cp);

        let range_items: BTreeSet<u32> = graph.range(2..=4).map(|cp| cp.height()).collect();
        assert_eq!(range_items, [2, 3, 4].into());
    }

    #[test]
    fn test_switch_forks() {
        let hash_0 = genesis_header();
        let hash_1 = header(hash_0.block_hash(), Some(1));
        let hash_2 = header(hash_1.block_hash(), Some(2));
        let hash_3 = header(hash_2.block_hash(), Some(3));
        let hash_3_alt = header(hash_2.block_hash(), Some(103));
        let hash_4_alt = header(hash_3_alt.block_hash(), Some(104));
        // Create blockgraph
        let changeset = ChangeSet {
            blocks: [
                (hash_0.block_hash(), (0u32, hash_0)),
                (hash_1.block_hash(), (1u32, hash_1)),
                (hash_2.block_hash(), (2u32, hash_2)),
                (hash_3.block_hash(), (3u32, hash_3)),
            ]
            .into(),
            edges: [
                (BlockHash::all_zeros(), hash_0.block_hash()),
                (hash_0.block_hash(), hash_1.block_hash()),
                (hash_1.block_hash(), hash_2.block_hash()),
                (hash_2.block_hash(), hash_3.block_hash()),
            ]
            .into(),
        };
        let mut graph = BlockGraph::from_changeset(changeset).unwrap().unwrap();

        // connect 3-alternate
        let _ = graph.apply_update(checkpoint([(2, hash_2), (3, hash_3_alt)])).unwrap();
        // Chain tip should change if hash_3_alt is smaller
        if hash_3_alt.block_hash() < hash_3.block_hash() {
            assert_eq!(graph.tip.hash(), hash_3_alt.block_hash());
        } else {
            assert_eq!(graph.tip.hash(), hash_3.block_hash());
        }

        // Now extend competing branch, we should correctly switch forks
        let _ = graph
            .apply_update(checkpoint([(2, hash_2), (3, hash_3_alt), (4, hash_4_alt)]))
            .unwrap();
        assert_eq!(graph.tip.hash(), hash_4_alt.block_hash());
    }

    #[test]
    fn test_apply_update_eq_ptr_optimization() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);
        const COUNT: u32 = 10;
        const INIT_HEIGHT: u32 = COUNT - 1;

        // Create a block graph
        let (cp, _headers) = extend_with_headers(graph.tip(), INIT_HEIGHT, 1);
        let _ = graph.apply_update(cp).unwrap();

        assert_eq!(graph.tip().height(), INIT_HEIGHT);
        assert_eq!(graph.blocks.len() as u32, COUNT);

        // Create update checkpoint based on clone of original graph.tip() and extend by 1 block
        let original_tip = graph.tip();
        let original_tip_hash = original_tip.hash();
        let new_height = INIT_HEIGHT + 1;
        let new_header = header(original_tip_hash, Some(new_height));
        let new_block_hash = new_header.block_hash();
        let update_checkpoint = original_tip
            .push(new_height, new_header)
            .expect("should be able to extend chain");

        // Record state before update
        let blocks_before = graph.blocks.len();
        let tip_height_before = graph.tip().height();

        // Apply the update - this should only connect 1 new block due to eq_ptr optimization
        let items_to_connect = graph.merge_chains(update_checkpoint.clone());
        assert_eq!(items_to_connect.len(), 1);
        let changeset = graph.apply_update(update_checkpoint).unwrap();

        // Verify that only 1 new block was processed in the changeset
        assert_eq!(
            changeset.blocks.len(),
            1,
            "Changeset should contain exactly 1 new block due to eq_ptr optimization"
        );

        // Verify the changeset contains the correct block
        assert_eq!(
            changeset.blocks,
            [(new_block_hash, (new_height, new_header))].into(),
            "Changeset should contain the new block connecting to the original tip"
        );
        assert_eq!(
            changeset.edges,
            [(original_tip_hash, new_block_hash)].into(),
            "Changeset should contain the edge connecting to the original tip"
        );

        // Verify that only 1 new block was processed
        assert_eq!(
            graph.blocks.len(),
            blocks_before + 1,
            "Should have added exactly 1 new block"
        );
        assert_eq!(
            graph.tip().height(),
            tip_height_before + 1,
            "Tip height should increase by 1"
        );
        assert_eq!(graph.tip().height(), new_height);
        assert_eq!(
            graph.tip().hash(),
            new_block_hash,
            "New tip should have the correct hash"
        );

        // Verify the chain integrity - should contain all (COUNT + 1) blocks
        assert_eq!(
            graph.iter().count(),
            (COUNT + 1) as usize,
            "Chain should contain expected checkpoints"
        );

        // Verify heights are correct in descending order
        for (i, checkpoint) in graph.iter().enumerate() {
            assert_eq!(
                checkpoint.height(),
                new_height - i as u32,
                "Heights should be in descending order"
            );
        }
    }

    #[test]
    fn test_apply_update_missing_parent_error() {
        // A checkpoint whose tail block has no common ancestor with the graph
        // should return ApplyUpdateError::MissingParent.
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        // Single-block checkpoint with no chain back to genesis
        let orphan = header(BlockHash::all_zeros(), Some(999));
        let cp = CheckPoint::new(1, orphan);

        assert_eq!(graph.apply_update(cp), Err(ApplyUpdateError::MissingParent));
    }

    #[test]
    fn test_apply_update_error_leaves_graph_unchanged() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let (cp, _headers) = extend_with_headers(graph.tip(), 3, 1);
        graph.apply_update(cp).unwrap();

        let before = graph.clone();
        let before_parents = graph.parents.clone();

        let orphan = header(BlockHash::all_zeros(), Some(999));
        assert_eq!(
            graph.apply_update(CheckPoint::new(4, orphan)),
            Err(ApplyUpdateError::MissingParent),
        );

        assert_eq!(graph, before, "a failed update must not mutate the graph");
        assert_eq!(graph.parents, before_parents);
        assert!(!graph.blocks.contains_key(&orphan.block_hash()));
    }

    #[test]
    fn test_connect_block_error_leaves_graph_unchanged() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        // A block at height 5 that can't be the parent of a lower block.
        let h5 = header(genesis.block_hash(), Some(5));
        graph.connect_block(5, h5, genesis.block_hash()).unwrap();

        let before = graph.clone();
        let before_parents = graph.parents.clone();

        let h3 = header(h5.block_hash(), Some(3));
        assert_eq!(
            graph.connect_block(3, h3, h5.block_hash()),
            Err(ConnectBlockError::ParentHeightNotSmaller),
        );

        assert_eq!(graph, before, "a failed connection must not mutate the graph");
        assert_eq!(graph.parents, before_parents);
        assert!(!graph.blocks.contains_key(&h3.block_hash()));
    }

    #[test]
    fn connect_block_rejects_reconnecting_root_at_a_different_height() {
        // Regression test: reconnecting an already-known hash at a
        // different height silently overwrote its recorded height in `self.blocks`, corrupting
        // `self.root`'s own invariant (root must always be at height 0) when the hash happened
        // to be genesis. Blocks are monotone/append-only, so this must be rejected instead.
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);
        let before = graph.clone();

        let unrelated_parent = header(BlockHash::all_zeros(), Some(1)).block_hash();
        assert_eq!(
            graph.connect_block(65535, genesis, unrelated_parent),
            Err(ConnectBlockError::HeightConflict {
                hash: genesis.block_hash(),
                existing_height: 0,
                new_height: 65535,
            }),
        );
        assert_eq!(graph, before, "a rejected reconnection must not mutate the graph");
        assert_eq!(
            graph.blocks.get(&genesis.block_hash()).map(|(h, _)| *h),
            Some(0),
            "root must remain at height 0"
        );
    }

    #[test]
    fn connect_block_rejects_reconnecting_known_hash_at_a_different_height() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let h1 = header(genesis.block_hash(), Some(1));
        graph.connect_block(1, h1, genesis.block_hash()).unwrap();
        let before = graph.clone();

        // Same hash (h1), but declared at height 2 this time instead of 1.
        assert_eq!(
            graph.connect_block(2, h1, genesis.block_hash()),
            Err(ConnectBlockError::HeightConflict {
                hash: h1.block_hash(),
                existing_height: 1,
                new_height: 2,
            }),
        );
        assert_eq!(graph, before, "a rejected reconnection must not mutate the graph");
    }

    #[test]
    fn apply_update_rejects_same_hash_at_conflicting_heights_in_one_batch() {
        // A caller can construct a `CheckPoint` chain with the same header value pushed at two
        // different heights, provided the second push is a gap (`push` only validates adjacent
        // links, not hash uniqueness across the whole chain), so this must be caught during
        // `apply_update` too, not just `connect_block`.
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let h1 = header(genesis.block_hash(), Some(1));
        let cp = graph
            .tip()
            .push(1, h1)
            .unwrap()
            .push(3, h1) // same value as height 1, now claimed at height 3 (a gap)
            .unwrap();

        let before = graph.clone();
        assert_eq!(
            graph.apply_update(cp),
            Err(ApplyUpdateError::ConnectBlock(ConnectBlockError::HeightConflict {
                hash: h1.block_hash(),
                existing_height: 1,
                new_height: 3,
            })),
        );
        assert_eq!(graph, before, "a failed apply_update must not mutate the graph");
    }

    #[test]
    fn connect_block_rejects_self_referential_parent() {
        // Regression test: a block declaring itself as its own parent
        // (`prev_hash == value.to_blockhash()`) slipped past `check_parent_height`, since the
        // block isn't in `self.blocks` yet at validation time, so its own (about-to-exist)
        // height looked like an unknown parent — which `check_parent_height` always allows.
        // This created an immediate self-loop cycle via the safe `connect_block` API.
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);
        let before = graph.clone();

        let h1 = header(genesis.block_hash(), Some(1));
        assert_eq!(
            graph.connect_block(1, h1, h1.block_hash()),
            Err(ConnectBlockError::ParentHeightNotSmaller),
        );
        assert_eq!(
            graph, before,
            "a rejected self-referential connect must not mutate the graph"
        );
    }

    #[test]
    fn connect_block_rejects_retroactive_child_height_violation() {
        // Regression test: connecting a block under an as-yet-unknown
        // parent hash is legal (a gapped/forward reference), but if that parent hash later gets
        // connected as an actual block, its height must still be strictly less than every
        // already-connected child that named it as parent. Without this check, the parent could
        // retroactively be assigned a height at or above its child's, corrupting the
        // parent/child height invariant everywhere else in the crate relies on.
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let parent_header = header(BlockHash::all_zeros(), Some(1));
        let unknown_parent = parent_header.block_hash();
        let child = header(unknown_parent, Some(2));
        graph.connect_block(10, child, unknown_parent).unwrap();

        let before = graph.clone();
        // Now connect `unknown_parent` for real, at a height that isn't less than its
        // already-recorded child's height (10).
        assert_eq!(
            graph.connect_block(10, parent_header, genesis.block_hash()),
            Err(ConnectBlockError::ChildHeightNotGreater {
                hash: unknown_parent,
                height: 10,
                child_hash: child.block_hash(),
                child_height: 10,
            }),
        );
        assert_eq!(graph, before, "a rejected connection must not mutate the graph");
    }

    #[test]
    fn connect_block_backfills_parents_for_a_retroactively_known_parent() {
        // Regression test: a child connected under an as-yet-unknown
        // parent hash (a gapped/forward reference) never had its `parents` entry backfilled once
        // that parent was later connected for real, leaving `parent()`'s backward walk unable to
        // find a parent that actually exists. `check_invariants` catches the resulting mismatch
        // between `next_hashes` and `parents`.
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        // `parent_header` genuinely extends genesis, so connecting it adjacent to genesis below
        // passes prev_blockhash validation; only the forward-referenced `child` connection is a gap.
        let parent_header = header(genesis.block_hash(), Some(1));
        let parent_hash = parent_header.block_hash();
        let child = header(parent_hash, Some(2));
        graph.connect_block(5, child, parent_hash).unwrap();

        // Now connect `parent_hash` for real, at a height that's still valid for the child.
        graph.connect_block(1, parent_header, genesis.block_hash()).unwrap();
        graph.check_invariants().unwrap();

        assert_eq!(
            graph
                .parents
                .get(&child.block_hash())
                .map(|s| s.iter().copied().collect::<Vec<_>>()),
            Some(vec![BlockId {
                height: 1,
                hash: parent_hash
            }]),
            "child's parents entry should be backfilled once the parent becomes known"
        );
    }

    #[test]
    fn apply_update_promotes_a_better_chain_left_by_connect_block_even_on_a_no_op_update() {
        // Regression test: `apply_update` only re-canonicalized starting
        // from the items *it* connected (`from_hash`); if the update contributed nothing new
        // (e.g. it just re-affirms the current tip), reconciliation was skipped entirely — even
        // though an earlier `connect_block` call (which deliberately never moves the tip) can
        // have already left a strictly better chain sitting in the graph, still unpromoted.
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let h1 = header(genesis.block_hash(), Some(1));
        graph.connect_block(1, h1, genesis.block_hash()).unwrap();
        assert_eq!(
            graph.tip().hash(),
            genesis.block_hash(),
            "connect_block must not move the tip"
        );

        // A no-op update: just the current tip, contributing no new items.
        let cp = CheckPoint::new(0, genesis);
        let changeset = graph.apply_update(cp).unwrap();
        assert!(changeset.is_empty());

        graph.check_invariants().unwrap();
        graph.check_best_tip().unwrap();
        assert_eq!(
            graph.tip().hash(),
            h1.block_hash(),
            "should promote the already-connected h1"
        );
    }

    #[test]
    fn apply_update_reorg_to_a_taller_sibling_does_not_fabricate_ancestry() {
        // Regression test: reconciliation rebuilt the tip by calling
        // `insert` on the *current* tip, which splices by height alone. That's only correct
        // when the new best chain actually descends from the old tip; if it's really a taller
        // *sibling* branch (its true ancestor is much further back), `insert` would instead
        // graft it onto the old tip's own lineage, fabricating ancestry that `next_hashes`/
        // `parents` never recorded — caught here by comparing against the `from_changeset`
        // roundtrip, which rebuilds `tip` from the graph's real edges rather than by splicing.
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        // A short chain that becomes the tip.
        let short = header(genesis.block_hash(), Some(1));
        graph
            .apply_update(CheckPoint::new(0, genesis).push(1, short).unwrap())
            .unwrap();
        assert_eq!(graph.tip().hash(), short.block_hash());

        // A taller, unrelated sibling connected directly under genesis (not under `short`).
        let tall = header(genesis.block_hash(), Some(2));
        graph.connect_block(100, tall, genesis.block_hash()).unwrap();
        assert_eq!(
            graph.tip().hash(),
            short.block_hash(),
            "connect_block must not move the tip"
        );

        // A no-op update should still discover and correctly promote `tall`.
        graph.apply_update(CheckPoint::new(0, genesis)).unwrap();

        assert_eq!(graph.tip().hash(), tall.block_hash());
        assert_eq!(graph.tip().height(), 100);
        assert_eq!(
            graph.tip().prev().map(|cp| cp.hash()),
            Some(genesis.block_hash()),
            "tall's real parent is genesis directly, not short"
        );
        graph.check_invariants().unwrap();
        graph.check_best_tip().unwrap();

        let recovered = BlockGraph::from_changeset(graph.initial_changeset()).unwrap().unwrap();
        assert_eq!(
            recovered, graph,
            "roundtrip must match: tip must reflect real ancestry"
        );
    }

    #[test]
    fn connect_block_rejects_second_block_at_height_zero() {
        // Regression test: height 0 is reserved for the graph's single
        // root, but nothing stopped a *different* hash from also being connected at height 0
        // under an unknown parent (unknown parents never block a connection, by design, for
        // gapped connections). That silently created a second "genesis", later rejected only on
        // a `from_changeset` roundtrip via `MultipleGenesisBlocks`, i.e. too late, since the graph
        // itself was already broken.
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);
        let before = graph.clone();

        let other = header(BlockHash::all_zeros(), Some(1));
        assert_eq!(
            graph.connect_block(0, other, BlockHash::all_zeros()),
            Err(ConnectBlockError::HeightZeroReservedForRoot {
                hash: other.block_hash()
            }),
        );
        assert_eq!(graph, before, "a rejected connection must not mutate the graph");
    }

    #[test]
    fn apply_update_finds_best_chain_on_an_unrelated_third_branch() {
        // Regression test: reconciliation used to search for a "fork
        // point" only relative to the update's own new items (via a backward walk from the new
        // hash), so it could miss an entirely unrelated, already-better branch elsewhere in the
        // graph — e.g. one connected by an earlier, unrelated update. Re-canonicalizing must
        // always search the whole graph from the root.
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        // Branch A: becomes the tip.
        let a = header(genesis.block_hash(), Some(1));
        graph
            .apply_update(CheckPoint::new(0, genesis).push(1, a).unwrap())
            .unwrap();
        assert_eq!(graph.tip().hash(), a.block_hash());

        // Branch B: taller than A, connected directly under genesis (a sibling of A).
        let b = header(genesis.block_hash(), Some(2));
        graph.connect_block(500, b, genesis.block_hash()).unwrap();
        assert_eq!(
            graph.tip().hash(),
            a.block_hash(),
            "connect_block must not move the tip"
        );

        // Branch C: a small, unrelated update that itself contributes a shorter chain than B.
        let c = header(a.block_hash(), Some(3));
        graph.apply_update(CheckPoint::new(1, a).push(2, c).unwrap()).unwrap();

        // Despite C being the most recently touched branch, B (found via a full search from
        // root) is still the true best chain.
        assert_eq!(graph.tip().hash(), b.block_hash());
        assert_eq!(graph.tip().height(), 500);
        graph.check_invariants().unwrap();
        graph.check_best_tip().unwrap();
    }

    #[test]
    fn connect_block_validates_adjacent_ok() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let h1 = header(genesis.block_hash(), Some(1));
        let cs = graph.connect_block(1, h1, genesis.block_hash()).unwrap();
        assert_eq!(cs.blocks, [(h1.block_hash(), (1u32, h1))].into());
    }

    #[test]
    fn connect_block_rejects_adjacent_prev_blockhash_mismatch() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        // h1's prev_blockhash points elsewhere, but we claim it connects to genesis at height 1.
        let unrelated = header(BlockHash::all_zeros(), Some(0)).block_hash();
        let h1 = header(unrelated, Some(1));

        let before = graph.clone();
        assert_eq!(
            graph.connect_block(1, h1, genesis.block_hash()),
            Err(ConnectBlockError::PrevBlockhashMismatch),
        );
        assert_eq!(graph, before, "a failed connection must not mutate the graph");
    }

    #[test]
    fn connect_block_gap_skips_prev_blockhash_check() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        // h3's prev_blockhash doesn't point to genesis, but height 3 is a gapped (non-adjacent)
        // connection to genesis, so the linkage check is skipped.
        let unrelated = header(BlockHash::all_zeros(), Some(0)).block_hash();
        let h3 = header(unrelated, Some(3));

        let cs = graph.connect_block(3, h3, genesis.block_hash()).unwrap();
        assert_eq!(cs.blocks, [(h3.block_hash(), (3u32, h3))].into());
    }

    #[test]
    fn connect_block_skips_prev_blockhash_check_for_unvalidatable_type() {
        // `BlockHash` never declares a `prev_blockhash` (`Block::prev_blockhash` returns `None`),
        // so an adjacent connection is accepted even when it couldn't possibly be verified.
        let genesis: BlockHash = Hash::hash(b"genesis");
        let mut graph = BlockGraph::from_genesis(genesis);

        let h1: BlockHash = Hash::hash(b"1");
        let cs = graph.connect_block(1, h1, genesis).unwrap();
        assert_eq!(cs.blocks, [(h1, (1u32, h1))].into());
    }

    #[test]
    fn apply_update_validates_adjacent_ok() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let (cp, _headers) = extend_with_headers(graph.tip(), 3, 1);
        let changeset = graph.apply_update(cp).unwrap();
        assert_eq!(changeset.blocks.len(), 3);
        assert_eq!(graph.tip().height(), 3);
    }

    #[test]
    fn apply_update_input_cannot_carry_a_broken_prev_blockhash_link() {
        // `CheckPoint::push`/`insert` now validate `prev_blockhash` automatically whenever a
        // value declares one (see `checkpoint::Block`), so a `CheckPoint<Header>` chain can no
        // longer be built with a broken adjacent link in the first place — there's nothing left
        // for `apply_update`'s own `check_prev_blockhash` to catch that wasn't already rejected
        // at construction time. `connect_block`'s check remains reachable, since it takes a raw
        // `prev_hash` directly rather than a validated `CheckPoint`.
        let genesis = genesis_header();
        let h1 = header(genesis.block_hash(), Some(1));
        let broken_h2 = header(BlockHash::all_zeros(), Some(2));

        let err = CheckPoint::new(0, genesis)
            .push(1, h1)
            .unwrap()
            .push(2, broken_h2)
            .unwrap_err();
        assert_eq!(err.value, broken_h2);
    }

    #[test]
    fn apply_update_allows_gap_without_prev_blockhash_check() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        // Height 3 is a gapped connection to genesis (heights 1, 2 are absent), so
        // `apply_update` doesn't require `h3.prev_blockhash()` to match genesis.
        let unrelated = header(BlockHash::all_zeros(), Some(0)).block_hash();
        let h3 = header(unrelated, Some(3));
        let cp = graph.tip().insert(3, h3);

        let changeset = graph.apply_update(cp).unwrap();
        assert_eq!(changeset.blocks, [(h3.block_hash(), (3u32, h3))].into());
    }

    #[test]
    fn apply_update_is_idempotent_when_reconciliation_moves_the_base_off_the_tip_chain() {
        // Regression test: `merge_chains` decides what needs connecting by
        // diffing the update against the *canonical tip chain* only, but the graph knows about
        // far more blocks than that chain. The update's root checkpoint declares no parent (it's
        // an anchor, not something to connect), which is fine while it still sits on the tip
        // chain — but a successful `apply_update` re-canonicalizes, and the new best chain can
        // legitimately no longer contain that base. Replaying the very same update then diffed
        // the base as "missing" and pushed it as an item with no declared parent, which
        // `apply_update` rejected with `MissingParent`, breaking idempotence.
        //
        // Found by the `apply_update` fuzz target
        // (crash-3c02bb90c7213ec78b000e66f1ab72c8fadf2f99).
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        // A branch that is *not* reachable forward from the root: `p` hangs off a parent hash
        // the graph has never seen, so `canonicalize` can't discover it yet.
        let unknown = header(BlockHash::all_zeros(), Some(90)).block_hash();
        let p = header(unknown, Some(91));
        graph.connect_block(10, p, unknown).unwrap();
        let f = header(p.block_hash(), Some(92));
        graph.connect_block(100, f, p.block_hash()).unwrap();

        // `b` becomes the tip.
        let b = header(genesis.block_hash(), Some(1));
        graph.apply_update(CheckPoint::new(0, genesis).insert(5, b)).unwrap();
        assert_eq!(
            graph.tip().block_id(),
            BlockId {
                height: 5,
                hash: b.block_hash()
            }
        );

        // `n` extends `b`. Record the edge `n -> f` up front (a gapped/forward-referenced
        // connection, made while `n` itself is still unknown) so that connecting `n` is what
        // finally makes the tall `f` branch reachable from the root.
        let n = header(b.block_hash(), Some(2));
        graph.connect_block(100, f, n.block_hash()).unwrap();

        let update = CheckPoint::new(5, b).push(6, n).unwrap();
        graph.apply_update(update.clone()).unwrap();
        graph.check_invariants().unwrap();
        graph.check_best_tip().unwrap();

        // Reconciliation promoted the taller `f` branch, whose real ancestry runs through `p`,
        // not through `n`/`b`. The update's base is no longer on the tip chain.
        assert_eq!(
            graph.tip().block_id(),
            BlockId {
                height: 100,
                hash: f.block_hash()
            }
        );
        assert!(graph.tip().get(5).is_none(), "the update's base is off the tip chain");

        // Replaying the identical update must still be an accepted no-op.
        let before = graph.clone();
        let changeset = graph
            .apply_update(update)
            .expect("replaying a successful update must not error");
        assert!(changeset.is_empty(), "replaying a successful update adds nothing");
        assert_eq!(
            graph, before,
            "replaying a successful update must not mutate the graph"
        );
    }

    // --- `check_invariants` / `check_best_tip` ---

    #[test]
    fn invariants_hold_for_linear_chain() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let (cp, _headers) = extend_with_headers(graph.tip(), 5, 1);
        graph.apply_update(cp).unwrap();

        graph.check_invariants().unwrap();
        graph.check_best_tip().unwrap();
    }

    #[test]
    fn invariants_hold_after_fork_reorg() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let h1a = header(genesis.block_hash(), Some(1));
        let h2a = header(h1a.block_hash(), Some(2));
        let h1b = header(genesis.block_hash(), Some(101));
        let h2b = header(h1b.block_hash(), Some(102));
        let h3b = header(h2b.block_hash(), Some(103));

        graph
            .apply_update(checkpoint([(0, genesis), (1, h1a), (2, h2a)]))
            .unwrap();
        graph.check_invariants().unwrap();
        graph.check_best_tip().unwrap();

        // Extend the shorter fork past the current best tip, forcing a reorg.
        graph
            .apply_update(checkpoint([(0, genesis), (1, h1b), (2, h2b), (3, h3b)]))
            .unwrap();
        graph.check_invariants().unwrap();
        graph.check_best_tip().unwrap();
        assert_eq!(graph.tip().hash(), h3b.block_hash());
    }

    #[test]
    fn invariants_hold_for_gapped_connection() {
        let genesis = genesis_header();
        let mut graph = BlockGraph::from_genesis(genesis);

        let h1 = header(genesis.block_hash(), Some(1));
        let h2 = header(h1.block_hash(), Some(2));
        let h3 = header(h2.block_hash(), Some(3));
        let cp = graph.tip().insert(3, h3);
        graph.apply_update(cp).unwrap();
        graph.check_invariants().unwrap();
        graph.check_best_tip().unwrap();

        // Fill the gap; `connect_block` alone must not move the tip.
        graph.connect_block(1, h1, genesis.block_hash()).unwrap();
        graph.connect_block(2, h2, h1.block_hash()).unwrap();
        graph.check_invariants().unwrap();
        assert_eq!(
            graph.tip().hash(),
            h3.block_hash(),
            "connect_block must not move the tip"
        );
    }

    // --- `canonicalize` visited-set regression, and `from_changeset` edge-height validation ---
    //
    // `canonicalize`'s tip-discovery BFS previously had no visited-set, so a diamond-shaped DAG
    // (legitimate: skip-chain edges deliberately give a block more than one recorded parent) was
    // re-traversed once per incoming path, and a self-referencing/cyclic edge in a `ChangeSet`
    // sent it into an infinite loop. Adding the visited-set fixed the hang, but a purely cyclic
    // graph then had no leaf at all, turning the hang into a panic in `canonicalize`. Changeset
    // edges are now validated for strictly-increasing parent/child height before the graph is
    // built (`check_changeset_edge_heights`), which rejects cycles outright — the only paths
    // that could otherwise introduce a cycle, since `connect_block`/`apply_update` already
    // enforce strictly-increasing parent height on every insertion.

    #[test]
    fn canonicalize_terminates_on_diamond_dag() {
        let g: BlockHash = Hash::hash(b"genesis");
        let h1a: BlockHash = Hash::hash(b"1a");
        let h1b: BlockHash = Hash::hash(b"1b");
        let h2: BlockHash = Hash::hash(b"2"); // two parents: h1a and h1b
        let h3: BlockHash = Hash::hash(b"3"); // sole leaf

        let changeset = ChangeSet {
            blocks: [
                (g, (0u32, g)),
                (h1a, (1u32, h1a)),
                (h1b, (1u32, h1b)),
                (h2, (2u32, h2)),
                (h3, (3u32, h3)),
            ]
            .into(),
            edges: [
                (BlockHash::all_zeros(), g),
                (g, h1a),
                (g, h1b),
                (h1a, h2),
                (h1b, h2),
                (h2, h3),
            ]
            .into(),
        };

        let graph = BlockGraph::from_changeset(changeset).unwrap().unwrap();
        graph.check_invariants().unwrap();
        graph.check_best_tip().unwrap();
        assert_eq!(graph.tip().hash(), h3);
    }

    #[test]
    fn from_changeset_rejects_cycle_even_with_an_escape_leaf() {
        let g: BlockHash = Hash::hash(b"genesis");
        let h_x: BlockHash = Hash::hash(b"x");
        let h_y: BlockHash = Hash::hash(b"y"); // hX -> hY -> hX forms a cycle
        let h_leaf: BlockHash = Hash::hash(b"leaf"); // a real leaf elsewhere in the graph

        let changeset = ChangeSet {
            blocks: [
                (g, (0u32, g)),
                (h_x, (1u32, h_x)),
                (h_y, (2u32, h_y)),
                (h_leaf, (1u32, h_leaf)),
            ]
            .into(),
            edges: [
                (BlockHash::all_zeros(), g),
                (g, h_x),
                (h_x, h_y),
                (h_y, h_x), // cycles back: parent height 2 >= child height 1
                (g, h_leaf),
            ]
            .into(),
        };

        // The presence of a valid leaf elsewhere must not let the cyclic edge slip through:
        // every edge is validated, not just the ones the tip-selection BFS happens to reach.
        assert_eq!(
            BlockGraph::from_changeset(changeset),
            Err(FromChangeSetError::InvalidEdgeHeight {
                parent: BlockId {
                    height: 2,
                    hash: h_y
                },
                child: BlockId {
                    height: 1,
                    hash: h_x
                },
            }),
        );
    }

    #[test]
    fn from_changeset_rejects_multiple_genesis_blocks() {
        // Regression test: with two distinct height-0 blocks, whichever
        // one isn't picked as the designated root can still end up reachable as some other
        // block's "best" recorded parent, so `canonicalize`'s backward walk can land on it
        // instead of the real root, smuggling an extra height-0 entry into the canonical items
        // and panicking `tip.extend` (which requires strictly-increasing height from the tip's
        // starting height of 0).
        let g1: BlockHash = Hash::hash(b"genesis-1");
        let g2: BlockHash = Hash::hash(b"genesis-2");

        let changeset = ChangeSet {
            blocks: [(g1, (0u32, g1)), (g2, (0u32, g2))].into(),
            edges: [(BlockHash::all_zeros(), g1), (BlockHash::all_zeros(), g2)].into(),
        };

        let (first, second) = if g1 < g2 { (g1, g2) } else { (g2, g1) };
        assert_eq!(
            BlockGraph::from_changeset(changeset),
            Err(FromChangeSetError::MultipleGenesisBlocks { first, second }),
        );
    }

    #[test]
    fn from_changeset_rejects_self_referencing_edge() {
        // Regression test: an infinite loop (and, before edge-height validation
        // was added, a follow-on panic in `canonicalize`): a self-referencing edge on genesis.
        let g: BlockHash = Hash::hash(b"genesis");
        let changeset = ChangeSet {
            blocks: [(g, (0u32, g))].into(),
            edges: [(BlockHash::all_zeros(), g), (g, g)].into(),
        };

        assert_eq!(
            BlockGraph::from_changeset(changeset),
            Err(FromChangeSetError::InvalidEdgeHeight {
                parent: BlockId { height: 0, hash: g },
                child: BlockId { height: 0, hash: g },
            }),
        );
    }

    #[test]
    fn from_changeset_rejects_backwards_height_edge() {
        let g: BlockHash = Hash::hash(b"genesis");
        let h1: BlockHash = Hash::hash(b"1");
        let h2: BlockHash = Hash::hash(b"2");

        // h2 is declared as h1's parent, despite being at a higher height.
        let changeset = ChangeSet {
            blocks: [(g, (0u32, g)), (h1, (1u32, h1)), (h2, (2u32, h2))].into(),
            edges: [(BlockHash::all_zeros(), g), (g, h2), (h2, h1)].into(),
        };

        assert_eq!(
            BlockGraph::from_changeset(changeset),
            Err(FromChangeSetError::InvalidEdgeHeight {
                parent: BlockId {
                    height: 2,
                    hash: h2
                },
                child: BlockId {
                    height: 1,
                    hash: h1
                },
            }),
        );
    }

    #[test]
    fn from_changeset_drops_edges_with_unknown_child() {
        let g: BlockHash = Hash::hash(b"genesis");
        let unknown: BlockHash = Hash::hash(b"unknown"); // never appears in `blocks`

        // Dangling edges whose child has no block data describe no real relationship; they must
        // be dropped from both `next_hashes` and `parents` rather than left dangling, regardless
        // of whether the declared parent (here, both an unknown hash and genesis, a known one)
        // has block data of its own.
        let changeset = ChangeSet {
            blocks: [(g, (0u32, g))].into(),
            edges: [(BlockHash::all_zeros(), g), (unknown, unknown), (g, unknown)].into(),
        };

        let graph = BlockGraph::from_changeset(changeset).unwrap().unwrap();
        graph.check_invariants().unwrap();
        graph.check_best_tip().unwrap();
        assert_eq!(graph.tip().hash(), g);
    }
}
