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
use core::fmt::Debug;
use core::ops::RangeBounds;

use bdk_chain::{
    bitcoin, local_chain::MissingGenesisError, BlockId, ChainOracle, Merge, ToBlockHash,
};
use bitcoin::{hashes::Hash, BlockHash};

use crate::collections::{BTreeSet, HashMap, HashSet};
use crate::CheckPoint;

/// Block graph.
#[derive(Debug, Clone)]
pub struct BlockGraph<T> {
    /// Nodes of `(Height, T)` in the block graph keyed by block hash.
    blocks: HashMap<BlockHash, (u32, T)>,
    /// Map of block hash to set of parent IDs
    /// `parents` is a set because a child can point to its own parent and/or a
    /// a more distant ancestor.
    parents: HashMap<BlockHash, BTreeSet<BlockId>>,
    /// `next_hashes` maps a block hash to the set of hashes extending from it.
    next_hashes: HashMap<BlockHash, HashSet<BlockHash>>,
    /// The root hash, aka genesis.
    root: BlockHash,
    /// The canonical chain tip.
    tip: CheckPoint<T>,
}

impl<T: ToBlockHash + Debug + Ord + Clone> BlockGraph<T> {
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
    /// 1. Finding the genesis block (height 0)
    /// 2. Building the graph structure with parent-child relationships
    /// 3. Determining the canonical chain tip
    /// 4. Constructing the canonical chain by traversing back from the tip
    ///
    /// # Errors
    ///
    /// Returns [`MissingGenesisError`] if no block at height 0 exists in the changeset.
    pub fn from_changeset(changeset: ChangeSet<T>) -> Result<Option<Self>, MissingGenesisError> {
        if changeset.blocks.is_empty() {
            return Ok(None);
        }
        let (_, genesis_value, _) = changeset
            .blocks
            .iter()
            .find(|(id, _, _)| id.height == 0)
            .ok_or(MissingGenesisError)?;

        let mut graph = Self::from_genesis(genesis_value.clone());

        for (block_id, value, parent_hash) in changeset.blocks {
            let BlockId { height, hash } = block_id;
            // Fill in block data.
            graph.blocks.insert(hash, (height, value));
            // Record that this hash extends from its parent.
            graph.next_hashes.entry(parent_hash).or_default().insert(hash);
            // Since changeset is an ordered set, we will have seen the parent already
            if let Some(block_id) = graph.block_id(&parent_hash) {
                graph.parents.entry(hash).or_default().insert(block_id);
            }
        }

        let items = graph.canonicalize(graph.root);
        graph.tip = graph.tip.extend(items).expect("tip height must only increase");

        debug_assert!(
            graph
                .tip
                .get(0)
                .is_some_and(|cp| cp.value().to_blockhash() == graph.root),
            "failed to canonicalize blockgraph"
        );

        Ok(Some(graph))
    }

    /// Canonicalize the [`BlockGraph`] starting from the given `root` and return a
    /// collection of `(height, value)` tuples in ascending height order.
    ///
    /// Note: The caller must ensure that `root` exists in the current active chain.
    fn canonicalize(&mut self, root: BlockHash) -> Vec<(u32, T)> {
        // Find the possible tips by exploring `.next_hashes` starting from the root.
        let mut tips = HashSet::<BlockHash>::new();
        let mut queue = vec![];
        queue.push(root);

        while let Some(hash) = queue.pop() {
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

        for (parent_hash, child_hashes) in &self.next_hashes {
            for &block_hash in child_hashes {
                // Get the original block data corresponding to this block hash.
                let (height, block_data) = self
                    .blocks
                    .get(&block_hash)
                    .cloned()
                    .expect("block must exist in graph");
                let block_id = BlockId {
                    height,
                    hash: block_hash,
                };
                changeset.blocks.insert((block_id, block_data, *parent_hash));
            }
        }

        changeset
    }

    /// Find the best parent [`BlockId`] of the given `hash` if it exists in `self.parents`.
    ///
    /// The "best" parent is the one with the highest order [`BlockId`].
    fn parent(&self, hash: &BlockHash) -> Option<BlockId> {
        self.parents.get(hash)?.iter().last().copied()
    }

    /// Locates the most recent common ancestor between `self.tip` and the given block `hash`,
    /// if one exists.
    fn find_fork_point(&self, hash: BlockHash) -> Option<BlockHash> {
        self.iter_block_graph(hash)
            .find_map(|(height, hash, _value)| match self.tip.get(height) {
                Some(checkpoint) if checkpoint.hash() == hash => Some(hash),
                _ => None,
            })
    }

    /// Iterate over `(height, blockhash, value)` tuples in the [`BlockGraph`] starting from the given [`BlockHash`].
    fn iter_block_graph(&self, from: BlockHash) -> impl Iterator<Item = (u32, BlockHash, T)> + '_ {
        let mut current_hash = Some(from);

        core::iter::from_fn(move || {
            let hash = current_hash?;
            let (height, value) = self.blocks.get(&hash).cloned()?;
            current_hash = self.parent(&hash).map(|id| id.hash);
            Some((height, hash, value))
        })
    }

    /// Connects a value of `T` having `block_id` which is connected to `prev_hash`.
    ///
    /// Note: This method only adds block data to the [`BlockGraph`] without updating the
    /// canonical chain tip. To do that use [`apply_update`](Self::apply_update).
    ///
    /// # Errors
    ///
    /// - If `value` doesn't produce the same block hash as `block_id.hash`.
    /// - If the parent having `prev_hash` exists in graph not at a strictly lesser height
    ///   than `block_id.height`.
    pub fn connect_block(
        &mut self,
        block_id: BlockId,
        value: T,
        prev_hash: BlockHash,
    ) -> Result<ChangeSet<T>, &'static str> {
        let mut changeset = ChangeSet::default();
        let height = block_id.height;
        let hash = block_id.hash;
        if value.to_blockhash() != hash {
            return Err("block hash mismatch");
        }
        // The same value exists in self.blocks
        if self
            .blocks
            .get(&hash)
            .is_some_and(|(existing_height, existing_value)| {
                existing_height == &height && existing_value == &value
            })
            // The same parent-child dependency exists
            && self.next_hashes.get(&prev_hash).is_some_and(|set| set.contains(&hash))
        {
            return Ok(changeset);
        }
        // Add block to graph
        self.blocks.insert(hash, (height, value.clone()));
        // Record that this block extends from its parent
        self.next_hashes.entry(prev_hash).or_default().insert(hash);
        // Record prev as a parent of this block
        if let Some(parent_id) = self.block_id(&prev_hash) {
            if parent_id.height >= height {
                return Err("value must connect to parent of smaller height");
            }
            self.parents.entry(hash).or_default().insert(parent_id);
        }
        changeset.blocks.insert(((height, hash).into(), value, prev_hash));

        Ok(changeset)
    }

    /// Applies a checkpoint update.
    ///
    /// - `prev_hash`: Indicates the hash of a parent block to which the root of the
    ///   checkpoint should connect.
    pub fn apply_update(
        &mut self,
        checkpoint: CheckPoint<T>,
        prev_hash: BlockHash,
    ) -> Result<ChangeSet<T>, &'static str>
    where
        T: ToBlockHash + Clone,
    {
        let mut changeset = ChangeSet::default();
        let mut items = self.merge_chains(checkpoint);
        items.reverse();
        let from_hash = items.first().map(|(id, _, _)| id.hash);
        // Connect each item in ascending height order. `prev` should only be None for
        // the first element, ie. the root of the checkpoint.
        for (height, value, prev) in items {
            changeset.merge(self.connect_block(height, value, prev.unwrap_or(prev_hash))?);
        }
        // To update the canonical chain we need to re-canonicalize the graph
        // starting from the most recent common ancestor (i.e. "fork point").
        if let Some(hash) = from_hash {
            if let Some(ancestor) = self.find_fork_point(hash) {
                let old_tip = self.tip.block_id();
                let mut new_tip = self.tip();
                for (height, value) in self.canonicalize(ancestor) {
                    new_tip = new_tip.insert(height, value.clone());
                    // If the best tip didn't change we're done
                    if new_tip.block_id() == old_tip {
                        break;
                    }
                }
                self.tip = new_tip;
            }
        }
        Ok(changeset)
    }

    /// TODO: We should be able to combine two BlockGraphs into one
    #[allow(unused)]
    pub fn apply_graph_update(&mut self, other: Self) {
        todo!();
    }
}

impl<T: Debug + Clone + PartialEq> PartialEq for BlockGraph<T> {
    fn eq(&self, other: &Self) -> bool {
        self.blocks == other.blocks
            && self.next_hashes == other.next_hashes
            && self.root == other.root
            && self.tip == other.tip
    }
}

impl<T: ToBlockHash + Debug + Ord + Clone> ChainOracle for BlockGraph<T> {
    type Error = core::convert::Infallible;

    fn get_chain_tip(&self) -> Result<BlockId, Self::Error> {
        Ok(self.tip().block_id())
    }

    fn is_block_in_chain(
        &self,
        block: BlockId,
        chain_tip: BlockId,
    ) -> Result<Option<bool>, Self::Error> {
        // `block` height must be within that of `chain_tip`.
        if block.height > chain_tip.height {
            return Ok(None);
        }
        // `chain_tip` must exist in chain.
        if self
            .tip
            .get(chain_tip.height)
            .is_none_or(|cp| cp.value().to_blockhash() != chain_tip.hash)
        {
            return Ok(None);
        }
        // A block of given height must exist in this chain, and the hashes must match.
        match self.tip.get(block.height) {
            Some(cp) => Ok(Some(cp.value().to_blockhash() == block.hash)),
            None => Ok(None),
        }
    }
}

/// A changeset representing modifications to a [`BlockGraph`].
///
/// Contains the set of blocks to be added to the graph, along with their parent relationships.
/// Each block entry is a tuple of `(block_id, block_data, parent_hash)`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(bound(
        deserialize = "T: Ord + serde::Deserialize<'de>",
        serialize = "T: Ord + serde::Serialize",
    ))
)]
pub struct ChangeSet<T> {
    /// Set of blocks to add to the graph.
    ///
    /// Each entry is a tuple of `(block_id, block_data, parent_hash)` where:
    /// - `block_id`: The block's identifier (height and hash)
    /// - `block_data`: The generic block data (typically implements [`ToBlockHash`])
    /// - `parent_hash`: The hash of the block this extends from
    pub blocks: BTreeSet<(BlockId, T, BlockHash)>,
}

impl<T> Default for ChangeSet<T> {
    fn default() -> Self {
        Self {
            blocks: Default::default(),
        }
    }
}

impl<T: Ord> Merge for ChangeSet<T> {
    fn merge(&mut self, other: Self) {
        self.blocks.extend(other.blocks)
    }
    fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

impl<T> BlockGraph<T>
where
    T: ToBlockHash + Debug + Ord + Clone,
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
                                update.value(),
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
                                            update.value(),
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
                                    update.value(),
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

#[cfg(test)]
mod test {
    use std::string::ToString;

    use super::*;

    use bitcoin::hashes::Hash;
    use bitcoin::{constants, Network};

    fn checkpoint<T>(entries: impl IntoIterator<Item = (u32, T)>) -> CheckPoint<T>
    where
        T: core::fmt::Debug + ToBlockHash,
    {
        CheckPoint::from_entries(entries).expect("failed to create CheckPoint")
    }

    #[test]
    fn test_from_genesis() {
        let genesis_block = constants::genesis_block(Network::Bitcoin);
        let header = genesis_block.header;
        let graph = BlockGraph::from_genesis(header);
        assert_eq!(graph.blocks.len(), 1);
        assert_eq!(graph.next_hashes.len(), 1);
        assert_eq!(graph.tip.iter().count(), 1);
    }

    #[test]
    fn test_apply_update() {
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"0"),
        };
        let mut graph = BlockGraph::from_genesis(genesis_block.hash);

        let hash_1 = Hash::hash(b"1");
        let hash_2 = Hash::hash(b"2");
        let hash_3 = Hash::hash(b"3");

        let mut cp = CheckPoint::new(0, genesis_block.hash);
        for (height, hash) in (1..=3).zip([hash_1, hash_2, hash_3]) {
            cp = cp.push(height, hash).unwrap();
        }

        let changeset = graph.apply_update(cp, genesis_block.hash).unwrap();

        // Verify the changeset contains the expected blocks
        assert_eq!(changeset.blocks.len(), 3);
        assert_eq!(
            changeset.blocks,
            [
                ((1, hash_1).into(), hash_1, genesis_block.hash),
                ((2, hash_2).into(), hash_2, hash_1),
                ((3, hash_3).into(), hash_3, hash_2),
            ]
            .into()
        );
    }

    // Test that we can iterate blocks of the main chain, and
    // the blocks are correct.
    #[test]
    fn iter_timechain() {
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"0"),
        };
        let mut graph = BlockGraph::from_genesis(genesis_block.hash);

        let mut blocks: Vec<BlockId> = vec![genesis_block];

        let mut cp = CheckPoint::new(0, genesis_block.hash);
        for height in 1u32..=3 {
            let hash = Hash::hash(height.to_be_bytes().as_slice());
            cp = cp.push(height, hash).unwrap();
            blocks.push(BlockId { height, hash });
        }
        let _ = graph.apply_update(cp, genesis_block.hash).unwrap();

        blocks.reverse();
        let tip_blocks = graph.iter().map(|cp| cp.block_id()).collect::<Vec<_>>();
        assert_eq!(tip_blocks, blocks);
    }

    #[test]
    fn test_initial_changeset() {
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"gen"),
        };
        let mut graph = BlockGraph::from_genesis(genesis_block.hash);

        let mut cp = CheckPoint::new(0, genesis_block.hash);
        for height in 1u32..=3 {
            let hash = Hash::hash(height.to_be_bytes().as_slice());
            cp = cp.push(height, hash).unwrap();
        }

        let _ = graph.apply_update(cp, genesis_block.hash).unwrap();

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
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"0"),
        };
        let mut graph = BlockGraph::from_genesis(genesis_block.hash);
        let mut tip = graph.tip();
        let block_1 = BlockId {
            height: 1,
            hash: Hash::hash(b"1"),
        };
        tip = tip.push(1, block_1.hash).unwrap();
        let changeset = graph.apply_update(tip, genesis_block.hash).unwrap();

        assert_eq!(changeset.blocks.len(), 1);
        assert_eq!(changeset.blocks, [(block_1, block_1.hash, genesis_block.hash)].into());
    }

    #[test]
    fn test_merge_chains_connect_two() {
        // case: connect 2
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"0"),
        };
        let mut graph = BlockGraph::from_genesis(genesis_block.hash);
        let mut tip = graph.tip();
        let block_1 = BlockId {
            height: 1,
            hash: Hash::hash(b"one"),
        };
        let block_2 = BlockId {
            height: 2,
            hash: Hash::hash(b"two"),
        };
        tip = tip.extend([(1, block_1.hash), (2, block_2.hash)]).unwrap();
        let changeset = graph.apply_update(tip, genesis_block.hash).unwrap();

        assert_eq!(changeset.blocks.len(), 2);
        assert_eq!(
            changeset.blocks,
            [
                (block_1, block_1.hash, genesis_block.hash),
                (block_2, block_2.hash, block_1.hash),
            ]
            .into(),
        );
    }

    #[test]
    fn test_is_block_in_chain() {
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"0"),
        };
        let block_1 = BlockId {
            height: 1,
            hash: Hash::hash(b"1"),
        };
        let block_2 = BlockId {
            height: 2,
            hash: Hash::hash(b"2"),
        };
        let changeset = ChangeSet {
            blocks: [
                (genesis_block, genesis_block.hash, BlockHash::all_zeros()),
                (block_1, block_1.hash, genesis_block.hash),
                (block_2, block_2.hash, block_1.hash),
            ]
            .into(),
        };
        let graph = BlockGraph::from_changeset(changeset).unwrap().unwrap();
        let chain_tip = graph.tip().block_id();
        assert_eq!(chain_tip, block_2);
        for block in [genesis_block, block_1, block_2] {
            assert!(matches!(graph.is_block_in_chain(block, chain_tip), Ok(Some(true))))
        }
        assert!(
            matches!(
                graph.is_block_in_chain(
                    BlockId {
                        height: 2,
                        hash: Hash::hash(b"2a")
                    },
                    chain_tip
                ),
                Ok(Some(false))
            ),
            "block of wrong hash cannot be in chain"
        );
        assert!(
            graph
                .is_block_in_chain(
                    BlockId {
                        height: 3,
                        hash: Hash::hash(b"3")
                    },
                    chain_tip
                )
                .unwrap()
                .is_none(),
            "block height past tip cannot be in chain"
        );
    }

    #[test]
    fn insert_older_block_should_be_canonical() {
        let genesis_hash = Hash::hash(b"0");
        let mut graph = BlockGraph::<BlockHash>::from_genesis(genesis_hash);
        let _genesis_block = BlockId {
            height: 0,
            hash: genesis_hash,
        };

        let mut cp = CheckPoint::new(0, genesis_hash);

        // Add blocks to graph, leaving a gap at height = 1
        for i in 2u32..=5 {
            let height = i;
            let hash = Hash::hash(height.to_string().as_bytes());
            cp = cp.insert(height, hash);
        }
        let _ = graph.apply_update(cp.clone(), genesis_hash).unwrap();

        // Now insert block 1
        let hash_1 = Hash::hash(b"1");
        let block_1 = BlockId {
            height: 1,
            hash: hash_1,
        };
        cp = cp.insert(1, hash_1);
        let block_2 = cp.get(2).map(|cp| cp.block_id()).expect("block_2 should exist in CP");
        let changeset = graph.apply_update(cp, genesis_hash).unwrap();

        // Verify changeset contains the expected block
        assert_eq!(
            changeset.blocks,
            [
                (block_1, block_1.hash, genesis_hash),
                (block_2, block_2.hash, block_1.hash),
            ]
            .into(),
            "Expected changeset to contain blocks 1 and 2"
        );

        // Canonical chain should remain unchanged
        let test_blocks = graph.iter().map(|cp| cp.block_id()).collect::<Vec<_>>();
        assert_eq!(test_blocks.len(), 6);
        for i in 0..=5 {
            let expected_height = i as u32;
            let expected_hash = if i == 0 {
                genesis_hash
            } else {
                Hash::hash(expected_height.to_string().as_bytes())
            };
            let test_block = test_blocks[test_blocks.len() - 1 - i];
            assert_eq!(test_block.height, expected_height);
            assert_eq!(test_block.hash, expected_hash);
        }
    }

    #[test]
    fn test_apply_update_single_block() {
        let genesis_hash = Hash::hash(b"gen");
        let mut graph = BlockGraph::from_genesis(genesis_hash);

        let block_1_hash = Hash::hash(b"1");
        let cp = CheckPoint::new(1, block_1_hash);

        let changeset = graph.apply_update(cp, genesis_hash).unwrap();

        assert_eq!(graph.tip().height(), 1);
        assert_eq!(graph.tip().hash(), block_1_hash);
        assert_eq!(graph.blocks.len(), 2);

        // Verify changeset
        assert_eq!(changeset.blocks.len(), 1);
        let expected_block = BlockId {
            height: 1,
            hash: block_1_hash,
        };
        assert_eq!(changeset.blocks, [(expected_block, block_1_hash, genesis_hash)].into());
    }

    #[test]
    fn test_apply_update_multiple_blocks() {
        let genesis_hash = Hash::hash(b"gen");
        let mut graph = BlockGraph::from_genesis(genesis_hash);

        let hash_1 = Hash::hash(b"1");
        let hash_2 = Hash::hash(b"2");
        let hash_3 = Hash::hash(b"3");
        let mut cp = CheckPoint::new(1, hash_1);
        cp = cp.push(2, hash_2).unwrap();
        cp = cp.push(3, hash_3).unwrap();

        let changeset = graph.apply_update(cp, genesis_hash).unwrap();

        assert_eq!(graph.tip().height(), 3);
        assert_eq!(graph.blocks.len(), 4);

        // Verify changeset contains all 3 blocks
        assert_eq!(changeset.blocks.len(), 3);
        let expected_blocks = [
            (
                BlockId {
                    height: 1,
                    hash: hash_1,
                },
                hash_1,
                genesis_hash,
            ),
            (
                BlockId {
                    height: 2,
                    hash: hash_2,
                },
                hash_2,
                hash_1,
            ),
            (
                BlockId {
                    height: 3,
                    hash: hash_3,
                },
                hash_3,
                hash_2,
            ),
        ];
        assert_eq!(changeset.blocks, expected_blocks.into());
    }

    #[test]
    fn test_apply_update_maintains_chain_integrity() {
        let genesis_hash = Hash::hash(b"0");
        let mut graph = BlockGraph::from_genesis(genesis_hash);

        let hash_1 = Hash::hash(b"1");
        let hash_2 = Hash::hash(b"2");
        let mut cp = CheckPoint::new(1, hash_1);
        cp = cp.push(2, hash_2).unwrap();

        graph.apply_update(cp, genesis_hash).unwrap();

        // Verify parent-child relationships
        assert!(graph.next_hashes.get(&genesis_hash).unwrap().contains(&hash_1));
        assert!(graph.next_hashes.get(&hash_1).unwrap().contains(&hash_2));
    }

    #[test]
    fn test_apply_update_updates_tip() {
        let genesis_hash = Hash::hash(b"0");
        let mut graph = BlockGraph::from_genesis(genesis_hash);

        let hash_1 = Hash::hash(b"1");
        let cp = CheckPoint::new(1, hash_1);

        graph.apply_update(cp, genesis_hash).unwrap();

        let tip_id = graph.tip().block_id();
        assert_eq!(tip_id.height, 1);
        assert_eq!(tip_id.hash, hash_1);
    }

    #[test]
    fn test_apply_update_extended_chain() {
        let genesis_hash = Hash::hash(b"gen");
        let mut graph = BlockGraph::from_genesis(genesis_hash);

        let hash_1 = Hash::hash(b"1");
        let hash_2 = Hash::hash(b"2");

        // First update
        let mut cp1 = CheckPoint::new(1, hash_1);
        cp1 = cp1.push(2, hash_2).unwrap();
        let changeset1 = graph.apply_update(cp1, genesis_hash).unwrap();

        assert_eq!(changeset1.blocks.len(), 2);

        // Second update extending from previous tip
        let prev_tip_hash = graph.tip().hash();
        let hash_3 = Hash::hash(b"3");
        let cp2 = CheckPoint::new(3, hash_3);
        let changeset2 = graph.apply_update(cp2, prev_tip_hash).unwrap();

        assert_eq!(graph.tip().height(), 3);
        assert_eq!(graph.blocks.len(), 4);

        // Verify second changeset
        assert_eq!(changeset2.blocks.len(), 1);
        assert_eq!(
            changeset2.blocks,
            [(
                BlockId {
                    height: 3,
                    hash: hash_3
                },
                hash_3,
                hash_2
            )]
            .into()
        );
    }

    #[test]
    fn test_connect_blocks() {
        let genesis_hash = Hash::hash(b"0");
        let mut graph = BlockGraph::from_genesis(genesis_hash);

        let hash_1 = Hash::hash(b"1");
        let hash_2 = Hash::hash(b"2");

        // Connect block 1
        let cs = graph.connect_block((1, hash_1).into(), hash_1, genesis_hash).unwrap();
        assert_eq!(cs.blocks, [((1, hash_1).into(), hash_1, genesis_hash)].into());
        assert!(
            graph
                .blocks
                .get(&hash_1)
                .is_some_and(|(height, value)| height == &1 && value == &hash_1),
            "Block 1 should exist"
        );
        assert!(
            graph.next_hashes.get(&genesis_hash).unwrap().contains(&hash_1),
            "Hash 1 should extend from hash 0"
        );
        assert_eq!(
            graph.parent(&hash_1).unwrap(),
            (0, genesis_hash).into(),
            "Block 0 should be parent of block 1"
        );

        // Connect block 2
        let cs = graph.connect_block((2, hash_2).into(), hash_2, hash_1).unwrap();
        assert_eq!(cs.blocks, [((2, hash_2).into(), hash_2, hash_1)].into());
        assert!(
            graph
                .blocks
                .get(&hash_2)
                .is_some_and(|(height, value)| height == &2 && value == &hash_2),
            "Block 2 should exist"
        );
        assert!(
            graph.next_hashes.get(&hash_1).unwrap().contains(&hash_2),
            "Hash 2 should extend from hash 1"
        );
        assert_eq!(
            graph.parent(&hash_2).unwrap(),
            (1, hash_1).into(),
            "Block 1 should be parent of block 2"
        );
    }

    #[test]
    fn test_canonicalize_selects_longest_chain() {
        let genesis_hash = Hash::hash(b"0");
        let mut graph = BlockGraph::from_genesis(genesis_hash);

        // Add two competing chains
        let hash_1a = Hash::hash(b"1a");
        let hash_2a = Hash::hash(b"2a");
        let hash_1b = Hash::hash(b"1b");

        let _ = graph
            .apply_update(checkpoint([(1, hash_1a), (2, hash_2a)]), genesis_hash)
            .unwrap();

        // Add shorter competing chain
        let _ = graph.apply_update(checkpoint([(1, hash_1b)]), genesis_hash).unwrap();

        // Longer chain should be canonical
        assert_eq!(graph.tip().block_id(), (2, hash_2a).into());
    }

    #[test]
    fn test_merge_chains_with_gap() {
        let genesis_hash = Hash::hash(b"0");
        let mut graph = BlockGraph::from_genesis(genesis_hash);

        let hash_3 = Hash::hash(b"3");
        let cp = CheckPoint::new(3, hash_3);

        let changeset = graph.apply_update(cp, genesis_hash).unwrap();

        // Should create a single block at height 3 connecting to genesis
        assert_eq!(changeset.blocks.len(), 1);
        assert_eq!(
            changeset.blocks,
            [(
                BlockId {
                    height: 3,
                    hash: hash_3
                },
                hash_3,
                genesis_hash
            )]
            .into()
        );
    }

    #[test]
    fn test_connect_same_block_twice_ok() {
        let genesis_hash = Hash::hash(b"0");
        let mut graph = BlockGraph::from_genesis(genesis_hash);

        let block_1_hash = Hash::hash(b"1");
        graph
            .connect_block((1, block_1_hash).into(), block_1_hash, genesis_hash)
            .unwrap();

        assert_eq!(graph.blocks.len(), 2);

        // Connecting same block again should be ok
        let cs = graph
            .connect_block((1, block_1_hash).into(), block_1_hash, genesis_hash)
            .unwrap();
        assert!(
            cs.is_empty(),
            "Same block connection should return an empty change set"
        );
    }

    #[test]
    fn test_block_id_retrieval() {
        let genesis_hash = Hash::hash(b"0");
        let mut graph = BlockGraph::from_genesis(genesis_hash);

        let hash_1 = Hash::hash(b"1");
        graph.connect_block((1, hash_1).into(), hash_1, genesis_hash).unwrap();

        let block_id = graph.block_id(&hash_1).unwrap();
        assert_eq!(block_id.height, 1);
        assert_eq!(block_id.hash, hash_1);
    }

    #[test]
    fn test_block_id_nonexistent() {
        let genesis_hash: BlockHash = Hash::hash(b"0");
        let graph = BlockGraph::from_genesis(genesis_hash);

        let nonexistent_hash = Hash::hash(b"nonexistent");
        assert!(graph.block_id(&nonexistent_hash).is_none());
    }

    #[test]
    fn test_genesis_block_retrieval() {
        let genesis_hash: BlockHash = Hash::hash(b"genesis");
        let graph = BlockGraph::from_genesis(genesis_hash);

        assert_eq!(graph.genesis_block(), genesis_hash);
    }

    #[test]
    fn test_range_blocks() {
        let genesis_hash = Hash::hash(b"0");
        let mut graph = BlockGraph::from_genesis(genesis_hash);

        let mut cp = graph.tip();

        for i in 1u32..=5 {
            let hash = Hash::hash(i.to_be_bytes().as_slice());
            cp = cp.push(i, hash).unwrap();
        }

        let _ = graph.apply_update(cp, genesis_hash);

        let range_items: BTreeSet<u32> = graph.range(2..=4).map(|cp| cp.height()).collect();
        assert_eq!(range_items, [2, 3, 4].into());
    }

    #[test]
    fn test_switch_forks() {
        let hash_0: BlockHash = Hash::hash(b"0");
        let hash_1: BlockHash = Hash::hash(b"1");
        let hash_2: BlockHash = Hash::hash(b"2");
        let hash_3: BlockHash = Hash::hash(b"3");
        let hash_3_alt: BlockHash = Hash::hash(b"3_alt");
        let hash_4_alt: BlockHash = Hash::hash(b"4_alt");
        // Create blockgraph
        let changeset = ChangeSet {
            blocks: [
                ((0, hash_0).into(), hash_0, BlockHash::all_zeros()),
                ((1, hash_1).into(), hash_1, hash_0),
                ((2, hash_2).into(), hash_2, hash_1),
                ((3, hash_3).into(), hash_3, hash_2),
            ]
            .into(),
        };
        let mut graph = BlockGraph::from_changeset(changeset).unwrap().unwrap();

        // connect 3-alternate
        let _ = graph.apply_update(checkpoint([(3, hash_3_alt)]), hash_2).unwrap();
        // Chain tip should change if hash_3_alt is smaller
        if hash_3_alt < hash_3 {
            assert_eq!(graph.tip.hash(), hash_3_alt);
        } else {
            assert_eq!(graph.tip.hash(), hash_3);
        }

        // Now extend competing branch, we should correctly switch forks
        let _ = graph.apply_update(checkpoint([(4, hash_4_alt)]), hash_3_alt).unwrap();
        assert_eq!(graph.tip.hash(), hash_4_alt);
    }

    #[test]
    fn test_apply_update_eq_ptr_optimization() {
        let genesis_hash = Hash::hash(b"0");
        let mut graph = BlockGraph::from_genesis(genesis_hash);
        let mut cp = graph.tip();
        const COUNT: u32 = 10;
        const INIT_HEIGHT: u32 = COUNT - 1;

        // Create a block graph
        for height in 1u32..=INIT_HEIGHT {
            let hash = Hash::hash(height.to_string().as_bytes());
            cp = cp.push(height, hash).unwrap();
        }
        let _ = graph.apply_update(cp, genesis_hash).unwrap();

        assert_eq!(graph.tip().height(), INIT_HEIGHT);
        assert_eq!(graph.blocks.len() as u32, COUNT);

        // Create update checkpoint based on clone of original graph.tip() and extend by 1 block
        let original_tip = graph.tip();
        let original_tip_hash = original_tip.hash();
        let new_height = INIT_HEIGHT + 1;
        let new_block_hash = Hash::hash(new_height.to_string().as_bytes());
        let update_checkpoint = original_tip
            .push(new_height, new_block_hash)
            .expect("should be able to extend chain");

        // Record state before update
        let blocks_before = graph.blocks.len();
        let tip_height_before = graph.tip().height();

        // Apply the update - this should only connect 1 new block due to eq_ptr optimization
        let items_to_connect = graph.merge_chains(update_checkpoint.clone());
        assert_eq!(items_to_connect.len(), 1);
        let changeset = graph.apply_update(update_checkpoint, original_tip_hash).unwrap();

        // Verify that only 1 new block was processed in the changeset
        assert_eq!(
            changeset.blocks.len(),
            1,
            "Changeset should contain exactly 1 new block due to eq_ptr optimization"
        );

        // Verify the changeset contains the correct block
        let expected_block = BlockId {
            height: new_height,
            hash: new_block_hash,
        };
        assert_eq!(
            changeset.blocks,
            [(expected_block, new_block_hash, original_tip_hash)].into(),
            "Changeset should contain the new block connecting to the original tip"
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
}
