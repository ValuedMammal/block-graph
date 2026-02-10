//! [`BlockGraph`].

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::fmt::Debug;
use core::ops::RangeBounds;

use bdk_chain::{
    bitcoin, local_chain::MissingGenesisError, BlockId, ChainOracle, Merge, ToBlockHash,
};
use bitcoin::{hashes::Hash, BlockHash};

use crate::collections::{BTreeSet, HashMap, HashSet};
use crate::CheckPoint;

/// Block graph.
#[derive(Debug)]
pub struct BlockGraph<T> {
    /// Nodes of `(Height, T)` in the block graph keyed by block hash.
    blocks: HashMap<BlockHash, (u32, T)>,
    /// `next_hashes` maps a block hash to the set of hashes extending from it.
    next_hashes: HashMap<BlockHash, HashSet<BlockHash>>,
    /// The root hash, aka genesis.
    root: BlockHash,
    /// The canonical list of blocks in the chain.
    tip: CheckPoint<T>,
}

impl<T: ToBlockHash + Debug + Ord + Clone> BlockGraph<T> {
    /// From genesis `data`.
    pub fn from_genesis(data: T) -> Self {
        let genesis_height = 0;
        let genesis_hash = data.to_blockhash();

        let mut blocks = HashMap::new();
        blocks.insert(genesis_hash, (genesis_height, data.clone()));
        let mut next_hashes = HashMap::new();
        next_hashes.insert(BlockHash::all_zeros(), [genesis_hash].into());
        let root = genesis_hash;
        let tip = CheckPoint::new(genesis_height, data);

        Self {
            blocks,
            next_hashes,
            root,
            tip,
        }
    }

    /// Get the chain tip block id.
    pub fn tip(&self) -> BlockId {
        let cp = self.tip.iter().next().expect("chain tip must be non-empty");
        BlockId {
            height: cp.height(),
            hash: cp.0.value.to_blockhash(),
        }
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
        let (_, genesis_data, _) = changeset
            .blocks
            .iter()
            .find(|(id, _, _)| id.height == 0)
            .ok_or(MissingGenesisError)?;
        let genesis_hash = genesis_data.to_blockhash();

        let mut graph = Self::from_genesis(genesis_data.clone());

        // Keep a map of block_hash -> parent block IDs for canonical chain reconstruction.
        let mut block_parents: HashMap<BlockHash, BTreeSet<BlockId>> = HashMap::new();

        for (block_id, data, parent_hash) in changeset.blocks {
            let BlockId { height, hash } = block_id;
            // Fill in block data.
            graph.blocks.insert(hash, (height, data));
            // Record that this hash extends from its parent.
            graph.next_hashes.entry(parent_hash).or_default().insert(hash);
            // `changeset.blocks` is an ordered set, so we will have included the parent already.
            // Store parent-child relationship for canonical chain reconstruction.
            if let Some(parent_block_id) = graph.block_id(&parent_hash) {
                block_parents.entry(hash).or_default().insert(parent_block_id);
            }
        }

        // Find the possible tips by exploring `.next_hashes` depth-first starting from the root.
        let mut tips = HashSet::<BlockHash>::new();
        let mut queue = Vec::with_capacity(graph.blocks.len());
        queue.push(genesis_hash);

        while let Some(hash) = queue.pop() {
            match graph.next_hashes.get(&hash) {
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
            .filter_map(|hash| graph.block_id(hash))
            .max_by_key(|b| (b.height, core::cmp::Reverse(b.hash)));

        let Some(best_block) = best_block else {
            debug_assert!(false, "failed to find best tip");
            return Ok(None);
        };

        // Now that we know the tip, populate the canonical chain by traversing
        // back to the root and collecting block data along the way.
        let mut canonical_block_data = vec![];
        let mut current_hash = Some(best_block.hash);
        while let Some(hash) = current_hash {
            // Get block data from graph.
            let (height, data) = match graph.blocks.get(&hash).cloned() {
                Some(value) => value,
                None => break,
            };
            canonical_block_data.push((height, data));
            // Get the next parent hash for traversal.
            // The canonical parent is the one with the highest ordered block ID.
            current_hash = block_parents
                .get(&hash)
                .and_then(|parents| parents.last().map(|id| id.hash));
        }

        let mut tip = graph.tip.clone();
        for (height, data) in canonical_block_data.into_iter().rev() {
            tip = tip.insert(height, data);
        }
        graph.tip = tip;

        debug_assert!(
            graph
                .tip
                .get(0)
                .is_some_and(|cp| cp.value().to_blockhash() == genesis_hash),
            "failed to canonicalize blockgraph"
        );

        Ok(Some(graph))
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

    /// Apply a chain update by integrating new blocks from the given tip.
    ///
    /// This method attempts to merge the new chain with the existing graph,
    /// potentially causing reorganizations if the new chain conflicts with the current tip.
    ///
    /// # Errors
    ///
    /// Returns [`CannotConnectError`] if the update cannot be connected to the existing chain.
    pub fn apply_update(&mut self, tip: CheckPoint<T>) -> Result<ChangeSet<T>, CannotConnectError>
    where
        T: ToBlockHash + Copy,
    {
        self.apply_update_connected_to(
            tip.iter()
                .map(|cp| (cp.height(), cp.value(), cp.prev().map(|cp| cp.hash()))),
        )
    }

    /// Apply an update using an iterator of block information.
    ///
    /// Each item in `blocks` represents a tuple of (block_height, block_data, parent_hash)
    /// where parent_hash is the block this update connects to (None for genesis connection).
    pub fn apply_update_connected_to<I>(
        &mut self,
        blocks: I,
    ) -> Result<ChangeSet<T>, CannotConnectError>
    where
        I: IntoIterator<Item = (u32, T, Option<BlockHash>)>,
        T: ToBlockHash + Copy,
    {
        let (changeset, disconnections) = self.merge_chains(blocks.into_iter())?;

        self.apply_changeset(changeset.clone(), disconnections);

        Ok(changeset)
    }

    /// Apply a changeset to the block graph.
    ///
    /// This method is infallible since it's only called after successfully merging chains.
    /// It updates both the block storage and the canonical chain tip.
    fn apply_changeset(&mut self, changeset: ChangeSet<T>, _disconnections: Vec<BlockId>) {
        // First, add all new blocks to the graph storage.
        for (BlockId { height, hash }, block_data, parent_hash) in changeset.blocks.iter() {
            self.blocks.insert(*hash, (*height, block_data.clone()));
            self.next_hashes.entry(*parent_hash).or_default().insert(*hash);
        }

        // Update the canonical chain tip. Any stale conflicts that exist in self.tip
        // are automatically purged by inserting the new block data.
        let mut updated_tip = self.tip.clone();
        for (BlockId { height, .. }, block_data, _) in changeset.blocks {
            updated_tip = updated_tip.insert(height, block_data);
        }
        self.tip = updated_tip;
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
        Ok(self.tip())
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
/// Each block entry is a tuple of (block_id, block_data, parent_hash).
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
    /// Each entry is a tuple of (block_id, block_data, parent_hash) where:
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

impl<T: Default + Ord> Merge for ChangeSet<T> {
    fn merge(&mut self, other: Self) {
        self.blocks.extend(other.blocks);
    }
    fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

impl<T> BlockGraph<T>
where
    T: ToBlockHash + Debug + Ord + Copy,
{
    /// Merge an update chain with the current chain and return the resulting changeset.
    ///
    /// This method compares the existing canonical chain with the proposed update chain
    /// to determine what changes are needed. It returns both the changeset of new blocks
    /// to add and any blocks that were disconnected due to reorganization.
    ///
    /// The merge algorithm:
    /// 1. Iterates through both chains from tip to genesis
    /// 2. Advances on the chain with higher block height
    /// 3. When heights are equal, compares block hashes for conflicts
    /// 4. Detects reorganizations and finds the point of agreement
    ///
    /// Note: This method calculates the merge without modifying `self`.
    ///
    /// # Errors
    ///
    /// Returns [`CannotConnectError`] if the chains cannot be connected (no point of agreement
    /// and no reorganization detected).
    fn merge_chains<I>(&self, update: I) -> Result<(ChangeSet<T>, Vec<BlockId>), CannotConnectError>
    where
        I: Iterator<Item = (u32, T, Option<BlockHash>)>,
    {
        let mut original_blocks = self.tip.iter().peekable();
        let mut update_blocks = update.peekable();

        let mut point_of_agreement = None;
        let mut has_reorg_occurred = false;
        let mut potentially_invalid_blocks = vec![];

        let mut changeset = ChangeSet::default();

        loop {
            match (original_blocks.peek(), update_blocks.peek()) {
                // We're done when all updates are processed.
                (_original, None) => break,
                // Error if attempting to overwrite the genesis hash.
                (_, Some(&(update_height, update, _))) if update_height == 0 => {
                    if update.to_blockhash() != self.root {
                        return Err(CannotConnectError(0));
                    }
                    point_of_agreement = Some(update_height);
                    break;
                }
                (Some(original_checkpoint), Some(&(update_height, update_data, parent_hash))) => {
                    assert!(update_height > 0, "must have non-zero update height");
                    let resolved_parent_hash = parent_hash.unwrap_or_else(|| {
                        self.range(0..update_height)
                            .next()
                            .map(|cp| cp.value().to_blockhash())
                            .expect("range must be non-empty")
                    });
                    let original_block_id = original_checkpoint.block_id();
                    let original_height = original_block_id.height;
                    let original_hash = original_block_id.hash;
                    let update_hash = update_data.to_blockhash();
                    let update_block_id = (update_height, update_hash).into();

                    match update_height.cmp(&original_height) {
                        // Update block at height not present in original chain.
                        Ordering::Greater => {
                            changeset.blocks.insert((
                                update_block_id,
                                update_data,
                                resolved_parent_hash,
                            ));
                            update_blocks.next();
                        }
                        // Original block at height not present in update chain.
                        Ordering::Less => {
                            potentially_invalid_blocks.push(original_block_id);
                            has_reorg_occurred = false;
                            original_blocks.next();
                        }
                        // Both chains have blocks at same height - compare hashes.
                        Ordering::Equal => {
                            if update_hash == original_hash {
                                // Blocks match - this is our point of agreement.
                                point_of_agreement = Some(original_height);
                                // Add edge if the parent doesn't exist in graph yet.
                                if !self.blocks.contains_key(&resolved_parent_hash) {
                                    changeset.blocks.insert((
                                        update_block_id,
                                        update_data,
                                        resolved_parent_hash,
                                    ));
                                }
                            } else {
                                // Hash mismatch - reorganization detected.
                                potentially_invalid_blocks.push(original_block_id);
                                has_reorg_occurred = true;
                                changeset.blocks.insert((
                                    update_block_id,
                                    update_data,
                                    resolved_parent_hash,
                                ));
                            }
                            original_blocks.next();
                            update_blocks.next();
                        }
                    }
                }
                (None, Some(..)) => {
                    unreachable!("original cannot be exhausted while there are updates to process, as that would imply a negative update height");
                }
            }
        }

        // Fail if no point of agreement is found after traversing the original chain,
        // unless there was an explicit reorganization (invalidation of existing blocks).
        if point_of_agreement.is_none() && !has_reorg_occurred {
            let suggested_height =
                potentially_invalid_blocks.last().copied().unwrap_or(self.tip()).height;
            return Err(CannotConnectError(suggested_height));
        }

        let disconnections = if has_reorg_occurred {
            potentially_invalid_blocks
        } else {
            vec![]
        };

        Ok((changeset, disconnections))
    }
}

/// Error indicating that a chain update cannot be connected to the existing graph.
///
/// This occurs when there is no point of agreement between the update chain and the
/// existing canonical chain, and no reorganization was detected.
///
/// Contains the height at which the connection was attempted.
#[derive(Debug)]
pub struct CannotConnectError(pub u32);

impl fmt::Display for CannotConnectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cannot connect; try include height: {}", self.0)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CannotConnectError {}

#[cfg(test)]
mod test {
    use super::*;

    use bitcoin::hashes::Hash;
    use bitcoin::{constants, Network};

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
            hash: Hash::hash(b"gen"),
        };
        let mut graph = BlockGraph::from_genesis(genesis_block.hash);

        let mut cp = CheckPoint::new(0, genesis_block.hash);
        for height in 1u32..=3 {
            let hash = Hash::hash(height.to_be_bytes().as_slice());
            cp = cp.push(height, hash).unwrap();
        }

        let cs = graph.apply_update(cp).unwrap();
        assert_eq!(cs.blocks.len(), 3);
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
        let _ = graph.apply_update(cp).unwrap();

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
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"0"),
        };
        let mut graph = BlockGraph::from_genesis(genesis_block.hash);
        let mut tip = graph.tip.clone();
        let block_1 = BlockId {
            height: 1,
            hash: Hash::hash(b"1"),
        };
        tip = tip.push(1, block_1.hash).unwrap();
        let changeset = graph.apply_update(tip).unwrap();

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
        let mut tip = graph.tip.clone();
        let block_1 = BlockId {
            height: 1,
            hash: Hash::hash(b"one"),
        };
        let block_2 = BlockId {
            height: 2,
            hash: Hash::hash(b"two"),
        };
        tip = tip.extend([(1, block_1.hash), (2, block_2.hash)]).unwrap();
        let changeset = graph.apply_update(tip).unwrap();

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
    fn test_merge_chains_error() {
        // case: error if no PoA
        // 0-x-B
        //  -A
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"0"),
        };
        let block_2 = BlockId {
            height: 2,
            hash: Hash::hash(b"2"),
        };
        let changeset = ChangeSet {
            blocks: [
                (genesis_block, genesis_block.hash, BlockHash::all_zeros()),
                (block_2, block_2.hash, genesis_block.hash),
            ]
            .into(),
        };
        let graph = BlockGraph::from_changeset(changeset).unwrap().unwrap();

        // Now try to insert block 1
        let block_1 = BlockId {
            height: 1,
            hash: Hash::hash(b"1"),
        };
        let update = CheckPoint::new(block_1.height, block_1.hash);
        let res = graph.merge_chains(
            update
                .iter()
                .map(|cp| (cp.height(), cp.value(), cp.prev().map(|cp| cp.hash()))),
        );
        assert!(matches!(
            res,
            Err(CannotConnectError(height)) if height == 2,
        ));
    }

    #[test]
    fn test_merge_chains_introduce_older_ok() {
        // 0-x-B
        // 0-A
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"0"),
        };
        let block_2 = BlockId {
            height: 2,
            hash: Hash::hash(b"2"),
        };
        let mut graph = BlockGraph::from_changeset(ChangeSet {
            blocks: [
                (genesis_block, genesis_block.hash, BlockHash::all_zeros()),
                (block_2, block_2.hash, genesis_block.hash),
            ]
            .into(),
        })
        .unwrap()
        .unwrap();

        // Now insert block 1, which is based on a clone of the original.
        let block_1 = BlockId {
            height: 1,
            hash: Hash::hash(b"1"),
        };
        let mut tip = graph.tip.get(0).unwrap();
        tip = tip.push(block_1.height, block_1.hash).unwrap();
        let changeset = graph.apply_update(tip).unwrap();
        assert_eq!(changeset.blocks.len(), 1);
        assert_eq!(changeset.blocks.first().unwrap().0, block_1);
    }

    #[test]
    fn test_merge_chains_evict_block() {
        // 0-A
        // 0-A'
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"0"),
        };
        let block_1 = BlockId {
            height: 1,
            hash: Hash::hash(b"1"),
        };
        let changeset = ChangeSet {
            blocks: [
                (genesis_block, genesis_block.hash, BlockHash::all_zeros()),
                (block_1, block_1.hash, genesis_block.hash),
            ]
            .into(),
        };

        let mut graph = BlockGraph::from_changeset(changeset).unwrap().unwrap();
        assert_eq!(graph.blocks.len(), 2);

        // Now invalidate the tip
        let mut tip = graph.tip.clone();
        let block_1a = BlockId {
            height: 1,
            hash: Hash::hash(b"1a"),
        };
        tip = tip.insert(1, block_1a.hash);

        let (changeset, disconnections) = graph
            .merge_chains(
                tip.iter()
                    .map(|cp| (cp.height(), cp.value(), cp.prev().map(|cp| cp.hash()))),
            )
            .unwrap();
        assert_eq!(
            changeset.blocks,
            [(block_1a, block_1a.hash, genesis_block.hash)].into()
        );
        assert_eq!(disconnections.len(), 1);

        // Now apply update
        let _ = graph.apply_update(tip).unwrap();
        assert_eq!(graph.tip(), block_1a);
    }

    #[test]
    fn test_merge_chains_evict_two_blocks() {
        // 0-A-B
        // 0-A'
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

        let mut graph = BlockGraph::from_changeset(changeset).unwrap().unwrap();
        assert_eq!(graph.blocks.len(), 3);

        // Now invalidate two blocks
        let mut tip = graph.tip.get(1).unwrap();
        let block_1a = BlockId {
            height: 1,
            hash: Hash::hash(b"1a"),
        };
        tip = tip.insert(1, block_1a.hash);

        let (changeset, disconnections) = graph
            .merge_chains(
                tip.iter()
                    .map(|cp| (cp.height(), cp.value(), cp.prev().map(|cp| cp.hash()))),
            )
            .unwrap();
        assert_eq!(
            changeset.blocks,
            [(block_1a, block_1a.hash, genesis_block.hash)].into()
        );
        assert_eq!(disconnections.len(), 2);

        let _ = graph.apply_update(tip).unwrap();
        assert_eq!(graph.tip(), block_1a);
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
        let chain_tip = graph.tip();
        assert_eq!(chain_tip, block_2);
        for block in [genesis_block, block_1, block_2] {
            assert!(matches!(graph.is_block_in_chain(block, graph.tip()), Ok(Some(true))))
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
    fn introduce_older_block_should_be_canonical() {
        // Start with a sparse chain
        let genesis_hash = Hash::hash(b"0");
        let mut graph = BlockGraph::<BlockHash>::from_genesis(genesis_hash);
        let genesis_block = BlockId {
            height: 0,
            hash: genesis_hash,
        };

        let mut cp = CheckPoint::new(0, genesis_hash);

        // Leave a gap at height = 1
        for i in 2u32..=10 {
            let height = i;
            let hash = Hash::hash(height.to_be_bytes().as_slice());
            cp = cp.insert(height, hash);
        }

        let _ = graph.apply_update(cp.clone()).unwrap();

        // Now insert block 1
        let hash_1 = Hash::hash(b"1");
        cp = cp.insert(1, hash_1);
        let block_1 = BlockId {
            height: 1,
            hash: hash_1,
        };
        let hash_2 = cp.get(2).unwrap().hash();
        let block_2 = graph.block_id(&hash_2).unwrap();

        let changeset = graph.apply_update(cp).unwrap();
        assert_eq!(
            changeset,
            ChangeSet {
                blocks: [(block_1, hash_1, genesis_block.hash), (block_2, hash_2, block_1.hash)]
                    .into(),
            }
        );

        // Test recover from initial changeset
        let init_cs = graph.initial_changeset();
        let recovered = BlockGraph::from_changeset(init_cs).unwrap().unwrap();
        assert_eq!(recovered, graph);
    }
}
