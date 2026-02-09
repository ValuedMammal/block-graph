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

/// Default capacity of the `BlockGraph` if not provided.
pub const DEFAULT_CAPACITY: usize = 1 << 20;

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

    /// Construct from a [`ChangeSet`]. Will be `None` if `changeset` is empty.
    ///
    /// # Errors
    ///
    /// `changeset.blocks.first()` must correspond to the "genesis block" or else a
    /// [`MissingGenesisError`] will occur.
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

        // Keep a map of block_hash -> parent(s).
        let mut parents: HashMap<BlockHash, BTreeSet<BlockId>> = HashMap::new();

        for (block_id, data, parent_hash) in changeset.blocks {
            let BlockId { height, hash } = block_id;
            // Fill in block data.
            graph.blocks.insert(hash, (height, data));
            // Record that this hash extends from its parent.
            graph.next_hashes.entry(parent_hash).or_default().insert(hash);
            // `changeset.blocks` is an ordered set, so we will have included the parent already.
            // Store it in `parents` for reference.
            if let Some(parent) = graph.block_id(&parent_hash) {
                parents.entry(hash).or_default().insert(parent);
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

        // Now that we know the tip we need to populate the canonical chain
        // by traversing back to the root and collecting block data along
        // the way.
        let mut block_data = vec![];
        let mut cur = Some(best_block.hash);
        while let Some(hash) = cur {
            // Get block data from graph.
            let (height, data) = match graph.blocks.get(&hash).cloned() {
                Some(value) => value,
                None => break,
            };
            block_data.push((height, data));
            // Get next parent hash.
            // The canonical parent is the one with the highest order block id.
            cur = parents.get(&hash).and_then(|parents| parents.last().map(|id| id.hash));
        }

        let mut tip = graph.tip.clone();
        for (height, data) in block_data.into_iter().rev() {
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

    /// Obtain an initial changeset. The initial changeset represents the difference between `self` and
    /// an empty [`BlockGraph`].
    pub fn initial_changeset(&self) -> ChangeSet<T> {
        let mut changeset = ChangeSet::default();

        for (parent_hash, extends) in &self.next_hashes {
            for &hash in extends {
                // Get the orginal block corresponding to `hash`.
                let (height, data) = self.blocks.get(&hash).cloned().expect("invariant");
                let id = BlockId { height, hash };
                changeset.blocks.insert((id, data, *parent_hash));
            }
        }

        changeset
    }

    /// Apply update.
    ///
    /// # Errors
    pub fn apply_update(&mut self, tip: CheckPoint<T>) -> Result<ChangeSet<T>, CannotConnectError>
    where
        T: ToBlockHash + Copy,
    {
        self.apply_update_connected_to(
            tip.iter()
                .map(|cp| (cp.height(), cp.value(), cp.prev().map(|cp| cp.hash()))),
        )
    }

    /// Applies an iterator of update blocks.
    ///
    /// Items of `blocks` represent the block height, block data, and block hash which the update
    /// connects to.
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

    /// Apply changeset. This should be infallible since we only call it after successfully
    /// merging chains.
    fn apply_changeset(&mut self, changeset: ChangeSet<T>, _disconnections: Vec<BlockId>) {
        // First add blocks to graph.
        for (BlockId { height, hash }, data, parent_hash) in changeset.blocks.iter() {
            self.blocks.insert(*hash, (*height, data.clone()));
            self.next_hashes.entry(*parent_hash).or_default().insert(*hash);
        }

        // Update the canonical chain tip. Any stale conflicts that exist in self.tip
        // are purged by inserting the new data.
        let mut new_tip = self.tip.clone();
        for (BlockId { height, .. }, data, _) in changeset.blocks {
            new_tip = new_tip.insert(height, data);
        }
        self.tip = new_tip;
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

/// Change set.
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
    /// The block data, as a set of of (id, data, parent_hash).
    ///
    /// `T` represents the generic block data, typically something that implements [`ToBlockHash`].
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
    /// Merges `update` with `self` and returns the resulting [`ChangeSet`], along with any
    /// disconnections that occurred.
    ///
    /// Note, this method finds the changeset between two chains if they *can be merged*, but
    /// returns without actually modifying `self`.
    ///
    /// To find the difference between the new chain and the original we iterate over both of them
    /// from the tip backwards in tandem, only advancing on the higher of the two chains by height
    /// (or both if equal). The critical logic happens when they have blocks at the same height.
    ///
    /// # Errors
    ///
    /// Returns a [`CannotConnectError`] if the chains don't connect, for example if there is
    /// no point of agreement (and no reorg occurred).
    fn merge_chains<I>(&self, update: I) -> Result<(ChangeSet<T>, Vec<BlockId>), CannotConnectError>
    where
        I: Iterator<Item = (u32, T, Option<BlockHash>)>,
    {
        let mut original_tip = self.tip.iter().peekable();
        let mut update_tip = update.peekable();

        let mut point_of_agreement = None;
        let mut is_prev_orig_invalid = false;
        let mut potentially_invalid_blocks = vec![];

        let mut changeset = ChangeSet::default();

        loop {
            match (original_tip.peek(), update_tip.peek()) {
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
                (Some(original), Some(&(update_height, update, parent_hash))) => {
                    assert!(update_height > 0, "must have non-zero update height");
                    let parent_hash = parent_hash.unwrap_or_else(|| {
                        self.range(0..update_height)
                            .next()
                            .map(|cp| cp.value().to_blockhash())
                            .expect("range must be non-empty")
                    });
                    let height = original.height();
                    let data = original.value();
                    let original_hash = data.to_blockhash();
                    let block_id = (height, original_hash).into();
                    let update_hash = update.to_blockhash();
                    let update_block_id = (update_height, update_hash).into();

                    match update_height.cmp(&height) {
                        // Update height that is not in original.
                        Ordering::Greater => {
                            changeset.blocks.insert((update_block_id, update, parent_hash));
                            update_tip.next();
                        }
                        // Original height not in update.
                        Ordering::Less => {
                            potentially_invalid_blocks.push(block_id);
                            is_prev_orig_invalid = false;
                            original_tip.next();
                        }
                        Ordering::Equal => {
                            // Compare block hashes.
                            if update_hash == original_hash {
                                point_of_agreement = Some(height);
                                // We may be adding an edge if the parent doesn't exist in graph.
                                if !self.blocks.contains_key(&parent_hash) {
                                    changeset.blocks.insert((update_block_id, update, parent_hash));
                                }
                            } else {
                                potentially_invalid_blocks.push(block_id);
                                is_prev_orig_invalid = true;
                                changeset.blocks.insert((update_block_id, update, parent_hash));
                            }
                            original_tip.next();
                            update_tip.next();
                        }
                    }
                }
                (None, Some(..)) => {
                    unreachable!("original cannot be exhausted while there are updates to process, as that would imply a negative update height");
                }
            }
        }

        // Fail if no point of agreement is found after traversing blocks of the original chain,
        // unless there's an explicit invalidation, meaning 1 or more blocks
        // were reorged.
        if point_of_agreement.is_none() && !is_prev_orig_invalid {
            let try_height =
                potentially_invalid_blocks.last().copied().unwrap_or(self.tip()).height;
            return Err(CannotConnectError(try_height));
        }

        let disconnections = if is_prev_orig_invalid {
            potentially_invalid_blocks
        } else {
            vec![]
        };

        Ok((changeset, disconnections))
    }
}

/// Happens when the chain being merged doesn't connect to the parent chain.
///
/// Includes the height of the parent chain that refused the connection.
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
        //  -A'
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

        // Now insert block 1 by applying checkpoint containing heights [0,1,2]
        // and agreement height = 2
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
