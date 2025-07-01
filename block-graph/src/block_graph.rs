//! [`BlockGraph`].

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::ops::RangeBounds;

use bdk_chain::{
    bitcoin, local_chain::MissingGenesisError, BlockId, CheckPoint, Merge, ToBlockHash,
};
use bitcoin::{hashes::Hash, BlockHash};
use skiplist::SkipList;

use crate::collections::{BTreeMap, HashMap, HashSet};

/// Default capacity of the `BlockGraph` if not provided.
pub const DEFAULT_CAPACITY: usize = 1 << 20;

/// Block graph.
#[derive(Debug)]
pub struct BlockGraph<T> {
    /// Nodes of `T` in the block graph keyed by block hash.
    blocks: HashMap<BlockHash, (u32, T)>,
    /// `next_hashes` maps a block hash to the set of hashes extending from it.
    next_hashes: HashMap<BlockHash, HashSet<BlockHash>>,
    /// The root hash, aka genesis.
    root: BlockHash,
    /// The canonical chain tip.
    tip: SkipList<T>,
}

impl<T: ToBlockHash + fmt::Debug + Ord + Clone> BlockGraph<T> {
    /// From genesis `block`.
    pub fn from_genesis(block: T) -> Self {
        Self::from_genesis_with_capacity(block, DEFAULT_CAPACITY)
    }

    /// From genesis `block` with the given capacity.
    ///
    /// - `cap`: How many nodes are expected to exist in the canonical chain over the lifetime
    ///   of this BlockGraph. The BlockGraph may grow to beyond the specified `cap`, however the
    ///   performance benefits can dimish as the internal skiplist becomes more densely populated.
    pub fn from_genesis_with_capacity(block: T, cap: usize) -> Self {
        let genesis_height = 0;
        let genesis_hash = block.to_blockhash();

        let mut blocks = HashMap::new();
        blocks.insert(genesis_hash, (genesis_height, block.clone()));
        let next_hashes = HashMap::new();
        let root = genesis_hash;
        let mut tip = SkipList::with_capacity(cap);
        tip.insert(genesis_height, block);

        Self {
            blocks,
            next_hashes,
            root,
            tip,
        }
    }

    /// Get the chain tip block id.
    pub fn tip(&self) -> BlockId {
        let (tip_height, block) = self.tip.iter().next().expect("chain tip must be non-empty");
        BlockId {
            height: *tip_height,
            hash: block.to_blockhash(),
        }
    }

    /// Get the value of a node in the block graph by a given `height`.
    pub fn get(&self, height: u32) -> Option<&T> {
        self.tip.get(height)
    }

    /// Iterate items of the canonical chain.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> {
        self.tip.iter().map(|(k, v)| (*k, v))
    }

    /// Iterate items of the canonical chain within a specified `range` of heights.
    pub fn range(&self, range: impl RangeBounds<u32>) -> impl Iterator<Item = (u32, &T)> {
        self.tip.range(range).map(|(k, v)| (*k, v))
    }

    /// Retrieve the block id of a given `hash` if it exists.
    pub fn block_id(&self, hash: &BlockHash) -> Option<BlockId> {
        self.blocks.get(hash).map(|(height, block)| BlockId {
            height: *height,
            hash: block.to_blockhash(),
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
        let (genesis, _par) = match changeset.blocks.get(&0).cloned() {
            Some(genesis) => genesis,
            None => return Err(MissingGenesisError),
        };
        let genesis_hash = genesis.to_blockhash();

        let mut graph = Self::from_genesis(genesis);

        // Keep a map of block_id -> prev_hash
        let prev_hashes: HashMap<BlockId, BlockHash> = changeset
            .blocks
            .iter()
            .map(|(&height, (block, par))| ((height, block.to_blockhash()).into(), par.hash))
            .collect();

        // The following algorithm is meant to deterministically find the tip of
        // the longest chain among a set of possible tips.

        // First fill in block nodes and next hashes.
        for (height, (block, par)) in changeset.blocks {
            let hash = block.to_blockhash();
            graph.blocks.insert(hash, (height, block));
            // Skip adding to `next_hashes` for the genesis block, since
            // it doesn't extend from anything.
            if height != 0 {
                graph.next_hashes.entry(par.hash).or_default().insert(hash);
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
            .flat_map(|hash| graph.block_id(hash))
            .max_by_key(|b| (b.height, core::cmp::Reverse(b.hash)));

        let best_block = match best_block {
            Some(b) => b,
            None => return Ok(None),
        };

        // Now that we know the tip we need to populate the canonical chain
        // by traversing back to the root and collecting items of `T` along the way.
        let blocks = &graph.blocks;
        let mut cur = best_block.hash;
        let graph_iter = core::iter::from_fn(|| {
            let (height, block) = blocks.get(&cur)?;
            let block_id = BlockId {
                height: *height,
                hash: block.to_blockhash(),
            };
            let prev = prev_hashes.get(&block_id).copied()?;
            cur = prev;
            Some((*height, block.clone()))
        });
        for (height, block) in graph_iter {
            graph.tip.insert(height, block);
        }

        Ok(Some(graph))
    }

    /// Obtain an initial changeset. The initial changeset represents the difference between `self` and
    /// an empty [`BlockGraph`].
    pub fn initial_changeset(&self) -> ChangeSet<T> {
        let mut blocks = BTreeMap::new();

        // Remember to include the root, since it doesn't have an entry in `next_hashes`.
        let (_, genesis) = self.blocks.get(&self.root).cloned().expect("must contain root");
        blocks.insert(
            0,
            (
                genesis,
                BlockId {
                    height: 0,
                    hash: BlockHash::all_zeros(),
                },
            ),
        );

        for (par_hash, extends) in &self.next_hashes {
            for hash in extends {
                // Get block
                let (height, block) = self.blocks.get(hash).cloned().expect("must have block");
                // Get parent id
                let par_id = self.block_id(par_hash).expect("must have block");

                blocks.insert(height, (block, par_id));
            }
        }

        ChangeSet { blocks }
    }

    /// Apply update.
    pub fn apply_update(&mut self, tip: CheckPoint<T>) -> Result<ChangeSet<T>, CannotConnectError>
    where
        T: ToBlockHash + Copy,
    {
        let (changeset, disconnections) = self.merge_chains(tip)?;

        self.apply_changeset(changeset.clone(), disconnections);

        Ok(changeset)
    }

    /// Apply changeset. This must not fail!
    fn apply_changeset(&mut self, changeset: ChangeSet<T>, disconnections: Vec<BlockId>) {
        // First add blocks to graph.
        for (height, (block, par)) in changeset.blocks.clone().into_iter() {
            let hash = block.to_blockhash();
            self.blocks.insert(hash, (height, block.clone()));
            // Skip adding to `next_hashes` for the genesis block, since
            // it doesn't extend from anything.
            if height > 0 {
                self.next_hashes.entry(par.hash).or_default().insert(hash);
            }
        }

        // If there are disconnections remove them from the canonical chain,
        // before applying the newly added blocks.
        for block_id in disconnections {
            let height = block_id.height;
            let _removed_value = self.tip.remove(height);
        }
        for (height, (block, _par)) in changeset.blocks {
            self.tip.insert(height, block);
        }
    }
}

impl<T: fmt::Debug + Clone + PartialEq> PartialEq for BlockGraph<T> {
    fn eq(&self, other: &Self) -> bool {
        self.blocks == other.blocks
            && self.next_hashes == other.next_hashes
            && self.root == other.root
            && self.tip == other.tip
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
    /// Map of `height -> (T, parent_id)`
    pub blocks: BTreeMap<u32, (T, BlockId)>,
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
    T: ToBlockHash + fmt::Debug + Ord + Copy,
{
    /// Get blocks of the canonical chain as a single [`CheckPoint`].
    ///
    /// This is here in order to remain interoperable with code that interfaces with
    /// [`LocalChain`](bdk_chain::local_chain::LocalChain), etc.
    fn checkpoint(&self) -> CheckPoint<T> {
        self.tip
            .iter()
            .rev()
            .fold(Option::<CheckPoint<T>>::None, |acc, (height, block)| match acc {
                Some(cp) => Some(cp.push(*height, *block).expect("blocks are in order")),
                None => Some(CheckPoint::new(*height, *block)),
            })
            .expect("chain tip must be non-empty")
    }

    /// Merges `update` with `self` and returns the resulting [`ChangeSet`].
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
    fn merge_chains(
        &self,
        update: CheckPoint<T>,
    ) -> Result<(ChangeSet<T>, Vec<BlockId>), CannotConnectError> {
        let cp = self.checkpoint();

        let mut original_tip = cp.iter().peekable();
        let mut update_tip = update.iter().peekable();

        let mut point_of_agreement = None;
        let mut is_prev_orig_invalid = false;
        let mut potentially_invalid_blocks = vec![];

        // The changes to be applied if merging the chains succeeds.
        let mut stage: BTreeMap<u32, (T, Option<BlockId>)> = BTreeMap::new();

        loop {
            match (original_tip.peek(), update_tip.peek()) {
                // We're done when all updates have been processed.
                (_, None) => break,
                // Compare heights.
                (Some(original), Some(update)) => {
                    match update.height().cmp(&original.height()) {
                        // Update height that is not in original.
                        Ordering::Greater => {
                            stage.insert(
                                update.height(),
                                (update.data(), update.prev().map(|cp| cp.block_id())),
                            );
                            update_tip.next();
                        }
                        // Original height not in update.
                        Ordering::Less => {
                            potentially_invalid_blocks.push(original.block_id());
                            is_prev_orig_invalid = false;
                            original_tip.next();
                        }
                        Ordering::Equal => {
                            // Compare block hashes.
                            if update.hash() == original.hash() {
                                point_of_agreement = Some(original.height());
                                // OPTIMIZATION: If we have the same underlying pointer we can guarantee
                                // no more blocks will be introduced.
                                if update.eq_ptr(original) {
                                    break;
                                }
                            } else {
                                // We have an explicit invalidation height. Update the changeset, as long
                                // as it does not conflict with the root (0).
                                if update.height() == 0 {
                                    return Err(CannotConnectError(0));
                                }
                                potentially_invalid_blocks.push(original.block_id());
                                is_prev_orig_invalid = true;
                                stage.insert(
                                    update.height(),
                                    (update.data(), update.prev().map(|l| l.block_id())),
                                );
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

        // Fail if no PoA is found after traversing blocks of the original chain,
        // unless there's an explicit invalidation, meaning 1 or more blocks
        // were reorged.
        if point_of_agreement.is_none()
            && !potentially_invalid_blocks.is_empty()
            && !is_prev_orig_invalid
        {
            let prev_orig = potentially_invalid_blocks.pop().expect("it wasn't empty");
            println!("Found no PoA, try height={}", prev_orig.height);
            return Err(CannotConnectError(prev_orig.height));
        }

        println!("PoA={:?}", point_of_agreement);

        // To produce a changeset we need to flatten the optional parent id, which we can do
        // by finding the next nearest block of the *original chain* that sits at a lower height
        // than the update block.
        let blocks = stage
            .into_iter()
            // We don't apply updates to the root (0).
            .filter(|&(height, _)| height > 0)
            .map(|(height, (block, par))| {
                let par = par.unwrap_or_else(|| {
                    println!("Finding parent of height={}", height);
                    self.tip
                        .range(..height)
                        .next()
                        .map(|&(height, block)| BlockId {
                            height,
                            hash: block.to_blockhash(),
                        })
                        .expect("range must not be empty")
                });
                (height, (block, par))
            })
            .collect();

        let mut disconnections = Vec::with_capacity(potentially_invalid_blocks.len());

        if is_prev_orig_invalid {
            disconnections = potentially_invalid_blocks;
        }

        Ok((ChangeSet { blocks }, disconnections))
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

    use bitcoin::{constants, Network};

    #[test]
    fn test_from_genesis() {
        let genesis_block = constants::genesis_block(Network::Bitcoin);
        let header = genesis_block.header;
        let graph = BlockGraph::from_genesis(header);
        assert_eq!(graph.blocks.len(), 1);
        assert!(graph.next_hashes.is_empty());
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
        dbg!(&graph);

        blocks.reverse();
        let tip_blocks = graph
            .tip
            .iter()
            .map(|&(height, hash)| BlockId { height, hash })
            .collect::<Vec<_>>();
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
        let mut tip = graph.checkpoint();
        let block_1 = BlockId {
            height: 1,
            hash: Hash::hash(b"1"),
        };
        tip = tip.push(1, block_1.hash).unwrap();
        let changeset = graph.apply_update(tip).unwrap();

        assert_eq!(
            changeset.blocks,
            [(block_1.height, (block_1.hash, genesis_block))].into()
        );
    }

    #[test]
    fn test_merge_chains_connect_two() {
        // case: connect 2
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"0"),
        };
        let mut graph = BlockGraph::from_genesis(genesis_block.hash);
        let mut tip = graph.checkpoint();
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
                (block_1.height, (block_1.hash, genesis_block)),
                (block_2.height, (block_2.hash, block_1))
            ]
            .into(),
        );
    }

    #[test]
    fn test_merge_chains_no_point_of_agreement_ok() {
        // case: connect 1 (no PoA) by strictly extending should be ok
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"0"),
        };
        let mut graph = BlockGraph::from_genesis(genesis_block.hash);
        let block_1 = BlockId {
            height: 1,
            hash: Hash::hash(b"one"),
        };
        let tip = CheckPoint::new(1, block_1.hash);
        let changeset = graph.apply_update(tip).unwrap();
        assert_eq!(
            changeset.blocks,
            [(block_1.height, (block_1.hash, genesis_block))].into()
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
        let mut graph = BlockGraph::from_genesis(genesis_block.hash);
        let block_2 = BlockId {
            height: 2,
            hash: Hash::hash(b"2"),
        };
        let update = CheckPoint::new(block_2.height, block_2.hash);
        let changeset = graph.apply_update(update).unwrap();
        assert!(!changeset.blocks.is_empty());
        // Now try to insert block 1
        let block_1 = BlockId {
            height: 1,
            hash: Hash::hash(b"1"),
        };
        let update = CheckPoint::new(block_1.height, block_1.hash);
        let res = graph.apply_update(update);
        assert!(matches!(
            res,
            Err(CannotConnectError(height)) if height == 2,
        ));
    }

    #[test]
    fn test_merge_chains_introduce_older_ok() {
        // This is ok?
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
                (block_2.height, (block_2.hash, genesis_block)),
                (genesis_block.height, (genesis_block.hash, BlockId::default())),
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
        let mut tip = graph.checkpoint().get(0).unwrap();
        tip = tip.push(block_1.height, block_1.hash).unwrap();
        let changeset = graph.apply_update(tip).unwrap();
        assert_eq!(changeset.blocks.len(), 1);
        assert_eq!(changeset.blocks.keys().next(), Some(&1));
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
                (genesis_block.height, (genesis_block.hash, BlockId::default())),
                (block_1.height, (block_1.hash, genesis_block)),
            ]
            .into(),
        };

        let mut graph = BlockGraph::from_changeset(changeset).unwrap().unwrap();
        assert_eq!(graph.blocks.len(), 2);

        // Now invalidate the tip
        let mut tip = graph.checkpoint();
        let block_1a = BlockId {
            height: 1,
            hash: Hash::hash(b"1a"),
        };
        tip = tip.insert(1, block_1a.hash);

        let (changeset, disconnections) = graph.merge_chains(tip.clone()).unwrap();
        assert_eq!(
            changeset.blocks,
            [(block_1a.height, (block_1a.hash, genesis_block))].into()
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
                (genesis_block.height, (genesis_block.hash, BlockId::default())),
                (block_1.height, (block_1.hash, genesis_block)),
                (block_2.height, (block_2.hash, block_1)),
            ]
            .into(),
        };

        let mut graph = BlockGraph::from_changeset(changeset).unwrap().unwrap();
        assert_eq!(graph.blocks.len(), 3);

        // Now invalidate two blocks
        let mut tip = graph.checkpoint().get(1).unwrap();
        let block_1a = BlockId {
            height: 1,
            hash: Hash::hash(b"1a"),
        };
        tip = tip.insert(1, block_1a.hash);

        let (changeset, disconnections) = graph.merge_chains(tip.clone()).unwrap();
        assert_eq!(
            changeset.blocks,
            [(block_1a.height, (block_1a.hash, genesis_block))].into()
        );
        assert_eq!(disconnections.len(), 2);

        let _ = graph.apply_update(tip).unwrap();
        assert_eq!(graph.tip(), block_1a);
    }
}
