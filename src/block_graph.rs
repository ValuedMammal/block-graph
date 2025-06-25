//! [`BlockGraph`].

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;

use crate::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use bitcoin::{block::Header, constants, hashes::Hash, BlockHash, Network};

use bdk_chain::{
    bdk_core::{BlockId, Merge},
    bitcoin,
    local_chain::MissingGenesisError,
};

use crate::List;

/// Block header id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockHeaderId {
    /// height
    pub height: u32,
    /// hash
    pub hash: BlockHash,
    /// header
    pub header: Header,
}

impl Default for BlockHeaderId {
    fn default() -> Self {
        let block = constants::genesis_block(Network::Bitcoin);
        let hash = block.block_hash();
        let header = block.header;
        Self {
            height: Default::default(),
            hash,
            header,
        }
    }
}

/// Trait for types that can derive a `BlockId`.
pub trait ToBlockId {
    /// Return the identity of the block.
    fn block_id(&self) -> BlockId;
}

impl ToBlockId for BlockHeaderId {
    fn block_id(&self) -> BlockId {
        BlockId {
            height: self.height,
            hash: self.hash,
        }
    }
}

impl ToBlockId for BlockId {
    fn block_id(&self) -> BlockId {
        *self
    }
}

/// Block graph.
#[derive(Debug)]
pub struct BlockGraph<T> {
    /// Nodes of `T` in the block graph keyed by block hash.
    blocks: HashMap<BlockHash, T>,
    /// Next hashes maps a block hash to the set of hashes extending from it.
    next_hashes: HashMap<BlockHash, HashSet<BlockHash>>,
    /// The root hash, aka genesis.
    root: BlockHash,
    /// The canonical chain tip.
    tip: List<T>,
}

impl<T: ToBlockId + fmt::Debug + Ord + Clone> BlockGraph<T> {
    /// From genesis `block`.
    pub fn from_genesis(block: T) -> Self {
        let block_id = block.block_id();
        assert_eq!(block_id.height, 0, "err: missing genesis");

        let mut blocks = HashMap::new();
        blocks.insert(block_id.hash, block.clone());
        let next_hashes = HashMap::new();
        let root = block_id.hash;
        let tip = List::new(block_id.height, block);

        Self {
            blocks,
            next_hashes,
            root,
            tip,
        }
    }

    /// Get the chain tip.
    pub fn tip(&self) -> List<T> {
        self.tip.clone()
    }

    /// Construct from a [`ChangeSet`]. Will be `None` if `changeset` is empty.
    ///
    /// # Errors
    ///
    /// `changeset.blocks.first()` must correspond to the "genesis block" or else a
    /// [`MissingGenesisError`] will occur.
    pub fn from_changeset(changeset: ChangeSet<T>) -> Result<Option<Self>, MissingGenesisError>
    where
        T: Default,
    {
        if changeset.is_empty() {
            return Ok(None);
        }
        let (genesis, _par) = changeset.blocks.iter().next().cloned().expect("must not be empty");
        let genesis_block = genesis.block_id();
        if genesis_block.height != 0 {
            return Err(MissingGenesisError);
        }

        let mut graph = Self::from_genesis(genesis);

        // Keep a map of block_id -> prev_hash
        let prev_hashes = changeset
            .blocks
            .iter()
            .map(|(b, par)| (b.block_id(), par.hash))
            .collect::<HashMap<_, _>>();

        // The following algorithm is meant to deterministically find the tip of
        // the longest chain among a set of possible tips.

        // First fill in block nodes and next hashes
        for (block, par) in changeset.blocks {
            let block_id = block.block_id();
            let hash = block_id.hash;
            graph.blocks.insert(hash, block);
            // Skip adding to `next_hashes` for the genesis block, since
            // it doesn't extend from anything.
            if block_id.height != 0 {
                graph.next_hashes.entry(par.hash).or_default().insert(hash);
            }
        }

        // Find the possible tips by exploring `.next_hashes` depth-first starting from the root.
        let mut tips = HashSet::<BlockHash>::new();
        let mut queue = Vec::with_capacity(graph.blocks.len());
        queue.push(genesis_block.hash);

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
            .flat_map(|hash| Some(graph.blocks.get(hash)?.block_id()))
            .max_by_key(|b| (b.height, core::cmp::Reverse(b.hash)));

        let best_block = match best_block {
            Some(b) => b,
            None => return Ok(None),
        };

        // Now that we know the tip we need to traverse back to the root,
        // collecting items of `T` along the way.
        let blocks = &graph.blocks;
        let mut cur = best_block.hash;
        let graph_iter = core::iter::from_fn(|| {
            let block = blocks.get(&cur)?;
            let prev = prev_hashes.get(&block.block_id()).copied()?;
            cur = prev;
            Some(block)
        });
        let blocks: Vec<T> = graph_iter.into_iter().cloned().collect();

        // Finally, construct the canonical List.
        let mut tip = graph.tip.clone();
        for block in blocks.into_iter().rev() {
            let height = block.block_id().height;
            tip = tip.insert(height, block);
        }
        graph.tip = tip;

        Ok(Some(graph))
    }

    /// Obtain an initial changeset. The initial changeset represents the difference between `self` and
    /// an empty [`BlockGraph`].
    pub fn initial_changeset(&self) -> ChangeSet<T> {
        let mut blocks = BTreeSet::new();

        // Remember to include the root, since it doesn't have an entry in `next_hashes`.
        let genesis_block = self.blocks.get(&self.root).cloned().expect("must contain root");
        blocks.insert((
            genesis_block,
            BlockId {
                height: 0,
                hash: BlockHash::all_zeros(),
            },
        ));

        for (par_hash, extends) in &self.next_hashes {
            for hash in extends {
                // Get block
                let block = self.blocks.get(hash).cloned().expect("must have block");
                // Get parent id
                let par_id = self.blocks.get(par_hash).expect("must have block").block_id();

                blocks.insert((block, par_id));
            }
        }

        ChangeSet { blocks }
    }

    /// Apply update.
    pub fn apply_update(&mut self, tip: List<T>) -> Result<ChangeSet<T>, CannotConnectError> {
        let (changeset, disconnections) = self.merge_chains(tip)?;

        self.apply_changeset(&changeset, disconnections);

        debug_assert!(self.check_changeset_is_applied(&changeset));

        Ok(changeset)
    }

    /// Apply changeset. This must not fail!
    fn apply_changeset(&mut self, changeset: &ChangeSet<T>, disconnections: Vec<BlockId>) {
        // First add blocks to graph.
        for (block, par) in changeset.blocks.clone().into_iter() {
            let block_id = block.block_id();
            let hash = block_id.hash;
            self.blocks.insert(hash, block);
            // Skip adding to `next_hashes` for the genesis block, since
            // it doesn't extend from anything.
            if block_id.height != 0 {
                self.next_hashes.entry(par.hash).or_default().insert(hash);
            }
        }

        // Now we need to update the canonical chain.
        let mut tip = self.tip.clone();

        // If there are disconnections remove them from the canonical `List`.
        if !disconnections.is_empty() {
            for (block, _) in changeset.blocks.iter().cloned() {
                tip = tip.insert(block.block_id().height, block);
            }
        } else {
            // Otherwise we can apply the changeset while retaining blocks of the original chain.
            tip = apply_changeset_to_tip(tip, changeset);
        }

        self.tip = tip;
    }

    /// Check changeset is applied, only used in debug mode.
    ///
    /// Note this function assumes that every element of `changeset` is also part of
    /// the canonical chain, although normally that's not a requirement of ChangeSet.
    #[allow(unused)]
    fn check_changeset_is_applied(&self, changeset: &ChangeSet<T>) -> bool {
        let mut cur = self.tip.clone();
        for (block, par) in changeset.blocks.iter().rev() {
            let block_id = block.block_id();
            // The block is present in the block graph
            if !self.blocks.contains_key(&block_id.hash) {
                return false;
            }
            // The block extends from the parent
            let empty_extends = HashSet::new();
            if block_id.height > 0 {
                let extends = self.next_hashes.get(&par.hash).unwrap_or(&empty_extends);
                if !extends.contains(&block_id.hash) {
                    return false;
                }
            }

            // The block is in the canonical List
            match cur.get(block_id.height) {
                Some(ls) => {
                    if &ls.value() != block {
                        return false;
                    }
                    cur = ls;
                }
                None => return false,
            }
        }
        true
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

/// Apply changeset to tip.
fn apply_changeset_to_tip<T>(mut tip: List<T>, changeset: &ChangeSet<T>) -> List<T>
where
    T: ToBlockId + fmt::Debug + Ord + Clone,
{
    if let Some((start_block, _)) = changeset.blocks.iter().next() {
        let start_height = start_block.block_id().height;

        let mut extension = BTreeMap::<u32, T>::new();
        let mut base = None;

        // Keep blocks belonging to `tip` that are above the start height.
        for cur in tip.iter() {
            if cur.height() < start_height {
                base = Some(cur);
                break;
            }
            extension.insert(cur.height(), cur.value());
        }

        for (block, _) in &changeset.blocks {
            let height = block.block_id().height;
            extension.insert(height, block.clone());
        }

        let new_tip = match base {
            Some(ls) => ls.extend(extension).expect("extension is strictly greater than base"),
            None => List::from_entries(extension).expect("valid List"),
        };

        debug_assert_eq!(new_tip.iter().last(), tip.get(0), "new tip must be a valid chain");

        tip = new_tip;
    }

    tip
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
    /// Set of `(block, parent_id)`.
    pub blocks: BTreeSet<(T, BlockId)>,
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
    T: ToBlockId + fmt::Debug + Ord + Clone,
{
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
        update: List<T>,
    ) -> Result<(ChangeSet<T>, Vec<BlockId>), CannotConnectError> {
        let mut original_tip = self.tip.iter().peekable();
        let mut update_tip = update.iter().peekable();

        let mut point_of_agreement = None;
        let mut is_prev_orig_invalid = false;
        let mut potentially_invalid_blocks = vec![];

        // The changes to be applied if merging the chains succeeds.
        let mut stage: BTreeSet<(T, Option<BlockId>)> = BTreeSet::new();

        loop {
            match (original_tip.peek(), update_tip.peek()) {
                // We're done when all updates have been processed.
                (_, None) => break,
                // Compare heights.
                (Some(original), Some(update)) => {
                    match update.height().cmp(&original.height()) {
                        // Update height that is not in original.
                        Ordering::Greater => {
                            stage.insert((update.value(), update.prev().map(|l| l.block_id())));
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
                                stage.insert((update.value(), update.prev().map(|l| l.block_id())));
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

        // To produce a changeset we need to "flatten" the optional parent id, which we can do
        // by finding the next nearest block of the *original chain* that sits at a lower height
        // than the update block.
        let blocks = stage
            .into_iter()
            // We don't apply updates to the root (0).
            .filter(|(b, _)| b.block_id().height > 0)
            .map(|(block, par)| {
                let height = block.block_id().height;
                let par = par.unwrap_or_else(|| {
                    println!("Finding parent of height={}", height);
                    self.tip
                        .range(..height)
                        .next()
                        .map(|ls| ls.block_id())
                        .expect("range must not be empty")
                });
                (block, par)
            })
            .collect();

        let mut disconnections = vec![];

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

    use crate::BlockHeaderId;
    use bitcoin::{constants, Network};

    #[test]
    fn test_from_genesis() {
        let genesis_block = constants::genesis_block(Network::Bitcoin);
        let header = genesis_block.header;
        let hash = genesis_block.block_hash();
        let header = BlockHeaderId {
            height: 0,
            hash,
            header,
        };
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
        let mut graph = BlockGraph::from_genesis(genesis_block);

        let mut ls = graph.tip.clone();
        for height in 1..=3 {
            let block = BlockId {
                height,
                hash: Hash::hash(height.to_be_bytes().as_slice()),
            };
            ls = ls.push(height, block).unwrap();
        }

        let cs = graph.apply_update(ls).unwrap();
        assert_eq!(cs.blocks.len(), 3);
    }

    #[test]
    fn iter_timechain() {
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"gen"),
        };
        let mut graph = BlockGraph::from_genesis(genesis_block);

        let mut blocks: Vec<BlockId> = vec![genesis_block];

        let mut ls = graph.tip.clone();
        for height in 1..=3 {
            let block = BlockId {
                height,
                hash: Hash::hash(height.to_be_bytes().as_slice()),
            };
            ls = ls.push(height, block).unwrap();
            blocks.push(block);
        }
        let _ = graph.apply_update(ls).unwrap();

        blocks.reverse();
        let tip_blocks = graph.tip.iter().map(|l| l.value()).collect::<Vec<_>>();
        assert_eq!(tip_blocks, blocks);
    }

    #[test]
    fn test_initial_changeset() {
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"gen"),
        };
        let mut graph = BlockGraph::from_genesis(genesis_block);

        let mut ls = graph.tip.clone();
        for height in 1..=3 {
            let block = BlockId {
                height,
                hash: Hash::hash(height.to_be_bytes().as_slice()),
            };
            ls = ls.push(height, block).unwrap();
        }

        let _ = graph.apply_update(ls).unwrap();

        // collect the initial changeset
        let cs = graph.initial_changeset();
        assert_eq!(cs.blocks.len(), 4);

        // now recover from changeset
        let recovered = BlockGraph::from_changeset(cs).unwrap().unwrap();
        assert_eq!(recovered, graph);
    }

    #[test]
    fn test_merge_chains_connect() {
        // case: connect 1
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"0"),
        };
        let mut graph = BlockGraph::from_genesis(genesis_block);
        let mut tip = graph.tip.clone();
        let block_1 = BlockId {
            height: 1,
            hash: Hash::hash(b"one"),
        };
        tip = tip.push(1, block_1).unwrap();
        let changeset = graph.apply_update(tip).unwrap();

        assert_eq!(changeset.blocks, [(block_1, genesis_block)].into());
    }

    #[test]
    fn test_merge_chains_connect_two() {
        // case: connect 2
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"0"),
        };
        let mut graph = BlockGraph::from_genesis(genesis_block);
        let mut tip = graph.tip.clone();
        let block_1 = BlockId {
            height: 1,
            hash: Hash::hash(b"one"),
        };
        let block_2 = BlockId {
            height: 2,
            hash: Hash::hash(b"two"),
        };
        tip = tip.extend([(1, block_1), (2, block_2)]).unwrap();
        let changeset = graph.apply_update(tip).unwrap();

        assert_eq!(changeset.blocks, [(block_1, genesis_block), (block_2, block_1)].into());
    }

    #[test]
    fn test_merge_chains_no_point_of_agreement_ok() {
        // case: connect 1 (no PoA) by strictly extending should be ok
        let genesis_block = BlockId {
            height: 0,
            hash: Hash::hash(b"0"),
        };
        let mut graph = BlockGraph::from_genesis(genesis_block);
        let block_1 = BlockId {
            height: 1,
            hash: Hash::hash(b"one"),
        };
        let update = List::new(1, block_1);
        let changeset = graph.apply_update(update).unwrap();
        assert_eq!(changeset.blocks, [(block_1, genesis_block)].into());
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
        let mut graph = BlockGraph::from_genesis(genesis_block);
        let block_2 = BlockId {
            height: 2,
            hash: Hash::hash(b"2"),
        };
        let update = List::new(block_2.height, block_2);
        let changeset = graph.apply_update(update).unwrap();
        assert!(!changeset.blocks.is_empty());
        // Now try to insert block 1
        let block_1 = BlockId {
            height: 1,
            hash: Hash::hash(b"1"),
        };
        let update = List::new(block_1.height, block_1);
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
            blocks: [(block_2, genesis_block), (genesis_block, BlockId::default())].into(),
        })
        .unwrap()
        .unwrap();

        // Now insert block 1, which is based on a clone of the original.
        let block_1 = BlockId {
            height: 1,
            hash: Hash::hash(b"1"),
        };
        let mut tip = graph.tip.get(0).unwrap();
        tip = tip.push(block_1.height, block_1).unwrap();
        let changeset = graph.apply_update(tip).unwrap();
        assert_eq!(changeset.blocks.len(), 1);
        assert_eq!(changeset.blocks.first().unwrap().0.height, 1);
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
            blocks: [(genesis_block, BlockId::default()), (block_1, genesis_block)].into(),
        };

        let mut graph = BlockGraph::from_changeset(changeset).unwrap().unwrap();
        assert_eq!(graph.blocks.len(), 2);

        // Now invalidate the tip
        let mut tip = graph.tip.clone();
        let block_1a = BlockId {
            height: 1,
            hash: Hash::hash(b"1a"),
        };
        tip = tip.insert(1, block_1a);

        let (changeset, disconnections) = graph.merge_chains(tip.clone()).unwrap();
        assert_eq!(changeset.blocks, [(block_1a, genesis_block)].into());
        assert_eq!(disconnections.len(), 1);

        // Now apply update
        let _ = graph.apply_update(tip).unwrap();
        assert_eq!(graph.tip.block_id(), block_1a);
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
                (genesis_block, BlockId::default()),
                (block_1, genesis_block),
                (block_2, block_1),
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
        tip = tip.insert(1, block_1a);

        let (changeset, disconnections) = graph.merge_chains(tip.clone()).unwrap();
        assert_eq!(changeset.blocks, [(block_1a, genesis_block)].into());
        assert_eq!(disconnections.len(), 2);

        let _ = graph.apply_update(tip).unwrap();
        assert_eq!(graph.tip.block_id(), block_1a);
    }
}

// TODO: perf. benchmark the code in `BlockGraph::from_changeset`.
