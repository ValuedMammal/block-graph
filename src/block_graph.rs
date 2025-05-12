//! [`BlockGraph`].

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use bdk_chain::local_chain::MissingGenesisError;
use bitcoin::{block::Header, constants, hashes::Hash, BlockHash, Network};

use bdk_chain::{
    bdk_core::{BlockId, Merge},
    bitcoin,
};

use crate::List;

/// Block header id
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
    /// Blocks.
    blocks: HashMap<BlockHash, T>,
    /// Next hashes.
    next_hashes: HashMap<BlockHash, HashSet<BlockHash>>,
    /// The root hash, aka genesis.
    root: BlockHash,
    /// List.
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

    /// From changeset. None if the `changeset` is empty.
    ///
    /// The first item in `changeset` must correspond to the "genesis block" or else a
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
        graph.apply_changeset(&changeset);

        Ok(Some(graph))
    }

    /// Initial changeset.
    pub fn initial_changeset(&self) -> ChangeSet<T> {
        let mut blocks = BTreeSet::new();

        // Remember to include the root, since it doesn't have a parent entry in `next_hashes`.
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
        let changeset = self.merge_chains(tip)?;

        self.apply_changeset(&changeset);

        debug_assert!(self.check_changeset_is_applied(&changeset));

        Ok(changeset)
    }

    /// Apply changeset.
    fn apply_changeset(&mut self, changeset: &ChangeSet<T>) {
        // Add blocks to graph.
        for (block, par) in changeset.blocks.iter().cloned() {
            let block_id = block.block_id();
            let hash = block_id.hash;
            self.blocks.insert(hash, block);
            // Skip adding to `next_hashes` for the genesis block, since
            // it doesn't extend from anything.
            if block_id.height != 0 {
                self.next_hashes.entry(par.hash).or_default().insert(hash);
            }
        }
        // Update List index.
        // TODO: it seems dangerous to unconditionally set the new tip. This works for now
        // because `apply_changeset` is only called after successfully merging chains.
        let tip = self.tip.clone();
        let new_tip = apply_changeset_to_tip(tip, changeset);
        self.tip = new_tip;
    }

    /// Check changeset is applied, only used in debug mode.
    ///
    /// Note this function assumes that every element of `changeset` is also part of
    /// the canonical chain, although normally that's not a requirement of ChangeSet.
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
            let extends = self.next_hashes.get(&par.hash).unwrap_or(&empty_extends);
            if !extends.contains(&block_id.hash) {
                return false;
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

/// Apply changeset to tip.
fn apply_changeset_to_tip<T>(mut tip: List<T>, changeset: &ChangeSet<T>) -> List<T>
where
    T: ToBlockId + fmt::Debug + Ord + Clone,
{
    if let Some((start_block, _)) = changeset.blocks.iter().next() {
        let start_height = start_block.block_id().height;

        let mut extension = BTreeMap::<u32, T>::new();
        let mut base = None;

        // Keep blocks of the existing chain that are above the start height.
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

impl<T> BlockGraph<T>
where
    T: ToBlockId + fmt::Debug + Ord + Clone,
{
    /// Merges `update` with `self` and returns the resulting [`ChangeSet`].
    ///
    /// Returns a [`CannotConnectError`] if the chains don't connect.
    fn merge_chains(&mut self, update: List<T>) -> Result<ChangeSet<T>, CannotConnectError> {
        // The changes to be applied if merging the chains succeeds.
        let mut stage: BTreeSet<(T, Option<BlockId>)> = BTreeSet::new();

        let mut orig = self.tip.iter();
        let mut update = update.iter();
        let mut curr_orig: Option<List<T>> = None;
        let mut curr_update: Option<List<T>> = None;
        let mut prev_orig: Option<List<T>> = None;
        let mut prev_update: Option<List<T>> = None;
        let mut point_of_agreement_found = false;
        let mut prev_orig_was_invalidated = false;

        let mut potentially_invalidated_heights = vec![];

        // If we can, we want to return the update tip as the new tip because this allows checkpoints
        // in multiple locations to keep the same `Arc` pointers when they are being updated from each
        // other using this function. We can do this as long as the update contains every
        // block's height of the original chain.
        // let mut is_update_height_superset_of_original = true;

        // To find the difference between the new chain and the original we iterate over both of them
        // from the tip backwards in tandem. We are always dealing with the highest one from either chain
        // first and move to the next highest. The crucial logic is applied when they have blocks at the
        // same height.
        loop {
            if curr_orig.is_none() {
                curr_orig = orig.next();
            }
            if curr_update.is_none() {
                curr_update = update.next();
            }

            match (curr_orig.as_ref(), curr_update.as_ref()) {
                // Update block that doesn't exist in the original chain
                (orig, Some(update)) if Some(update.height()) > orig.map(|o| o.height()) => {
                    stage.insert((update.value(), update.prev().map(|l| l.value().block_id())));
                    prev_update = curr_update.take();
                }
                // Original block that isn't in the update
                (Some(orig), update) if Some(orig.height()) > update.map(|u| u.height()) => {
                    // this block might be gone if an earlier block gets invalidated
                    potentially_invalidated_heights.push(orig.height());
                    prev_orig_was_invalidated = false;
                    prev_orig = curr_orig.take();

                    // is_update_height_superset_of_original = false;

                    // OPTIMIZATION: we have run out of update blocks so we don't need to continue
                    // iterating because there's no possibility of adding anything to changeset.
                    if update.is_none() {
                        break;
                    }
                }
                (Some(orig), Some(update)) => {
                    if orig.value().block_id().hash == update.value().block_id().hash {
                        // We found the connection point. We require that the previous (i.e.
                        // higher because we are iterating backwards) block in the original chain was
                        // invalidated (if it exists). This ensures that there is an unambiguous point of
                        // connection to the original chain from the update chain (i.e. we know the
                        // precisely which original blocks are invalid).
                        if !prev_orig_was_invalidated && !point_of_agreement_found {
                            if let (Some(prev_orig), Some(_prev_update)) =
                                (&prev_orig, &prev_update)
                            {
                                return Err(CannotConnectError(prev_orig.height()));
                            }
                        }
                        point_of_agreement_found = true;
                        prev_orig_was_invalidated = false;
                        // OPTIMIZATION 2 -- if we have the same underlying pointer at this point, we
                        // can guarantee that no older blocks are introduced.
                        if Arc::as_ptr(&orig.0) == Arc::as_ptr(&update.0) {
                            break;
                        }
                    } else {
                        // We have an invalidation height so we set the height to the updated hash and
                        // also purge all the original chain block hashes above this block.
                        stage.insert((update.value(), update.prev().map(|l| l.value().block_id())));
                        prev_orig_was_invalidated = true;
                    }
                    prev_update = curr_update.take();
                    prev_orig = curr_orig.take();
                }
                (None, None) => {
                    break;
                }
                _ => {
                    unreachable!("compiler cannot tell that everything has been covered")
                }
            }
        }

        // When we don't have a point of agreement you can imagine it is implicitly the
        // genesis block so we need to do the final connectivity check which in this case
        // just means making sure the entire original chain was invalidated.
        if !prev_orig_was_invalidated && !point_of_agreement_found {
            if let Some(prev_orig) = prev_orig {
                return Err(CannotConnectError(prev_orig.height()));
            }
        }

        let blocks = stage
            .into_iter()
            // We don't apply changes to the genesis block.
            .filter(|(block, _)| block.block_id().height > 0)
            .map(|(block, mut par)| {
                // Find a suitable parent if needed.
                if par.is_none() {
                    let height = block.block_id().height;
                    let ls = self.tip.range(..height).next();
                    par = ls.map(|l| l.value().block_id());
                }

                (block, par.expect("Every non-root Node has a parent"))
            })
            .collect();

        Ok(ChangeSet { blocks })
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

        // obtain the initial changeset
        let cs = graph.initial_changeset();
        assert_eq!(cs.blocks.len(), 4);

        // now recover from changeset
        let recovered = BlockGraph::from_changeset(cs).unwrap().unwrap();
        assert_eq!(recovered, graph);
    }
}
