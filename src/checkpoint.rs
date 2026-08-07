//! [`CheckPoint`]

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::RangeBounds;

use bitcoin::BlockHash;

/// Block ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId {
    /// height
    pub height: u32,
    /// block hash
    pub hash: BlockHash,
}

impl From<(u32, BlockHash)> for BlockId {
    fn from(tup: (u32, BlockHash)) -> Self {
        Self {
            height: tup.0,
            hash: tup.1,
        }
    }
}

/// Implemented by types that can provide a [`BlockHash`] identifying a block.
pub trait ToBlockHash {
    /// To block hash
    fn to_blockhash(&self) -> BlockHash;
}

impl ToBlockHash for BlockHash {
    fn to_blockhash(&self) -> BlockHash {
        *self
    }
}

impl ToBlockHash for bitcoin::block::Header {
    fn to_blockhash(&self) -> BlockHash {
        self.block_hash()
    }
}

/// Implemented by types that carry the hash of their predecessor block.
///
/// Used by [`CheckPoint::push_connected`] to validate chain connectivity.
pub trait HasPrevBlockhash {
    /// Prev block hash
    fn prev_blockhash(&self) -> BlockHash;
}

impl HasPrevBlockhash for bitcoin::block::Header {
    fn prev_blockhash(&self) -> BlockHash {
        self.prev_blockhash
    }
}

/// Error returned by [`CheckPoint::push_connected`].
///
/// Carries both the rejected value and the checkpoint that was the tip at the
/// time of the failed push, so callers can inspect or retry.
#[derive(Debug)]
pub struct ConnectError<T> {
    /// The checkpoint that was the current tip when the push failed.
    pub checkpoint: CheckPoint<T>,
    /// The value that could not be pushed.
    pub value: T,
}

/// A cheaply cloneable, singly-linked list of block nodes.
///
/// Skip pointers enable O(log n) height lookups, based on Bitcoin Core's `GetSkipHeight`.
#[derive(Debug)]
pub struct CheckPoint<T>(Arc<Node<T>>);

impl<T> Clone for CheckPoint<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

/// Internal node for [`CheckPoint`].
#[derive(Debug)]
struct Node<T> {
    height: u32,
    hash: BlockHash,
    value: T,
    prev: Option<Arc<Node<T>>>,
    skip: Option<Arc<Node<T>>>,
    index: u32,
}

impl<T> Drop for Node<T> {
    fn drop(&mut self) {
        // Iteratively drop `prev` links to prevent stack overflow on deep chains.
        let mut current = self.prev.take();
        while let Some(arc_node) = current {
            match Arc::into_inner(arc_node) {
                Some(mut node) => {
                    current = node.prev.take();
                    // Let `node` drop here: prev is None so Drop won't recurse, and
                    // `node.skip` has its refcount decremented automatically.
                }
                None => break,
            }
        }
    }
}

impl<T> PartialEq for CheckPoint<T> {
    fn eq(&self, other: &Self) -> bool {
        self.iter()
            .map(|cp| cp.block_id())
            .eq(other.iter().map(|cp| cp.block_id()))
    }
}

impl<T: ToBlockHash> CheckPoint<T> {
    /// Create a genesis checkpoint.
    pub fn new(height: u32, value: T) -> Self {
        let hash = value.to_blockhash();
        Self(Arc::new(Node {
            height,
            hash,
            value,
            prev: None,
            skip: None,
            index: 0,
        }))
    }

    /// Construct a checkpoint chain from an iterator of `(height, value)` entries in
    /// ascending height order.
    pub fn from_entries(entries: impl IntoIterator<Item = (u32, T)>) -> Result<Self, Option<Self>> {
        let mut iter = entries.into_iter();
        let (height, value) = iter.next().ok_or(None)?;
        let cp = Self::new(height, value);
        cp.extend(iter).map_err(Some)
    }

    /// Push a new block onto the tip of the chain.
    ///
    /// Returns `Err(self)` if `height` is not strictly greater than the current tip.
    /// Skip pointers are computed automatically.
    pub fn push(self, height: u32, value: T) -> Result<Self, Self> {
        if self.height() >= height {
            return Err(self);
        }
        let new_index = self.0.index + 1;
        let skip_target = get_skip_index(new_index);
        let skip_node = self.checkpoint_at_index(skip_target).map(|cp| cp.0);
        let hash = value.to_blockhash();
        Ok(Self(Arc::new(Node {
            height,
            hash,
            value,
            prev: Some(self.0),
            skip: skip_node,
            index: new_index,
        })))
    }

    /// Extend the chain with an iterator of `(height, value)` entries.
    ///
    /// Returns `Err(self_clone)` if any entry has a non-increasing height.
    pub fn extend(self, items: impl IntoIterator<Item = (u32, T)>) -> Result<Self, Self> {
        let mut curr = self.clone();
        for (height, value) in items {
            curr = curr.push(height, value).map_err(|_| self.clone())?;
        }
        Ok(curr)
    }
}

impl<T: ToBlockHash + HasPrevBlockhash> CheckPoint<T> {
    /// Push a new block onto the chain, with optional `prev_blockhash` validation.
    ///
    /// Returns `Err(ConnectError)` in two cases:
    /// - `height <= self.height()` — the height is not strictly greater than the current tip
    /// - `height == self.height() + 1 && value.prev_blockhash() != self.hash()` — the block
    ///   is directly adjacent but does not link to the current tip
    ///
    /// Sparse pushes (gap > 1) are allowed without a `prev_blockhash` check, since
    /// intermediate blocks are simply absent from the checkpoint chain.
    pub fn push_connected(self, height: u32, value: T) -> Result<Self, ConnectError<T>> {
        // Height must always be strictly greater.
        if self.height() >= height {
            return Err(ConnectError {
                checkpoint: self,
                value,
            });
        }
        // For adjacent blocks only, validate that prev_blockhash links correctly.
        // Sparse pushes (gap > 1) skip this check: intermediate blocks are absent.
        if height == self.height() + 1 && value.prev_blockhash() != self.hash() {
            return Err(ConnectError {
                checkpoint: self,
                value,
            });
        }
        // Preconditions are satisfied; push cannot fail.
        match self.push(height, value) {
            Ok(cp) => Ok(cp),
            Err(_) => unreachable!("height was verified to be strictly greater"),
        }
    }

    /// Extend the chain with connected `(height, value)` entries.
    ///
    /// Each entry must satisfy the same preconditions as [`push_connected`](Self::push_connected).
    /// On failure, the error contains the last successfully built checkpoint and the
    /// first rejected value.
    pub fn extend_connected(
        self,
        items: impl IntoIterator<Item = (u32, T)>,
    ) -> Result<Self, ConnectError<T>> {
        let mut curr = self;
        for (height, value) in items {
            curr = curr.push_connected(height, value)?;
        }
        Ok(curr)
    }
}

impl<T: ToBlockHash + HasPrevBlockhash + Clone> CheckPoint<T> {
    /// Insert `value` at `height`, enforcing `prev_blockhash` connectivity.
    ///
    /// Behaves like [`insert`](Self::insert) but additionally validates the
    /// `prev_blockhash` link between adjacent checkpoints:
    ///
    /// - If a checkpoint at `height` already exists with the same hash, returns `self`
    ///   unchanged.
    /// - If the checkpoint at `height - 1` has a hash that conflicts with
    ///   `value.prev_blockhash()`, that checkpoint is *displaced* (evicted along with
    ///   everything above it).
    /// - After inserting, any tail blocks that were above the insertion point are
    ///   re-validated via [`push_connected`](Self::push_connected). The first block
    ///   that no longer connects causes the chain to be truncated there; the last
    ///   valid checkpoint is returned.
    ///
    /// # Panics
    ///
    /// Panics if insertion implies a different genesis hash than the one anchoring
    /// the chain (height 0 is immutable).
    #[must_use]
    pub fn insert_connected(self, height: u32, value: T) -> Self {
        let mut cp = self.clone();
        let mut tail: Vec<(u32, T)> = vec![];

        let base = loop {
            // Genesis (height 0) must remain immutable.
            if cp.height() == 0 {
                let implied_genesis = match height {
                    0 => Some(value.to_blockhash()),
                    1 => Some(value.prev_blockhash()),
                    _ => None,
                };
                if let Some(hash) = implied_genesis {
                    assert_eq!(hash, cp.hash(), "inserted data implies a different genesis");
                }
            }

            if cp.height() > height {
                // Above insertion: collect for potential re-insertion.
                // Connectivity is re-validated during the rebuild step below.
                tail.push((cp.height(), cp.value().clone()));
            } else if cp.height() == height {
                if cp.hash() == value.to_blockhash() {
                    return self; // Already present — no change.
                }
                // Hash conflict at this height: evict everything at and above.
                tail.clear();
            } else if cp.height() + 1 == height && value.prev_blockhash() != cp.hash() {
                // Displacement: the adjacent-below checkpoint conflicts with
                // `value.prev_blockhash()`. Evict it along with everything above;
                // it will not be used as the base.
                tail.clear();
            } else {
                // cp.height() < height with no conflict: this is our base.
                break Some(cp);
            }

            match cp.prev() {
                Some(prev) => cp = prev,
                None => break None,
            }
        };

        // Append the inserted block then restore ascending order.
        tail.push((height, value));
        let ascending: Vec<(u32, T)> = tail.into_iter().rev().collect();

        // Rebuild the chain above the base. `push_connected` re-validates adjacency:
        // any tail block whose `prev_blockhash` conflicts with its predecessor causes
        // the chain to be truncated and the last valid checkpoint is returned.
        let seed = match base {
            Some(base_cp) => base_cp,
            None => {
                // No base: the chain root itself was displaced.
                // Start a new chain from the lowest tail item (the inserted value).
                let mut iter = ascending.into_iter();
                let (height, value) = iter.next().expect("tail always contains the inserted value");
                let mut curr = CheckPoint::new(height, value);
                for (height, value) in iter {
                    curr = match curr.push_connected(height, value) {
                        Ok(cp) => cp,
                        Err(ConnectError { checkpoint, .. }) => return checkpoint,
                    };
                }
                return curr;
            }
        };

        let mut curr = seed;
        for (height, value) in ascending {
            curr = match curr.push_connected(height, value) {
                Ok(cp) => cp,
                Err(ConnectError { checkpoint, .. }) => return checkpoint,
            };
        }
        curr
    }
}

impl<T> CheckPoint<T> {
    /// The [`BlockId`] (height + hash) of this checkpoint.
    pub fn block_id(&self) -> BlockId {
        BlockId {
            height: self.0.height,
            hash: self.0.hash,
        }
    }

    /// The block height.
    pub fn height(&self) -> u32 {
        self.0.height
    }

    /// The block hash.
    pub fn hash(&self) -> BlockHash {
        self.0.hash
    }

    /// The block value stored at this checkpoint.
    pub fn value(&self) -> &T {
        &self.0.value
    }

    /// 0-based position in the chain (genesis = 0).
    pub fn index(&self) -> u32 {
        self.0.index
    }

    /// Total number of checkpoints in the chain.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        (self.0.index + 1) as usize
    }

    /// The immediately previous checkpoint, or `None` at genesis.
    pub fn prev(&self) -> Option<CheckPoint<T>> {
        self.0.prev.clone().map(CheckPoint)
    }

    fn skip(&self) -> Option<CheckPoint<T>> {
        self.0.skip.clone().map(CheckPoint)
    }

    /// Iterate checkpoints from tip toward genesis.
    pub fn iter(&self) -> CheckPointIter<T> {
        self.clone().into_iter()
    }

    /// Find the checkpoint at `height` using skip pointers — O(log n).
    pub fn get(&self, height: u32) -> Option<Self> {
        if height > self.height() {
            return None;
        }
        let mut current = self.clone();
        while current.height() > height {
            current = match current.skip() {
                Some(skip_cp) if skip_cp.height() >= height => skip_cp,
                _ => current.prev()?,
            };
        }
        (current.height() == height).then_some(current)
    }

    /// Find the checkpoint at position `target_index` using skip pointers.
    fn checkpoint_at_index(&self, target_index: u32) -> Option<Self> {
        if target_index > self.0.index {
            return None;
        }
        let mut current = self.clone();
        while current.0.index > target_index {
            let cur_idx = current.0.index;
            let skip_idx = get_skip_index(cur_idx);
            let prev_skip_idx = get_skip_index(cur_idx.saturating_sub(1));
            current = match current.skip() {
                Some(skip_cp)
                    if skip_idx == target_index
                        || (skip_idx > target_index
                            && !(prev_skip_idx < skip_idx.saturating_sub(2)
                                && prev_skip_idx >= target_index)) =>
                {
                    skip_cp
                }
                _ => current.prev()?,
            };
        }
        (current.index() == target_index).then_some(current)
    }

    /// Walk back from this checkpoint to the highest checkpoint at or below `target_height`.
    ///
    /// Returns `None` when the chain's lowest checkpoint is itself above `target_height`.
    fn walk_to_floor(&self, target_height: u32) -> Option<Self> {
        let mut curr = self.clone();
        while curr.height() > target_height {
            let skip = curr.skip();
            let take_skip = match &skip {
                Some(skip_cp) if skip_cp.height() < target_height => false,
                Some(skip_cp) if skip_cp.height() == target_height => true,
                // Skip lands above the target. Prefer prev's skip when it makes a
                // strictly bigger jump (>2 heights lower than current's skip) and
                // still reaches the target — riding prev's skip next iteration is
                // then faster overall.
                Some(skip_cp) => match curr.prev().and_then(|p| p.skip()) {
                    Some(prev_skip_cp) => {
                        let skip_gap = skip_cp.height().saturating_sub(prev_skip_cp.height());
                        !(skip_gap > 2 && prev_skip_cp.height() >= target_height)
                    }
                    None => true,
                },
                None => false,
            };
            curr = if take_skip { skip? } else { curr.prev()? };
        }
        Some(curr)
    }

    /// Iterate checkpoints within a height range in tip-to-genesis order.
    pub fn range<R>(&self, range: R) -> impl Iterator<Item = CheckPoint<T>>
    where
        R: RangeBounds<u32>,
    {
        let start_bound = range.start_bound().cloned();
        let end_bound = range.end_bound().cloned();

        // Quick seek to the end bound instead of linear traversal
        let end = match end_bound {
            core::ops::Bound::Included(inc_bound) => self.walk_to_floor(inc_bound),
            core::ops::Bound::Excluded(exc_bound) => {
                exc_bound.checked_sub(1).and_then(|b| self.walk_to_floor(b))
            }
            core::ops::Bound::Unbounded => Some(self.clone()),
        };

        end.into_iter()
            .flat_map(IntoIterator::into_iter)
            .take_while(move |cp| match start_bound {
                core::ops::Bound::Included(inc_bound) => cp.height() >= inc_bound,
                core::ops::Bound::Excluded(exc_bound) => cp.height() > exc_bound,
                core::ops::Bound::Unbounded => true,
            })
    }

    /// `true` if both handles point to the same allocated node.
    pub fn eq_ptr(&self, other: &Self) -> bool {
        Arc::as_ptr(&self.0) == Arc::as_ptr(&other.0)
    }
}

impl<T: Clone + PartialEq + ToBlockHash + core::fmt::Debug> CheckPoint<T> {
    /// Insert or replace a checkpoint at `height`.
    ///
    /// If an identical value already exists at `height`, the chain is unchanged.
    /// If a conflicting value exists, all checkpoints above it are dropped.
    /// Panics if trying to replace the genesis block.
    #[must_use]
    pub fn insert(self, height: u32, value: T) -> Self {
        let mut cp = self.clone();
        let mut tail: Vec<(u32, T)> = vec![];
        let base = loop {
            if cp.height() == height {
                if cp.value() == &value {
                    return self;
                }
                assert_ne!(cp.height(), 0, "cannot replace genesis block");
                tail = vec![];
                break cp.prev().expect("can't be called on genesis block");
            }
            if cp.height() < height {
                break cp;
            }
            tail.push((cp.height(), cp.value().clone()));
            cp = cp.prev().expect("will break before genesis block");
        };
        base.extend(core::iter::once((height, value)).chain(tail.into_iter().rev()))
            .expect("tail is in order")
    }
}

/// Iterator over a [`CheckPoint`] chain from tip toward genesis.
pub struct CheckPointIter<T> {
    current: Option<Arc<Node<T>>>,
}

impl<T> Iterator for CheckPointIter<T> {
    type Item = CheckPoint<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current.clone()?;
        self.current.clone_from(&current.prev);
        Some(CheckPoint(current))
    }
}

impl<T> IntoIterator for CheckPoint<T> {
    type Item = CheckPoint<T>;
    type IntoIter = CheckPointIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        CheckPointIter {
            current: Some(self.0),
        }
    }
}

/// Compute the skip-pointer target index for the node at `index`.
fn get_skip_index(index: u32) -> u32 {
    // Clears the lowest set bit of `n`. `wrapping_sub` avoids integer casts and
    // makes `invert_lowest_one(0) == 0` explicit (the odd-index branch relies on
    // this when `index == 3`: `invert_lowest_one(invert_lowest_one(2)) + 1 = 1`).
    fn invert_lowest_one(n: u32) -> u32 {
        n & n.wrapping_sub(1)
    }

    if index < 2 {
        return 0;
    }
    if index & 1 == 0 {
        invert_lowest_one(index)
    } else {
        invert_lowest_one(invert_lowest_one(index - 1)) + 1
    }
}

#[cfg(test)]
mod test {
    use bitcoin::block::Header;
    use bitcoin::hashes::Hash;
    use bitcoin::{pow, TxMerkleNode};

    use super::*;
    use crate::collections::BTreeMap;

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

    fn genesis() -> Header {
        header(BlockHash::all_zeros(), Some(0))
    }

    /// Build a dense chain of `count` blocks (heights `0..count`) linked by `BlockHash`.
    fn dense_chain(count: u32) -> CheckPoint<BlockHash> {
        let mut cp = CheckPoint::new(0, Hash::hash(b"genesis"));
        for height in 1..count {
            let hash = Hash::hash(height.to_be_bytes().as_slice());
            cp = cp.push(height, hash).unwrap();
        }
        cp
    }

    // --- Invariants: `push_connected` / `extend_connected` / `ConnectError` ---

    #[test]
    fn push_connected_rejects_non_increasing_height() {
        let cp = CheckPoint::new(5, genesis());
        let h = header(genesis().block_hash(), Some(1));

        let err = cp.clone().push_connected(5, h).unwrap_err();
        assert_eq!(err.checkpoint.height(), 5, "tip is returned unchanged on error");
        assert_eq!(err.value, h);

        let err = cp.push_connected(4, h).unwrap_err();
        assert_eq!(err.value, h);
    }

    #[test]
    fn push_connected_rejects_adjacent_prev_blockhash_mismatch() {
        let gen = genesis();
        let cp = CheckPoint::new(0, gen);
        // `h`'s prev_blockhash doesn't match `gen`'s hash.
        let h = header(BlockHash::all_zeros(), Some(1));

        let err = cp.clone().push_connected(1, h).unwrap_err();
        assert_eq!(err.checkpoint.hash(), gen.block_hash());
        assert_eq!(err.value, h);
    }

    #[test]
    fn push_connected_accepts_adjacent_match() {
        let gen = genesis();
        let h = header(gen.block_hash(), Some(1));

        let cp = CheckPoint::new(0, gen).push_connected(1, h).unwrap();
        assert_eq!(cp.hash(), h.block_hash());
    }

    #[test]
    fn push_connected_skips_check_on_gap() {
        let gen = genesis();
        // `h`'s prev_blockhash doesn't match `gen`, but height 2 is a gapped push
        // (height 1 is absent), so the linkage check doesn't apply.
        let h = header(BlockHash::all_zeros(), Some(2));

        let cp = CheckPoint::new(0, gen).push_connected(2, h).unwrap();
        assert_eq!(cp.hash(), h.block_hash());
    }

    #[test]
    fn extend_connected_stops_at_first_broken_link() {
        let gen = genesis();
        let h1 = header(gen.block_hash(), Some(1));
        let broken_h2 = header(BlockHash::all_zeros(), Some(2)); // doesn't extend h1

        let err = CheckPoint::new(0, gen)
            .extend_connected([(1, h1), (2, broken_h2)])
            .unwrap_err();
        assert_eq!(err.checkpoint.height(), 1, "should retain the successfully pushed h1");
        assert_eq!(err.checkpoint.hash(), h1.block_hash());
        assert_eq!(err.value, broken_h2);
    }

    // --- Invariants: `insert_connected` ---

    #[test]
    fn insert_connected_noop_on_identical_value() {
        let gen = genesis();
        let h1 = header(gen.block_hash(), Some(1));
        let cp = CheckPoint::new(0, gen).push_connected(1, h1).unwrap();

        let same = cp.clone().insert_connected(1, h1);
        assert!(same.eq_ptr(&cp), "identical insert must return the chain unchanged");
    }

    #[test]
    fn insert_connected_evicts_above_on_hash_conflict_at_height() {
        let gen = genesis();
        let h1 = header(gen.block_hash(), Some(1));
        let h2 = header(h1.block_hash(), Some(2));
        let cp = CheckPoint::new(0, gen)
            .push_connected(1, h1)
            .unwrap()
            .push_connected(2, h2)
            .unwrap();

        // A conflicting header at height 1 evicts both h1 and h2 (which sits above it).
        let h1_conflict = header(gen.block_hash(), Some(101));
        let cp = cp.insert_connected(1, h1_conflict);
        assert_eq!(cp.height(), 1);
        assert_eq!(cp.hash(), h1_conflict.block_hash());
    }

    #[test]
    fn insert_connected_displaces_adjacent_below_on_prev_blockhash_conflict() {
        let gen = genesis();
        let h1 = header(gen.block_hash(), Some(1));
        // h3 is gapped (height 2 absent), so its prev_blockhash was never validated.
        let h3 = header(BlockHash::all_zeros(), Some(3));
        let cp = CheckPoint::new(0, gen).push_connected(1, h1).unwrap().insert(3, h3);

        // h2's prev_blockhash doesn't match h1's hash: h1 is displaced (along with h3,
        // which sits above it), rather than becoming h2's validated parent.
        let h2 = header(BlockHash::all_zeros(), Some(2));
        let cp = cp.insert_connected(2, h2);
        assert_eq!(cp.height(), 2);
        assert_eq!(cp.hash(), h2.block_hash());
        assert!(cp.get(1).is_none(), "h1 should have been displaced");
    }

    #[test]
    fn insert_connected_truncates_tail_that_no_longer_connects() {
        let gen = genesis();
        // h3 is gapped (height 2 absent), so its prev_blockhash was never validated
        // against a real predecessor.
        let h3 = header(BlockHash::all_zeros(), Some(3));
        let cp = CheckPoint::new(0, gen).insert(3, h3);

        // Filling the gap at height 2 makes h3 adjacent to h2; since h3's prev_blockhash
        // doesn't actually point to h2, the rebuild truncates the chain at h2.
        let h2 = header(gen.block_hash(), Some(2));
        let cp = cp.insert_connected(2, h2);
        assert_eq!(cp.height(), 2, "h3 must be dropped since it no longer connects");
        assert_eq!(cp.hash(), h2.block_hash());
    }

    #[test]
    #[should_panic(expected = "inserted data implies a different genesis")]
    fn insert_connected_panics_on_genesis_mismatch() {
        let cp = CheckPoint::new(0, genesis());
        let other_genesis = header(BlockHash::all_zeros(), Some(999));
        let _ = cp.insert_connected(0, other_genesis);
    }

    // --- Skip-pointer machinery ---

    #[test]
    fn get_matches_linear_scan() {
        const COUNT: u32 = 500;
        let cp = dense_chain(COUNT);

        let expected: BTreeMap<u32, BlockHash> =
            cp.iter().map(|c| (c.height(), c.hash())).collect();
        for height in 0..COUNT {
            assert_eq!(
                cp.get(height).map(|c| c.hash()),
                expected.get(&height).copied(),
                "mismatch at height {height}"
            );
        }
        assert!(cp.get(COUNT).is_none(), "height past tip must be None");
    }

    #[test]
    fn checkpoint_at_index_matches_naive_traversal() {
        const COUNT: u32 = 500;
        let cp = dense_chain(COUNT);

        let expected: BTreeMap<u32, BlockId> =
            cp.iter().map(|c| (c.index(), c.block_id())).collect();
        for (&index, &block_id) in &expected {
            assert_eq!(
                cp.checkpoint_at_index(index).map(|c| c.block_id()),
                Some(block_id),
                "mismatch at index {index}"
            );
        }
        assert!(cp.checkpoint_at_index(cp.index() + 1).is_none());
    }

    #[test]
    fn walk_to_floor_matches_naive_traversal() {
        // Build a chain with gaps so some target heights aren't directly present.
        let gen: BlockHash = Hash::hash(b"genesis");
        let heights: [u32; 6] = [0, 3, 4, 8, 15, 16];
        let mut cp = CheckPoint::new(0, gen);
        for &height in &heights[1..] {
            let hash = Hash::hash(height.to_be_bytes().as_slice());
            cp = cp.insert(height, hash);
        }

        for target in 0..=20u32 {
            let expected = cp.iter().find(|c| c.height() <= target).map(|c| c.block_id());
            assert_eq!(
                cp.walk_to_floor(target).map(|c| c.block_id()),
                expected,
                "mismatch for target height {target}"
            );
        }
    }

    #[test]
    fn range_matches_linear_filter() {
        // A gapped chain stresses the bound-seeking logic in `range` beyond a dense chain.
        let gen: BlockHash = Hash::hash(b"genesis");
        let heights: [u32; 5] = [0, 2, 5, 6, 10];
        let mut cp = CheckPoint::new(0, gen);
        for &height in &heights[1..] {
            let hash = Hash::hash(height.to_be_bytes().as_slice());
            cp = cp.insert(height, hash);
        }
        let all: Vec<u32> = cp.iter().map(|c| c.height()).collect();

        let got: Vec<u32> = cp.range(0..11).map(|c| c.height()).collect();
        assert_eq!(got, all);

        let got: Vec<u32> = cp.range(3..7).map(|c| c.height()).collect();
        let expected: Vec<u32> = all.iter().copied().filter(|h| (3..7).contains(h)).collect();
        assert_eq!(got, expected);

        let got: Vec<u32> = cp.range(6..=10).map(|c| c.height()).collect();
        let expected: Vec<u32> = all.iter().copied().filter(|h| *h >= 6).collect();
        assert_eq!(got, expected);

        let got: Vec<u32> = cp.range(100..200).map(|c| c.height()).collect();
        assert!(got.is_empty());
    }
}
