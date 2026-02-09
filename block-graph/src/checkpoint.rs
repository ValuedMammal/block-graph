//! [`CheckPoint`] is a cheaply cloneable, singly-linked list.

use alloc::sync::Arc;
use alloc::vec;
use core::fmt;
use core::fmt::Debug;
use core::ops::RangeBounds;

use bdk_chain::{bitcoin, BlockId, ToBlockHash};
use bitcoin::BlockHash;

// TODO:
// - Implement Node::skip
// - Improve `get` efficiency
// - Add Node::index that stores the numeric index of a checkpoint starting from 0
// - Change to `get_skip_index` (instead of height), since in a "sparse" chain we can't assume that all heights are present
// - As we're now tracking the index, we should be able to implement `CheckPoint::len` for free
// - `range` could be improve to utilize fast search to the end bound
// - Store the blockhash inside Node, to avoid continually re-hashing

/// CheckPoint, guaranteed to have at least 1 element.
#[derive(Debug)]
pub struct CheckPoint<T>(pub(crate) Arc<Node<T>>);

/// Node containing both a key and value. The key is referred to as `height`.
#[derive(Debug)]
pub(crate) struct Node<T> {
    pub height: u32,
    pub hash: BlockHash,
    pub value: T,
    pub prev: Option<Arc<Node<T>>>,
    // pointer to a Node further back
    pub skip: Option<Arc<Node<T>>>,
    pub index: usize,
}

impl<T> Drop for Node<T> {
    fn drop(&mut self) {
        let mut current = self.prev.take();
        while let Some(node) = current {
            match Arc::into_inner(node) {
                Some(mut node) => {
                    current = node.prev.take();
                }
                None => break,
            }
        }
    }
}

impl<T> CheckPoint<T>
where
    T: Debug + ToBlockHash,
{
    /// Start a new list with `value` at the head.
    pub fn new(height: u32, value: T) -> Self {
        Self(Arc::new(Node {
            height,
            hash: value.to_blockhash(),
            value,
            prev: None,
            skip: None,
            index: 0,
        }))
    }

    /// Construct from an iterator of (height, value) entries.
    pub fn from_entries<I>(entries: I) -> Result<Self, Option<Self>>
    where
        I: IntoIterator<Item = (u32, T)>,
    {
        let mut entries = entries.into_iter();
        let (k, v) = entries.next().ok_or(None)?;
        let cp = Self::new(k, v);

        Ok(cp.extend(entries)?)
    }

    /// Extend with an iterator of (height, value) entries.
    pub fn extend<I>(self, items: I) -> Result<Self, Self>
    where
        I: IntoIterator<Item = (u32, T)>,
    {
        let mut cp = self;
        for (k, v) in items.into_iter() {
            cp = cp.push(k, v)?;
        }
        Ok(cp)
    }

    /// Push a `T` value onto the head of the list.
    ///
    /// Error if the height is not strictly greater than self.
    pub fn push(self, height: u32, value: T) -> Result<Self, Self> {
        if self.height() >= height {
            return Err(self);
        }

        let new_index = self.0.index + 1;

        // Calculate skip pointer using index-based logic
        let skip_index = get_skip_index(new_index);
        let skip_node = self.get_index(skip_index).map(|cp| cp.0);
        debug_assert!(skip_node.is_some(), "Each new node must have a skip");

        let node = Node {
            height,
            hash: value.to_blockhash(),
            value,
            prev: Some(self.0),
            skip: skip_node,
            index: new_index,
        };

        Ok(Self(Arc::new(node)))
    }

    /// Insert.
    ///
    /// Note this allows replacing the value of an entry. This panics if the caller attempts to
    /// replace the root of the current list.
    pub fn insert(self, height: u32, value: T) -> Self
    where
        T: Clone + PartialEq,
    {
        let mut cur = self.clone();
        let mut tail = vec![];

        // Look for the next nearest node smaller than the given height.
        let base = loop {
            if cur.height() == height {
                if cur.value() == value {
                    return self;
                }
                // If we're replacing a value, the tail is by implication invalid.
                tail = vec![];
                break cur.prev().expect("cannot replace root");
            }
            // We found our base.
            if cur.height() < height {
                break cur;
            }
            tail.push((cur.height(), cur.value()));
            cur = cur.prev().expect("will break before root");
        };

        base.extend(core::iter::once((height, value)).chain(tail.into_iter().rev()))
            .expect("tail must be in order")
    }

    /// Range.
    pub fn range(&self, range: impl RangeBounds<u32>) -> impl Iterator<Item = CheckPoint<T>> {
        let start_bound = range.start_bound().cloned();
        let end_bound = range.end_bound().cloned();
        self.iter()
            .skip_while(move |cp| match end_bound {
                core::ops::Bound::Included(included) => cp.height() > included,
                core::ops::Bound::Excluded(excluded) => cp.height() >= excluded,
                core::ops::Bound::Unbounded => false,
            })
            .take_while(move |cp| match start_bound {
                core::ops::Bound::Included(included) => cp.height() >= included,
                core::ops::Bound::Excluded(excluded) => cp.height() > excluded,
                core::ops::Bound::Unbounded => true,
            })
    }
}

impl<T> CheckPoint<T> {
    /// Prev.
    pub fn prev(&self) -> Option<Self> {
        self.0.prev.clone().map(Self)
    }

    /// Height.
    pub fn height(&self) -> u32 {
        self.0.height
    }

    /// Hash.
    pub fn hash(&self) -> BlockHash {
        self.0.hash
    }

    /// BlockId.
    pub fn block_id(&self) -> BlockId {
        BlockId {
            height: self.0.height,
            hash: self.0.hash,
        }
    }

    /// Value.
    pub fn value(&self) -> T
    where
        T: Clone,
    {
        self.0.value.clone()
    }

    /// Return the skip CheckPoint of this CheckPoint.
    fn skip(&self) -> Option<Self> {
        self.0.skip.clone().map(Self)
    }

    /// Get the current index of this checkpoint.
    pub fn index(&self) -> usize {
        self.0.index
    }

    /// Whether `self` and `other` share the same underlying pointer.
    pub fn eq_ptr(&self, other: &Self) -> bool {
        Arc::as_ptr(&self.0) == Arc::as_ptr(&other.0)
    }

    /// Iter.
    pub fn iter(&self) -> CheckPointIter<T> {
        CheckPointIter {
            cur: Some(self.0.clone()),
        }
    }

    /// Get the [`CheckPoint`] at the given `height` if it exists.
    pub fn get(&self, height: u32) -> Option<Self> {
        if height > self.height() {
            return None;
        }
        let mut current = self.clone();
        while current.height() > height {
            current = if let Some(skip_cp) = current.skip() {
                if skip_cp.height() >= height {
                    skip_cp
                } else {
                    current.prev()?
                }
            } else {
                current.prev()?
            };
        }
        (current.height() == height).then_some(current)
    }

    /// Get the [`CheckPoint`] at the given `index` if it exists.
    fn get_index(&self, target_index: usize) -> Option<Self> {
        if target_index > self.0.index {
            return None;
        }
        let mut current = self.clone();
        while current.0.index > target_index {
            let current_index = current.0.index;
            let skip_index = get_skip_index(current_index);
            let prev_index = current_index.saturating_sub(1);
            let prev_skip_index = get_skip_index(prev_index);
            current = if let Some(skip_cp) = current.skip() {
                if skip_index == target_index
                    // Only follow skip_cp if `prev_skip_index` isn't better
                    || (skip_index > target_index
                        && !(prev_skip_index < skip_index.saturating_sub(2)
                            && prev_skip_index >= target_index))
                {
                    skip_cp
                } else {
                    current.prev()?
                }
            } else {
                current.prev()?
            }
        }
        (current.index() == target_index).then_some(current)
    }
}

impl<T> Clone for CheckPoint<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T: fmt::Debug + Clone + PartialEq> PartialEq for CheckPoint<T> {
    fn eq(&self, other: &Self) -> bool {
        self.iter()
            .map(|cp| (cp.height(), cp.value()))
            .eq(other.iter().map(|cp| (cp.height(), cp.value())))
    }
}

/// CheckPoint iter.
pub struct CheckPointIter<T> {
    cur: Option<Arc<Node<T>>>,
}

impl<T> Iterator for CheckPointIter<T> {
    type Item = CheckPoint<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.cur.clone()?;
        self.cur.clone_from(&cur.prev);

        Some(CheckPoint(cur))
    }
}

/// Flips the lowest bit of `n` from 1 to 0 and returns the result.
fn invert_lowest_1(n: i32) -> i32 {
    n & (n - 1)
}

/// Compute what index to jump back to with the skip pointer.
/// ref: `GetSkipHeight` <https://github.com/bitcoin/bitcoin/blob/2d6426c296ea43e62980d87d94fde0e94318a341/src/chain.cpp#L83>
fn get_skip_index(index: usize) -> usize {
    if index < 2 {
        return 0;
    }
    let n = index as i32;
    let ret = if n & 1 == 0 {
        invert_lowest_1(n)
    } else {
        // Handle odd case separately
        invert_lowest_1(invert_lowest_1(n - 1)) + 1
    };
    ret.try_into().expect("n should be >= 0")
}

#[cfg(test)]
mod test {
    use super::*;

    use bitcoin::hashes::Hash;

    #[test]
    fn test_get_with_100_elements() {
        let hash_0 = BlockHash::all_zeros();
        let mut cp = CheckPoint::new(0, hash_0);
        let mut expected_values = vec![hash_0];

        for height in 1u32..100 {
            let value: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
            expected_values.push(value);
            cp = cp.push(height, value).unwrap();
        }

        // Test getting elements at various heights
        for test_height in 0..100 {
            let result_cp = cp.get(test_height).expect("cp should exist");
            assert_eq!(result_cp.height(), test_height);
            assert_eq!(result_cp.value(), expected_values[test_height as usize]);
        }

        // Test getting non-existent heights
        assert!(cp.get(100).is_none(), "height 100 should not exist");
        assert!(cp.get(150).is_none(), "height 150 should not exist");
    }

    #[test]
    fn test_get_skip_index_even() {
        // Test skip index calculation for even indices
        assert_eq!(get_skip_index(0), 0);
        assert_eq!(get_skip_index(2), 0);
        assert_eq!(get_skip_index(4), 0);
        assert_eq!(get_skip_index(6), 4);
        assert_eq!(get_skip_index(8), 0);

        // Test some larger values
        assert_eq!(get_skip_index(32), 0);
        assert_eq!(get_skip_index(64), 0);
        assert_eq!(get_skip_index(256 + 2), 256);
        assert_eq!(get_skip_index(1024 + 4), 1024);
    }

    #[test]
    fn test_get_skip_index_odd() {
        // Test skip index calculation for odd indices
        assert_eq!(get_skip_index(3), 1);
        assert_eq!(get_skip_index(5), 1);
        assert_eq!(get_skip_index(9), 1);
        assert_eq!(get_skip_index(15), 9);
    }

    #[test]
    fn test_skip_height_validity() {
        let mut cp = CheckPoint::new(0, BlockHash::all_zeros());

        for height in 1u32..100 {
            let value: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
            cp = cp.push(height, value).unwrap();
        }

        // For each node verify that a skip pointer exists and points to a node
        // with a smaller height that is further back in the chain
        for height in 1..100 {
            let test_cp = cp.get(height).expect("checkpoint should exist");
            let skip_cp = test_cp.skip().expect("every non-zero node should have a skip");
            println!("height={}, skip_cp_height={}", height, skip_cp.height());
            assert!(
                skip_cp.height() < height,
                "skip_cp height must be less than current cp height"
            );
        }
    }

    #[test]
    fn test_skip_index_validity() {
        let init_height = 0;
        let mut cp = CheckPoint::<BlockHash>::new(init_height, BlockHash::all_zeros());

        // Create sparse chain where heights are multiples of 50
        for i in 1..20 {
            let height: u32 = 50 * (i as u32);
            let value: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
            cp = cp.push(height, value).unwrap();
        }

        // For each node verify that a skip pointer exists and points to a node
        // with a smaller height that is further back in the chain
        for index in 0..20 {
            let result_cp = cp.get_index(index).expect("checkpoint should exist");
            assert_eq!(result_cp.index(), index, "indices must increment sequentially");
            if index == 0 {
                assert_eq!(result_cp.height(), init_height, "initial height must be index 0");
                assert!(result_cp.skip().is_none(), "index 0 cp should not have skip");
            }
            let height = result_cp.height();
            let skip_cp = result_cp.skip();
            if index > 0 {
                assert!(skip_cp.is_some(), "each non-zero index must have a skip");
            }
            let skip_cp_height = skip_cp.as_ref().map(CheckPoint::height);
            if let Some(skip_cp_height) = skip_cp_height {
                assert!(
                    cp.get(skip_cp_height).is_some(),
                    "skip_cp height must exist in the chain"
                );
            }
            assert!(
                skip_cp_height < Some(height),
                "skip_cp height must be less than current cp height"
            );
            println!(
                "index={}, height={}, skip_cp_height={:?}",
                index, height, skip_cp_height,
            );
        }
    }

    #[test]
    fn test_index_with_non_contiguous_heights() {
        let test_value = BlockHash::all_zeros();
        let mut cp = CheckPoint::new(0, test_value);
        assert_eq!(cp.get_index(0).unwrap(), cp);

        let test_heights = [10, 25, 100, 500];

        // Push some checkpoints with non-contiguous heights (sparse)
        for height in test_heights {
            cp = cp.push(height, test_value).unwrap();
        }

        assert!(cp.skip().is_some(), "head should have skip pointer");

        // Test `get` at expected heights
        for (expect_index, expect_height) in (1..).zip(test_heights) {
            let result_cp = cp.get(expect_height).expect("checkpoint 0 should exist");
            assert_eq!(result_cp.height(), expect_height);
            assert_eq!(result_cp.index(), expect_index);
        }

        // Test non-existent index
        assert!(cp.get(11).is_none(), "height 11 should not exist");
    }
}
