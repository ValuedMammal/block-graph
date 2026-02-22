//! [`CheckPoint`] is a cheaply cloneable, singly-linked list implementation with skip pointers.
//!
//! This module implements a skip list-like data structure optimized for blockchain data,
//! where each checkpoint represents a block at a specific height. The skip pointers
//! enable efficient traversal and lookup operations.

use alloc::sync::Arc;
use alloc::vec;
use core::fmt::Debug;
use core::ops::RangeBounds;

use bdk_chain::{bitcoin, BlockId, ToBlockHash};
use bitcoin::BlockHash;

/// A checkpoint in the blockchain, representing a linked list node with skip pointers.
///
/// CheckPoint is guaranteed to have at least one element and provides efficient
/// traversal through both sequential (prev) and skip pointers for faster lookups.
/// Each checkpoint contains block data and maintains connections to previous blocks.
#[derive(Debug)]
pub struct CheckPoint<T>(pub(crate) Arc<Node<T>>);

/// Internal node structure for the CheckPoint linked list.
///
/// Contains the block data along with pointers for efficient traversal:
/// - `prev`: Points to the immediately previous block in the chain
/// - `skip`: Points to a block further back, enabling logarithmic-time lookups
/// - `index`: The position of this node in the chain (0-based)
#[derive(Debug)]
pub(crate) struct Node<T> {
    /// Block height (also used as the key for ordering)
    pub height: u32,
    /// Block hash for this checkpoint
    pub hash: BlockHash,
    /// The generic block data
    pub value: T,
    /// Pointer to the immediately previous node in the chain
    pub prev: Option<Arc<Node<T>>>,
    /// Skip pointer to a node further back, enabling fast traversal
    pub skip: Option<Arc<Node<T>>>,
    /// 0-based index of this node in the chain
    pub index: u32,
}

impl<T> Drop for Node<T> {
    /// Custom drop implementation to prevent stack overflow on deeply nested chains.
    ///
    /// Iteratively drops nodes to avoid recursive destruction that could
    /// cause stack overflow on long chains.
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
    /// Create a new CheckPoint starting a new chain with the given block data.
    ///
    /// This creates the genesis checkpoint at the specified height with the provided value.
    /// The resulting checkpoint will have index 0 and no skip or prev pointers.
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

    /// Construct a CheckPoint chain from an iterator of (height, value) entries.
    ///
    /// The entries must be provided in ascending height order. Returns an error
    /// if the iterator is empty or if heights are not strictly increasing.
    ///
    /// # Errors
    ///
    /// Returns `Err(None)` if the iterator is empty, or `Err(Some(checkpoint))`
    /// if the heights are not in strictly increasing order.
    pub fn from_entries<I>(entries: I) -> Result<Self, Option<Self>>
    where
        I: IntoIterator<Item = (u32, T)>,
    {
        let mut entry_iter = entries.into_iter();
        let (initial_height, initial_value) = entry_iter.next().ok_or(None)?;
        let cp = Self::new(initial_height, initial_value);
        Ok(cp.extend(entry_iter)?)
    }

    /// Extend the checkpoint chain with additional (height, value) entries.
    ///
    /// All entries must have heights strictly greater than the current tip.
    /// This method processes entries sequentially, building the chain forward.
    ///
    /// # Errors
    ///
    /// Returns the original checkpoint if any entry has a height not strictly
    /// greater than the previous entry.
    pub fn extend<I>(self, items: I) -> Result<Self, Self>
    where
        I: IntoIterator<Item = (u32, T)>,
    {
        let mut cp = self;
        for (height, value) in items.into_iter() {
            cp = cp.push(height, value)?;
        }
        Ok(cp)
    }

    /// Add a new block to the front of the checkpoint chain.
    ///
    /// Creates a new checkpoint with the given height and value, linking it to
    /// the current chain. The new checkpoint becomes the new tip of the chain.
    /// Skip pointers are automatically calculated for efficient traversal.
    ///
    /// # Errors
    ///
    /// Returns the original checkpoint unchanged if the new height is not
    /// strictly greater than the current tip height.
    pub fn push(self, height: u32, value: T) -> Result<Self, Self> {
        if self.height() >= height {
            return Err(self);
        }

        let new_index = self.0.index + 1;

        // Calculate skip pointer using index-based logic for efficient lookups
        let skip_target_index = get_skip_index(new_index);
        let skip_node = self.checkpoint_at_index(skip_target_index).map(|cp| cp.0);
        debug_assert!(skip_node.is_some(), "Each new node must have a skip pointer");

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

    /// Insert or replace a block at the specified height in the chain.
    ///
    /// This method allows inserting a block at any valid height, potentially
    /// reorganizing the chain. If a block already exists at the given height:
    /// - If the value is identical, returns the original chain unchanged
    /// - If the value differs, replaces it and invalidates all subsequent blocks
    ///
    /// # Panics
    ///
    /// Panics if attempting to replace the root (genesis) block of the chain.
    pub fn insert(self, height: u32, value: T) -> Self
    where
        T: Clone + PartialEq,
    {
        let mut cp = self.clone();
        let mut tail_blocks = vec![];

        // Traverse backwards to find the insertion point
        let base = loop {
            if cp.height() == height {
                if cp.value() == value {
                    // Value is identical, no change needed
                    return self;
                }
                // Replacing existing value invalidates the tail
                tail_blocks = vec![];
                break cp.prev().expect("cannot replace genesis block");
            }
            // Found our insertion base
            if cp.height() < height {
                break cp;
            }
            // Collect blocks that will be re-added after insertion
            tail_blocks.push((cp.height(), cp.value()));
            cp = cp.prev().expect("will break before root");
        };

        base.extend(core::iter::once((height, value)).chain(tail_blocks.into_iter().rev()))
            .expect("tail blocks must be in ascending height order")
    }

    /// Create an iterator over checkpoints within the specified height range.
    ///
    /// Returns checkpoints in descending height order (tip to genesis direction)
    /// that fall within the given range bounds. Supports all standard range types.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use block_graph::CheckPoint;
    /// # use bdk_chain::bitcoin;
    /// # use bitcoin::BlockHash;
    /// # use bitcoin::hashes::Hash;
    /// # let checkpoint = CheckPoint::<BlockHash>::new(0, Hash::hash(b"0"));
    /// // Get checkpoints between heights 10 and 20 (inclusive)
    /// for cp in checkpoint.range(10..=20) {
    ///     println!("Height: {}", cp.height());
    /// }
    /// ```
    pub fn range(&self, range: impl RangeBounds<u32>) -> impl Iterator<Item = CheckPoint<T>> {
        let start_bound = range.start_bound().cloned();
        let end_bound = range.end_bound().cloned();
        self.iter()
            .skip_while(move |checkpoint| match end_bound {
                core::ops::Bound::Included(included) => checkpoint.height() > included,
                core::ops::Bound::Excluded(excluded) => checkpoint.height() >= excluded,
                core::ops::Bound::Unbounded => false,
            })
            .take_while(move |checkpoint| match start_bound {
                core::ops::Bound::Included(included) => checkpoint.height() >= included,
                core::ops::Bound::Excluded(excluded) => checkpoint.height() > excluded,
                core::ops::Bound::Unbounded => true,
            })
    }
}

impl<T> CheckPoint<T> {
    /// Get the immediately previous checkpoint in the chain.
    ///
    /// Returns `None` if this is the genesis checkpoint (root of the chain).
    pub fn prev(&self) -> Option<Self> {
        self.0.prev.clone().map(Self)
    }

    /// Get the block height of this checkpoint.
    pub fn height(&self) -> u32 {
        self.0.height
    }

    /// Get the block hash of this checkpoint.
    pub fn hash(&self) -> BlockHash {
        self.0.hash
    }

    /// Get the [`BlockId`] for this checkpoint.
    pub fn block_id(&self) -> BlockId {
        BlockId {
            height: self.0.height,
            hash: self.0.hash,
        }
    }

    /// Get a clone of the block data stored in this checkpoint.
    pub fn value(&self) -> T
    where
        T: Clone,
    {
        self.0.value.clone()
    }

    /// Get the skip pointer checkpoint for efficient backward traversal.
    ///
    /// The skip pointer points to a checkpoint further back in the chain,
    /// enabling logarithmic-time lookups. Returns `None` for the genesis block.
    fn skip(&self) -> Option<Self> {
        self.0.skip.clone().map(Self)
    }

    /// Get the 0-based index of this [`CheckPoint`] in the chain.
    ///
    /// The root has index 0, and each subsequent [`CheckPoint`] increments the index.
    pub fn index(&self) -> u32 {
        self.0.index
    }

    /// Get the length of the list, i.e. the total count of checkpoint nodes
    /// that are linked.
    #[allow(clippy::len_without_is_empty)] // CheckPoint can't be empty
    pub fn len(&self) -> usize {
        (self.0.index + 1) as usize
    }

    /// Check if two checkpoints share the same underlying memory allocation.
    ///
    /// This is more efficient than value equality checking when you need to
    /// test if two checkpoint references point to the exact same node.
    pub fn eq_ptr(&self, other: &Self) -> bool {
        Arc::as_ptr(&self.0) == Arc::as_ptr(&other.0)
    }

    /// Create an iterator that traverses the checkpoint chain from tip to genesis.
    ///
    /// The iterator yields checkpoints in descending height order, starting from
    /// this checkpoint and following prev pointers back to the genesis.
    pub fn iter(&self) -> CheckPointIter<T> {
        CheckPointIter {
            current: Some(self.0.clone()),
        }
    }

    /// Get the checkpoint at the specified height if it exists.
    ///
    /// Returns `None` if no checkpoint exists at the requested height or if
    /// the height is greater than this checkpoint's height.
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

    /// Look up a checkpoint at the specified index using skip pointers.
    ///
    /// This is an internal method that uses skip pointers to efficiently
    /// navigate to a checkpoint at a specific 0-based index position.
    ///
    /// Returns `None` if the target index is greater than this checkpoint's index.
    fn checkpoint_at_index(&self, target_index: u32) -> Option<Self> {
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

impl<T: Clone + PartialEq> PartialEq for CheckPoint<T> {
    /// Compare two checkpoint chains for value equality.
    ///
    /// Two chains are equal if they contain the same sequence of (height, value) pairs,
    /// regardless of their internal pointer structure.
    fn eq(&self, other: &Self) -> bool {
        self.iter()
            .map(|cp| (cp.height(), cp.value()))
            .eq(other.iter().map(|cp| (cp.height(), cp.value())))
    }
}

/// Iterator for traversing a CheckPoint chain from tip to genesis.
///
/// Yields checkpoints in descending height order by following prev pointers.
pub struct CheckPointIter<T> {
    /// Current node being processed (None when iteration is complete)
    current: Option<Arc<Node<T>>>,
}

impl<T> Iterator for CheckPointIter<T> {
    type Item = CheckPoint<T>;

    /// Get the next checkpoint in the chain (moving toward genesis).
    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current.clone()?;
        self.current.clone_from(&current.prev);

        Some(CheckPoint(current))
    }
}

/// Clear the lowest set bit in the given number.
///
/// This is a bit manipulation utility used in skip list index calculations.
/// For example: `invert_lowest_1(6)` (binary 110) returns 4 (binary 100).
fn invert_lowest_1(n: i32) -> i32 {
    n & (n - 1)
}

/// Calculate the target index for the skip pointer at the given index.
///
/// This function implements the skip list algorithm used in Bitcoin Core's chain indexing.
/// It ensures that skip pointers create an efficient logarithmic lookup structure.
///
/// The algorithm:
/// - For even indices: Clear the lowest set bit
/// - For odd indices: Apply the even algorithm to (index - 1), then again to the result,
///   and finally add 1.
///
/// ref: `GetSkipHeight` in Bitcoin Core <https://github.com/bitcoin/bitcoin/blob/2d6426c296ea43e62980d87d94fde0e94318a341/src/chain.cpp#L83>
fn get_skip_index(index: u32) -> u32 {
    assert!(index < i32::MAX as u32, "index cannot exceed");
    if index < 2 {
        return 0;
    }
    let index: i32 = index.try_into().expect("index should cast to i32");
    let skip_index = if index & 1 == 0 {
        invert_lowest_1(index)
    } else {
        invert_lowest_1(invert_lowest_1(index - 1)) + 1
    };
    skip_index.try_into().expect("skip_index should be non-negative")
}

#[cfg(test)]
mod test {
    use super::*;

    use bitcoin::hashes::Hash;

    #[test]
    fn test_get_with_100_elements() {
        let genesis_hash = BlockHash::all_zeros();
        let mut checkpoint = CheckPoint::new(0, genesis_hash);
        let mut expected_values = vec![genesis_hash];

        for height in 1u32..100 {
            let block_hash: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
            expected_values.push(block_hash);
            checkpoint = checkpoint.push(height, block_hash).unwrap();
        }

        // Test getting elements at various heights
        for test_height in 0..100 {
            let result_checkpoint = checkpoint.get(test_height).expect("checkpoint should exist");
            assert_eq!(result_checkpoint.height(), test_height);
            assert_eq!(result_checkpoint.value(), expected_values[test_height as usize]);
        }

        // Test getting non-existent heights
        assert!(checkpoint.get(100).is_none(), "height 100 should not exist");
        assert!(checkpoint.get(150).is_none(), "height 150 should not exist");
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
        let mut checkpoint = CheckPoint::new(0, BlockHash::all_zeros());

        for height in 1u32..100 {
            let block_hash: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
            checkpoint = checkpoint.push(height, block_hash).unwrap();
        }

        // For each node verify that a skip pointer exists and points to a node
        // with a smaller height that is further back in the chain
        for height in 1..100 {
            let test_checkpoint = checkpoint.get(height).expect("checkpoint should exist");
            let skip_checkpoint = test_checkpoint
                .skip()
                .expect("every non-zero node should have a skip pointer");
            println!(
                "height={}, skip_checkpoint_height={}",
                height,
                skip_checkpoint.height()
            );
            assert!(
                skip_checkpoint.height() < height,
                "skip checkpoint height must be less than current checkpoint height"
            );
        }
    }

    #[test]
    fn test_skip_index_validity() {
        let initial_height = 0;
        let mut checkpoint = CheckPoint::<BlockHash>::new(initial_height, BlockHash::all_zeros());

        // Create sparse chain where heights are multiples of 50
        for i in 1..20 {
            let height: u32 = 50 * (i as u32);
            let block_hash: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
            checkpoint = checkpoint.push(height, block_hash).unwrap();
        }

        // For each node verify that a skip pointer exists and points to a node
        // with a smaller height that is further back in the chain
        for index in 0..20 {
            let result_checkpoint =
                checkpoint.checkpoint_at_index(index).expect("checkpoint should exist");
            assert_eq!(result_checkpoint.index(), index, "indices must increment sequentially");
            if index == 0 {
                assert_eq!(
                    result_checkpoint.height(),
                    initial_height,
                    "initial height must be at index 0"
                );
                assert!(
                    result_checkpoint.skip().is_none(),
                    "index 0 checkpoint should not have skip pointer"
                );
            }
            let height = result_checkpoint.height();
            let skip_checkpoint = result_checkpoint.skip();
            if index > 0 {
                assert!(
                    skip_checkpoint.is_some(),
                    "each non-zero index must have a skip pointer"
                );
            }
            let skip_checkpoint_height = skip_checkpoint.as_ref().map(CheckPoint::height);
            if let Some(skip_height) = skip_checkpoint_height {
                assert!(
                    checkpoint.get(skip_height).is_some(),
                    "skip checkpoint height must exist in the chain"
                );
            }
            assert!(
                skip_checkpoint_height < Some(height),
                "skip checkpoint height must be less than current checkpoint height"
            );
            println!(
                "index={}, height={}, skip_checkpoint_height={:?}",
                index, height, skip_checkpoint_height,
            );
        }
    }

    #[test]
    fn test_index_with_non_contiguous_heights() {
        let test_block_hash = BlockHash::all_zeros();
        let mut checkpoint = CheckPoint::new(0, test_block_hash);
        assert_eq!(checkpoint.checkpoint_at_index(0).unwrap(), checkpoint);

        let test_heights = [10, 25, 100, 500];

        // Push some checkpoints with non-contiguous heights (sparse chain)
        for height in test_heights {
            checkpoint = checkpoint.push(height, test_block_hash).unwrap();
        }

        assert!(checkpoint.skip().is_some(), "head should have skip pointer");

        // Test `get` at expected heights
        for (expected_index, expected_height) in (1..).zip(test_heights) {
            let result_checkpoint =
                checkpoint.get(expected_height).expect("checkpoint should exist");
            assert_eq!(result_checkpoint.height(), expected_height);
            assert_eq!(result_checkpoint.index(), expected_index);
        }

        // Test non-existent height
        assert!(checkpoint.get(11).is_none(), "height 11 should not exist");
    }

    #[test]
    fn test_checkpoint_len() {
        // Test single-element CheckPoint has length 1
        let mut checkpoint = CheckPoint::new(0, BlockHash::all_zeros());
        assert_eq!(checkpoint.len(), 1, "single-element checkpoint should have length 1");

        // Test that length increases by 1 for each push
        for height in 1u32..=10 {
            let block_hash: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
            checkpoint = checkpoint.push(height, block_hash).unwrap();
            let expected_len = (height + 1) as usize;
            assert_eq!(
                checkpoint.len(),
                expected_len,
                "checkpoint should have length {} after pushing height {}",
                expected_len,
                height
            );
        }

        // Final verification: chain with heights 0-10 should have length 11
        assert_eq!(checkpoint.len(), 11, "final checkpoint should have length 11");
    }
}
