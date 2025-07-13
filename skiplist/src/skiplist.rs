//! [`SkipList`].

// Loosely based on <https://github.com/JP-Ellis/rust-skiplist>.

use alloc::boxed::Box;
use core::ops::RangeBounds;

use rand::rngs::ThreadRng;
use rand::Rng;

mod node;

pub use node::SkipListIter;

/// Nodes that are internal to the skiplist.
type Node<T> = node::Node<(u32, T)>;

/// Skip list.
#[derive(Debug)]
pub struct SkipList<T> {
    head: Box<Node<T>>,
    levels: usize,
    len: usize,
    rng: ThreadRng,
}

impl<T: Clone + PartialEq> PartialEq for SkipList<T> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().cloned().eq(other.iter().cloned())
    }
}

impl<T> SkipList<T> {
    /// Create a new skiplist with the given capacity.
    ///
    /// This will compute the number of levels from the desired capacity, for example given
    /// a `cap` of 10^6 nodes we:
    ///
    /// - Use a probability `p = 0.5` of finding a node at the next level
    /// - Find the log (base 2) of `cap`, or about 20 levels.
    ///
    /// It is possible for the skiplist to grow beyond the specified `cap`, but note that
    /// the supposed performance benefits can diminish as higher levels grow more densely
    /// packed.
    pub fn with_capacity(cap: usize) -> Self {
        let levels = core::cmp::max(1, (cap as f64).log2().round() as usize);
        let head = Box::new(Node::new_head(levels));
        let len = 0;
        let rng = rand::thread_rng();

        Self {
            head,
            levels,
            len,
            rng,
        }
    }

    /// Return the number of elements in the skiplist.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the skiplist is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Generate a new random level.
    ///
    /// All new nodes begin at the bottom-most level (0), and we use a coin flip
    /// to determine whether to promote the node to the next level. This makes
    /// the occurrence of "tall" nodes increasingly improbable.
    fn level(&mut self) -> usize {
        let mut level = 0;
        let p = 0.5;
        while level + 1 < self.levels {
            if !self.rng.gen_bool(p) {
                break;
            }
            level += 1;
        }
        level
    }

    /// Search for a value by a given `height` if it exists.
    pub fn get(&self, height: u32) -> Option<&T> {
        let seek_node = node::Seek::new(height).seek(&self.head);
        if let Some(target_node) = seek_node.next_ref() {
            if target_node.key() == Some(height) {
                return target_node.value();
            }
        }
        None
    }

    /// Get a reference to the node at the given `index`.
    ///
    /// Here index refers not to the value of node's key, but rather the value of the i-th
    /// element of the collection.
    ///
    /// Note also that since nodes are arranged in descending key order, the node at index 0
    /// should always return the node with the highest key.
    fn get_index(&self, index: usize) -> Option<&Node<T>> {
        if index >= self.len {
            return None;
        }

        self.head.advance(index + 1)
    }

    /// Insert `T` at the target `height`.
    ///
    /// Returns `None` if the value was newly inserted, or some old value if the value was replaced.
    pub fn insert(&mut self, height: u32, value: T) -> Option<T> {
        let level = self.level();

        let insert =
            node::InsertOrReplace::new(height, value, |k, v| Box::new(Node::new((k, v), level)));
        let head = self.head.as_mut();
        match insert.seek_and_insert_or_replace(head, head.level) {
            Ok(..) => {
                self.len += 1;
                None
            }
            Err(old_value) => Some(old_value),
        }
    }

    /// Remove the value of the given `height` if it exists.
    pub fn remove(&mut self, height: u32) -> Option<T> {
        let remove = node::Remove::new(height);
        let head = self.head.as_mut();
        match remove.seek_and_remove(head, head.level) {
            Ok((mut node, _)) => {
                self.len -= 1;
                let (_key, value) = node.value.take()?;
                Some(value)
            }
            Err(..) => None,
        }
    }

    /// Iterate over nodes in the skiplist.
    pub fn iter(&self) -> SkipListIter<(u32, T)> {
        SkipListIter::from_head(self.head.as_ref(), self.len)
    }

    /// Range.
    pub fn range(&self, range: impl RangeBounds<u32>) -> impl Iterator<Item = &(u32, T)> {
        use core::ops::Bound;
        let start: u32 = match range.start_bound().cloned() {
            Bound::Included(k) => k,
            Bound::Excluded(k) => k + 1,
            Bound::Unbounded => u32::MIN,
        };
        let end: u32 = match range.end_bound().cloned() {
            Bound::Included(k) => k,
            Bound::Excluded(k) => k - 1,
            Bound::Unbounded => u32::MAX,
        };

        let node = node::Seek::new(end).seek(self.head.as_ref());

        // Iterate nodes of the bottom level while the keys are in the intended range.
        let mut cur = node.next_ref();
        core::iter::from_fn(move || {
            let node = cur?;
            let key = node.key()?;
            if key < start {
                return None;
            }
            cur = node.next_ref();
            node.value.as_ref()
        })
    }
}

impl<T> core::ops::Index<usize> for SkipList<T> {
    type Output = (u32, T);

    fn index(&self, index: usize) -> &Self::Output {
        let node = self.get_index(index).expect("index out of bounds");
        node.value.as_ref().expect("node must not be head")
    }
}

#[cfg(test)]
#[allow(unused)]
mod test {
    use super::*;

    use bdk_chain::bitcoin;
    use bdk_chain::BlockId;
    use bitcoin::hashes::Hash;
    use bitcoin::BlockHash;

    use crate::skiplist::node::SkipListIter;
    use std::collections::{BTreeMap, BTreeSet};

    // Helper to display elements of SkipList.
    fn print_skiplist<T>(node: &Node<T>) {
        for level in (0..node.level).rev() {
            println!("\n===== Level {level} =====");
            let (level_node, _) = node.advance_while(level, |n, _| {
                if n.value.is_none() {
                    assert!(n.is_head());
                    print!("HEAD");
                } else {
                    let key = n.key().unwrap();
                    let dist = n.links_len[level];
                    print!(" - K[{}] d({})", key, dist);
                }
                true
            });
            if level_node.value.is_some() {
                print!(
                    " - K[{}] d({})",
                    level_node.key().unwrap(),
                    level_node.links_len[level]
                );
            }
        }
        println!();
    }

    #[test]
    fn test_skiplist_insert() {
        //
        // 0-1-2-3-...-H
        use node::SkipListIter;
        let exp_len = 50;
        let mut skiplist = SkipList::<BlockHash>::with_capacity(exp_len);

        for i in 0..exp_len {
            let height = i as u32;
            let hash: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
            let insert_res = skiplist.insert(height, hash);
            assert!(insert_res.is_none(), "the value was newly inserted");
        }
        assert_eq!(skiplist.len, exp_len);
        skiplist.head.check();
        print_skiplist(&skiplist.head);

        // Test range
        let exp_range = 5..=13;
        let keys: BTreeSet<u32> = skiplist.range(exp_range.clone()).map(|(k, v)| *k).collect();
        assert_eq!(keys, exp_range.into_iter().collect::<BTreeSet<_>>());
    }

    #[test]
    fn test_skiplist_iter() {
        // 0-1-2-3-...-H
        let exp_len = 50;
        let mut skiplist = SkipList::<BlockHash>::with_capacity(exp_len);

        let mut values = BTreeMap::new();

        for i in 0..exp_len {
            let height = i as u32;
            let hash: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
            values.insert(height, hash);
            skiplist.insert(height, hash);
        }

        let mut iter = skiplist.iter();

        for (exp_height, exp_value) in iter {
            assert_eq!(values.get(exp_height), Some(exp_value));
        }
    }

    #[test]
    fn test_skiplist_range() {
        // 0-1-2-3-...-H
        let exp_len = 50;
        let mut skiplist = SkipList::<BlockHash>::with_capacity(exp_len);

        for i in 0..exp_len {
            let height = i as u32;
            let hash: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
            skiplist.insert(height, hash);
        }
        let exp_range = 5..=13;
        let keys: BTreeSet<u32> = skiplist.range(exp_range.clone()).map(|(k, v)| *k).collect();
        assert_eq!(keys, exp_range.collect::<BTreeSet<_>>());
    }

    #[test]
    fn test_get() {
        //
        // 0-1-2-3-...-H
        let exp_len = 100;
        let mut skiplist = SkipList::<BlockHash>::with_capacity(exp_len);

        let mut test_hash = None;

        for i in 0..exp_len {
            let height = i as u32;
            let hash: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
            if height == 13 {
                test_hash = Some(hash);
            }
            assert!(skiplist.insert(height, hash).is_none());
        }

        assert_eq!(skiplist.len, exp_len);

        let val = skiplist.get(13).copied();
        assert_eq!(val, test_hash);
    }

    #[test]
    fn test_sparse_list() {
        //
        // 0-1-3-5-...-H
        let exp_cap = 100;
        let mut skiplist = SkipList::<BlockHash>::with_capacity(exp_cap);

        for i in (0..).filter(|i| i % 2 == 1).take(10) {
            let height = i as u32;
            let hash: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
            assert!(skiplist.insert(height, hash).is_none());
        }

        print_skiplist(&skiplist.head);
        assert_eq!(skiplist.len, 10);
    }

    #[test]
    fn test_skiplist_remove() {
        //
        // 0-1-x-5-...-H
        let exp_len = 50;
        let mut skiplist = SkipList::<BlockHash>::with_capacity(exp_len);

        for i in 0..exp_len {
            let height = i as u32;
            let hash: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
            assert!(skiplist.insert(height, hash).is_none());
        }
        assert_eq!(skiplist.len, exp_len);

        // Test `remove`
        for height in [0, 1, 2] {
            let removed_value = skiplist.remove(height).expect("should return removed value");
            skiplist.head.check();
        }

        let missing_height = exp_len as u32;
        assert!(skiplist.remove(missing_height).is_none());
    }

    #[test]
    fn test_skiplist_index() {
        let mut skiplist = SkipList::<i32>::with_capacity(100);

        assert!(skiplist.get_index(0).is_none());

        // Insert 10 values
        for key in 0..10 {
            let value = 1 << (key as i32);
            skiplist.insert(key, value);
        }

        assert_eq!(skiplist.len(), 10);

        // Index operation should not panic
        for test_index in 0..10 {
            let _value = skiplist[test_index];
        }

        assert!(skiplist.get_index(skiplist.len).is_none());
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_skiplist_index_out_of_bounds() {
        let mut skiplist = SkipList::<i32>::with_capacity(100);
        assert!(skiplist.is_empty());

        let _no_value = skiplist[0];
    }
}
