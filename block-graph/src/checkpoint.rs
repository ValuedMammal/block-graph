//! [`CheckPoint`] is a cheaply cloneable, singly-linked list.

use alloc::sync::Arc;
use alloc::vec;
use core::fmt;
use core::fmt::Debug;
use core::ops::RangeBounds;

// TODO:
// - implement pksip
// - bench perf

/// CheckPoint, guaranteed to have at least 1 element.
#[derive(Debug)]
pub struct CheckPoint<T>(pub(crate) Arc<Node<T>>);

impl<T> CheckPoint<T>
where
    T: Debug,
{
    /// Start a new list with `value` at the head.
    pub fn new(height: u32, value: T) -> Self {
        Self(Arc::new(Node {
            height,
            value,
            prev: None,
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
        let node = Node {
            height,
            value,
            prev: Some(self.0),
        };
        Ok(Self(Arc::new(node)))
    }

    /// Prev.
    pub fn prev(&self) -> Option<Self> {
        self.0.prev.clone().map(Self)
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

    /// Height.
    pub fn height(&self) -> u32 {
        self.0.height
    }

    /// Value.
    pub fn value(&self) -> T
    where
        T: Clone,
    {
        self.0.value.clone()
    }

    /// Whether `self` and `other` share the same underlying pointer.
    pub fn eq_ptr(&self, other: &Self) -> bool {
        Arc::as_ptr(&self.0) == Arc::as_ptr(&other.0)
    }

    /// Iter.
    pub fn iter(&self) -> ListIter<T> {
        ListIter {
            cur: Some(self.0.clone()),
        }
    }

    /// Get.
    pub fn get(&self, height: u32) -> Option<Self> {
        self.range(height..=height).next()
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

    /// Compute what height to jump back to with the skip.
    fn get_skip_height(&self) -> u32 {
        let height = self.0.height;
        if height < 2 {
            return 0;
        }
        let n = height as i32;
        // Handle odd case separately
        let ret = if n % 2 == 0 {
            invert_lowest_1(n)
        } else {
            invert_lowest_1(invert_lowest_1(n - 1)) + 1
        };
        ret.try_into().expect("n should be >= 0")
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

/// Node containing both a key and value. The key is referred to as `height`.
#[derive(Debug)]
pub(crate) struct Node<T> {
    pub height: u32,
    pub value: T,
    pub prev: Option<Arc<Node<T>>>,
    // TODO
    // pub skip: Option<Arc<Node<T>>>,
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

/// CheckPoint iter.
pub struct ListIter<T> {
    cur: Option<Arc<Node<T>>>,
}

impl<T> Iterator for ListIter<T> {
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_get_with_100_elements() {
        let mut cp = CheckPoint::new(0, 0);

        for height in 1..100 {
            let value = height;
            cp = cp.push(height, value).unwrap();
        }

        // Test getting elements at various heights
        for i in 0..100 {
            let result_cp = cp.get(i).expect("should find height");
            assert_eq!(result_cp.height(), i);
            assert_eq!(result_cp.value(), i);
        }

        // Test getting non-existent heights
        assert!(cp.get(100).is_none(), "height 100 should not exist");
        assert!(cp.get(150).is_none(), "height 150 should not exist");
    }

    #[test]
    fn test_get_skip_height_even() {
        let mut cp = CheckPoint::new(0, 0);

        for height in 1..100 {
            let value = height;
            cp = cp.push(height, value).unwrap();
        }

        // 2 & (2 - 1)
        //     0000 0010
        // &   0000 0001
        // res 0000 0000
        let test_height = 2;
        let test_cp = cp.get(test_height).unwrap();
        let result = dbg!(test_cp.get_skip_height());
        assert_eq!(result, 0);

        // 4 & (4 - 1)
        //     0000 0100
        // &   0000 0011
        // res 0000 0000
        let test_height = 4;
        let test_cp = cp.get(test_height).unwrap();
        let result = dbg!(test_cp.get_skip_height());
        assert_eq!(result, 0);

        // 6 & (6 - 1)
        //     0000 0110
        // &   0000 0101
        // res 0000 0100
        let test_height = 6;
        let test_cp = cp.get(test_height).unwrap();
        let result = dbg!(test_cp.get_skip_height());
        assert_eq!(result, 4);

        // 8 & (8 - 1)
        //   0000 1000
        // & 0000 0111
        // result:
        let test_height = 8;
        let test_cp = cp.get(test_height).unwrap();
        let result = dbg!(test_cp.get_skip_height());
        assert_eq!(result, 0);
    }

    #[test]
    fn test_get_skip_height_odd() {
        let mut cp = CheckPoint::new(0, 0);

        for height in 1..100 {
            let value = height;
            cp = cp.push(height, value).unwrap();
        }

        // 3 is odd: invert_lowest_1(invert_lowest_1(3 - 1)) + 1
        // invert_lowest_1(2) = 2 & 1 = 0
        // invert_lowest_1(0) = 0 & -1 = 0
        // result: 0 + 1 = 1
        let test_height = 3;
        let test_cp = cp.get(test_height).unwrap();
        let result = dbg!(test_cp.get_skip_height());
        assert_eq!(result, 1);

        // 5 is odd: invert_lowest_1(invert_lowest_1(5 - 1)) + 1
        // invert_lowest_1(4) = 4 & 3 = 0
        // invert_lowest_1(0) = 0 & -1 = 0
        // result: 0 + 1 = 1
        let test_height = 5;
        let test_cp = cp.get(test_height).unwrap();
        let result = dbg!(test_cp.get_skip_height());
        assert_eq!(result, 1);

        // 9 is odd: invert_lowest_1(invert_lowest_1(9 - 1)) + 1
        // invert_lowest_1(8) = 8 & 7 = 0
        // invert_lowest_1(0) = 0 & -1 = 0
        // result: 0 + 1 = 1
        let test_height = 9;
        let test_cp = cp.get(test_height).unwrap();
        let result = dbg!(test_cp.get_skip_height());
        assert_eq!(result, 1);

        // 15 is odd: invert_lowest_1(invert_lowest_1(15 - 1)) + 1
        // invert_lowest_1(14) = 14 & 13 = 12
        // invert_lowest_1(12) = 12 & 11 = 8
        // result: 8 + 1 = 9
        let test_height = 15;
        let test_cp = cp.get(test_height).unwrap();
        let result = dbg!(test_cp.get_skip_height());
        assert_eq!(result, 9);
    }
}
