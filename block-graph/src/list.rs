//! [`List`] is a cheaply cloneable, singly-linked list.

use alloc::sync::Arc;
use alloc::vec;
use core::fmt;
use core::ops::RangeBounds;

/// List, guaranteed to have at least 1 element.
#[derive(Debug)]
pub struct List<T>(pub(crate) Arc<Node<T>>);

impl<T: fmt::Debug> List<T> {
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
        let ls = Self::new(k, v);

        Ok(ls.extend(entries)?)
    }

    /// Extend with an iterator of (height, value) entries.
    pub fn extend<I>(self, items: I) -> Result<Self, Self>
    where
        I: IntoIterator<Item = (u32, T)>,
    {
        let mut ls = self;
        for (k, v) in items.into_iter() {
            ls = ls.push(k, v)?;
        }
        Ok(ls)
    }

    /// Push an element of `T` onto the head of the list.
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
        self.iter().find(|l| l.height() == height)
    }

    /// Range.
    pub fn range(&self, range: impl RangeBounds<u32>) -> impl Iterator<Item = List<T>> {
        let start_bound = range.start_bound().cloned();
        let end_bound = range.end_bound().cloned();
        self.iter()
            .skip_while(move |ls| match end_bound {
                core::ops::Bound::Included(included) => ls.height() > included,
                core::ops::Bound::Excluded(excluded) => ls.height() >= excluded,
                core::ops::Bound::Unbounded => false,
            })
            .take_while(move |ls| match start_bound {
                core::ops::Bound::Included(included) => ls.height() >= included,
                core::ops::Bound::Excluded(excluded) => ls.height() > excluded,
                core::ops::Bound::Unbounded => true,
            })
    }
}

impl<T> Clone for List<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T: fmt::Debug + Clone + PartialEq> PartialEq for List<T> {
    fn eq(&self, other: &Self) -> bool {
        let a = self.iter().map(|l| (l.height(), l.value()));
        let b = other.iter().map(|l| (l.height(), l.value()));
        a.eq(b)
    }
}

/// Node containing both a key and value. The key is referred to as `height`.
#[derive(Debug)]
pub(crate) struct Node<T> {
    pub height: u32,
    pub value: T,
    pub prev: Option<Arc<Node<T>>>,
}

// We do this to avoid recursively dropping `Arc`s,
// which can happen whenever a `List` object goes out of scope
// and the default destructor for `Node` is run.
impl<T> Drop for Node<T> {
    fn drop(&mut self) {
        let mut cur = self.prev.take();
        while let Some(this) = cur {
            match Arc::try_unwrap(this).ok() {
                Some(mut node) => {
                    cur = node.prev.take();
                    core::mem::forget(node);
                }
                None => break,
            }
        }
    }
}

/// List iter.
pub struct ListIter<T> {
    cur: Option<Arc<Node<T>>>,
}

impl<T> Iterator for ListIter<T> {
    type Item = List<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.cur.clone()?;
        self.cur.clone_from(&cur.prev);

        Some(List(cur))
    }
}
