//! A node in the skiplist.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::mem;
use core::ptr;
use core::ptr::NonNull;

type Link<T> = Option<NonNull<Node<T>>>;

/// A node in the SkipList.
#[derive(Debug)]
pub struct Node<T> {
    /// The value stored in the node, will be `None` for the head node.
    pub value: Option<T>,
    /// The level of the node, i.e. how "high" the node reaches.
    pub level: usize,
    /// The predecessor node at level 0.
    pub prev: Link<T>,
    /// Links to the next node at every level. The length of the vector is always
    /// `level + 1`.
    pub links: Vec<Link<T>>,
    /// The corresponding length of each link. For a given node, `links_len[level]`
    /// tells you how many nodes are skipped by following the link at that level.
    pub links_len: Vec<usize>,
}

impl<T> Node<T> {
    /// Construct a new head node with the specified number of `levels`.
    pub fn new_head(levels: usize) -> Self {
        Self {
            value: None,
            prev: None,
            level: levels - 1,
            links: vec![None; levels],
            links_len: vec![0; levels],
        }
    }

    /// Construct a new node with `value` at a given `level`.
    pub fn new(value: T, level: usize) -> Self {
        Self {
            value: Some(value),
            prev: None,
            level,
            links: vec![None; level + 1],
            links_len: vec![0; level + 1],
        }
    }

    /// Whether this [`Node`] is a head node. By definition the head
    /// has no predecessor.
    pub fn is_head(&self) -> bool {
        self.prev.is_none()
    }

    /// Next node reference at the bottom most level.
    pub fn next_ref(&self) -> Option<&Node<T>> {
        // SAFETY: ptr to `Node` is always convertible to a reference.
        unsafe { self.links[0].as_ref().map(|p| p.as_ref()) }
    }

    /// Next node at `level`.
    pub fn next(&self, level: usize) -> Option<&Node<T>> {
        // SAFETY: ptr to `Node` is always convertible to a reference.
        unsafe { self.links[level].as_ref().map(|p| p.as_ref()) }
    }

    /// Next `&mut` node at `level`.
    pub fn next_mut(&mut self, level: usize) -> Option<&mut Node<T>> {
        // SAFETY: ptr to `Node` is always convertible to a reference.
        unsafe { self.links[level].as_mut().map(|p| p.as_mut()) }
    }

    /// Move to the last node reachable from this node (includes all levels).
    pub fn last(&self) -> &Self {
        (0..=self.level)
            .rev()
            .fold(self, |node, level| node.advance_while(level, |_, _| true).0)
    }

    /// Takes the next node and returns it. The `prev` of the taken node will be set to `None`.
    pub fn take_tail(&mut self) -> Option<Box<Self>> {
        // SAFETY: `p` must be valid and non-null.
        // SAFETY: `Box::from_raw` must only be called once on the same `*mut T`.
        unsafe {
            self.links[0].take().map(|p| {
                let mut next = Box::from_raw(p.as_ptr());
                next.prev = None;
                self.links_len[0] = 0;
                next
            })
        }
    }

    /// Replaces the next node of `self` with `new_next` and returns the old.
    pub fn replace_tail(&mut self, mut new_next: Box<Self>) -> Option<Box<Self>> {
        let mut old_next = self.take_tail();
        if let Some(old_next) = old_next.as_mut() {
            old_next.prev = None;
        }
        new_next.prev = NonNull::new(self as *mut _);
        self.links[0] = NonNull::new(Box::into_raw(new_next));
        assert!(self.next(0).is_some());
        self.links_len[0] = 1;

        old_next
    }

    /// Insert the `new_node` immediately after this node and return a `&mut` reference to it.
    pub fn insert_next(&mut self, mut new_node: Box<Self>) -> &mut Self {
        if let Some(tail) = self.take_tail() {
            new_node.replace_tail(tail);
        }
        self.replace_tail(new_node);

        self.next_mut(0).expect("must have a tail")
    }

    /// Take the node right after this node.
    pub fn take_next(&mut self) -> Option<Box<Self>> {
        let mut ret = self.take_tail()?;
        if let Some(new_tail) = ret.take_tail() {
            self.replace_tail(new_tail);
        }

        Some(ret)
    }

    /// Continue moving at the specified `level` while the predicate is true.
    ///
    /// This method takes a `pred` closure which given a reference to the current and next node
    /// determines whether to advance.
    ///
    /// Returns A tuple `(node, distance)` representing the target node and distance moved.
    pub fn advance_while(
        &self,
        level: usize,
        mut pred: impl FnMut(&Self, &Self) -> bool,
    ) -> (&Self, usize) {
        let mut current = self;
        let mut traveled = 0;
        loop {
            match current.next_if(level, &mut pred) {
                Ok((node, steps)) => {
                    current = node;
                    traveled += steps;
                }
                Err(node) => return (node, traveled),
            }
        }
    }

    /// Return a reference to the next node at `level` if the given predicate is true.
    ///
    /// - `pred` takes a reference to the current node and the next node.
    ///
    /// If no next node satisfies the predicate, then just returns `self`.
    pub fn next_if(
        &self,
        level: usize,
        pred: impl FnOnce(&Self, &Self) -> bool,
    ) -> Result<(&Self, usize), &Self> {
        // SAFETY: ptr to `Node` is always convertible to a reference.
        let next = unsafe { self.links[level].and_then(|p| p.as_ptr().as_ref()) };
        match next {
            Some(next) if pred(self, next) => Ok((next, self.links_len[level])),
            _ => Err(self),
        }
    }

    /// Continue moving at the specified `level` mutably while the predicate is true.
    ///
    /// - `pred` takes a reference to current and next node.
    ///
    /// Returns a tuple `(node, distance)` representing the target node and the total distance
    /// moved.
    pub fn advance_while_mut(
        &mut self,
        level: usize,
        mut pred: impl FnMut(&Self, &Self) -> bool,
    ) -> (&mut Self, usize) {
        let mut current = self;
        let mut traveled = 0;
        loop {
            match current.next_if_mut(level, &mut pred) {
                Ok((node, steps)) => {
                    current = node;
                    traveled += steps;
                }
                Err(node) => return (node, traveled),
            }
        }
    }

    /// Return a reference to the next node at `level` if the given predicate is true.
    ///
    /// - `pred` takes reference to the current node and the next node.
    ///
    /// If no next node satisfies the predicate, then just returns `self`.
    pub fn next_if_mut(
        &mut self,
        level: usize,
        pred: impl FnOnce(&Self, &Self) -> bool,
    ) -> Result<(&mut Self, usize), &mut Self> {
        // SAFETY: ptr to `Node` is always convertible to a reference.
        let next = unsafe { self.links[level].and_then(|p| p.as_ptr().as_mut()) };
        match next {
            Some(next) if pred(self, next) => Ok((next, self.links_len[level])),
            _ => Err(self),
        }
    }

    /// Compute the distance to `target` at the specified level.
    ///
    /// If no target is given, return the distance from this node to the last reachable node at
    /// level.
    pub fn distance_at_level(&self, level: usize, target: Option<&Self>) -> Result<usize, ()> {
        match target {
            Some(target) => {
                let (node, distance) = self.advance_while(level, |cur, _| !ptr::eq(cur, target));
                if !ptr::eq(node, target) {
                    return Err(());
                }
                Ok(distance)
            }
            None => {
                let (node, distance) = self.advance_while(level, |_, _| true);
                Ok(distance + node.links_len[level])
            }
        }
    }

    /// Move for `max_distance` steps and return a reference to the target node if it exists.
    pub fn advance(&self, max_distance: usize) -> Option<&Self> {
        let level = self.level;
        let mut node = self;
        let mut traveled = 0;

        for level in (0..=level).rev() {
            let (level_node, _) = node.advance_while(level, |cur_node, _| {
                let step = cur_node.links_len[level];
                if step + traveled <= max_distance {
                    traveled += step;
                    true
                } else {
                    false
                }
            });
            node = level_node;
        }

        if traveled == max_distance {
            Some(node)
        } else {
            None
        }
    }
}

/// Alias for a node in the [`SkipList`](super::SkipList).
type SkipListNode<V> = Node<(u32, V)>;

impl<V> SkipListNode<V> {
    /// Get the `key` of this node.
    pub fn key(&self) -> Option<u32> {
        self.value.as_ref().map(|&(k, _)| k)
    }

    /// Get a reference to the value of this node.
    pub fn value(&self) -> Option<&V> {
        self.value.as_ref().map(|(_, v)| v)
    }

    /// Get a mutable reference to the value of this node.
    pub fn value_mut(&mut self) -> Option<&mut V> {
        self.value.as_mut().map(|(_, v)| v)
    }
}

/// Type used to quickly search for a value in `SkipList`.
pub struct Seek<T> {
    key: u32,
    marker: core::marker::PhantomData<T>,
}

impl<T> Seek<T> {
    /// New with a target `key`.
    pub fn new(key: u32) -> Self {
        Self {
            key,
            marker: core::marker::PhantomData,
        }
    }

    /// Seek to the nearest node matching the target key.
    ///
    /// When traversing the skiplist we advance at each level while the *next key*
    /// is greater than the target key. The "target node" will be the `.next_ref` node
    /// following the returned node (not the node itself). This enables us to continue
    /// traversing at lower levels without overshooting the target.
    pub fn seek<'a>(&self, node: &'a SkipListNode<T>) -> &'a SkipListNode<T> {
        let target_key = self.key;
        let levels = node.level;
        let mut cur_node = node;

        for level in (0..levels).rev() {
            let (level_node, _) = cur_node
                .advance_while(level, |_, next| next.key().map_or(true, |k| k > target_key));
            cur_node = level_node;
        }

        cur_node
    }
}

/// Type used for inserting values into `SkipList`.
pub struct InsertOrReplace<T, F> {
    key: u32,
    value: T,
    make_node: F,
}

impl<T, F> InsertOrReplace<T, F>
where
    F: FnOnce(u32, T) -> Box<SkipListNode<T>>,
{
    /// Create new [`InsertOrReplace`] with the given `key`, `value`, and `make_node` fn.
    pub fn new(key: u32, value: T, make_node: F) -> Self {
        Self {
            key,
            value,
            make_node,
        }
    }

    /// Seek to the target node and insert (or replace) the value.
    /// Return a mutable reference to the new node,
    /// or else and `Err` containing the old value if the value was replaced.
    pub fn seek_and_insert_or_replace(
        self,
        node: &mut SkipListNode<T>,
        level: usize,
    ) -> Result<(&mut SkipListNode<T>, usize), T> {
        // SAFETY: `level_head` must point to a valid Node, so it is safe to dereference.
        unsafe {
            let (level_head, distance_this_level) =
                node.advance_while_mut(level, |_, next| next.key().map_or(true, |k| k > self.key));
            let level_head_ptr = level_head as *mut _;
            if level == 0 {
                let node = self.insert_or_replace(level_head)?;
                Ok((node, distance_this_level))
            } else {
                // Recurse at (level - 1)...
                let (node, distance_after_head) =
                    self.seek_and_insert_or_replace(level_head, level - 1)?;
                let level_head = &mut *level_head_ptr;
                Self::insertion_fixup(level, level_head, distance_after_head, node);
                Ok((node, distance_this_level + distance_after_head))
            }
        }
    }

    /// Act on the node by inserting a value if it didn't exist, returning a mutable reference
    /// to it, or else an `Err` containing the old value if the value was already present.
    fn insert_or_replace(self, node: &mut SkipListNode<T>) -> Result<&mut SkipListNode<T>, T> {
        let target_key = self.key;
        // A value was present at the target node, replace it and return the old value.
        if let Some(target_node) = node.next_mut(0) {
            if let Some(node_key) = target_node.key() {
                if node_key == target_key {
                    let old_value =
                        mem::replace(target_node.value_mut().expect("must have value"), self.value);
                    return Err(old_value);
                }
            }
        }
        // Install the new node at the target key by inserting it right after `node`.
        let new_node = (self.make_node)(self.key, self.value);
        node.insert_next(new_node);

        Ok(node.next_mut(0).expect("we just inserted it"))
    }

    /// Fixes links of the specified `level` after insertion.
    ///
    /// Places the new node after `level_head` (if `level` is within that of the new_node).
    fn insertion_fixup(
        level: usize,
        level_head: &mut SkipListNode<T>,
        distance_from_parent: usize,
        new_node: &mut SkipListNode<T>,
    ) {
        if level == 0 {
            return;
        }
        if level <= new_node.level {
            new_node.links[level] = level_head.links[level];
            level_head.links[level] = NonNull::new(new_node);
            let old_len = level_head.links_len[level];
            new_node.links_len[level] = old_len - distance_from_parent;
            level_head.links_len[level] = distance_from_parent + 1;
        } else {
            level_head.links_len[level] += 1;
        }
    }
}

/// Type used to remove a value from `SkipList`.
pub struct Remove<T> {
    key: u32,
    marker: core::marker::PhantomData<T>,
}

impl<T> Remove<T> {
    /// New with the target key.
    pub fn new(key: u32) -> Self {
        Self {
            key,
            marker: core::marker::PhantomData,
        }
    }

    pub fn seek_and_remove(
        self,
        node: &mut SkipListNode<T>,
        level: usize,
    ) -> Result<(Box<SkipListNode<T>>, usize), ()> {
        // SAFETY: `level_head` must point to a valid Node, so it is safe to dereference.
        unsafe {
            let (level_head, distance_this_level) =
                node.advance_while_mut(level, |_, next| next.key().map_or(true, |k| k > self.key));
            let level_head_ptr = level_head as *mut _;
            if level == 0 {
                let node = self.remove(level_head)?;
                Ok((node, distance_this_level))
            } else {
                // Recurse at (level - 1)...
                let (mut node, distance_after_head) =
                    self.seek_and_remove(level_head, level - 1)?;
                let level_head = &mut *level_head_ptr;
                Self::removal_fixup(level, level_head, &mut node);
                Ok((node, distance_this_level + distance_after_head))
            }
        }
    }

    // Act on the node by removing the value after `level_node` if it matches
    // the target key.
    pub fn remove(&self, level_node: &mut SkipListNode<T>) -> Result<Box<SkipListNode<T>>, ()> {
        let target_key = self.key;

        match level_node.next_ref() {
            Some(n) if n.key() == Some(target_key) => {
                Ok(level_node.take_next().expect("`next_ref` was not None"))
            }
            _ => Err(()),
        }
    }

    /// Fix links of the given `level` after removal.
    fn removal_fixup(
        level: usize,
        level_head: &mut SkipListNode<T>,
        removed_node: &mut Box<SkipListNode<T>>,
    ) {
        if level == 0 {
            return;
        }
        if level <= removed_node.level {
            assert!(
                ptr::eq(level_head.links[level].unwrap().as_ptr(), removed_node.as_ref()),
                "level_head must be linked to the removed_node"
            );
            level_head.links[level] = removed_node.links[level];
            level_head.links_len[level] += removed_node.links_len[level];
        }
        level_head.links_len[level] -= 1;
    }
}

impl<T> Node<T> {
    /// Check the integrity of the skiplist.
    #[allow(unused)]
    pub(crate) fn check(&self) {
        assert!(self.is_head());
        assert!(self.value.is_none());
        let mut cur_node = Some(self);
        let mut len = 0;
        while let Some(node) = cur_node {
            // Check the integrity of node.
            assert_eq!(node.level + 1, node.links.len());
            assert_eq!(node.level + 1, node.links_len.len());
            if !node.is_head() {
                assert!(node.value.is_some());
            }
            // Check link at level 0
            if let Some(next_node) = node.next(0) {
                len += 1;
                assert!(ptr::eq(next_node.prev.unwrap().as_ptr(), node));
            }
            cur_node = node.next(0);
        }

        for level in 1..=self.level {
            let mut length_sum = 0;
            let mut cur_node = Some(self);
            while let Some(node) = cur_node {
                length_sum += node.links_len[level];
                let next_node = unsafe { node.links[level].as_ref().map(|p| p.as_ref()) };
                assert_eq!(
                    node.links_len[level],
                    node.distance_at_level(level - 1, next_node).unwrap(),
                    "Node gives different distance at level {} and level {}",
                    level,
                    level - 1
                );

                cur_node = next_node;
            }

            assert_eq!(length_sum, len);
        }
    }
}

/// An iterator for SkipList allowing both forwards and backwards iteration.
#[derive(Debug)]
pub struct SkipListIter<'a, T> {
    first: Option<&'a Node<T>>,
    last: Option<&'a Node<T>>,
    size: usize,
}

impl<'a, T> SkipListIter<'a, T> {
    /// Construct a [`SkipListIter`] from a head node.
    ///
    /// Note: The caller is responsible for ensuring there are `len` nodes after `head`.
    pub(crate) fn from_head(head: &'a Node<T>, len: usize) -> Self {
        if len == 0 {
            Self {
                first: None,
                last: None,
                size: 0,
            }
        } else {
            let first = head.next_ref();
            let last = first.as_ref().map(|n| n.last());
            Self {
                first,
                last,
                size: len,
            }
        }
    }
}

impl<'a, T> Iterator for SkipListIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let cur_node = self.first?;
        if ptr::eq(cur_node, self.last.map_or(ptr::null(), |v| v as *const _)) {
            self.first = None;
            self.last = None;
        } else {
            self.first = cur_node.next_ref();
        }
        self.size -= 1;
        cur_node.value.as_ref()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

impl<T> DoubleEndedIterator for SkipListIter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let last_node = self.last?;
        if ptr::eq(last_node, self.first.map_or(ptr::null(), |v| v as *const _)) {
            self.first = None;
            self.last = None;
        } else {
            // SAFETY: ptr to `Node` is always convertible to a reference.
            unsafe {
                self.last = last_node.prev.as_ref().map(|p| p.as_ref());
            }
        }
        self.size -= 1;
        last_node.value.as_ref()
    }
}

impl<T> ExactSizeIterator for SkipListIter<'_, T> {}

/// Node iter.
#[derive(Debug)]
pub struct NodeIter<'a, T> {
    current: Option<&'a Node<T>>,
}

impl<'a, T> Iterator for NodeIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.current?;
        self.current = node.next_ref();
        node.value.as_ref()
    }
}
