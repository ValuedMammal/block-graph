//! `block_graph`

#![warn(missing_docs)]
#![no_std]

#[macro_use]
#[allow(unused_imports)]
extern crate alloc;

#[cfg(feature = "std")]
#[macro_use]
extern crate std;

mod block_graph;
pub use block_graph::*;

/// Additional methods on `CheckPoint`
pub(crate) trait CheckPointExt {
    type Value;
    /// Obtain the inner value of this `CheckPoint`
    fn value(&self) -> Self::Value;
}

impl<T: Clone> CheckPointExt for bdk_chain::CheckPoint<T> {
    type Value = T;
    fn value(&self) -> T {
        self.data()
    }
}

#[cfg(feature = "std")]
pub(crate) mod collections {
    #![allow(unused)]
    pub use std::collections::*;
}

#[cfg(not(feature = "std"))]
pub(crate) mod collections {
    #![allow(unused)]
    pub type HashMap<K, V> = alloc::collections::BTreeMap<K, V>;
    pub type HashSet<T> = alloc::collections::BTreeSet<T>;
    pub use alloc::collections::*;
}
