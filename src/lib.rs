//! `block-graph`

#![warn(missing_docs)]
#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
#[macro_use]
extern crate std;

mod block_graph;
mod list;

pub use block_graph::*;
pub use list::*;

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
