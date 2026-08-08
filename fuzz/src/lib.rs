//! Shared fuzzing infrastructure for `block_graph`.
//!
//! Everything here operates on `T = BlockHash` (never raw bytes as block data): the graph is
//! keyed by `BlockHash`, so random 32-byte hashes almost never match an existing block, and the
//! fuzzer would burn its whole budget on early-return paths. Instead we generate operation
//! sequences over a small, fixed universe of hashes (see [`hash_from_id`]), which keeps
//! same-height tips reachable and exercises the graph's actual decision points.

use std::collections::HashMap;

use bitcoin::hashes::Hash;
use bitcoin::BlockHash;
use block_graph::{BlockGraph, ChangeSet, CheckPoint, ConnectBlockError};

/// Map a small id to a distinct, non-zero [`BlockHash`].
///
/// Two deliberate properties: (a) it is never [`BlockHash::all_zeros`], which the crate uses as
/// the sentinel predecessor of genesis; (b) each id maps to a distinct hash, so hash collisions
/// can't happen here — but the small, 256-value id space makes it cheap for two tips to land at
/// the *same height*, which is what actually exercises the
/// `max_by_key(|id| (id.height, Reverse(id.hash)))` tie-break in `canonicalize` (the `Reverse
/// (hash)` half only ever runs once two candidates already tie on height).
pub fn hash_from_id(id: u8) -> BlockHash {
    let mut bytes = [0u8; 32];
    bytes[0] = 0xff;
    bytes[31] = id;
    BlockHash::from_byte_array(bytes)
}

/// A single graph-mutating operation.
#[derive(Debug, arbitrary::Arbitrary)]
pub enum Op {
    /// Connect a new block under a parent the graph already knows.
    Connect {
        /// Index into `Harness::known`, modulo its length.
        parent_idx: u8,
        /// Added to the parent's height to get the new block's height.
        height_delta: u8,
        /// Id of the new block's hash.
        id: u8,
    },
    /// Connect under a possibly-unknown parent hash (exercises the error paths).
    ConnectLoose {
        /// Id of the (possibly unknown) parent's hash.
        parent_id: u8,
        /// Height of the new block.
        height: u16,
        /// Id of the new block's hash.
        id: u8,
    },
}

/// One block to append to an [`Update`]'s checkpoint chain.
#[derive(Debug, arbitrary::Arbitrary)]
pub struct UpdateBlock {
    /// Added to the running height to get this block's height.
    pub height_delta: u8,
    /// Id of this block's hash.
    pub id: u8,
}

/// A checkpoint chain to apply via `apply_update`.
#[derive(Debug, arbitrary::Arbitrary)]
pub struct Update {
    /// Index into `Harness::known`, modulo its length; the update chain forks from here.
    pub base_idx: u8,
    /// Occasionally start from a hash the graph has never seen (exercises `MissingParent`).
    pub base_unknown: Option<u8>,
    /// Blocks to append, in ascending height order.
    pub blocks: Vec<UpdateBlock>,
}

/// A full fuzz scenario: a sequence of `connect_block` ops followed by a sequence of updates.
#[derive(Debug, arbitrary::Arbitrary)]
pub struct Scenario {
    /// `connect_block`-style operations to seed the graph.
    pub ops: Vec<Op>,
    /// `apply_update`-style checkpoint chains.
    pub updates: Vec<Update>,
}

/// Test harness wrapping a [`BlockGraph<BlockHash>`] plus bookkeeping needed to drive [`Op`]s
/// and [`Update`]s against real, already-known blocks.
pub struct Harness {
    /// The graph under test.
    pub graph: BlockGraph<BlockHash>,
    /// Known block hashes in insertion order; index 0 is genesis.
    pub known: Vec<BlockHash>,
    /// Mirror of hash -> height, so the target can predict expected outcomes.
    pub heights: HashMap<BlockHash, u32>,
    /// Mirror of parent hash -> `(child hash, child height)` pairs connected under it, so the
    /// target can predict `ChildHeightNotGreater` outcomes.
    pub children: HashMap<BlockHash, Vec<(BlockHash, u32)>>,
}

impl Harness {
    /// Start a fresh harness from a fixed genesis (id 0).
    pub fn new() -> Self {
        let genesis = hash_from_id(0);
        let mut heights = HashMap::new();
        heights.insert(genesis, 0);
        Self {
            graph: BlockGraph::from_genesis(genesis),
            known: vec![genesis],
            heights,
            children: HashMap::new(),
        }
    }

    /// Record a newly-connected hash so later ops/updates can reference it.
    pub fn record(&mut self, hash: BlockHash, height: u32) {
        if !self.heights.contains_key(&hash) {
            self.known.push(hash);
        }
        self.heights.insert(hash, height);
    }

    /// Record every block in a [`ChangeSet`] returned by `apply_update`, whose newly-connected
    /// blocks `apply_op` never sees.
    pub fn record_changeset(&mut self, changeset: &ChangeSet<BlockHash>) {
        for (&hash, &(height, _)) in &changeset.blocks {
            self.record(hash, height);
        }
    }

    /// Resolve an [`Op`] to the concrete `(height, hash, parent_hash)` it connects.
    ///
    /// Callers that need to issue the identical underlying `connect_block` call twice (e.g. to
    /// check idempotence) must resolve once and reuse the result, rather than resolving the same
    /// `Op` again later: `Op::Connect`'s `parent_idx` indexes into `known` modulo its length,
    /// which grows as a side effect of a successful connect, so re-resolving after the fact can
    /// silently pick a different parent.
    pub fn resolve(&self, op: &Op) -> (u32, BlockHash, BlockHash) {
        match *op {
            Op::Connect {
                parent_idx,
                height_delta,
                id,
            } => {
                let parent_hash = self.known[parent_idx as usize % self.known.len()];
                let parent_height = self.heights[&parent_hash];
                let height = parent_height.saturating_add(height_delta as u32);
                (height, hash_from_id(id), parent_hash)
            }
            Op::ConnectLoose {
                parent_id,
                height,
                id,
            } => (height as u32, hash_from_id(id), hash_from_id(parent_id)),
        }
    }

    /// Apply a single [`Op`] to the graph.
    pub fn apply_op(&mut self, op: &Op) -> Result<ChangeSet<BlockHash>, ConnectBlockError> {
        let (height, hash, parent_hash) = self.resolve(op);

        let result = self.graph.connect_block(height, hash, parent_hash);
        if result.is_ok() {
            self.record(hash, height);
            self.children.entry(parent_hash).or_default().push((hash, height));
        }
        result
    }

    /// Build a checkpoint chain for an [`Update`], forking from a known (or deliberately
    /// unknown) base, in ascending height order.
    ///
    /// Pushes that would violate strictly-increasing height (i.e. a zero `height_delta`)
    /// are skipped rather than aborting the whole chain.
    pub fn build_update(&self, update: &Update) -> Option<CheckPoint<BlockHash>> {
        let (base_height, base_hash) = match update.base_unknown {
            Some(id) => (0u32, hash_from_id(id)),
            None => {
                let hash = self.known[update.base_idx as usize % self.known.len()];
                (self.heights[&hash], hash)
            }
        };

        let mut cp = CheckPoint::new(base_height, base_hash);
        for block in update.blocks.iter().take(64) {
            let height = cp.height().saturating_add(block.height_delta as u32);
            let hash = hash_from_id(block.id);
            cp = match cp.push(height, hash) {
                Ok(next) => next,
                Err(err) => err.checkpoint,
            };
        }
        Some(cp)
    }
}

impl Default for Harness {
    fn default() -> Self {
        Self::new()
    }
}

/// Assert `res` is `Ok`, panicking with `ctx` (typically a `Debug`-formatted scenario) included
/// so a crash artifact shows which invariant broke and what operation sequence caused it.
pub fn assert_ok(res: Result<(), String>, ctx: &str) {
    if let Err(e) = res {
        panic!("{e}\ncontext: {ctx}");
    }
}
