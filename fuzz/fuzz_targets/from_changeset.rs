#![no_main]

use bitcoin::BlockHash;
use block_graph::{
    ApplyChangeSetError, BlockGraph, ChangeSet, ConnectBlockError, FromChangeSetError,
};
use libfuzzer_sys::fuzz_target;

use block_graph_fuzz::hash_from_id;

/// Directly-constructed, possibly-corrupt/adversarial changeset input.
///
/// `ChangeSet` is the persistence format, so `from_changeset` must handle arbitrary,
/// attacker-influenced data without panicking: two distinct blocks at height 0 (rejected as
/// `MultipleGenesisBlocks`), an edge whose child is at a lower height than its parent (rejected
/// as `Apply(InvalidEdgeHeight)`), a cycle (also rejected, since every cycle contains at least one
/// such edge), an edge referencing a missing block, and a block with no incoming edge should all
/// be reachable shapes here.
///
/// Every other `ApplyChangeSetError`/`ConnectBlockError` variant is unreachable through this
/// single-changeset-onto-fresh-genesis path specifically, not just for `T = BlockHash`:
/// `apply_changeset`'s checks that compare against *pre-existing* graph state can only fire when
/// `self` already holds data that conflicts with the incoming changeset, but here `self` is a bare
/// `from_genesis` graph before `apply_changeset` runs, and `RawChangeSet` can't encode two
/// different heights for the same hash. `ParentHeightNotSmaller` is `connect_block`-only and
/// never returned by the changeset validation path at all. `BlockHashMismatch` and the
/// edge-adjacency flavor of `PrevBlockhashMismatch` are additionally unreachable for
/// `T = BlockHash` specifically: `to_blockhash`/`prev_blockhash` are the identity/always-`None`,
/// so `RawChangeSet` can't misconstruct either check's input.
#[derive(Debug, arbitrary::Arbitrary)]
struct RawChangeSet {
    /// `(id, height)` pairs describing blocks.
    blocks: Vec<(u8, u16)>,
    /// `(parent id, child id)` pairs describing edges.
    edges: Vec<(u8, u8)>,
}

fuzz_target!(|raw: RawChangeSet| {
    let mut changeset = ChangeSet::<BlockHash>::default();
    for &(id, height) in raw.blocks.iter().take(256) {
        let hash = hash_from_id(id);
        // T = BlockHash, so the stored value must equal the key (`to_blockhash` is the identity).
        changeset.blocks.insert(hash, (height as u32, hash));
    }
    for &(parent_id, child_id) in raw.edges.iter().take(256) {
        changeset
            .edges
            .insert((hash_from_id(parent_id), hash_from_id(child_id)));
    }

    let ctx = format!("{raw:?}");

    match BlockGraph::from_changeset(changeset.clone()) {
        Ok(Some(graph)) => {
            block_graph_fuzz::assert_ok(graph.check_invariants(), &ctx);
            block_graph_fuzz::assert_ok(graph.check_best_tip(), &ctx);

            // Roundtrip: a valid graph's own changeset must reconstruct an equal graph.
            let recovered = BlockGraph::from_changeset(graph.initial_changeset())
                .expect("roundtripping a valid graph's changeset must not error")
                .expect("a valid graph's initial changeset is never empty");
            assert_eq!(recovered, graph, "roundtrip mismatch\ncontext: {ctx}");
            block_graph_fuzz::assert_ok(recovered.check_invariants(), &ctx);
        }
        Ok(None) => {
            assert!(
                changeset.blocks.is_empty(),
                "Ok(None) implies an empty changeset\ncontext: {ctx}"
            );
        }
        Err(FromChangeSetError::MissingGenesis) => {}
        Err(FromChangeSetError::MultipleGenesisBlocks { first, second }) => {
            assert_ne!(
                first, second,
                "MultipleGenesisBlocks must name two distinct blocks\ncontext: {ctx}"
            );
        }
        Err(FromChangeSetError::Apply(ApplyChangeSetError::InvalidEdgeHeight { parent, child })) => {
            assert!(
                parent.height >= child.height,
                "InvalidEdgeHeight should only be raised for a non-increasing edge\ncontext: {ctx}"
            );
        }
        Err(FromChangeSetError::Apply(ApplyChangeSetError::BlockHashMismatch { .. })) => {
            unreachable!(
                "T = BlockHash's to_blockhash is the identity, so RawChangeSet can't construct a \
                 mismatch\ncontext: {ctx}"
            );
        }
        Err(FromChangeSetError::Apply(ApplyChangeSetError::ConnectBlock(err))) => match err {
            ConnectBlockError::ParentHeightNotSmaller => unreachable!(
                "connect_block-only variant, never returned by the changeset validation path\ncontext: {ctx}"
            ),
            ConnectBlockError::PrevBlockhashMismatch => unreachable!(
                "T = BlockHash never declares a prev_blockhash, so this can't be raised\ncontext: {ctx}"
            ),
            ConnectBlockError::HeightConflict { .. } => unreachable!(
                "requires self to already hold the hash at a conflicting height, but self is a \
                 bare from_genesis graph here\ncontext: {ctx}"
            ),
            ConnectBlockError::ChildHeightNotGreater { .. } => unreachable!(
                "requires self to already know a child of this hash, but self is a bare \
                 from_genesis graph here\ncontext: {ctx}"
            ),
            ConnectBlockError::HeightZeroReservedForRoot { .. } => unreachable!(
                "the single height-0 entry is already consumed as genesis before apply_changeset \
                 runs\ncontext: {ctx}"
            ),
        },
    }
});
