#![no_main]

use bitcoin::BlockHash;
use block_graph::{BlockGraph, ChangeSet, FromChangeSetError};
use libfuzzer_sys::fuzz_target;

use block_graph_fuzz::hash_from_id;

/// Directly-constructed, possibly-corrupt/adversarial changeset input.
///
/// `ChangeSet` is the persistence format, so `from_changeset` must handle arbitrary,
/// attacker-influenced data without panicking: two distinct blocks at height 0 (rejected as
/// `MultipleGenesisBlocks`), an edge whose child is at a lower height than its parent (rejected
/// as `InvalidEdgeHeight`), a cycle (also rejected, since every cycle contains at least one such
/// edge), an edge referencing a missing block, and a block with no incoming edge should all be
/// reachable shapes here. `InconsistentPrevBlockhash` is not reachable with `T = BlockHash`,
/// since `Block::prev_blockhash` always returns `None` for it; it only fires for a `T` that
/// declares a `prev_blockhash` (e.g. `Header`), which isn't fuzzed here.
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
        Err(FromChangeSetError::InvalidEdgeHeight { parent, child }) => {
            assert!(
                parent.height >= child.height,
                "InvalidEdgeHeight should only be raised for a non-increasing edge\ncontext: {ctx}"
            );
        }
        Err(FromChangeSetError::InconsistentPrevBlockhash { .. }) => {
            unreachable!(
                "T = BlockHash never declares a prev_blockhash, so this can't be raised\ncontext: {ctx}"
            );
        }
    }
});
