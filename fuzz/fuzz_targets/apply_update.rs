#![no_main]

use std::collections::HashSet;

use bitcoin::BlockHash;
use block_graph::{BlockGraph, BlockId};
use block_graph_fuzz::{Harness, Scenario};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|scenario: Scenario| {
    let mut harness = Harness::new();
    let ctx_scenario = format!("{scenario:?}");

    // Seed the graph; connect errors here are expected (e.g. `ParentHeightNotSmaller`) and not
    // interesting for this target, which is about `apply_update`.
    for op in scenario.ops.iter().take(256) {
        let _ = harness.apply_op(op);
    }

    for (i, update) in scenario.updates.iter().take(16).enumerate() {
        let Some(cp) = harness.build_update(update) else {
            continue;
        };
        let ctx = format!("update[{i}] = {update:?}\nscenario: {ctx_scenario}");

        let before = harness.graph.clone();
        let result = harness.graph.apply_update(cp.clone());

        match result {
            Err(_) => {
                // Atomicity: a failed update must leave the graph exactly as it was.
                assert_eq!(
                    harness.graph, before,
                    "a failed apply_update must not mutate the graph\ncontext: {ctx}"
                );
            }
            Ok(changeset) => {
                harness.record_changeset(&changeset);
                block_graph_fuzz::assert_ok(harness.graph.check_invariants(), &ctx);
                block_graph_fuzz::assert_ok(harness.graph.check_best_tip(), &ctx);

                // Idempotence: applying the identical update again is a no-op.
                let before_repeat = harness.graph.clone();
                match harness.graph.apply_update(cp) {
                    Ok(cs) => assert!(
                        cs.is_empty(),
                        "repeating a successful apply_update should return an empty changeset\ncontext: {ctx}"
                    ),
                    Err(e) => panic!("repeating a successful apply_update errored: {e}\ncontext: {ctx}"),
                }
                assert_eq!(
                    harness.graph, before_repeat,
                    "repeating a successful apply_update must not mutate the graph\ncontext: {ctx}"
                );

                // Roundtrip through `initial_changeset`.
                let recovered = BlockGraph::from_changeset(harness.graph.initial_changeset())
                    .expect("roundtripping a valid graph's changeset must not error")
                    .expect("a valid graph's initial changeset is never empty");
                assert_eq!(recovered, harness.graph, "roundtrip mismatch\ncontext: {ctx}");
                block_graph_fuzz::assert_ok(recovered.check_invariants(), &ctx);

                // `is_block_in_chain` agreement.
                let chain_tip = harness.graph.tip().block_id();
                let tip_chain_hashes: HashSet<BlockHash> =
                    harness.graph.iter().map(|cp| cp.hash()).collect();
                for cp in harness.graph.iter() {
                    assert_eq!(
                        harness.graph.is_block_in_chain(cp.block_id(), chain_tip),
                        Some(true),
                        "tip-chain block should report as in-chain\ncontext: {ctx}"
                    );
                }
                for (&hash, &height) in &harness.heights {
                    if tip_chain_hashes.contains(&hash) {
                        continue; // already checked above
                    }
                    let block = BlockId { height, hash };
                    // Below the tip height, the canonical chain may have a *gap* at `height`
                    // (a gapped/sparse connection never filled it in), in which case there's
                    // nothing to compare against and the real answer is `None`, not `Some(false)`.
                    let expected = if height > chain_tip.height {
                        None
                    } else {
                        harness.graph.tip().get(height).map(|_| false)
                    };
                    assert_eq!(
                        harness.graph.is_block_in_chain(block, chain_tip),
                        expected,
                        "is_block_in_chain mismatch for {block:?}\ncontext: {ctx}"
                    );
                }
            }
        }
    }
});
