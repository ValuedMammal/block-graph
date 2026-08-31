#![no_main]

use bitcoin::BlockHash;
use block_graph::BlockGraph;
use block_graph_fuzz::{Harness, Scenario};
use libfuzzer_sys::fuzz_target;

/// Drive a [`Scenario`] into a fresh [`Harness`] the same way `apply_update.rs` does, ignoring
/// per-step errors — only the resulting graph matters for the merge properties checked here.
fn build_graph(scenario: &Scenario) -> BlockGraph<BlockHash> {
    let mut harness = Harness::new();
    for op in scenario.ops.iter().take(256) {
        let _ = harness.apply_op(op);
    }
    for update in scenario.updates.iter().take(16) {
        let Some(cp) = harness.build_update(update) else {
            continue;
        };
        let _ = harness.graph.apply_update(cp);
    }
    harness.graph
}

fuzz_target!(|scenarios: (Scenario, Scenario, Scenario)| {
    let (scenario_a, scenario_b, scenario_c) = scenarios;
    let graph_a = build_graph(&scenario_a);
    let graph_b = build_graph(&scenario_b);
    let graph_c = build_graph(&scenario_c);

    let ctx = format!("a: {scenario_a:?}\nb: {scenario_b:?}\nc: {scenario_c:?}");

    // Commutativity: when both orders succeed, the merged graphs must agree. Success/failure is
    // not asserted to be symmetric — `validate_changeset` runs against a different `self` in each
    // direction, so there's no reason it must be.
    let mut ab = graph_a.clone();
    let ab_ok = ab.apply_block_graph(&graph_b).is_ok();
    let mut ba = graph_b.clone();
    let ba_ok = ba.apply_block_graph(&graph_a).is_ok();
    if ab_ok && ba_ok {
        assert_eq!(ab, ba, "merge should be commutative\ncontext: {ctx}");
        block_graph_fuzz::assert_ok(ab.check_invariants(), &ctx);
        block_graph_fuzz::assert_ok(ab.check_best_tip(), &ctx);
    }

    // Idempotence: replaying a successful merge is a no-op.
    if ab_ok {
        let before = ab.clone();
        match ab.apply_block_graph(&graph_b) {
            Ok(cs) => assert!(
                cs.is_empty(),
                "repeating a successful merge should return an empty changeset\ncontext: {ctx}"
            ),
            Err(e) => panic!("repeating a successful merge errored: {e}\ncontext: {ctx}"),
        }
        assert_eq!(
            ab, before,
            "repeating a successful merge must not mutate the graph\ncontext: {ctx}"
        );
    }

    // Associativity-flavored check across three graphs: (a ∪ b) ∪ c vs (a ∪ c) ∪ b.
    let mut abc = graph_a.clone();
    let abc_ok = abc.apply_block_graph(&graph_b).is_ok() && abc.apply_block_graph(&graph_c).is_ok();
    let mut acb = graph_a.clone();
    let acb_ok = acb.apply_block_graph(&graph_c).is_ok() && acb.apply_block_graph(&graph_b).is_ok();
    if abc_ok && acb_ok {
        assert_eq!(
            abc, acb,
            "merging b then c should agree with merging c then b\ncontext: {ctx}"
        );
        block_graph_fuzz::assert_ok(abc.check_invariants(), &ctx);
        block_graph_fuzz::assert_ok(abc.check_best_tip(), &ctx);
    }
});
