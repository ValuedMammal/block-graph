#![no_main]

use bitcoin::BlockHash;
use block_graph::{BlockGraph, ChangeSet};
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

/// `false` if `a` and `b` bind any shared hash to two different heights.
///
/// `ChangeSet::merge` is append-only/first-write-wins: it's only a commutative, associative,
/// apply-order-independent union on compatible input. On incompatible input it resolves silently
/// toward `self`, and its `debug_assert` would fire (the fuzz profile builds with
/// `-Cdebug-assertions`) — so every changeset-level law below is guarded by this.
fn compatible(a: &ChangeSet<BlockHash>, b: &ChangeSet<BlockHash>) -> bool {
    a.blocks
        .iter()
        .all(|(hash, (height, _))| b.blocks.get(hash).is_none_or(|(h, _)| h == height))
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

    // --- Changeset-level laws (`ChangeSet::merge`), guarded by `compatible` ---
    let a = graph_a.initial_changeset();
    let b = graph_b.initial_changeset();
    let c = graph_c.initial_changeset();

    if compatible(&a, &b) {
        let mut ab = a.clone();
        ab.merge(b.clone());
        let mut ba = b.clone();
        ba.merge(a.clone());
        assert_eq!(
            ab, ba,
            "changeset merge should be commutative on compatible input\ncontext: {ctx}"
        );

        let mut ab_again = ab.clone();
        ab_again.merge(b.clone());
        assert_eq!(
            ab_again, ab,
            "repeating a changeset merge should be idempotent\ncontext: {ctx}"
        );

        // Homomorphism: `from_changeset(a.merge(b))` must agree with `from_changeset(a)` followed
        // by `apply_changeset(b)`, both on success/failure and on the resulting graph. `a` is a
        // real graph's own changeset, so it's never empty and always loads.
        let mut sequential = BlockGraph::from_changeset(a.clone())
            .expect("a real graph's own changeset must load")
            .expect("a real graph's changeset is never empty");
        let sequential_ok = sequential.apply_changeset(b.clone()).is_ok();

        let merged = BlockGraph::from_changeset(ab.clone());
        let merged_ok = matches!(merged, Ok(Some(_)));

        assert_eq!(
            merged_ok, sequential_ok,
            "merged-union and sequential application should agree on success/failure on \
             compatible input\ncontext: {ctx}"
        );
        if let Ok(Some(merged_graph)) = merged {
            if sequential_ok {
                assert_eq!(
                    merged_graph, sequential,
                    "merging then loading should agree with sequential application\ncontext: {ctx}"
                );
                block_graph_fuzz::assert_ok(merged_graph.check_invariants(), &ctx);
                block_graph_fuzz::assert_ok(merged_graph.check_best_tip(), &ctx);
            }
        }
    }

    if compatible(&a, &b) && compatible(&a, &c) && compatible(&b, &c) {
        let mut abc = a.clone();
        abc.merge(b.clone());
        abc.merge(c.clone());
        let mut acb = a.clone();
        acb.merge(c.clone());
        acb.merge(b.clone());
        assert_eq!(
            abc, acb,
            "changeset merge should be associative on compatible input\ncontext: {ctx}"
        );
    }
});
