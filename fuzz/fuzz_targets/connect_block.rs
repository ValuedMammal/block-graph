#![no_main]

use bitcoin::BlockHash;
use block_graph::ConnectBlockError;
use block_graph_fuzz::{Harness, Op};
use libfuzzer_sys::fuzz_target;

/// The class of outcome a connect attempt should produce.
///
///`ChildHeightNotGreater`'s reported `child_hash`/`child_height` are unspecified
/// when more than one already-connected child conflicts (the library iterates a `HashSet`),
/// so this only tracks the error *kind*.
#[derive(Debug, PartialEq)]
enum Expected {
    Ok,
    HeightZeroReservedForRoot,
    HeightConflict,
    ChildHeightNotGreater,
    ParentHeightNotSmaller,
}

/// Predict the expected connect outcome, mirroring `connect_block`'s own validation order.
///
/// `check_not_second_genesis` (only the root may be connected at height 0), then
/// `check_height_unchanged` (a known hash must always be connected at the same height), then
/// `check_no_child_height_violation` (connecting a hash for the first time must not retroactively
/// precede an already-connected child that named it as parent), then the self-parent guard (a
/// block can't be its own parent), then `check_parent_height` (a known parent must be strictly
/// lower than the new block). An unknown parent or unknown hash never blocks a connection on its
/// own.
fn expect_outcome(
    harness: &Harness,
    height: u32,
    hash: BlockHash,
    parent_hash: BlockHash,
) -> Expected {
    let root = harness.known[0];
    if height == 0 && hash != root {
        return Expected::HeightZeroReservedForRoot;
    }

    match harness.heights.get(&hash) {
        Some(&existing_height) if existing_height != height => return Expected::HeightConflict,
        Some(_) => {}
        None => {
            let conflicts_with_a_child = harness.children.get(&hash).is_some_and(|children| {
                children.iter().any(|&(_, child_height)| height >= child_height)
            });
            if conflicts_with_a_child {
                return Expected::ChildHeightNotGreater;
            }
        }
    }

    let parent_height = if parent_hash == hash {
        Some(height)
    } else {
        harness.heights.get(&parent_hash).copied()
    };
    match parent_height {
        Some(parent_height) if parent_height >= height => Expected::ParentHeightNotSmaller,
        _ => Expected::Ok,
    }
}

fuzz_target!(|ops: Vec<Op>| {
    let mut harness = Harness::new();
    let ctx_ops = format!("{ops:?}");

    for (i, op) in ops.iter().take(256).enumerate() {
        let ctx = format!("op[{i}] = {op:?}\nall ops: {ctx_ops}");

        // Resolve once: `Op::Connect`'s `parent_idx` indexes into `known` modulo its length,
        // which grows on a successful connect, so the repeat call below must reuse this exact
        // tuple rather than re-resolving `op` against the now-mutated harness.
        let (height, hash, parent_hash) = harness.resolve(op);
        let expected = expect_outcome(&harness, height, hash, parent_hash);

        let before = harness.graph.clone();
        let result = harness.graph.connect_block(height, hash, parent_hash);

        let matches_expected = matches!(
            (&result, &expected),
            (Ok(_), Expected::Ok)
                | (
                    Err(ConnectBlockError::HeightZeroReservedForRoot { .. }),
                    Expected::HeightZeroReservedForRoot
                )
                | (Err(ConnectBlockError::HeightConflict { .. }), Expected::HeightConflict)
                | (
                    Err(ConnectBlockError::ChildHeightNotGreater { .. }),
                    Expected::ChildHeightNotGreater
                )
                | (
                    Err(ConnectBlockError::ParentHeightNotSmaller),
                    Expected::ParentHeightNotSmaller
                )
        );
        assert!(
            matches_expected,
            "connect outcome {result:?} doesn't match expected {expected:?}\ncontext: {ctx}"
        );

        match &result {
            Ok(_) => {
                harness.record(hash, height);
                harness.children.entry(parent_hash).or_default().push((hash, height));
                block_graph_fuzz::assert_ok(harness.graph.check_invariants(), &ctx);

                // Idempotence: re-issuing the identical connect is a no-op.
                let before_repeat = harness.graph.clone();
                match harness.graph.connect_block(height, hash, parent_hash) {
                    Ok(cs) => assert!(
                        cs.is_empty(),
                        "repeating a successful connect should return an empty changeset\ncontext: {ctx}"
                    ),
                    Err(e) => panic!("repeating a successful connect errored: {e}\ncontext: {ctx}"),
                }
                assert_eq!(
                    harness.graph, before_repeat,
                    "repeating a successful connect must not mutate the graph\ncontext: {ctx}"
                );
            }
            Err(_) => {
                // Atomicity: a failed connect must leave the graph exactly as it was.
                assert_eq!(
                    harness.graph, before,
                    "a failed connect must not mutate the graph\ncontext: {ctx}"
                );
            }
        }
    }
});
