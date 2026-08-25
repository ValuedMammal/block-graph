#![no_main]

use bitcoin::BlockHash;
use block_graph::CheckPoint;
use block_graph_fuzz::hash_from_id;
use libfuzzer_sys::fuzz_target;

/// A single [`CheckPoint`] operation, differentially tested against a naive
/// `Vec<(u32, BlockHash)>` model of the same chain.
#[derive(Debug, arbitrary::Arbitrary)]
enum CpOp {
    /// Push onto the tip; `height_delta` of 0 exercises the non-increasing-height rejection.
    Push {
        height_delta: u8,
        id: u8,
    },
    /// Insert at an arbitrary height (height 0 is skipped: replacing genesis panics by design).
    Insert {
        height: u16,
        id: u8,
    },
    Get(u16),
    Range(u16, u16),
}

fuzz_target!(|ops: Vec<CpOp>| {
    let genesis_hash = hash_from_id(0);
    let mut cp = CheckPoint::new(0u32, genesis_hash);
    // Ascending, one entry per height — mirrors the checkpoint chain.
    let mut model: Vec<(u32, BlockHash)> = vec![(0, genesis_hash)];

    let ctx_ops = format!("{ops:?}");

    for (i, op) in ops.iter().take(256).enumerate() {
        let ctx = format!("op[{i}] = {op:?}\nall ops: {ctx_ops}");

        match *op {
            CpOp::Push { height_delta, id } => {
                let height = cp.height().saturating_add(height_delta as u32);
                let hash = hash_from_id(id);
                match cp.clone().push(height, hash) {
                    Ok(next) => {
                        cp = next;
                        model.push((height, hash));
                    }
                    Err(_) => assert_eq!(
                        height_delta, 0,
                        "push should only fail when height doesn't strictly increase\ncontext: {ctx}"
                    ),
                }
            }
            CpOp::Insert { height, id } => {
                let height = height as u32;
                if height == 0 {
                    continue;
                }
                let hash = hash_from_id(id);
                let existing = model.iter().find(|(h, _)| *h == height).map(|(_, hash)| *hash);
                let before = cp.clone();
                let next = cp.clone().insert(height, hash);

                match existing {
                    Some(existing_hash) if existing_hash == hash => {
                        assert!(
                            next.eq_ptr(&before),
                            "inserting an identical value must return the chain unchanged\ncontext: {ctx}"
                        );
                    }
                    Some(_) => {
                        // Conflict at `height`: everything above it is truncated.
                        model.retain(|(h, _)| *h < height);
                        model.push((height, hash));
                    }
                    None => {
                        // A true gap: splice in without disturbing entries above or below.
                        let pos = model.partition_point(|(h, _)| *h < height);
                        model.insert(pos, (height, hash));
                    }
                }
                cp = next;
            }
            CpOp::Get(height) => {
                let height = height as u32;
                let expected = model.iter().find(|(h, _)| *h == height).copied();
                let got = cp.get(height).map(|c| (c.height(), c.hash()));
                assert_eq!(got, expected, "get({height}) mismatch\ncontext: {ctx}");
            }
            CpOp::Range(a, b) => {
                let (a, b) = (a as u32, b as u32);
                if a > b {
                    continue;
                }
                let expected: Vec<(u32, BlockHash)> = model
                    .iter()
                    .rev()
                    .filter(|(h, _)| (a..b).contains(h))
                    .copied()
                    .collect();
                let got: Vec<(u32, BlockHash)> =
                    cp.range(a..b).map(|c| (c.height(), c.hash())).collect();
                assert_eq!(got, expected, "range({a}..{b}) mismatch\ncontext: {ctx}");
            }
        }

        assert_eq!(cp.len(), model.len(), "len() mismatch\ncontext: {ctx}");
        assert_eq!(
            cp.index() as usize,
            model.len() - 1,
            "index() mismatch\ncontext: {ctx}"
        );
    }
});
