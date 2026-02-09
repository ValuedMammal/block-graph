//! Benchmark `CheckPoint::get` with a query height near the middle of checkpoint of varying lengths.

use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};

use bdk_chain::bitcoin;
use bitcoin::{hashes::Hash, BlockHash};
use block_graph::CheckPoint;

// Call `get` with the given query.
fn bench_cp(cp: &CheckPoint<BlockHash>, query_height: u32) {
    assert!(cp.get(query_height).is_some(), "`query_height` should exist in `cp`");
}

fn checkpoint_2_pow_17(c: &mut Criterion) {
    let mut cp = CheckPoint::new(0, BlockHash::all_zeros());

    let len: u32 = 1 << 17;
    let query_height = len / 2;

    for i in 1..len {
        let height = i;
        let value: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
        cp = cp.push(height, value).unwrap();
    }

    c.bench_function("checkpoint_2_pow_17", move |b| {
        b.iter(|| bench_cp(black_box(&cp), black_box(query_height)));
    });
}

fn checkpoint_2_pow_18(c: &mut Criterion) {
    let mut cp = CheckPoint::new(0, BlockHash::all_zeros());

    let len: u32 = 1 << 18;
    let query_height = len / 2;

    for i in 1..len {
        let height = i;
        let value: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
        cp = cp.push(height, value).unwrap();
    }

    c.bench_function("checkpoint_2_pow_18", move |b| {
        b.iter(|| bench_cp(black_box(&cp), black_box(query_height)));
    });
}

fn checkpoint_2_pow_19(c: &mut Criterion) {
    let mut cp = CheckPoint::new(0, BlockHash::all_zeros());

    let len: u32 = 1 << 19;
    let query_height = len / 2;

    for i in 1..len {
        let height = i;
        let value: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
        cp = cp.push(height, value).unwrap();
    }

    c.bench_function("checkpoint_2_pow_19", move |b| {
        b.iter(|| bench_cp(black_box(&cp), black_box(query_height)));
    });
}

fn checkpoint_2_pow_20(c: &mut Criterion) {
    let mut cp = CheckPoint::new(0, BlockHash::all_zeros());

    let len: u32 = 1 << 20;
    let query_height = len / 2;

    for i in 1..len {
        let height = i;
        let value: BlockHash = Hash::hash(height.to_be_bytes().as_slice());
        cp = cp.push(height, value).unwrap();
    }

    c.bench_function("checkpoint_2_pow_20", move |b| {
        b.iter(|| bench_cp(black_box(&cp), black_box(query_height)));
    });
}

criterion_group!(
    benches,
    checkpoint_2_pow_17,
    checkpoint_2_pow_18,
    checkpoint_2_pow_19,
    checkpoint_2_pow_20
);
criterion_main!(benches);
