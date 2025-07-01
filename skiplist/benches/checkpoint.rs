use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};

use bitcoin::hashes::Hash;

use bdk_chain::{bitcoin, CheckPoint};

// Call `get` with the given query.
fn bench_cp(cp: &CheckPoint, q: u32) {
    assert!(cp.get(q).is_some());
}

fn checkpoint(c: &mut Criterion) {
    const CT: usize = 100_000;

    let mut cp = CheckPoint::new(0, Hash::hash(b"0"));

    // CheckPoint results
    // 100k: 380 us

    for i in 1..CT {
        let height = i as u32;
        let hash = Hash::hash(height.to_be_bytes().as_slice());
        cp = cp.push(height, hash).unwrap();
    }
    assert_eq!(cp.clone().iter().count(), CT);

    c.bench_function("checkpoint", move |b| {
        b.iter(|| bench_cp(black_box(&cp), black_box(100)));
    });
}

criterion_group!(benches, checkpoint);
criterion_main!(benches);
