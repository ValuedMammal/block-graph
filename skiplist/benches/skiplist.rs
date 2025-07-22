use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};

use bitcoin::hashes::Hash;
use bitcoin::BlockHash;

use bdk_chain::bitcoin;
use skiplist::SkipList;

// Call `get` with the given query.
fn bench_skiplist(skiplist: &SkipList<BlockHash>, q: u32) {
    assert!(skiplist.get(q).is_some());
}

fn skiplist(c: &mut Criterion) {
    const CT: usize = 100_000;

    let mut skiplist = SkipList::with_capacity(CT);

    for i in 0..CT {
        let height = i as u32;
        let hash = BlockHash::all_zeros();
        let _ = skiplist.insert(height, hash);
    }
    assert_eq!(skiplist.len(), CT);

    c.bench_function("skiplist", move |b| {
        b.iter(|| bench_skiplist(black_box(&skiplist), black_box(2100)));
    });
}

criterion_group!(benches, skiplist);
criterion_main!(benches);
