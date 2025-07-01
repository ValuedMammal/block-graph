use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};

use bitcoin::hashes::Hash;
use bitcoin::BlockHash;

use bdk_chain::bitcoin;
use skiplist::SkipList;

fn do_bench(skiplist: &SkipList<BlockHash>, r: impl core::ops::RangeBounds<u32>) {
    let it = skiplist.range(r);
    assert_eq!(it.count(), 2_000);
}

fn skiplist_range(c: &mut Criterion) {
    const CT: usize = 100_000;

    let mut skiplist = SkipList::<BlockHash>::with_capacity(CT);

    // SkipList results
    // 10k: 15 us
    // 100k: 7 us

    for i in 0..CT {
        let height = i as u32;
        let _ = skiplist.insert(height, BlockHash::all_zeros());
    }
    assert_eq!(skiplist.len(), CT);

    c.bench_function("skiplist_range", move |b| {
        let start = 49_000;
        let end = 51_000;
        b.iter(|| do_bench(black_box(&skiplist), black_box(start..end)));
    });
}

criterion_group!(benches, skiplist_range);
criterion_main!(benches);
