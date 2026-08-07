use std::hint::black_box;

use bdk_chain::bitcoin;
use bitcoin::hashes::Hash;
use bitcoin::BlockHash;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

const CT: u32 = 1000;

type BlockGraph = block_graph::BlockGraph<BlockHash>;

fn bench_apply_update(c: &mut Criterion) {
    let mut changeset = block_graph::ChangeSet::default();

    let mut prev_hash = BlockHash::all_zeros();

    for i in 0..CT {
        let height = i;
        let hash = Hash::hash(height.to_string().as_bytes());
        changeset.blocks.insert(hash, (height, hash));
        changeset.edges.insert((prev_hash, hash));
        prev_hash = hash;
    }

    // Create BlockGraph
    let graph = BlockGraph::from_changeset(changeset).unwrap().unwrap();

    // Connect next block to tip
    let tip = graph.tip();
    let update = tip.push(CT + 1, Hash::hash(b"update")).unwrap();

    c.bench_function("apply_update", move |b| {
        b.iter(|| {
            let mut graph = graph.clone();
            graph
                .apply_update(black_box(update.clone()))
                .expect("failed to apply update");
            let tip = graph.tip();
            assert_eq!(tip.height(), CT + 1);
            assert_eq!(tip.hash(), Hash::hash(b"update"));
        });
    });
}

criterion_group!(benches, bench_apply_update);
criterion_main!(benches);
