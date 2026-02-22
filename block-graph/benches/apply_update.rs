use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};

use bitcoin::hashes::Hash;
use bitcoin::BlockHash;

use bdk_chain::bitcoin;
use bdk_chain::BlockId;

const CT: u32 = 1000;

type BlockGraph = block_graph::BlockGraph<BlockHash>;
type CheckPoint = block_graph::CheckPoint<BlockHash>;

fn bench_apply_update(c: &mut Criterion) {
    let mut changeset = block_graph::ChangeSet::default();

    let mut prev_hash = BlockHash::all_zeros();

    for i in 0..CT {
        let height = i;
        let hash = Hash::hash(height.to_string().as_bytes());
        let block_id = BlockId { height, hash };
        changeset.blocks.insert((block_id, hash, prev_hash));
        prev_hash = hash;
    }

    // Create BlockGraph
    let graph = BlockGraph::from_changeset(changeset).unwrap().unwrap();

    // Connect next block to tip
    let tip = graph.tip();
    let tip_hash = tip.hash();
    let update = CheckPoint::new(CT + 1, Hash::hash(b"update"));

    c.bench_function("apply_update", move |b| {
        b.iter(|| {
            let mut graph = graph.clone();
            graph
                .apply_update(black_box(update.clone()), black_box(tip_hash))
                .expect("failed to apply update");
            let tip = graph.tip();
            assert_eq!(tip.height(), CT + 1);
            assert_eq!(tip.hash(), Hash::hash(b"update"));
        });
    });
}

criterion_group!(benches, bench_apply_update);
criterion_main!(benches);
