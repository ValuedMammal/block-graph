use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};

use bitcoin::hashes::Hash;
use bitcoin::BlockHash;

use bdk_chain::bitcoin;
use bdk_chain::BlockId;

use block_graph::{BlockGraph, ChangeSet};

const CT: usize = 10_000;

// Construct `BlockGraph` from the given changeset.
fn bench_from_changeset(changeset: ChangeSet<BlockHash>) {
    let graph = BlockGraph::from_changeset(changeset)
        .expect("must contain genesis")
        .expect("failed to construct BlockGraph from changeset");
    assert_eq!(graph.iter().count(), CT);
}

fn from_changeset(c: &mut Criterion) {
    let mut changeset = block_graph::ChangeSet::default();

    let mut parent_hash = BlockHash::all_zeros();

    for i in 0..CT {
        let height = i as u32;
        let hash = Hash::hash(height.to_be_bytes().as_slice());
        let block_id = BlockId { height, hash };
        changeset.blocks.insert((block_id, hash, parent_hash));
        // update next parent id.
        parent_hash = hash;
    }

    c.bench_function("from_changeset", move |b| {
        b.iter(|| bench_from_changeset(black_box(changeset.clone())));
    });
}

criterion_group!(benches, from_changeset);
criterion_main!(benches);
