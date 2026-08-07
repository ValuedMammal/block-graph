use std::hint::black_box;

use bitcoin::block::Header;
use bitcoin::hashes::Hash;
use bitcoin::pow;
use bitcoin::{BlockHash, TxMerkleNode};
use block_graph::{BlockGraph, ChangeSet};
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

const CT: usize = 10_000;

fn header(prev_blockhash: BlockHash, nonce: Option<u32>) -> Header {
    Header {
        version: bitcoin::block::Version::default(),
        merkle_root: TxMerkleNode::all_zeros(),
        time: 1234567,
        bits: pow::Target::MAX_ATTAINABLE_REGTEST.to_compact_lossy(),
        nonce: nonce.unwrap_or_default(),
        prev_blockhash,
    }
}

// Construct `BlockGraph` from the given changeset.
fn bench_from_changeset(changeset: ChangeSet<Header>) {
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
        let h = header(parent_hash, Some(height));
        changeset.blocks.insert(h.block_hash(), (height, h));
        changeset.edges.insert((parent_hash, h.block_hash()));
        // update next parent id.
        parent_hash = h.block_hash();
    }

    c.bench_function("from_changeset", move |b| {
        b.iter(|| bench_from_changeset(black_box(changeset.clone())));
    });
}

criterion_group!(benches, from_changeset);
criterion_main!(benches);
