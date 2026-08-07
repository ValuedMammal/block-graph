use std::hint::black_box;

use bitcoin::block::Header;
use bitcoin::hashes::Hash;
use bitcoin::pow;
use bitcoin::{BlockHash, TxMerkleNode};
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

const CT: u32 = 1000;

type BlockGraph = block_graph::BlockGraph<Header>;

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

fn bench_apply_update(c: &mut Criterion) {
    let mut changeset = block_graph::ChangeSet::default();

    let mut prev_hash = BlockHash::all_zeros();
    let mut prev_header = header(prev_hash, Some(0));

    for i in 0..CT {
        let height = i;
        let h = header(prev_hash, Some(height));
        changeset.blocks.insert(h.block_hash(), (height, h));
        changeset.edges.insert((prev_hash, h.block_hash()));
        prev_hash = h.block_hash();
        prev_header = h;
    }

    // Create BlockGraph
    let graph = BlockGraph::from_changeset(changeset).unwrap().unwrap();

    // Connect next block to tip
    let tip = graph.tip();
    let update_header = header(prev_header.block_hash(), Some(CT + 1));
    let update = tip.push(CT + 1, update_header).unwrap();

    c.bench_function("apply_update", move |b| {
        b.iter(|| {
            let mut graph = graph.clone();
            graph
                .apply_update(black_box(update.clone()))
                .expect("failed to apply update");
            let tip = graph.tip();
            assert_eq!(tip.height(), CT + 1);
            assert_eq!(tip.hash(), update_header.block_hash());
        });
    });
}

criterion_group!(benches, bench_apply_update);
criterion_main!(benches);
