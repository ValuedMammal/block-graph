use std::hint::black_box;

use bitcoin::block::Header;
use bitcoin::hashes::Hash;
use bitcoin::pow;
use bitcoin::{BlockHash, TxMerkleNode};
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use block_graph::BlockId;

const CT: usize = 50_000;

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

fn is_block_in_chain(c: &mut Criterion) {
    // Initialize blockgraph
    let genesis = header(BlockHash::all_zeros(), Some(0));
    let mut block_graph = BlockGraph::from_genesis(genesis);
    let mut cp = block_graph.tip();

    // Insert block into blockgraph
    for height in 1..=CT as u32 {
        let h = header(cp.hash(), Some(height));
        cp = cp.push(height, h).unwrap();
    }
    let _ = block_graph.apply_update(cp).unwrap();

    assert_eq!(block_graph.iter().count(), CT + 1);

    let chain_tip = block_graph.tip().block_id();
    let test_height = 13;
    let test_block = BlockId {
        height: test_height,
        hash: block_graph.get(test_height).unwrap().hash(),
    };

    c.bench_function("is_block_in_chain", move |b| {
        b.iter(|| {
            let result = block_graph.is_block_in_chain(black_box(test_block), black_box(chain_tip));
            assert!(matches!(result, Some(true)));
        });
    });
}

criterion_group!(benches, is_block_in_chain);
criterion_main!(benches);
