//! The benchmark creates a `BlockGraph<Header>` with 100 items
//! then creates an alternate chain with tip height 101 and connected to genesis.
//! We bench performance of connecting the new blocks, best tip selection, and setting the new tip.

use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};

use bitcoin::block::Header;
use bitcoin::hashes::Hash;
use bitcoin::pow;
use bitcoin::{BlockHash, TxMerkleNode};

use bdk_chain::bitcoin;

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

fn one_hundred_block_reorg(c: &mut Criterion) {
    // Create a BlockGraph with 101 elements (tip height of 100)
    let genesis_hash: BlockHash = Hash::hash(b"0");
    let genesis_header = header(BlockHash::all_zeros(), None);
    let mut block_graph = BlockGraph::from_genesis(genesis_header);
    let genesis_cp = block_graph.tip();
    let mut cp = genesis_cp.clone();

    // Insert blocks 1-100 into blockgraph
    for height in 1..=100u32 {
        let prev_blockhash = cp.hash();
        let header = header(prev_blockhash, None);
        cp = cp.push(height, header).unwrap();
    }
    let _ = block_graph.apply_update(cp, genesis_hash).unwrap();
    assert_eq!(block_graph.iter().count(), 101);

    // Create other checkpoint chain with blocks 1'-101'
    let mut other_cp = genesis_cp;
    for height in 1..=101u32 {
        let prev_hash = other_cp.hash();
        let other_header = header(prev_hash, Some(height));
        other_cp = other_cp.push(height, other_header).unwrap();
    }
    let expected_tip_hash = other_cp.hash();

    c.bench_function("100_block_reorg", move |b| {
        // Apply the checkpoint update and verify reorg occurred
        b.iter(|| {
            let mut test_block_graph = block_graph.clone();
            let _ = test_block_graph
                .apply_update(black_box(other_cp.clone()), black_box(genesis_hash))
                .unwrap();
            let new_tip = test_block_graph.tip();
            assert_eq!(new_tip.block_id(), (101, expected_tip_hash).into());
        });
    });
}

criterion_group!(benches, one_hundred_block_reorg);
criterion_main!(benches);
