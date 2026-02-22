use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};

use bitcoin::hashes::Hash;
use bitcoin::BlockHash;

use bdk_chain::BlockId;
use bdk_chain::{bitcoin, ChainOracle};

const CT: usize = 50_000;

type BlockGraph = block_graph::BlockGraph<BlockHash>;

fn is_block_in_chain(c: &mut Criterion) {
    // Initialize blockgraph
    let mut block_graph = BlockGraph::from_genesis(Hash::hash(b"0"));
    let mut cp = block_graph.tip();
    let tip_hash = cp.hash();

    // Insert block into blockgraph
    for height in 1..=CT as u32 {
        let hash: BlockHash = Hash::hash(height.to_string().as_bytes());
        cp = cp.push(height, hash).unwrap();
    }
    let _ = block_graph.apply_update(cp, tip_hash).unwrap();

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
            assert!(matches!(result, Ok(Some(true))));
        });
    });
}

criterion_group!(benches, is_block_in_chain);
criterion_main!(benches);
