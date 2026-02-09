use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};
use std::sync::Arc;

use bitcoin::hashes::Hash;
use bitcoin::BlockHash;

use bdk_chain::bitcoin;
use bdk_chain::{BlockId, CanonicalizationParams};
use bitcoin::{absolute::LockTime, transaction, Amount, ScriptBuf, Transaction, TxIn, TxOut};

const CT: usize = 2_000;

use block_graph::CheckPoint;

type TxGraph = bdk_chain::TxGraph<BlockId>;
type BlockGraph = block_graph::BlockGraph<BlockHash>;

// Run benchmark.
fn bench_canonical_view_txs(tx_graph: &TxGraph, chain: &BlockGraph) {
    assert_eq!(
        tx_graph
            .canonical_view(chain, chain.tip(), CanonicalizationParams::default())
            .txs()
            .count(),
        CT,
    );
}

fn canonical_view_txs(c: &mut Criterion) {
    // Initialize blockgraph
    let mut chain = BlockGraph::from_genesis(Hash::hash(b"0"));

    // Initialize txgraph
    let mut tx_graph = TxGraph::default();

    for height in 1..=CT as u32 {
        // Insert block into chain
        let hash = BlockHash::all_zeros();
        let id = BlockId { height, hash };
        let chain_tip = chain.tip();
        let _ = chain
            .apply_update(
                CheckPoint::from_entries([
                    (chain_tip.height, chain_tip.hash),
                    (id.height, id.hash),
                ])
                .unwrap(),
            )
            .unwrap();

        // Insert tx into txgraph
        let tx = Transaction {
            version: transaction::Version::ONE,
            lock_time: LockTime::from_consensus(height),
            input: vec![TxIn::default()],
            output: vec![TxOut {
                value: Amount::ZERO,
                script_pubkey: ScriptBuf::new_op_return([0xb1, 0x0c]),
            }],
        };
        let txid = tx.compute_txid();
        let _ = tx_graph.insert_tx(Arc::new(tx));

        // Insert anchor
        let anchor = id;
        let _ = tx_graph.insert_anchor(txid, anchor);
    }

    assert_eq!(chain.iter().count(), CT + 1);

    c.bench_function("canonical_view_txs", move |b| {
        b.iter(|| bench_canonical_view_txs(black_box(&tx_graph), black_box(&chain)));
    });
}

criterion_group!(benches, canonical_view_txs);
criterion_main!(benches);
