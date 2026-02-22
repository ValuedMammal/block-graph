<!-- block-graph -->

## Performance Summary

To reproduce:
```
cargo bench -p block_graph
```

### Time to search for a middle entry in a `CheckPoint<BlockHash>` containing $n$ elements

| N | time (ns) |
|---|---|
| 2<sup>17</sup> | 19.0 |
| 2<sup>18</sup> | 19.0 |
| 2<sup>19</sup> | 20.7 |
| 2<sup>20</sup> | 20.8 |

---

### Time to construct `BlockGraph<BlockHash>` from a `ChangeSet` containing $n$ elements

| N | time (ms) |
|---|---|
| 10,000 | 4.5 |

---

### Time to find result of `BlockGraph::is_block_in_chain` for a given query height

| N | query | time (ns) |
|---|---|---|
| 50,000 | 13 | 180 |

---

### Time to apply update to `BlockGraph<BlockHash>`

| N | time (us) |
|---|---|
| 1000 | 72 |

---

### 100 block reorg test for `BlockGraph<Header>`

| N | time (us) |
|---|---|
| 100 | 128 |

---

<!-- Benchmark `range` -->

## BlockGraph: Main Features

- **Monotone Structure**  
BlockGraph is monotone: once a block is added, it is never removed or modified. The graph only grows, preserving all historical data and supporting append-only semantics.

- **Sparse Chain Support**  
Blocks can be added at intermittent heights, allowing BlockGraph to represent forks, missing blocks, and partial chains.

- **Canonical Tip Selection**  
BlockGraph always determines the tip of the "best" (canonical) chain. The canonical tip is selected deterministically based on chain height and block hash, ensuring a unique and consistent view of the active chain.

- **Efficient Traversal**  
The structure provides efficient iteration over the blocks of the canonical chain (from tip to genesis), enabling fast traversal and inspection of the main chain.
Complexity approaches $O(log(n))$ for `get` and `insert` operations thanks to the internal skiplist representation.

- **Fork and Reorg Handling**  
BlockGraph naturally supports forks and chain reorgs, allowing robust handling of blockchain updates and competing branches.

- **ChangeSet and Update Application**   
BlockGraph can compute and apply changes as a result of "merging" with an updated chain tip, making it easy to synchronize, update, or roll back the chain as needed.
BlockGraph holds a `CheckPoint` internally, making it interoperable with other types in the BDK ecosystem.
