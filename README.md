<!-- block-graph -->

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
BlockGraph can be updated by (and transformed into) a `CheckPoint`, making it interoperable with other types in the BDK ecosystem.
