<!-- block-graph -->

`BlockGraph` is a structure designed to efficiently manage a directed acyclic graph (DAG) of Bitcoin blocks. It provides mechanisms to store, retrieve, and iterate over blocks while maintaining the integrity of the graph and supporting features like forks, reorgs, and sparse chains. Below are the key aspects and benefits:

---

### 1. **Efficient Storage and Retrieval of Block Data**
- **Hash-Based Indexing**:
  - Blocks are stored in a `HashMap` (`blocks`) where each block is indexed by its unique hash. This allows for **O(1)** average-time complexity for block lookups.
- **Parent-Child Relationships**:
  - The `next_hashes` map tracks the children of each block, enabling efficient traversal of the graph and quick identification of tips (blocks with no children).
- **Immutable Connections**:
  - Once a block is connected, its edges are immutable, ensuring the integrity of the graph.
  - The graph only grows as new blocks are added, ensuring that no data is lost or modified after being added.
- **Flexible Block Types**:
  - The `BlockGraph` is generic over the block type (`T`), enabling it to store and manage different types of block data, such as headers or full blocks.
- **Canonical Chain Iteration**:
  - The `iter` method allows for efficient iteration over blocks in descending height order from the tip of the canonical chain. This is achieved by following parent-child relationships stored in the graph.
  - Each block has a set of one or more parents, enabling sparse chains and the insertion of historic block data.

---

### 2. **Infallible Block Connection**
- **Validation of Connections**:
  - Blocks may be connected as long as the addition of an edge would not result in cyclic or invalid dependencies.
- **Support for Forks and Orphans**:
  - The graph supports multiple forks by maintaining a `tips` set, which tracks all candidate tips. Orphan blocks (blocks with no in-graph parent) can also be added without disrupting the graph.

---

### 3. **Dynamic Tip Management**
- **Dynamic Tip Updates**:
  - The `tips` set is dynamically updated as blocks are connected, ensuring that it always contains the current candidate tips.
- **Chain Validation**:
  - The `is_valid_chain` method ensures that a chain connects to the genesis block, either directly or indirectly, making it easy to validate the integrity of any chain in the graph.
- **Best Tip Selection**:
  - The `compare_tips` method dynamically determines the best chain by selecting the longest chain by height. In case of ties, the chain with the smallest block hash is chosen, ensuring determinism.
- **Chain Oracle**:
  - As a chain oracle, the `BlockGraph` can return the tip of the active chain using `get_chain_tip`. The `is_block_in_chain` method is used to determine if a given block is reachable from the tip of the active chain.

---

### 4. **Reorganization and Recovery**
- **Reindexing**:
  - The `reindex` method recalculates the set of valid tips and determines the best chain when the graph's state becomes inconsistent (e.g., after out-of-order connections). This ensures that the graph can recover from disruptions.
- **ChangeSet Support**:
  - The `ChangeSet` structure records changes made to the graph, enabling easy application of changes.
