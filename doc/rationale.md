# BlockGraph design rationale

This document outlines the rationale behind the design choices made in the implementation of BlockGraph.

## Rationale

**Why develop BlockGraph when we already have LocalChain?**

Currently changes to LocalChain are [not monotone](<https://github.com/bitcoindevkit/bdk_wallet/issues/148>).
Why is this a desirable property of the local chain? So that we don't lose the ability to query the chain for historical information about a block whether or not it is part of the best chain. Also since the change set is directly implicated in this effort it makes sense to design a new type that can be adopted slowly and minimize disruption to existing APIs in BDK.

**Why model a DAG?**

This enables forks and branches to occur, which happen naturally in a distributed network. In contrast with LocalChain which can only support a single canonical chain of blocks - if a reorg occurs, the information about disconnected blocks is simply lost.

**Why introduce a skiplist data structure?**

Statistically it's more performant (due to having a stack of linked lists enabling us to "skip" over large sections of the list at higher levels), at the cost of potentially more memory (since we're now dealing with a stack of linked lists instead of only one).

The skiplist isn't a hard requirement of BlockGraph, per se, but the attractive performance profile may help to facilitate adoption.

**Alternatives**

Some alternatives have been considered for implementing a skiplist, discussed here

- [`JP-Ellis/rust-skiplist`](https://github.com/JP-Ellis/rust-skiplist) Pure rust implementation of a skiplist. We decided against using this or developing an in-house solution because it is thought that similar performance can be achieved by adding the necessary features to the existing `CheckPoint` type.
- [`crossbeam-skiplist`](https://github.com/crossbeam-rs/crossbeam/tree/master/crossbeam-skiplist). Skiplist implementation that features lock-free concurrency based on ["epoch-based memory reclamation"](https://github.com/crossbeam-rs/crossbeam/blob/master/crossbeam-epoch/src/lib.rs), aka garbage collection. Looks promising as a dependency but probably not worth the effort of reimplementing the thing from scratch.
- ☑️ Achieve skiplist-like behavior by adding additional pointers to the `Node` structure in a singly linked list. This is mostly inpired by Bitcoin Core's [`CBlockIndex`](https://github.com/bitcoin/bitcoin/blob/v29.2/src/chain.h#L140) class. It's a novel approach for looking up historical block data in logarithmic time.

## Glossary of terms

- **Graph**: A structure with edges and vertices. The BlockGraph is a graph in which the points are blocks and the edges are block hashes. To be precise, the block graph is also rooted, directed, and acyclic, making it resemble an _out-tree_ or [arborescence](<https://en.wikipedia.org/wiki/Arborescence_(graph_theory)>).

- **Tip**: A point in the graph from which no edge extends. In BlockGraph the tip represents the block height and hash of the latest block in the chain (or a stale block if it was never built upon).

- **Root**: The point that all edges point away from. By definition nothing precedes it. In BlockGraph the root corresponds to the node at height 0, or the so-called "genesis" block.

- **Valid Chain**: A chain where the root is reachable by traversing backward from any intermediate node along the chain.

- **Canonical Chain**: A valid chain that is also the heaviest, which in terms of Bitcoin protocol means it has the most accumulated proof of work. Because BlockGraph is allowed to be sparse, attempting to accumulate the work involved in producing every block may undershoot the true quantity, therefore we use a simplified heuristic that the longest chain by height is also the chain of most work.
