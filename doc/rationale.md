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

**Why is the use of `unsafe` pervasive throughout the `node` module?**

The use of `unsafe` is used to break the lifetime constraint on `&self` when calling methods on Node and is only necessary when traversing nodes of the skiplist. The unsafety is contained within a safe API, meaning that if implemented properly, a user of SkipList should never run the risk of undefined behavior. In fact this is a common theme in the rust standard library with collections such as `LinkedList` and `Vec`.

For example:

The function `advance_while_mut` intends to return a `&mut` reference to `Node`. To do so, we need to continuously compare two adjacent nodes (by reference) to decide whether to advance.

The problem is that we can't hold a shared reference and an exclusive reference to `self` simultaneously (even though we never mutate a value before the method returns). In safe rust, a reference to a local variable isn't allowed to escape the scope of its referent. Normally this makes sense to prevent pointers from dangling, etc, however we make an exception in order to use a method like `advance_while_mut`, and we guarantee its safety by ensuring that instances of Node always begin as a pointer that is non-null and properly aligned (the pointer always points to a valid Node instance).

We get around this limitation by reinterpreting a value of `Node` through the use of raw pointers in order to obtain a unique reference with a lifetime that can expand to the scope in which it is used, i.e. an [unbounded lifetime](<https://doc.rust-lang.org/nomicon/unbounded-lifetimes.html#unbounded-lifetimes>). In addition the `Node` type is never exposed outside of the library, so the scope of any reference to Node cannot outlive the SkipList itself.

**The CheckPoint type is cheaply cloneable and thread safe, meaning we can pass around a reference to the chain tip, query it from multiple threads, and so on. Why can't SkipList be more like CheckPoint?**

The reason we're able to do this with CheckPoint is that CheckPoint just wraps a `CPInner`, that is, it is functionally equivalent to the inner node structure. Perhaps surprisingly, iterating a CheckPoint yields items of... well, CheckPoint. But unlike a checkpoint, `SkipList<T>` when iterated does not yield items of itself, rather it yields items of `&(u32, T)`. This is analogous to getting the height (`u32`) and block data (`T`) from a `CheckPoint<T>`.

As a high level data structure SkipList does not behave like any other Node, because it contains extra information related to the levels and capacity, it holds a pointer to the head node, as well as the logic of level generation (`rng`). On the plus side, we can convert the canonical chain of BlockGraph into a CheckPoint if we choose (a singly linked-list is enough if all we need is to iterate blocks from the tip), allowing us to pass it directly to a chain source just as you would when updating a LocalChain.

**Alternatives?**

The implementation of BlockGraph is largely inspired by [JP-Ellis/rust-skiplist](<https://github.com/JP-Ellis/rust-skiplist/>) which was chosen because of its relative popularity on crates.io. Other implementations that have been considered are... TODO

## Glossary of terms

- **Graph**: A structure with edges and vertices. The BlockGraph is a graph in which the points are blocks and the edges are block hashes. To be precise, the block graph is also rooted, directed, and acyclic, making it resemble an _out-tree_ or [arborescence](<https://en.wikipedia.org/wiki/Arborescence_(graph_theory)>).

- **Tip**: A point in the graph from which no edge extends. In BlockGraph the tip represents the block height and hash of the latest block in the chain (or a stale block if it was never built upon).

- **Root**: The point that all edges point away from. By definition nothing precedes it. In BlockGraph the root corresponds to the node at height 0, or the so-called "genesis" block.

- **Valid Chain**: A chain where the root is reachable by traversing backward from any intermediate node along the chain.

- **Canonical Chain**: A valid chain that is also the heaviest, which in terms of Bitcoin protocol means it has the most accumulated proof of work. Because BlockGraph is allowed to be sparse, attempting to accumulate the work involved in producing every block may undershoot the true quantity, therefore we use a simplified heuristic that the longest chain by height is also the chain of most work.
