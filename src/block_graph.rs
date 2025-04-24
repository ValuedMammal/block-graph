use std::collections::{BTreeSet, HashMap, HashSet};

use bitcoin::{block::Header, constants, hashes::Hash, BlockHash, Network, Target, Work};

/// Block id
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct BlockId {
    /// height
    pub height: u32,
    /// hash
    pub hash: BlockHash,
}

impl Default for BlockId {
    fn default() -> Self {
        Self {
            height: Default::default(),
            hash: constants::genesis_block(Network::Bitcoin).block_hash(),
        }
    }
}

/// Block header id
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockHeaderId {
    /// height
    pub height: u32,
    /// hash
    pub hash: BlockHash,
    /// header
    pub header: Header,
}

impl Default for BlockHeaderId {
    fn default() -> Self {
        let block = constants::genesis_block(Network::Bitcoin);
        let hash = block.block_hash();
        let header = block.header;
        Self {
            height: Default::default(),
            hash,
            header,
        }
    }
}

/// trait ToBlockId
pub trait ToBlockId {
    /// Return the identity of the block
    fn block_id(&self) -> BlockId;

    /// Return the [`Work`] of the block
    ///
    /// Default implementation is the work equivalent of [`Target::MAX_ATTAINABLE_MAINNET`].
    fn work(&self) -> Work {
        Target::MAX_ATTAINABLE_MAINNET.to_work()
    }
}

impl ToBlockId for BlockHeaderId {
    fn block_id(&self) -> BlockId {
        BlockId {
            height: self.height,
            hash: self.hash,
        }
    }
    fn work(&self) -> Work {
        Target::from(self.header.bits).to_work()
    }
}

/// A node in the `BlockGraph`
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Node<T> {
    /// Block data
    data: T,

    /// Connections
    // In a sparse chain a node may connect to multiple previous blocks
    conn: BTreeSet<BlockId>,

    /// Optimization: cache the accumulated work in the node, to avoid recomputing it.
    // TODO: this might not be a great design, because a sparse chain will naturally have less
    // work, so can potentially be gamed by introducing a shorter but heavier chain.
    acc: Option<Work>,
}

impl<T: ToBlockId> Node<T> {
    /// New from `data` and block id of the connection `conn`
    pub fn new(data: T, conn: BlockId) -> Self {
        Self {
            data,
            conn: [conn].into(),
            acc: None,
        }
    }

    /// Return the block identity of this node
    pub fn block_id(&self) -> BlockId {
        self.data.block_id()
    }

    /// Return the height of this node
    pub fn height(&self) -> u32 {
        self.block_id().height
    }

    /// Return the block hash of this node
    pub fn hash(&self) -> BlockHash {
        self.block_id().hash
    }

    /// Iterate over all connections of this node
    pub fn connections(&self) -> impl Iterator<Item = &BlockId> {
        self.conn.iter()
    }

    /// Return the *most recent* connection of this node, i.e. the highest block
    /// by height that this node connects to.
    pub fn connected_at(&self) -> BlockId {
        self.conn.iter().last().copied().expect("node must have a connection")
    }

    /// Return the work of this node
    pub fn work(&self) -> Work {
        self.data.work()
    }

    /// Return the accumulated work of this node
    pub fn acc_work(&self) -> Option<Work> {
        self.acc
    }
}

/// Block graph
///
/// A rooted, directed, acyclic graph where the nodes are (generic) blocks/headers and the
/// edges are block IDs.
#[derive(Debug, Clone)]
pub struct BlockGraph<T> {
    /// Nodes by block hash
    blocks: HashMap<BlockHash, Node<T>>,
    /// Next hashes, the set of blocks that connect to a given node
    next_hashes: HashMap<BlockHash, HashSet<BlockHash>>,
    /// A pointer to the active chain
    tip: BlockHash,
    /// Candidate tips (forks)
    ///
    /// Note that a candidate tip must also be a [valid chain](Self::is_valid_chain).
    tips: HashSet<BlockHash>,
}

// TODO: implement all the desired features
// - connect blocks
// - ? update canonical index
// - iterate blocks of the main chain
// - update best chain
// - impl to/from change set
// - make sure adding 2 values of Work doesn't overflow
// - simple BlockId chain

// Ways to connect a block
// case: extend main chain
// 0
// 0--A

// case: extend a candidate tip
// 0--A--B
// 0--A--Bh

// case: update to new tip (reorg)
// 0--A
// 0--Ah--Bh

// case: insert block into main chain
// 2 operations: connect-block (A->0), add-dependency (B->A)
// 0--o--B
// 0--A--B

// case: add orphan block (no parent)
// 0--A--B
//      -Bh

// how to manage tips
// HashSet<BlockHash>
// when a block is connected
//   remove its parent from tips
//   add this block to tips
// cache the acc. work for each node
// consider priority queue, BinaryHeap for efficient retrieval of the best tip

// optimize for fast indexing
// reintroduce canonical index as a contiguous array of Arc<T>
// periodically do work to reconstruct the index
// actions that would trigger reconstruction include
// - reorgs
// - introducing earlier blocks

impl<T: ToBlockId + Clone> BlockGraph<T> {
    /// Construct from genesis block
    ///
    /// Panics if the height of the genesis block is not 0.
    pub fn from_genesis(block: T) -> Self {
        let genesis_block = block.block_id();
        let genesis_height = genesis_block.height;
        assert_eq!(genesis_height, 0, "genesis block must be root of graph");

        let genesis_work = block.work();
        let genesis_hash = genesis_block.hash;
        let conn = BlockId {
            height: 0,
            hash: BlockHash::all_zeros(),
        };

        let mut blocks = HashMap::new();
        let mut next_hashes = HashMap::new();
        let mut tips = HashSet::new();

        // populate blocks
        let mut node = Node::new(block, conn);
        node.acc = Some(genesis_work);
        blocks.insert(genesis_hash, node);
        // populate edges
        next_hashes.insert(genesis_hash, HashSet::new());
        let tip = genesis_hash;
        // set tip
        tips.insert(tip);

        Self {
            blocks,
            next_hashes,
            tip,
            tips,
        }
    }

    /// Add an edge to the graph. Will be `None` if no entry is found in graph for the given
    /// `hash`.
    ///
    /// Adding an edge signifies that the Node at `hash` connects to the parent chain at `conn`.
    /// We can draw a number of implications as a result:
    ///
    /// - If the node at `hash` is reachable from the tip, we can deduce that `conn` is also
    ///   reachable from the tip.
    pub fn add_dependency(&mut self, hash: BlockHash, conn: BlockId) -> Option<ChangeSet<T>> {
        let mut change = ChangeSet::default();

        let node = self.blocks.get_mut(&hash)?;
        if node.conn.insert(conn) {
            change.blocks.push((node.data.clone(), conn));
        }

        // Update edges
        self.next_hashes.entry(conn.hash).or_default().insert(hash);

        Some(change)
    }

    /// Connect a `block` to the graph which connects to the node referenced by `conn`.
    pub fn connect_block(&mut self, block: T, conn: BlockId) -> ChangeSet<T> {
        let mut changeset = ChangeSet::default();
        let block_id = block.block_id();
        let hash = block_id.hash;

        // Avoid connecting a node to itself or a node at a greater height
        if block_id == conn || conn.height > block_id.height {
            return changeset;
        }
        assert!(block_id.height > conn.height);

        // An entry exists for this hash, add the edge and return
        if self.blocks.contains_key(&hash) {
            return self.add_dependency(hash, conn).expect("node must exist");
        }

        // Add this node, and record the fact that this block builds upon the previous
        self.blocks.insert(hash, Node::new(block.clone(), conn));
        self.next_hashes.entry(conn.hash).or_default().insert(hash);
        changeset.blocks.push((block, conn));

        self.accumulate_work(&hash); // why not?

        // Remove the parent from tips if present
        let tip_extended = self.tips.remove(&conn.hash);
        // If nothing builds on this block, considering adding it as a tip
        if !self.next_hashes.contains_key(&hash) {
            // If the block extends an existing tip, it necessarily becomes the new tip.
            // Or if the tip forms a valid chain it may be a fork contender. A valid
            // chain is one that has a common ancestor with the main chain.
            if tip_extended || self.is_valid_chain(&hash) {
                assert!(self.tips.insert(hash));
            }
        }

        // If the block extends the current tip, update the tip and return
        if conn.hash == self.tip().hash {
            self.tip = hash;
            return changeset;
        }

        // Re-evaluate tips to find the best one
        self.update_best_chain();

        changeset
    }

    /// Update `self` to the best chain.
    ///
    /// Called whenever a block is connected. Should be done very efficiently.
    //
    // TODO: The best chain is defined as...
    fn update_best_chain(&mut self) {
        let cur_height = self.tip().height;

        // Find best tip by height
        let best_block = self
            .tips
            .iter()
            .flat_map(|hash| Some(self.blocks.get(hash)?.block_id()))
            .max_by_key(|block| block.height)
            .expect("must have a valid tip");

        // Update self tip if there's a new best block
        if best_block.height > cur_height {
            self.tip = best_block.hash;
        }
    }

    /// Applies the given changeset
    pub fn apply_changeset(&mut self, changeset: ChangeSet<T>) {
        for (block, conn) in changeset.blocks {
            let _ = self.connect_block(block, conn);
        }
    }

    /// Obtain the initial [`ChangeSet`] of this block graph. The initial changeset
    /// is the difference between `self` and an empty graph.
    pub fn initial_changeset(&self) -> ChangeSet<T> {
        let mut blocks = vec![];
        for node in self.blocks.values() {
            for &conn in node.connections() {
                blocks.push((node.data.clone(), conn));
            }
        }

        ChangeSet { blocks }
    }

    /// Return the accumulated work of node at `hash` if it is part of a valid chain.
    ///
    /// Returns `None` if this node is not present in graph, or if its parent
    /// has no accumulated work of its own. A valid chain should always have
    /// accumulated work.
    fn _acc_work(&self, hash: &BlockHash) -> Option<Work> {
        let node = self.blocks.get(hash)?;
        let node_work = node.work();
        if let Some(parent) = self.blocks.get(&node.connected_at().hash) {
            if let Some(acc) = parent.acc_work() {
                return Some(acc + node_work);
            }
        }
        None
    }

    /// Compute the accumulated work of the node at `hash` and update the corresponding entry
    /// in the block graph.
    #[allow(unused)]
    fn accumulate_work(&mut self, hash: &BlockHash) {
        if let Some(acc) = self._acc_work(hash) {
            self.blocks.entry(*hash).and_modify(|n| {
                n.acc = Some(acc);
            });
        }
    }

    /// Get the total accumulated [`Work`] of the active chain
    #[allow(unused)]
    fn total_pow(&self) -> Work {
        self.blocks
            .get(&self.tip)
            .and_then(|n| n.acc)
            .expect("chain of tip must have accumulated work")
    }

    /// Get the genesis node
    pub fn genesis_node(&self) -> &Node<T> {
        // TODO: this may not be the most efficient. maybe store the "root" hash?
        let genesis_hash = self.get(0).expect("must have genesis block").hash;
        self.blocks.get(&genesis_hash).expect("must have node")
    }

    /// Get block id by height
    pub fn get(&self, height: u32) -> Option<BlockId> {
        self.iter().map(|item| item.block_id()).find(|b| b.height == height)
    }

    /// Get a node in the graph by hash
    pub fn node(&self, hash: &BlockHash) -> Option<&Node<T>> {
        self.blocks.get(hash)
    }

    /// Get the chain tip block id
    pub fn tip(&self) -> BlockId {
        self.get_chain_tip()
    }

    /// Iterate elements of `&T` in reverse height order starting from the tip of the active chain.
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        let mut cur = self.tip;
        core::iter::from_fn(move || {
            let node = self.blocks.get(&cur)?;
            let item = &node.data;
            cur = node.connected_at().hash;
            Some(item)
        })
    }

    /// The chain formed by `tip` is a valid chain iff it shares a common ancestor
    /// with the main chain.
    fn is_valid_chain(&self, hash: &BlockHash) -> bool {
        let chain_tip = self.tip();
        let mut cur = *hash;
        while let Some(node) = self.blocks.get(&cur) {
            if self.is_block_in_chain(node.block_id(), chain_tip).unwrap_or(false) {
                return true;
            }
            cur = node.connected_at().hash;
        }
        false
    }
}

/// Chain oracle
pub trait ChainOracle {
    /// Get chain tip
    fn get_chain_tip(&self) -> BlockId;
    /// Is block in chain with chain tip
    fn is_block_in_chain(&self, block: BlockId, chain_tip: BlockId) -> Option<bool>;
}

impl<T: ToBlockId + Clone> ChainOracle for BlockGraph<T> {
    fn get_chain_tip(&self) -> BlockId {
        let hash = self.tip;
        let height = self.blocks.get(&hash).expect("should have tip node").height();

        BlockId { height, hash }
    }

    fn is_block_in_chain(&self, block: BlockId, chain_tip: BlockId) -> Option<bool> {
        // The block cannot be in chain if its height is not within `chain_tip`
        if block.height > chain_tip.height {
            return None;
        }
        // We can't determine if block is in chain if `chain_tip` is
        // not part of the best chain
        if self.get(chain_tip.height) != Some(chain_tip) {
            return None;
        }
        // If `block` is reachable from the tip and the hashes match, then it
        // is also part of the best chain.
        self.get(block.height).map(|b| b.hash == block.hash)
    }
}

/// Records changes and additions to the block graph
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChangeSet<T> {
    /// tuples of (block, connected_at)
    blocks: Vec<(T, BlockId)>,
}

impl<T> Default for ChangeSet<T> {
    fn default() -> Self {
        Self {
            blocks: Default::default(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use bitcoin::Target;
    use bitcoin::{block, consensus, hashes};

    /// Generate a Header with a unique nonce `n` and constant difficulty
    fn generate_header(prev_blockhash: BlockHash, n: u32) -> Header {
        Header {
            version: block::Version::default(),
            prev_blockhash,
            merkle_root: hashes::Hash::hash(b"merkle-node"),
            time: 123,
            bits: Target::MAX_ATTAINABLE_REGTEST.to_compact_lossy(),
            nonce: n,
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    struct SimpleHeader {
        height: u32,
        hash: BlockHash,
        prev_hash: BlockHash,
        work: Work,
    }

    impl ToBlockId for SimpleHeader {
        fn block_id(&self) -> BlockId {
            BlockId {
                height: self.height,
                hash: self.hash,
            }
        }
        fn work(&self) -> Work {
            self.work
        }
    }

    impl SimpleHeader {
        fn genesis() -> Self {
            Self {
                height: 0,
                hash: bitcoin::constants::genesis_block(Network::Regtest).block_hash(),
                prev_hash: BlockHash::all_zeros(),
                work: Target::MAX_ATTAINABLE_REGTEST.to_work(),
            }
        }

        fn new(height: u32, header: Header) -> Self {
            Self {
                height,
                hash: header.block_hash(),
                prev_hash: header.prev_blockhash,
                work: Target::from(header.bits).to_work(),
            }
        }
    }

    #[test]
    fn header_chain() {
        let mut prev_hash = BlockHash::all_zeros();
        let work = Target::MAX_ATTAINABLE_REGTEST.to_work();

        let mut headers = vec![];

        for height in 0u32..5 {
            let hash: BlockHash = bitcoin::hashes::Hash::hash(height.to_be_bytes().as_slice());
            let header = SimpleHeader {
                height,
                hash,
                prev_hash,
                work,
            };
            headers.push(header);
            // update prev hash
            prev_hash = hash;
        }

        let g = headers[0];
        let mut graph = BlockGraph::from_genesis(g);
        for header in headers.into_iter().skip(1) {
            let cs = graph.connect_block(header, graph.tip());
            assert!(!cs.blocks.is_empty());
        }
        // dbg!(&graph);
        assert_eq!(graph.blocks.len(), 5);
    }

    #[test]
    fn from_genesis() {
        let genesis_block = constants::genesis_block(Network::Bitcoin);
        let header = genesis_block.header;
        let hash = genesis_block.block_hash();
        let header = BlockHeaderId {
            height: 0,
            hash,
            header,
        };
        let graph = BlockGraph::from_genesis(header);
        assert_eq!(graph.blocks.len(), 1);
        assert_eq!(graph.tips.len(), 1);
    }

    #[test]
    fn connect_block() {
        let genesis_block = constants::genesis_block(Network::Signet);
        let header = genesis_block.header;
        let hash = genesis_block.block_hash();
        let header = BlockHeaderId {
            height: 0,
            hash,
            header,
        };
        let mut graph = BlockGraph::from_genesis(header);

        let header_1: Header = consensus::encode::deserialize_hex("00000020f61eee3b63a380a477a063af32b2bbc97c9ff9f01f2c4225e973988108000000f575c83235984e7dc4afc1f30944c170462e84437ab6f2d52e16878a79e4678bd1914d5fae77031eccf40700")
            .unwrap();
        let header_1 = BlockHeaderId {
            height: 1,
            hash: header_1.block_hash(),
            header: header_1,
        };
        let tip = graph.tip();
        let cs = graph.connect_block(header_1, tip);
        assert!(!cs.blocks.is_empty());
        // dbg!(&res);
        // dbg!(&graph);
        assert_eq!(graph.blocks.len(), 2);
        assert_eq!(graph.tip().height, header_1.height);
        assert_eq!(graph.tip().hash, header_1.hash);
    }

    #[test]
    fn graph_should_handle_connect_same_block() {
        let mut graph = BlockGraph::from_genesis(SimpleHeader::genesis());
        let genesis_header = graph.genesis_node().data;
        let og_tip = graph.tip();

        // try reconnect the genesis block
        let cs = graph.connect_block(genesis_header, og_tip);
        assert!(cs.blocks.is_empty());

        // connect block 1 for real
        let header = generate_header(og_tip.hash, 1);
        let header_1 = SimpleHeader {
            height: 1,
            hash: header.block_hash(),
            prev_hash: og_tip.hash,
            work: Target::from(header.bits).to_work(),
        };
        let cs = graph.connect_block(header_1, og_tip);
        assert!(!cs.blocks.is_empty());
        assert_eq!(graph.tip().height, 1);

        // Now reconnect the same header to the original tip
        let cs = graph.connect_block(header_1, og_tip);
        assert!(cs.blocks.is_empty());

        // connect the current tip to itself
        let tip = graph.tip();
        let node = graph.node(&tip.hash).unwrap().clone();
        let cs = graph.connect_block(node.data, tip);
        assert!(cs.blocks.is_empty());
    }

    #[test]
    fn iter_timechain() {
        let mut prev_hash = BlockHash::all_zeros();
        let work = Target::MAX_ATTAINABLE_REGTEST.to_work();

        let mut headers: Vec<SimpleHeader> = vec![];

        for height in 0u32..5 {
            let hash: BlockHash = hashes::Hash::hash(height.to_be_bytes().as_slice());
            let header = SimpleHeader {
                height,
                hash,
                prev_hash,
                work,
            };
            headers.push(header);
            // update prev hash
            prev_hash = hash;
        }

        let g = headers[0];
        let mut graph = BlockGraph::from_genesis(g);
        for header in headers.into_iter().skip(1) {
            let _ = graph.connect_block(header, graph.tip());
        }
        // dbg!(&graph);
        assert_eq!(graph.blocks.len(), 5);

        let blocks: Vec<SimpleHeader> = graph.iter().copied().collect();
        assert_eq!(blocks.len(), 5);
    }

    #[test]
    fn update_best_chain() {
        let mut graph = BlockGraph::from_genesis(SimpleHeader::genesis());
        let og_tip = graph.tip();

        // connect block 1
        let header = generate_header(og_tip.hash, 1);
        let header_1 = SimpleHeader::new(1, header);
        let _ = graph.connect_block(header_1, og_tip);
        assert_eq!(graph.tip().height, 1);
        let og_tip_1 = graph.tip();

        // connect fork tip block 1h
        let mut header_h = header;
        header_h.nonce = 2;
        let header_1h = SimpleHeader::new(1, header_h);
        assert_ne!(header_1h, header_1);
        let _ = graph.connect_block(header_1h, og_tip);
        assert_eq!(graph.blocks.len(), 3);
        assert_eq!(graph.tips.len(), 2);
        // no reorg yet
        assert_eq!(graph.tip(), og_tip_1);

        // now build on top of the fork tip
        let header = generate_header(header_1h.hash, 2);
        let header_2h = SimpleHeader::new(2, header);
        let header_1h_block = BlockId {
            height: header_1h.height,
            hash: header_1h.hash,
        };
        let _ = graph.connect_block(header_2h, header_1h_block);
        assert_eq!(graph.tip().height, 2);
        assert_eq!(graph.tip(), header_2h.block_id());
        // dbg!(&graph);
    }

    #[test]
    fn test_total_pow() {
        let genesis_header = SimpleHeader::genesis();
        let mut graph = BlockGraph::from_genesis(genesis_header);

        let mut prev_hash = graph.genesis_node().hash();

        let init_target: Target = Target::MAX_ATTAINABLE_REGTEST;
        assert_eq!(graph.total_pow(), init_target.to_work());
        let mut exp_work = init_target.to_work();

        for n in 0..5 {
            let header = generate_header(prev_hash, n);
            let hash = header.block_hash();
            // sum the expected work
            let trg: Target = header.bits.into();
            exp_work = exp_work + trg.to_work();
            // connect this block
            let height = n + 1;
            let to_connect = SimpleHeader::new(height, header);
            let _ = graph.connect_block(to_connect, graph.tip());
            // update previous hash
            prev_hash = hash;
        }

        assert!(graph.total_pow() > init_target.to_work());
        assert_eq!(graph.total_pow(), exp_work);
    }

    #[test]
    fn connect_orphan_block() {
        let genesis_header = SimpleHeader::genesis();
        let mut graph = BlockGraph::from_genesis(genesis_header);

        let mut prev_hash = graph.genesis_node().hash();

        for n in 0..3 {
            let header = generate_header(prev_hash, n);
            let hash = header.block_hash();
            // connect this block
            let height = n + 1;
            let to_connect = SimpleHeader::new(height, header);
            let _ = graph.connect_block(to_connect, graph.tip());
            // update previous hash
            prev_hash = hash;
        }

        assert_eq!(graph.blocks.len(), 4);
        assert_eq!(graph.tips.len(), 1);
        let og_tip = graph.tip();

        // Now connect an orphan, i.e. a block with no in-graph parents
        let conn = BlockId {
            height: 2,
            hash: hashes::Hash::hash(b"no-parent"),
        };
        let header = generate_header(conn.hash, 42);
        let header_3a = SimpleHeader::new(3, header);
        let _ = graph.connect_block(header_3a, conn);
        // dbg!(&graph.tips);
        assert_eq!(graph.blocks.len(), 5);
        assert_eq!(graph.tips.len(), 1, "we should not have added any new tips");
        assert_eq!(graph.tip(), og_tip);
    }
}
