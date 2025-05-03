use std::cmp;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

use bitcoin::{block::Header, constants, hashes::Hash, BlockHash, Network};

/// Block id.
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
}

impl ToBlockId for BlockHeaderId {
    fn block_id(&self) -> BlockId {
        BlockId {
            height: self.height,
            hash: self.hash,
        }
    }
}

impl ToBlockId for BlockId {
    fn block_id(&self) -> BlockId {
        *self
    }
}

/// A node in the block graph.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Node<T> {
    /// Block data.
    data: T,

    /// Set of the node's parents.
    ///
    /// In a sparse chain a node may connect to multiple previous blocks.
    conn: BTreeSet<BlockId>,
}

impl<T: ToBlockId> Node<T> {
    /// New from `data` and block id of the connection `conn`
    pub fn new(data: T, conn: BlockId) -> Self {
        Self {
            data,
            conn: [conn].into(),
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

    /// Return the block data of this node
    pub fn data(&self) -> &T {
        &self.data
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
}

/// `BlockGraph`.
///
/// Internally modeled as a directed, acyclic graph (rooted in the genesis block) where
/// the nodes are blocks and the edges are hashes.
#[derive(Debug, Clone)]
pub struct BlockGraph<T> {
    /// Nodes by block hash
    blocks: HashMap<BlockHash, Node<T>>,
    /// Next hashes, represents the set of blocks that extend a given node
    next_hashes: HashMap<BlockHash, HashSet<BlockHash>>,
    /// The root hash, aka genesis
    root: BlockHash,
    /// Hash of the current chain tip
    tip: BlockHash,
    /// Candidate tips (forks)
    ///
    /// By definition a candidate tip must be a [valid chain](Self::is_valid_chain).
    tips: HashSet<BlockHash>,
}

// Features
// - Connect blocks to graph
// - Iterate blocks of the main chain
// - Update best chain
// - impl to/from ChangeSet
// - sparse chain
// - handles reorgs
// - HeaderId chain
// - BlockId chain

// Ways to connect a block
// case: extend main chain
// 0
// 0--A

// case: extend a candidate tip
// 0--A--B
// 0--A--Ba

// case: switch forks (reorg)
// 0--1--A
// 0--1--Aa--Ba

// case: add orphan block (no parent)
// 0--A--B
//      -Bh

// case: insert block into main chain
// 2 operations: connect-block (A->0), add-dependency (B->A)
// 0-----B
// 0--A--B

// How to manage tips
// HashSet<BlockHash>
// when a block is connected
//   remove its parent from tips
//   add this block to tips
// reevaluate valid tips to find the best one

// Can we optimize for faster indexing?
// reintroduce canonical index as a contiguous array of Arc<T>
// periodically do work to reconstruct the index
// actions that trigger a reindex?
// - connecting blocks out of order

// Definition of "best" chain
// The best chain is the longest by block height. The reason not to use Work is because
// in a sparse chain the accumulated work may be artificially low, so going by work alone
// could cause us to wrongfully switch to a shorter but heavier chain.

// If two tips are tied for longest chain, then we tie-break by blockhash
// (rationale: smaller hash -> more work)
// if we guess wrong, we'll just resume on whichever chain is extended

impl<T: ToBlockId + Clone> BlockGraph<T> {
    /// Construct from genesis block
    ///
    /// Panics if the height of the genesis block is not 0.
    pub fn from_genesis(block: T) -> Self {
        let genesis_block = block.block_id();
        assert_eq!(genesis_block.height, 0, "genesis block must be root of graph");

        let genesis_hash = genesis_block.hash;
        let conn = BlockId {
            height: 0,
            hash: BlockHash::all_zeros(),
        };

        let mut blocks = HashMap::new();
        let mut next_hashes = HashMap::new();
        let mut tips = HashSet::new();

        // populate blocks
        let node = Node::new(block, conn);
        blocks.insert(genesis_hash, node);
        // populate edges
        next_hashes.entry(genesis_hash).or_default();
        // store the root hash
        let root = genesis_hash;
        // set tip
        let tip = root;
        tips.insert(tip);

        Self {
            blocks,
            next_hashes,
            root,
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

        // Avoid connecting a node to itself or introducing an invalid dependency
        if block_id == conn || conn.height > block_id.height {
            return changeset;
        }
        assert!(block_id.height > conn.height);

        // An entry exists for this hash, add the edge and return
        if self.blocks.contains_key(&hash) {
            return self.add_dependency(hash, conn).expect("node must exist");
        }

        // Add this node, and record the fact that this block extends from the previous
        self.blocks.insert(hash, Node::new(block.clone(), conn));
        self.next_hashes.entry(conn.hash).or_default().insert(hash);
        changeset.blocks.push((block, conn));

        // Remove the parent from tips if present
        let tip_extended = self.tips.remove(&conn.hash);
        // If nothing extends from this block, considering adding it as a tip
        if !self.next_hashes.contains_key(&hash) {
            // If the block extends a previous tip, it necessarily becomes the new tip
            // or if the tip forms a valid chain it may be a fork contender.
            if tip_extended || self.is_valid_chain(&hash) {
                println!("Inserting tip {} {}", block_id.height, block_id.hash);
                assert!(self.tips.insert(hash));
            }
        }

        // If the block extends the current tip, update the tip and return
        if conn.hash == self.tip {
            self.tip = hash;
            return changeset;
        }

        // Reevaluate tips to find the best one
        self.compare_tips();

        changeset
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

    /// Construct from [`ChangeSet`].
    ///
    /// Errors if changeset is empty, or does not contain a genesis block.
    pub fn from_changeset(changeset: ChangeSet<T>) -> Result<Self, MissingGenesisError>
    where
        T: Ord,
    {
        let blocks: BTreeSet<(T, BlockId)> = changeset.blocks.into_iter().collect();

        let (genesis, _) = blocks.first().cloned().ok_or(MissingGenesisError)?;

        if genesis.block_id().height != 0 {
            return Err(MissingGenesisError);
        }

        let mut graph = BlockGraph::from_genesis(genesis);

        for (block, conn) in blocks {
            let _ = graph.connect_block(block, conn);
        }

        Ok(graph)
    }

    /// Get the genesis node
    pub fn genesis_node(&self) -> &Node<T> {
        self.blocks.get(&self.root).expect("must have node")
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

    /// Iterate elements of `&T` in descending height order from the tip of the active chain.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.iter_blocks(&self.tip)
    }

    /// Iterate elements of `&T` in descending height order from the chain defined by
    /// the given `hash`.
    ///
    /// This method returns an iterator that visits nodes of the graph beginning at
    /// `hash` and each node of the parent chain which it extends.
    ///
    /// Note that the returned block items may or may not belong to the best chain.
    pub fn iter_blocks(&self, hash: &BlockHash) -> impl Iterator<Item = &T> {
        let mut cur = *hash;
        core::iter::from_fn(move || {
            let node = self.blocks.get(&cur)?;
            let item = &node.data;
            cur = node.connected_at().hash;
            Some(item)
        })
    }

    /// Reindex the block graph. This should be done whenever it's uncertain what the best chain is.
    ///
    /// If a new tip is found it returns the old tip, otherwise None.
    pub fn reindex(&mut self) -> Option<BlockHash> {
        let mut valid_tips = HashSet::new();

        // Find the possible tips. A tip is any node from which no edge extends.
        let possible_tips = self.blocks.keys().filter(|&hash| !self.next_hashes.contains_key(hash));

        // Keep only the ones that make up a valid chain.
        for &tip_hash in possible_tips {
            if self.is_valid_chain(&tip_hash) {
                println!("Valid chain {}", tip_hash);
                valid_tips.insert(tip_hash);
            }
        }

        self.tips = valid_tips;

        // Of the valid tips pick the best one.
        self.compare_tips()
    }

    /// Whether the chain of the given `hash` constitutes a valid chain.
    ///
    /// The possible chain with tip `hash` is a valid chain if it connects to genesis
    /// either directly or indirectly through one of its parents.
    fn is_valid_chain(&self, hash: &BlockHash) -> bool {
        let genesis_block = self.genesis_node().block_id();
        let chain_tip = self.tip();

        let mut visited = HashSet::new();
        let mut to_validate = vec![*hash];

        while let Some(hash) = to_validate.pop() {
            if let Some(node) = self.blocks.get(&hash) {
                // If a parent is found to be in the best chain, we can deduce
                // that this is a valid chain.
                if let Some(true) = self.is_block_in_chain(node.block_id(), chain_tip) {
                    return true;
                }
                // If validity can't be determined from the tip then we have to do an
                // exhaustive search to see if it contains the genesis block.
                if self
                    .iter_blocks(&node.hash())
                    .any(|item| item.block_id() == genesis_block)
                {
                    return true;
                }
                let connections =
                    node.connections().map(|b| b.hash).filter(|&hash| visited.insert(hash));
                to_validate.extend(connections);
            }
        }

        false
    }

    /// Compare all tips of `self` and set the new best hash, returning the old tip if it changed.
    fn compare_tips(&mut self) -> Option<BlockHash> {
        let best_block = self
            .tips
            .iter()
            .flat_map(|hash| Some(self.blocks.get(hash)?.block_id()))
            .max_by(|a, b| {
                // Compare by height
                a.height
                    .cmp(&b.height)
                    // Tie-break by hash, smaller is better since it implies more work.
                    .then_with(|| match a.hash.cmp(&b.hash) {
                        cmp::Ordering::Less => cmp::Ordering::Greater,
                        cmp::Ordering::Greater => cmp::Ordering::Less,
                        _ => unreachable!("must not have duplicate tips"),
                    })
            })?;

        // Update to the best tip if needed
        if self.tip() == best_block {
            println!("Compare tips resulted in no change");
            None
        } else {
            println!("Setting new tip {} {}", best_block.height, best_block.hash);
            let old_tip = self.tip;
            self.tip = best_block.hash;
            Some(old_tip)
        }
    }
}

/// Error caused by lack of a genesis block
#[derive(Debug)]
pub struct MissingGenesisError;

impl fmt::Display for MissingGenesisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "block graph must have genesis node")
    }
}

impl std::error::Error for MissingGenesisError {}

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

impl<T> ChangeSet<T> {
    /// Merge
    pub fn merge(&mut self, other: Self) {
        self.blocks.extend(other.blocks);
    }
    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
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
            merkle_root: hashes::Hash::hash(b"merkle-root"),
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
    }

    impl ToBlockId for SimpleHeader {
        fn block_id(&self) -> BlockId {
            BlockId {
                height: self.height,
                hash: self.hash,
            }
        }
    }

    impl SimpleHeader {
        fn genesis() -> Self {
            Self {
                height: 0,
                hash: bitcoin::constants::genesis_block(Network::Regtest).block_hash(),
                prev_hash: BlockHash::all_zeros(),
            }
        }

        fn new(height: u32, header: Header) -> Self {
            Self {
                height,
                hash: header.block_hash(),
                prev_hash: header.prev_blockhash,
            }
        }
    }

    #[test]
    fn header_chain() {
        let mut prev_hash = BlockHash::all_zeros();

        let mut headers = vec![];

        for height in 0u32..5 {
            let hash: BlockHash = bitcoin::hashes::Hash::hash(height.to_be_bytes().as_slice());
            let header = SimpleHeader {
                height,
                hash,
                prev_hash,
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
    fn graph_should_handle_connecting_same_block() {
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

        let mut headers: Vec<SimpleHeader> = vec![];

        for height in 0u32..5 {
            let hash: BlockHash = hashes::Hash::hash(height.to_be_bytes().as_slice());
            let header = SimpleHeader {
                height,
                hash,
                prev_hash,
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

        // try reindex with debug prints
        graph.reindex();
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

    #[test]
    fn test_reindex() {
        // case: test tiebreak for best chain by block hash
        let blocks: Vec<BlockId> = [
            (0, "b90ca5b5653eabdc3341c6f96b3b80689cdd1bd6870265adfe17c8172501b98c"),
            (1, "73654ed17dba11dfe673300a31f9b4602ba735f4cce897ad9d719c965184b8fc"),
            (2, "b4f6ae6edf5a32fb6d9322563bde4b760472849556e78cac335ba711f33c173d"),
            (3, "020ee6a7037876579694f33ae98a6d2936213a557c0d8adae77a30ed6f4f0687"),
            (4, "5f19f2d6af102537e6db3d4e87dc4b8ebc4770c984d1f924e62450727f73bd89"),
        ]
        .into_iter()
        .map(|(height, s)| {
            let hash = s.parse().unwrap();
            BlockId { height, hash }
        })
        .collect();

        let genesis = blocks[0];
        let mut graph = BlockGraph::from_genesis(genesis);

        for i in (0..3).skip(1) {
            let block = blocks[i];
            let _ = graph.connect_block(block, graph.tip());
        }

        assert_eq!(graph.blocks.len(), 3);
        assert_eq!(graph.tip().height, 2);

        // now introduce a fork contender block_2a with a "smaller" hash
        let block_1 = graph.get(1).unwrap();
        let block_2 = graph.get(2).unwrap();
        let mut block_2a = block_2;
        block_2a.hash = "a4f6ae6edf5a32fb6d9322563bde4b760472849556e78cac335ba711f33c173d"
            .parse()
            .unwrap();
        assert!(block_2a.hash < block_2.hash);

        let cs = graph.connect_block(block_2a, block_1);
        assert!(!cs.is_empty(), "block_2a should connect");

        // reindex
        graph.tip = genesis.hash;
        graph.tips.clear();

        graph.reindex();

        assert_eq!(graph.tip(), block_2a);

        // case: test remove valid tips and recover them
        let mut graph = BlockGraph::from_genesis(genesis);
        for i in (0..5).skip(1) {
            let block = blocks[i];
            let _ = graph.connect_block(block, graph.tip());
        }

        assert_eq!(graph.tip().height, 4);
        let exp_tip = graph.tip();

        // reindex
        graph.tip = genesis.hash;
        graph.tips.clear();

        graph.reindex();

        assert_eq!(graph.tip(), exp_tip);
    }

    #[test]
    fn test_is_valid_chain() {
        // create a graph with 3 distinct but valid chains
        let root = "5f19f2d6af102537e6db3d4e87dc4b8ebc4770c984d1f924e62450727f73bd89"
            .parse()
            .unwrap();
        let genesis_block = BlockId {
            height: 0,
            hash: root,
        };
        let heights = [1, 2, 3];
        let hashes: Vec<BlockHash> = vec![
            "4d08b5fa27745754a02dbc3cf4cb348223bc543d7f75e36ffbcfb9dda9919715"
                .parse()
                .unwrap(),
            "2b97e32191a25284730c814ca77f3055df9670301b1fcc1e5ddb4c4efea7087a"
                .parse()
                .unwrap(),
            "1e64eb10b47460346bebac7d8a23db82f470fd53c3f61a6daa4b2e5df52e6508"
                .parse()
                .unwrap(),
        ];
        let mut graph = BlockGraph::from_genesis(genesis_block);
        for (&height, &hash) in heights.iter().zip(&hashes) {
            let block = BlockId { height, hash };
            let _ = graph.connect_block(block, graph.tip());
        }
        assert_eq!(graph.tip().height, 3);

        // insert blocks of a new chain also with tip height 3
        let hashes: Vec<BlockHash> = vec![
            "5b05130a385e58e6192af1c5abdf79abe782ae784783476336483cfa67c1aaf3"
                .parse()
                .unwrap(),
            "1c3b0be0aba92d5345faec0c07b70a0261bbb1a6532eee76ea81d32483bfa7e4"
                .parse()
                .unwrap(),
            "345241109d2e0b3be9e61ef017bfb9ca939e53994129e91f58a2e0d367578c77"
                .parse()
                .unwrap(),
        ];
        for (&height, &hash) in heights.iter().zip(&hashes) {
            let block = BlockId { height, hash };
            let par = if height == 1 {
                genesis_block
            } else {
                BlockId {
                    height: height - 1,
                    hash: hashes[(height - 2) as usize],
                }
            };
            let _ = graph.connect_block(block, par);
        }
        assert_eq!(graph.tip().height, 3);
        assert_eq!(graph.tips.len(), 2);
        assert_eq!(graph.blocks.len(), 7);

        // repeat again for a total of 10 blocks
        let hashes: Vec<BlockHash> = vec![
            "055445553b85fca3ee9632fa0a9e90d582789fdcb88ef9486d8ed29b2d3302ca"
                .parse()
                .unwrap(),
            "78a18b4420f09c8e1aa312f7d40721c579d3b8c28da12ded5a6c00f249915f07"
                .parse()
                .unwrap(),
            "162b270c510de742033b83ab5bc50e509f2064ae24f74d9845827438f1096ed9"
                .parse()
                .unwrap(),
        ];
        for (&height, &hash) in heights.iter().zip(&hashes) {
            let block = BlockId { height, hash };
            let par = if height == 1 {
                genesis_block
            } else {
                BlockId {
                    height: height - 1,
                    hash: hashes[(height - 2) as usize],
                }
            };
            let _ = graph.connect_block(block, par);
        }
        assert_eq!(graph.blocks.len(), 10);

        // we should have three unique tips
        assert_eq!(graph.tips.len(), 3);
        // height of tip is unchanged
        assert_eq!(graph.tip().height, 3);
        // smallest of the tips from above
        assert_eq!(
            graph.tip.to_string(),
            "1e64eb10b47460346bebac7d8a23db82f470fd53c3f61a6daa4b2e5df52e6508"
        );

        // now extend the best chain
        let hash = "230972c4e5148d7b6ca2f2be745896e89ca5e9b66b876bf519e6030fb9cc0e36"
            .parse()
            .unwrap();
        let block_4 = BlockId { height: 4, hash };
        let _ = graph.connect_block(block_4, graph.tip());
        assert_eq!(graph.tip().height, 4);

        // reindex
        graph.tip = genesis_block.hash;
        graph.tips.clear();

        // should recover original tips
        graph.reindex();

        assert_eq!(graph.tips.len(), 3);
        assert_eq!(graph.tip(), block_4);
    }
}
