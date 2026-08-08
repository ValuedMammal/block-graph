# `block-graph-fuzz`

`cargo-fuzz` harnesses for [`block_graph`](../) to find logic bugs: panics, hangs, structural corruption, and violations of the library's stated design goals.

## Running

`cargo-fuzz` requires a nightly toolchain. Use the `just` recipes from the repo root:

```sh
just fuzz-build                  # build all four targets
just fuzz apply_update 60        # run one target for 60s
just check-fuzzing               # clippy the main crate's `--cfg fuzzing` path
```

Building a target with plain `cargo build` inside `fuzz/` (rather than `cargo fuzz`) requires setting `RUSTFLAGS="--cfg fuzzing"` yourself, or the harness won't compile. On Apple Silicon, if ASan is uncooperative, `-s none` loses nothing of value here — there's no `unsafe` in the crate, and libFuzzer's `-rss_limit_mb` still guards OOM independently.

Targets: `connect_block`, `apply_update`, `from_changeset`, `checkpoint_ops`.

Crash artifacts land in `fuzz/artifacts/<target>/`. Reproduce with:

```sh
cd fuzz && cargo +nightly fuzz run <target> artifacts/<target>/<crash-file>
```

Every assertion in these targets carries a `Debug`-formatted scenario in its panic message (see
`block_graph_fuzz::assert_ok`), so an artifact tells you *which* invariant broke and *what*
operation sequence caused it without re-running anything.

## How the fuzzers see the graph

**We never feed raw bytes as block data.** The graph is keyed by `BlockHash`; a random 32-byte
parent hash never matches an existing block, so a naive byte-oriented fuzzer would burn 100% of its
budget on early-return paths. Instead, [`src/lib.rs`](src/lib.rs) generates *operation sequences*
over blocks the graph already knows about:

- `hash_from_id(u8)` maps a small id to a distinct, never-zero `BlockHash`. Never zero because
  `BlockHash::all_zeros()` is the crate's sentinel predecessor of genesis; only 256 ids so that
  hash collisions and near-ties are *reachable*, which is what exercises the
  `max_by_key(|id| (id.height, Reverse(id.hash)))` tie-break in `canonicalize`.
- `T = BlockHash` for every target. `Block::to_blockhash` for `BlockHash` is the identity, which
  skips a double-SHA256 per block (a large exec/s win) and lets the fuzzer choose hash *ordering*
  directly. Real `Header`s would only be needed to exercise the `prev_blockhash`-validation code
  paths (`Block::prev_blockhash` returns `Some` for `Header`, `None` for `BlockHash`), which aren't
  covered yet.
- `Op` / `Update` / `Scenario` index into a running `Vec` of known hashes **modulo its length**, so
  a generated parent is always real. `Op::ConnectLoose` and `Update::base_unknown` deliberately
  escape that, to exercise the unknown-parent and `MissingParent` paths.
- A `height_delta` of `0` is allowed on purpose (it must produce `ParentHeightNotSmaller`), and
  values `> 1` produce *gapped* connections, which are legal and are their own code path.

Collection sizes are capped in each target (ops at 256, updates at 16, blocks at 64) so exec time
stays bounded and the fuzzer doesn't waste effort exploring size.

## The invariants

Two checkers live *inside* the main crate — the harness is an external crate and can't see private
fields. They're gated on `#[cfg(any(test, fuzzing))]`: `cargo fuzz` injects `--cfg fuzzing` into
the whole build graph automatically, and the `test` half means the crate's own unit tests can call
them (and `just check` type-checks and lints them) with no flags set. They return
`Result<(), String>` rather than panicking, so each target decides how to report.

### `check_invariants()` — structural, must hold at **all** times

Asserted after every successful operation in every graph target.

1. **Root is genesis.** `self.root` exists in `blocks` at height 0.
2. **Values are self-consistent.** For every `(hash, (_, value))` in `blocks`,
   `value.to_blockhash() == hash`. Catches a block stored under the wrong key.
3. **Edges point at real blocks, and point forward.** For every edge `(parent, child)` in
   `next_hashes`: `child` exists in `blocks`, and if `parent` exists in `blocks` then
   `height(parent) < height(child)`. `parent` may legitimately be absent — that's the
   `BlockHash::all_zeros()` sentinel below genesis, and the gapped/forward-referenced connection
   case. Strict height monotonicity on every edge is also what makes cycles unrepresentable.
4. **`parents` is exactly the inverse of `next_hashes`, restricted to known parents.** Checked in
   both directions. `parents` drives `parent()`, `iter_block_graph()`, and `canonicalize`'s
   backward walk, so one-directional drift here reconstructs the wrong ancestry with no other
   symptom.
5. **The tip chain is a real chain.** Walking `self.tip` tip-to-genesis: heights strictly decrease,
   every checkpoint's `hash()` equals `value().to_blockhash()`, every `(height, hash)` matches an
   entry in `blocks`, and the last item is at height 0 with hash `self.root`.

### `check_best_tip()` — the longest-chain rule

BFS every block reachable from `self.root` via `next_hashes`, then assert `self.tip.block_id()` is
the maximum by key `(height, Reverse(hash))` — the same key `canonicalize` uses (longest chain by
height, ties broken toward the smaller hash as a proxy for "more work"). Comparing over *all*
reachable blocks rather than just leaves is safe precisely because of invariant 3: a parent is
always strictly lower than its child, so no interior node can be at the maximum height.

**This is asserted only in `apply_update` and `from_changeset`, never in `connect_block`.**
`connect_block` intentionally does not move the tip; something else reconciles it later. Asserting
it there would be asserting a rule the API doesn't claim to follow.

### Cross-cutting properties asserted by the targets

Beyond the two checkers, each target asserts the behavioral contracts the crate documents:

- **Atomicity** — on `Err`, the graph must equal a clone taken before the call. This is the
  validate-then-mutate contract.
- **Idempotence** — repeating an identical successful `connect_block`/`apply_update` returns an
  empty `ChangeSet` and leaves the graph bit-identical.
- **Changeset roundtrip** — `from_changeset(graph.initial_changeset())` reconstructs an *equal*
  graph. This is the strongest single assertion in the suite: it pits the incremental path against
  the from-scratch rebuild path, so any ancestry the incremental path fabricated shows up as a
  mismatch.
- **Error-kind prediction** (`connect_block`) — the target independently predicts the exact error
  *kind* for every connect attempt, mirroring the library's validation order. Disagreement in
  either direction is a bug: either the library accepted something it shouldn't, or the documented
  rules aren't what's actually implemented.
- **Differential model** (`checkpoint_ops`) — `CheckPoint`'s skip-pointer code (`get_skip_index`,
  `checkpoint_at_index`, `walk_to_floor`) has nested `saturating_sub`s and negated compound
  conditions. `get`/`range`/`len`/`index` are compared against a naive `Vec<(u32, BlockHash)>` scan
  after every mutation, and `insert` semantics (identical value => `eq_ptr`, unchanged; conflicting
  value => truncate everything above; gap => splice without disturbing neighbours) are mirrored in
  the model.

## The targets

| Target | Input | Focus |
|---|---|---|
| `connect_block` | `Vec<Op>` | Incremental API: per-op error-kind prediction, atomicity, idempotence, `check_invariants` |
| `apply_update` | `Scenario` (ops + updates) | Reorg machinery: `check_best_tip`, roundtrip, `is_block_in_chain` agreement |
| `from_changeset` | raw `(id, height)` blocks + `(parent, child)` edges | Corrupt/adversarial persistence: must never panic or hang |
| `checkpoint_ops` | `Vec<CpOp>` | Differential test of `CheckPoint`'s skip-pointer arithmetic |

Notes on the two that need the most care:

**`from_changeset`** is the highest-yield target and should stay that way: `ChangeSet` is the
*persistence format*, so it consumes data that may be corrupt or attacker-influenced. The target
asserts the call returns `Ok(Some(_))`, `Ok(None)`, or one of the defined `FromChangeSetError`
variants — and **never panics, never hangs**. Each error variant is also checked for internal
consistency (e.g. `InvalidEdgeHeight` is only raised for a genuinely non-increasing edge). Shapes
that must stay reachable: two distinct blocks at height 0, an edge whose child is lower than its
parent, a cycle, an edge referencing a block absent from `blocks`, and a block with no incoming
edge.

**`apply_update`** additionally asserts `is_block_in_chain` agreement: every checkpoint on the tip
chain reports `Some(true)`; every other known block at or below tip height reports `Some(false)`
*if the canonical chain actually has an entry at that height*, and `None` otherwise (a gapped chain
has nothing at that height to compare against); anything above tip height reports `None`.

## Findings

When a target finds something, the workflow is: minimize, add a regression test to the main crate's
`test` module, fix, then append an entry to `fuzz-findings.md`. Keep the invariant list
above and the checkers in [`src/block_graph.rs`](../src/block_graph.rs) in sync — if a new
invariant is added, document *why* it holds here, not just that it's checked.
