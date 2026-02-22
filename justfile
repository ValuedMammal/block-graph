_default:
    @just --list --unsorted

alias b := build
alias f := fmt
alias c := check
alias t := test

# Build workspace
build:
    cargo build --workspace

# Rustfmt
fmt:
    cargo +nightly fmt

# Check workspace, rustfmt, and clippy
check:
    cargo +nightly fmt --all -- --check
    cargo check --workspace --all-targets --all-features
    cargo clippy --all-targets --all-features -- -Dwarnings

# Run a block-graph unit test of a given name
test-name name="":
    cargo test -p block_graph --no-fail-fast --all-features --lib -- block_graph::test::{{name}} --exact --show-output

# Run block-graph unit tests
test:
    cargo test -p block_graph --no-fail-fast --all-features --lib -- checkpoint::test
    cargo test -p block_graph --no-fail-fast --all-features --lib -- block_graph::test

# Run block-graph benchmarks
bench:
    cargo bench -p block_graph --bench checkpoint
    cargo bench -p block_graph --bench from_changeset
    cargo bench -p block_graph --bench is_block_in_chain
    cargo bench -p block_graph --bench apply_update
    cargo bench -p block_graph --bench reorg
