default:
    @just --list

alias b := build
alias c := check
alias f := fmt
alias t := test

check:
    cargo check --workspace --all-targets --all-features
    cargo +nightly fmt --all -- --check
    cargo clippy --all-targets --all-features -- -Dwarnings

fmt:
    cargo +nightly fmt

build:
    cargo build --workspace

test:
    cargo test -p block_graph --no-fail-fast --all-features --lib -- checkpoint::test
    cargo test -p block_graph --no-fail-fast --all-features --lib -- block_graph::test

bench:
    cargo bench -p block_graph --bench block_graph_checkpoint
    cargo bench -p block_graph --bench from_changeset
    cargo bench -p block_graph --bench canonical_view
