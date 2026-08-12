_default:
    @just --list --unsorted

alias b := build
alias f := fmt
alias c := check
alias t := test
alias tn := test-name

# Build workspace
build:
    cargo build --workspace --all-targets --all-features

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

# Run all unit tests
test:
    cargo test -p block_graph --no-fail-fast --all-features --lib --

# Run block-graph benchmarks
bench:
    cargo bench -p block_graph --bench from_changeset
    cargo bench -p block_graph --bench is_block_in_chain
    cargo bench -p block_graph --bench apply_update
    cargo bench -p block_graph --bench reorg

# Build all fuzz targets
fuzz-build:
    cd fuzz && cargo +nightly fuzz build

# Run a fuzz target, e.g. `just fuzz apply_update 60`
fuzz target="from_changeset" secs="30":
    cd fuzz && cargo +nightly fuzz run {{target}} -- -max_total_time={{secs}} -max_len=4096 -rss_limit_mb=4096 -timeout=10

# Coverage-minimize a target's local corpus
fuzz-cmin target="from_changeset":
    cd fuzz && cargo +nightly fuzz cmin {{target}} -- -rss_limit_mb=4096 -timeout=10

# Lint the `fuzzing` cfg path of the main crate
check-fuzzing:
    RUSTFLAGS="--cfg fuzzing" cargo clippy --all-targets --all-features -- -Dwarnings
