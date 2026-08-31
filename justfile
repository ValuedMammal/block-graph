# Default recipe
_default:
    @just --list --unsorted

# ======================== #
# Aliases                  #
# ======================== #

alias b := build
alias f := fmt
alias c := check
alias d := doc
alias m := msrv
alias p := pre-push
alias t := test
alias tn := test-name

# ======================== #
# Toolchains               #
# ======================== #

# Nightly toolchain
nightly := 'nightly-2026-08-01'

# Stable toolchain
stable := '1.98.0'

# MSRV toolchain
msrv := '1.85.0'

# ======================== #
# Recipes                  #
# ======================== #

# Build workspace
build:
    cargo +{{stable}} build --workspace --all-targets --all-features

# Check MSRV
msrv:
    cargo +{{msrv}} build --lib --tests --examples --no-default-features
    cargo +{{msrv}} build --lib --tests --examples --all-features

# Rustfmt
fmt:
    cargo +{{nightly}} fmt

# Build and check docs
doc:
    RUSTDOCFLAGS='-Dwarnings' cargo +{{stable}} doc --workspace --all-features --no-deps

# Check formatting, compilation, linting
check:
    @echo $(git rev-list -1 HEAD)
    cargo +{{nightly}} fmt --all -- --check
    cargo +{{stable}} check --no-default-features
    cargo +{{stable}} check --all-targets --all-features
    cargo +{{stable}} clippy --all-targets --all-features -- -Dwarnings

# Run a block-graph unit test of a given name
test-name name="":
    cargo +{{stable}} test -p block_graph --no-fail-fast --all-features --lib -- {{name}} --show-output

# Run all tests
test:
    cargo +{{stable}} test -p block_graph --no-fail-fast --all-features

# Pre-commit push checks (build, test, lint)
pre-push: msrv check build test doc

# Run block-graph benchmarks
bench:
    cargo +{{stable}} bench -p block_graph --bench from_changeset
    cargo +{{stable}} bench -p block_graph --bench is_block_in_chain
    cargo +{{stable}} bench -p block_graph --bench apply_update
    cargo +{{stable}} bench -p block_graph --bench reorg

# Build all fuzz targets
fuzz-build:
    cd fuzz && cargo +{{nightly}} fmt && cargo +{{nightly}} fuzz build

# Run a fuzz target, e.g. `just fuzz apply_update 60`
fuzz target="from_changeset" secs="30":
    cd fuzz && cargo +{{nightly}} fuzz run {{target}} -- -max_total_time={{secs}} -max_len=4096 -rss_limit_mb=4096 -timeout=10

# Coverage-minimize a target's local corpus
fuzz-cmin target="from_changeset":
    cd fuzz && cargo +{{nightly}} fuzz cmin {{target}} -- -rss_limit_mb=4096 -timeout=10

# Lint the `fuzzing` cfg path of the main crate
check-fuzzing:
    RUSTFLAGS="--cfg fuzzing" cargo +{{stable}} clippy --all-targets --all-features -- -Dwarnings
