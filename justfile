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
    cargo test --workspace --all-features -- --test-threads=1
