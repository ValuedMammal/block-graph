default:
    @just --list

alias b := build
alias c := check
alias f := fmt
alias t := test

check:
    cargo check --all-targets --all-features
    cargo +nightly fmt --all -- --check
    cargo clippy --all-targets --all-features -- -Dwarnings

fmt:
    cargo +nightly fmt

build:
    cargo build

test:
    cargo test --all-targets --all-features
