#!/usr/bin/env sh
# Builds both SBF programs and runs the Mollusk CU comparison.
# `run.sh sizes` instead builds one .so per codec and prints section sizes.
set -e
cd "$(dirname "$0")"

if [ "$1" = "sizes" ]; then
    cd program
    for v in none tape five8 bs58; do
        if [ "$v" = none ]; then
            cargo-build-sbf -- --no-default-features
        else
            cargo-build-sbf -- --no-default-features --features "$v"
        fi
        cp target/deploy/cu_bench.so "target/deploy/size_$v.so"
    done
    ls -l target/deploy/size_*.so
    exit 0
fi

(cd program && cargo-build-sbf)
(cd pda-program && cargo-build-sbf)
cp pda-program/target/deploy/pda_bench.so program/target/deploy/
(cd runner && cargo build --release)
SBF_OUT_DIR="$PWD/program/target/deploy" RUST_LOG=off runner/target/release/cu-runner
