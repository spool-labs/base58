//! tape-base58 against five8 and bs58, per dispatched path
//!
//! The tape rows pin each path in turn, since it resolves at run time. five8
//! picks its instruction set at compile time instead, so the five8 rows are
//! whatever that crate's `cfg(target_feature)` selected for this build, and
//! the comparison has to be run twice to mean anything:
//!
//! ```text
//! cargo bench --bench against                        # as each ships
//! RUSTFLAGS="-C target-cpu=native" cargo bench --bench against
//! ```
//!
//! The first is the deployment question — a validator building with no target
//! flags gets tape's AVX-512 and five8's scalar path, and that gap is a real
//! property of runtime dispatch rather than an artifact. The second is the
//! algorithm question, where both reach their vector paths. Report both:
//! quoting only the first overstates us, quoting only the second hides an
//! advantage a real deployment gets.
//!
//! Under `target-cpu=native` LLVM also auto-vectorises tape's portable arm, so
//! the `tape_portable` rows stop meaning portable. Do not mix baselines.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tape_base58::{MAX_ENCODED_32, MAX_ENCODED_64};

/// What a tape path is called, and the marker that pins it
struct Path {
    name: &'static str,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    marker: Option<u8>,
}

#[cfg(target_arch = "x86_64")]
fn paths() -> Vec<Path> {
    use tape_base58::testing::{available, AVX2, PORTABLE, WIDE};
    let widest = available();
    let mut runnable = vec![Path {
        name: "tape_portable",
        marker: Some(PORTABLE),
    }];
    if widest == AVX2 || widest == WIDE {
        runnable.push(Path {
            name: "tape_avx2",
            marker: Some(AVX2),
        });
    }
    if widest == WIDE {
        runnable.push(Path {
            name: "tape_avx512",
            marker: Some(WIDE),
        });
    }
    runnable
}

#[cfg(not(target_arch = "x86_64"))]
fn paths() -> Vec<Path> {
    vec![Path {
        name: "tape_native",
        marker: None,
    }]
}

#[cfg(target_arch = "x86_64")]
fn pin(path: &Path) {
    if let Some(marker) = path.marker {
        tape_base58::testing::force(marker);
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn pin(_path: &Path) {}

const KEY: [u8; 32] = [
    24, 243, 6, 223, 230, 153, 210, 8, 92, 137, 123, 67, 164, 197, 79, 196, 125, 43, 183, 85, 103,
    91, 232, 167, 73, 131, 104, 131, 0, 101, 214, 231,
];

fn signature() -> [u8; 64] {
    let mut bytes = [0u8; 64];
    for (at, slot) in bytes.iter_mut().enumerate() {
        *slot = (at as u8).wrapping_mul(53).wrapping_add(7) | 1;
    }
    bytes
}

fn noise(seed: u64, into: &mut [u8]) {
    let mut state = seed | 1;
    for slot in into.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *slot = (state >> 24) as u8;
    }
}

fn fixed(criterion: &mut Criterion) {
    let signature = signature();
    let mut key_out = [0u8; MAX_ENCODED_32];
    let key_len = tape_base58::encode_32(&KEY, &mut key_out);
    let key_text = key_out[..key_len].to_vec();
    let mut signature_out = [0u8; MAX_ENCODED_64];
    let signature_len = tape_base58::encode_64(&signature, &mut signature_out);
    let signature_text = signature_out[..signature_len].to_vec();

    let mut key_bytes = [0u8; 32];
    let mut signature_bytes = [0u8; 64];

    let mut group = criterion.benchmark_group("against");

    for path in paths() {
        group.bench_function(BenchmarkId::new("encode_32", path.name), |bencher| {
            pin(&path);
            bencher.iter(|| tape_base58::encode_32(black_box(&KEY), black_box(&mut key_out)))
        });
        group.bench_function(BenchmarkId::new("encode_64", path.name), |bencher| {
            pin(&path);
            bencher.iter(|| {
                tape_base58::encode_64(black_box(&signature), black_box(&mut signature_out))
            })
        });
        group.bench_function(BenchmarkId::new("decode_32", path.name), |bencher| {
            pin(&path);
            bencher.iter(|| tape_base58::decode_32(black_box(&key_text), black_box(&mut key_bytes)))
        });
        group.bench_function(BenchmarkId::new("decode_64", path.name), |bencher| {
            pin(&path);
            bencher.iter(|| {
                tape_base58::decode_64(black_box(&signature_text), black_box(&mut signature_bytes))
            })
        });
    }

    let mut five8_key = [0u8; five8::BASE58_ENCODED_32_MAX_LEN];
    let mut five8_signature = [0u8; five8::BASE58_ENCODED_64_MAX_LEN];
    group.bench_function(BenchmarkId::new("encode_32", "five8"), |bencher| {
        bencher.iter(|| five8::encode_32(black_box(&KEY), black_box(&mut five8_key)))
    });
    group.bench_function(BenchmarkId::new("encode_64", "five8"), |bencher| {
        bencher.iter(|| five8::encode_64(black_box(&signature), black_box(&mut five8_signature)))
    });
    group.bench_function(BenchmarkId::new("decode_32", "five8"), |bencher| {
        bencher.iter(|| five8::decode_32(black_box(&key_text), black_box(&mut key_bytes)))
    });
    group.bench_function(BenchmarkId::new("decode_64", "five8"), |bencher| {
        bencher
            .iter(|| five8::decode_64(black_box(&signature_text), black_box(&mut signature_bytes)))
    });

    group.bench_function(BenchmarkId::new("encode_32", "bs58"), |bencher| {
        bencher.iter(|| bs58::encode(black_box(&KEY)).into_vec())
    });
    group.bench_function(BenchmarkId::new("encode_64", "bs58"), |bencher| {
        bencher.iter(|| bs58::encode(black_box(&signature)).into_vec())
    });
    group.bench_function(BenchmarkId::new("decode_32", "bs58"), |bencher| {
        bencher.iter(|| bs58::decode(black_box(&key_text)).into_vec())
    });
    group.bench_function(BenchmarkId::new("decode_64", "bs58"), |bencher| {
        bencher.iter(|| bs58::decode(black_box(&signature_text)).into_vec())
    });

    group.finish();

    #[cfg(target_arch = "x86_64")]
    tape_base58::testing::force(tape_base58::testing::UNKNOWN);
}

/// Widths the any-length path sees, the last being a Solana packet
const SIZES: [usize; 4] = [32, 128, 512, 1232];

fn variable(criterion: &mut Criterion) {
    // five8 has no any-length API, so this half is against bs58 alone.
    let mut group = criterion.benchmark_group("against_variable");
    for size in SIZES {
        let mut input = vec![0u8; size];
        noise(size as u64 + 1, &mut input);
        let mut out = vec![0u8; tape_base58::encoded_len(size)];
        let len = tape_base58::encode(&input, &mut out).expect("encode");
        let text = out[..len].to_vec();
        let mut back = vec![0u8; tape_base58::decoded_len(text.len())];

        group.bench_function(BenchmarkId::new("encode_tape", size), |bencher| {
            bencher.iter(|| tape_base58::encode(black_box(&input), black_box(&mut out)))
        });
        group.bench_function(BenchmarkId::new("encode_bs58", size), |bencher| {
            bencher.iter(|| bs58::encode(black_box(&input)).into_vec())
        });
        group.bench_function(BenchmarkId::new("decode_tape", size), |bencher| {
            bencher.iter(|| tape_base58::decode(black_box(&text), black_box(&mut back)))
        });
        group.bench_function(BenchmarkId::new("decode_bs58", size), |bencher| {
            bencher.iter(|| bs58::decode(black_box(&text)).into_vec())
        });
    }
    group.finish();
}

criterion_group!(benches, fixed, variable);
criterion_main!(benches);
