//! Times every path the running machine can execute, side by side
//!
//! The codec picks the widest instruction set at first use, so an ordinary
//! run only ever times one path. These pin each in turn, which is what makes
//! a vector path's gain against the portable one readable on one machine.
//!
//! One bench at a time, governor lifted, and the median of several rounds.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tape_base58::{
    decode_32, decode_32_batch, decode_64, decode_64_batch, encode_32, encode_32_batch, encode_64,
    encode_64_batch, MAX_ENCODED_32, MAX_ENCODED_64,
};

/// Inputs a batch bench converts in one call
const BATCH: usize = 64;

/// What a path is called, and the marker that pins it
///
/// Only x86-64 has more than one path to pin, so elsewhere the marker is
/// carried but never read.
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
        name: "portable",
        marker: Some(PORTABLE),
    }];
    if widest == AVX2 || widest == WIDE {
        runnable.push(Path {
            name: "avx2",
            marker: Some(AVX2),
        });
    }
    if widest == WIDE {
        runnable.push(Path {
            name: "avx512",
            marker: Some(WIDE),
        });
    }
    runnable
}

/// Every other architecture resolves its path at build time, so there is one
#[cfg(not(target_arch = "x86_64"))]
fn paths() -> Vec<Path> {
    vec![Path {
        name: "native",
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

fn key() -> [u8; 32] {
    [
        24, 243, 6, 223, 230, 153, 210, 8, 92, 137, 123, 67, 164, 197, 79, 196, 125, 43, 183, 85,
        103, 91, 232, 167, 73, 131, 104, 131, 0, 101, 214, 231,
    ]
}

fn signature() -> [u8; 64] {
    let mut bytes = [0u8; 64];
    for (at, slot) in bytes.iter_mut().enumerate() {
        *slot = (at as u8).wrapping_mul(53).wrapping_add(7) | 1;
    }
    bytes
}

fn compare(criterion: &mut Criterion) {
    let key = key();
    let signature = signature();

    let mut key_text = [0u8; MAX_ENCODED_32];
    let key_len = encode_32(&key, &mut key_text);
    let key_text = &key_text[..key_len];

    let mut signature_text = [0u8; MAX_ENCODED_64];
    let signature_len = encode_64(&signature, &mut signature_text);
    let signature_text = &signature_text[..signature_len];

    let keys = [key; BATCH];
    let signatures = [signature; BATCH];
    let key_texts: Vec<&[u8]> = (0..BATCH).map(|_| key_text).collect();
    let signature_texts: Vec<&[u8]> = (0..BATCH).map(|_| signature_text).collect();

    let mut key_out = [0u8; MAX_ENCODED_32];
    let mut signature_out = [0u8; MAX_ENCODED_64];
    let mut key_bytes = [0u8; 32];
    let mut signature_bytes = [0u8; 64];
    let mut encode_batch_keys = vec![0u8; BATCH * MAX_ENCODED_32];
    let mut encode_batch_signatures = vec![0u8; BATCH * MAX_ENCODED_64];
    let mut lengths = [0usize; BATCH];
    let mut decode_batch_keys = vec![[0u8; 32]; BATCH];
    let mut decode_batch_signatures = vec![[0u8; 64]; BATCH];

    let mut group = criterion.benchmark_group("paths");
    for path in paths() {
        group.bench_function(BenchmarkId::new("encode_32", path.name), |bencher| {
            pin(&path);
            bencher.iter(|| encode_32(black_box(&key), black_box(&mut key_out)))
        });
        group.bench_function(BenchmarkId::new("encode_64", path.name), |bencher| {
            pin(&path);
            bencher.iter(|| encode_64(black_box(&signature), black_box(&mut signature_out)))
        });
        group.bench_function(BenchmarkId::new("decode_32", path.name), |bencher| {
            pin(&path);
            bencher.iter(|| decode_32(black_box(key_text), black_box(&mut key_bytes)))
        });
        group.bench_function(BenchmarkId::new("decode_64", path.name), |bencher| {
            pin(&path);
            bencher.iter(|| decode_64(black_box(signature_text), black_box(&mut signature_bytes)))
        });

        group.bench_function(BenchmarkId::new("encode_32_batch", path.name), |bencher| {
            pin(&path);
            bencher.iter(|| {
                encode_32_batch(
                    black_box(&keys),
                    black_box(&mut encode_batch_keys),
                    black_box(&mut lengths),
                )
            })
        });
        group.bench_function(BenchmarkId::new("encode_64_batch", path.name), |bencher| {
            pin(&path);
            bencher.iter(|| {
                encode_64_batch(
                    black_box(&signatures),
                    black_box(&mut encode_batch_signatures),
                    black_box(&mut lengths),
                )
            })
        });
        group.bench_function(BenchmarkId::new("decode_32_batch", path.name), |bencher| {
            pin(&path);
            bencher
                .iter(|| decode_32_batch(black_box(&key_texts), black_box(&mut decode_batch_keys)))
        });
        group.bench_function(BenchmarkId::new("decode_64_batch", path.name), |bencher| {
            pin(&path);
            bencher.iter(|| {
                decode_64_batch(
                    black_box(&signature_texts),
                    black_box(&mut decode_batch_signatures),
                )
            })
        });
    }
    group.finish();

    #[cfg(target_arch = "x86_64")]
    tape_base58::testing::force(tape_base58::testing::UNKNOWN);
}

criterion_group!(benches, compare);
criterion_main!(benches);
