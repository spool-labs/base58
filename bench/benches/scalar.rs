//! Times the portable codec directly, which the default path may not reach
//!
//! On a machine with a vector path the crate's entry points route around parts
//! of `scalar`, so timing them there measures the vector code instead. These go
//! straight at the portable functions so a change to them shows up.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tape_base58::testing::{
    scalar_decode_32, scalar_decode_64, scalar_encode_32, scalar_encode_64,
};
use tape_base58::{MAX_ENCODED_32, MAX_ENCODED_64};

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

fn scalar(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("scalar");
    let signature = signature();

    let mut key_text = [0u8; MAX_ENCODED_32];
    let key_len = scalar_encode_32(&KEY, &mut key_text);
    let mut signature_text = [0u8; MAX_ENCODED_64];
    let signature_len = scalar_encode_64(&signature, &mut signature_text);

    let mut key_out = [0u8; MAX_ENCODED_32];
    group.bench_function("encode_32", |bencher| {
        bencher.iter(|| scalar_encode_32(black_box(&KEY), black_box(&mut key_out)))
    });

    let mut signature_out = [0u8; MAX_ENCODED_64];
    group.bench_function("encode_64", |bencher| {
        bencher.iter(|| scalar_encode_64(black_box(&signature), black_box(&mut signature_out)))
    });

    let mut key_bytes = [0u8; 32];
    group.bench_function("decode_32", |bencher| {
        bencher
            .iter(|| scalar_decode_32(black_box(&key_text[..key_len]), black_box(&mut key_bytes)))
    });

    let mut signature_bytes = [0u8; 64];
    group.bench_function("decode_64", |bencher| {
        bencher.iter(|| {
            scalar_decode_64(
                black_box(&signature_text[..signature_len]),
                black_box(&mut signature_bytes),
            )
        })
    });

    // A key that is all zero bytes takes the longest path through the leading
    // zero scans at both ends, which the ordinary inputs above never touch.
    let zeros = [0u8; 32];
    group.bench_function("encode_32_zeros", |bencher| {
        bencher.iter(|| scalar_encode_32(black_box(&zeros), black_box(&mut key_out)))
    });

    group.finish();
}

criterion_group!(benches, scalar);
criterion_main!(benches);
