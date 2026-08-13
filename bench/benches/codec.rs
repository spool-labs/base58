use criterion::{black_box, criterion_group, criterion_main, Criterion};

const KEY: [u8; 32] = [
    24, 243, 6, 223, 230, 153, 210, 8, 92, 137, 123, 67, 164, 197, 79, 196, 125, 43, 183, 85, 103,
    91, 232, 167, 73, 131, 104, 131, 0, 101, 214, 231,
];
const KEY_TEXT: &[u8] = b"2gPihUTjt3FJqf1VpidgrY5cZ6PuyMccGVwQHRfjMPZG";

fn signature() -> [u8; 64] {
    let mut bytes = [0u8; 64];
    for (at, slot) in bytes.iter_mut().enumerate() {
        *slot = (at as u8).wrapping_mul(53).wrapping_add(7) | 1;
    }
    bytes
}

fn codec(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("tape-base58");
    let signature = signature();
    let mut signature_text = [0u8; tape_base58::MAX_ENCODED_64];
    let written = tape_base58::encode_64(&signature, &mut signature_text);

    let mut key_out = [0u8; tape_base58::MAX_ENCODED_32];
    group.bench_function("encode_32", |bencher| {
        bencher.iter(|| tape_base58::encode_32(black_box(&KEY), black_box(&mut key_out)))
    });

    let mut signature_out = [0u8; tape_base58::MAX_ENCODED_64];
    group.bench_function("encode_64", |bencher| {
        bencher
            .iter(|| tape_base58::encode_64(black_box(&signature), black_box(&mut signature_out)))
    });

    let mut key_bytes = [0u8; 32];
    group.bench_function("decode_32", |bencher| {
        bencher.iter(|| tape_base58::decode_32(black_box(KEY_TEXT), black_box(&mut key_bytes)))
    });

    let mut signature_bytes = [0u8; 64];
    group.bench_function("decode_64", |bencher| {
        bencher.iter(|| {
            tape_base58::decode_64(
                black_box(&signature_text[..written]),
                black_box(&mut signature_bytes),
            )
        })
    });

    let keys = [KEY; 64];
    let mut batch_out = [0u8; 64 * tape_base58::MAX_ENCODED_32];
    let mut lengths = [0usize; 64];
    group.bench_function("encode_32_batch", |bencher| {
        bencher.iter(|| {
            tape_base58::encode_32_batch(
                black_box(&keys),
                black_box(&mut batch_out),
                black_box(&mut lengths),
            )
        })
    });

    let texts = [KEY_TEXT; 64];
    let mut decoded = [[0u8; 32]; 64];
    group.bench_function("decode_32_batch", |bencher| {
        bencher.iter(|| tape_base58::decode_32_batch(black_box(&texts), black_box(&mut decoded)))
    });

    let signature_texts = [&signature_text[..written]; 64];
    let mut decoded_signatures = [[0u8; 64]; 64];
    group.bench_function("decode_64_batch", |bencher| {
        bencher.iter(|| {
            tape_base58::decode_64_batch(
                black_box(&signature_texts),
                black_box(&mut decoded_signatures),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, codec);
criterion_main!(benches);
