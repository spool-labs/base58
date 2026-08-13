//! The 32-byte encoders in one binary against the portable tail
//!
//! Every row is handed the same words and leading-zero count, so the
//! portable row is the tail of `encode_32` rather than the whole entry point;
//! timing the entry point against rows handed pre-read words would charge
//! the read to one side only. Each row is asserted against the shipping
//! scalar encoder over a corpus before it is timed, so a wrong answer fails
//! the bench instead of posting a fast number.
//!
//! See BENCHMARKS.md's "Four traps": one binary, no `target-cpu=native`, and
//! nothing read across builds below fifteen percent.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tape_base58::testing::{
    scalar_encode_32, scalar_leading_zero_bytes, scalar_sum_32, scalar_to_words, scalar_write_32,
};
use tape_base58::MAX_ENCODED_32;

const KEY: [u8; 32] = [
    24, 243, 6, 223, 230, 153, 210, 8, 92, 137, 123, 67, 164, 197, 79, 196, 125, 43, 183, 85, 103,
    91, 232, 167, 73, 131, 104, 131, 0, 101, 214, 231,
];

/// Every leading-zero count, 64 pseudorandom values, all-zero and the key
fn corpus() -> Vec<[u8; 32]> {
    let mut inputs = Vec::new();
    for zeros in 0..=32 {
        let mut input = [0xFFu8; 32];
        for slot in input.iter_mut().take(zeros) {
            *slot = 0;
        }
        inputs.push(input);
    }
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..64 {
        let mut input = [0u8; 32];
        for chunk in input.chunks_exact_mut(8) {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            chunk.copy_from_slice(&state.to_le_bytes());
        }
        inputs.push(input);
    }
    inputs.push([0u8; 32]);
    inputs.push(KEY);
    inputs
}

/// Hold a row to the shipping scalar encoder over the whole corpus
fn assert_row(name: &str, run: impl Fn(&[u32; 8], usize, &mut [u8]) -> usize) {
    for input in corpus() {
        let words = scalar_to_words::<32, 8>(&input);
        let zeros = scalar_leading_zero_bytes(&input);
        let mut expected = [0u8; MAX_ENCODED_32];
        let expected_len = scalar_encode_32(&input, &mut expected);
        let mut produced = [0u8; MAX_ENCODED_32];
        let produced_len = run(&words, zeros, &mut produced);
        assert_eq!(
            (produced_len, &produced[..produced_len]),
            (expected_len, &expected[..expected_len]),
            "row {name} disagrees on {input:?}",
        );
    }
}

fn encoders(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("encode32");
    let words = scalar_to_words::<32, 8>(&KEY);
    let zeros = scalar_leading_zero_bytes(&KEY);
    let mut out = [0u8; MAX_ENCODED_32];

    group.bench_function("entry_point", |bencher| {
        bencher.iter(|| tape_base58::encode_32(black_box(&KEY), black_box(&mut out)))
    });

    assert_row("portable_tail", |words, zeros, out| {
        scalar_write_32(scalar_sum_32(words), zeros, out)
    });
    group.bench_function("portable_tail", |bencher| {
        bencher.iter(|| {
            scalar_write_32(
                scalar_sum_32(black_box(&words)),
                black_box(zeros),
                black_box(&mut out[..]),
            )
        })
    });

    #[cfg(target_arch = "x86_64")]
    {
        use tape_base58::testing::{
            available, avx2_encode_32, avx2_encode_32_entry, wide_encode_32, wide_encode_32_entry,
            AVX2, WIDE,
        };
        let widest = available();
        if widest == AVX2 || widest == WIDE {
            // SAFETY: gated on what the machine reported, and every buffer is
            // a whole `MAX_ENCODED_32` slot.
            assert_row("avx2", |words, zeros, out| unsafe {
                avx2_encode_32(words, zeros, out)
            });
            group.bench_function("avx2", |bencher| {
                bencher.iter(|| unsafe {
                    avx2_encode_32(black_box(&words), black_box(zeros), black_box(&mut out[..]))
                })
            });
            group.bench_function("avx2_entry", |bencher| {
                bencher.iter(|| unsafe {
                    avx2_encode_32_entry(black_box(&KEY), black_box(&mut out[..]))
                })
            });
        }
        if widest == WIDE {
            // SAFETY: as above.
            assert_row("avx512", |words, zeros, out| unsafe {
                wide_encode_32(words, zeros, out)
            });
            group.bench_function("avx512", |bencher| {
                bencher.iter(|| unsafe {
                    wide_encode_32(black_box(&words), black_box(zeros), black_box(&mut out[..]))
                })
            });
            group.bench_function("avx512_entry", |bencher| {
                bencher.iter(|| unsafe {
                    wide_encode_32_entry(black_box(&KEY), black_box(&mut out[..]))
                })
            });
        }
    }

    group.finish();
}

criterion_group!(benches, encoders);
criterion_main!(benches);
