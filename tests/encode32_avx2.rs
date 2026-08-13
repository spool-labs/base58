//! Holds the vector 32-byte encoders to the portable one
//!
//! These reach each encoder directly rather than through dispatch, so every
//! one is covered even where routing sends a key elsewhere — the wide
//! encoder no longer runs through dispatch on a half-width machine at all.

#![cfg(target_arch = "x86_64")]

use tape_base58::testing::{
    available, avx2_encode_32, avx2_encode_32_entry, scalar_encode_32, scalar_leading_zero_bytes,
    scalar_to_words, wide_encode_32, wide_encode_32_entry, AVX2, WIDE,
};
use tape_base58::MAX_ENCODED_32;

/// Deterministic bytes, so a failure is reproducible without a seed to carry
fn noise(seed: u64, into: &mut [u8]) {
    let mut state = seed | 1;
    for slot in into.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *slot = (state >> 24) as u8;
    }
}

/// Every leading-zero count against every byte fill, noise, and both ends
fn corpus() -> Vec<[u8; 32]> {
    let mut inputs = Vec::new();
    for zeros in 0..=32 {
        for fill in [0x01, 0x7F, 0xFF] {
            let mut input = [fill; 32];
            for slot in input.iter_mut().take(zeros) {
                *slot = 0;
            }
            inputs.push(input);
        }
    }
    for seed in 0..256 {
        let mut input = [0u8; 32];
        noise(seed, &mut input);
        inputs.push(input);
    }
    inputs.push([0u8; 32]);
    inputs.push([0xFF; 32]);
    inputs
}

fn check(name: &str, needs: u8, run: impl Fn(&[u8; 32], &[u32; 8], usize, &mut [u8]) -> usize) {
    let widest = available();
    let is_runnable = match needs {
        WIDE => widest == WIDE,
        _ => widest == AVX2 || widest == WIDE,
    };
    if !is_runnable {
        return;
    }
    for input in corpus() {
        let words = scalar_to_words::<32, 8>(&input);
        let zeros = scalar_leading_zero_bytes(&input);
        let mut expected = [0u8; MAX_ENCODED_32];
        let expected_len = scalar_encode_32(&input, &mut expected);
        let mut produced = [0u8; MAX_ENCODED_32];
        let produced_len = run(&input, &words, zeros, &mut produced);
        assert_eq!(
            (produced_len, &produced[..produced_len]),
            (expected_len, &expected[..expected_len]),
            "{name} disagrees on {input:?}",
        );
    }
}

// SAFETY: every call is gated on what the machine reported, and every buffer
// is a whole `MAX_ENCODED_32` slot.

#[test]
fn vector_encoder_matches_portable() {
    check("avx2", AVX2, |_, words, zeros, out| unsafe {
        avx2_encode_32(words, zeros, out)
    });
}

#[test]
fn vector_entry_matches_portable() {
    check("avx2_entry", AVX2, |input, _, _, out| unsafe {
        avx2_encode_32_entry(input, out)
    });
}

#[test]
fn wide_encoder_matches_portable() {
    check("avx512", WIDE, |_, words, zeros, out| unsafe {
        wide_encode_32(words, zeros, out)
    });
}

#[test]
fn wide_entry_matches_portable() {
    check("avx512_entry", WIDE, |input, _, _, out| unsafe {
        wide_encode_32_entry(input, out)
    });
}
