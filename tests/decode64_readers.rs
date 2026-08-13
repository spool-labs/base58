//! Holds the wide 64-byte reader to the four-lane one it no longer replaces
//!
//! Dispatch routes every machine's 64-byte decode through the four-lane
//! kernels now, so this differential is the wide reader's only exercise.
//! It stays correct here in case an unmeasured part wants it back.

#![cfg(target_arch = "x86_64")]

use tape_base58::testing::{
    available, avx2_read_64, avx2_words_from_limbs_64, wide_read_64, wide_words_from_limbs_64, WIDE,
};
use tape_base58::{encode_64, MAX_ENCODED_64};

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

/// Signatures across every leading-zero count, noise, and both ends
fn corpus() -> Vec<[u8; 64]> {
    let mut inputs = Vec::new();
    for zeros in 0..=64 {
        let mut input = [0xFFu8; 64];
        for slot in input.iter_mut().take(zeros) {
            *slot = 0;
        }
        inputs.push(input);
    }
    for seed in 0..256 {
        let mut input = [0u8; 64];
        noise(seed, &mut input);
        inputs.push(input);
    }
    inputs.push([0u8; 64]);
    inputs
}

// Reads each corpus encoding with both readers and compares limbs and words
#[test]
fn wide_reader_matches_four_lane() {
    if available() != WIDE {
        return;
    }
    for input in corpus() {
        let mut text = [0u8; MAX_ENCODED_64];
        let length = encode_64(&input, &mut text);
        let encoded = &text[..length];

        let mut wide_limbs = [0u32; 18];
        let mut narrow_limbs = [0u32; 18];
        // SAFETY: gated on what the machine reported, and the encoding is
        // between 64 and 88 bytes, which both readers have room for.
        let is_wide_read = unsafe { wide_read_64(encoded, &mut wide_limbs) };
        let is_narrow_read = unsafe { avx2_read_64(encoded, &mut narrow_limbs) };
        assert!(is_wide_read && is_narrow_read, "a reader refused {input:?}");
        assert_eq!(wide_limbs, narrow_limbs, "limbs differ on {input:?}");

        let mut wide_words = [0u64; 16];
        let mut narrow_words = [0u64; 16];
        // SAFETY: as above.
        unsafe {
            wide_words_from_limbs_64(&wide_limbs, &mut wide_words);
            avx2_words_from_limbs_64(&narrow_limbs, &mut narrow_words);
        }
        assert_eq!(wide_words, narrow_words, "words differ on {input:?}");
    }
}
