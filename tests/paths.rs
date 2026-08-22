//! Holds every path this machine can run to the portable one, byte for byte
//!
//! The codec picks the widest instruction set at first use, so an ordinary
//! test run only ever exercises one path. These pin each in turn and compare
//! it against both the reference and the portable path, which is the only way
//! a vector path gets checked at all on a machine that would not choose it.
//!
//! The paths share one atomic, so the tests hold a lock rather than run at
//! once.

#![cfg(target_arch = "x86_64")]

use std::sync::{Mutex, MutexGuard};

use tape_base58::testing::{
    available, avx2_place_multiply_add, avx2_shed_upward, avx2_shed_words_upward, force,
    wide_place_multiply_add, wide_shed_upward, wide_shed_words_upward, AVX2, PORTABLE, UNKNOWN,
    WIDE,
};
use tape_base58::{
    decode_32, decode_32_batch, decode_64, decode_64_batch, encode_32, encode_64, BatchError,
    DecodeError, MAX_ENCODED_32, MAX_ENCODED_64,
};

static PINNED: Mutex<()> = Mutex::new(());

/// The paths this machine can run, widest last
fn paths() -> Vec<u8> {
    let widest = available();
    let mut runnable = vec![PORTABLE];
    if widest == AVX2 || widest == WIDE {
        runnable.push(AVX2);
    }
    if widest == WIDE {
        runnable.push(WIDE);
    }
    runnable
}

/// Pin a path for the duration of the guard
fn pinned() -> MutexGuard<'static, ()> {
    match PINNED.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

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

fn encoded_32(input: &[u8; 32]) -> Vec<u8> {
    let mut out = [0u8; MAX_ENCODED_32];
    let len = encode_32(input, &mut out) as usize;
    out[..len].to_vec()
}

fn encoded_64(input: &[u8; 64]) -> Vec<u8> {
    let mut out = [0u8; MAX_ENCODED_64];
    let len = encode_64(input, &mut out) as usize;
    out[..len].to_vec()
}

// every path encodes what the reference does, at every count of leading zeros
#[test]
fn encoding_matches() {
    let _guard = pinned();
    for path in paths() {
        force(path);
        for zeros in 0..=32usize {
            let mut key = [0u8; 32];
            noise(zeros as u64 + 1, &mut key[zeros..]);
            for slot in key[zeros..].iter_mut() {
                *slot |= 1;
            }
            assert_eq!(
                encoded_32(&key),
                bs58::encode(key).into_vec(),
                "path {path}, key with {zeros} leading zeros"
            );
        }
        for zeros in 0..=64usize {
            let mut signature = [0u8; 64];
            noise(zeros as u64 + 7, &mut signature[zeros..]);
            for slot in signature[zeros..].iter_mut() {
                *slot |= 1;
            }
            assert_eq!(
                encoded_64(&signature),
                bs58::encode(signature).into_vec(),
                "path {path}, signature with {zeros} leading zeros"
            );
        }
    }
    force(UNKNOWN);
}

// every path decodes what the reference encoded, at every length
#[test]
fn decoding_matches() {
    let _guard = pinned();
    for path in paths() {
        force(path);
        for zeros in 0..=32usize {
            let mut key = [0u8; 32];
            noise(zeros as u64 + 3, &mut key[zeros..]);
            let text = bs58::encode(key).into_vec();
            let mut back = [0u8; 32];
            decode_32(&text, &mut back).expect("decode");
            assert_eq!(back, key, "path {path}, {} characters", text.len());
        }
        for zeros in 0..=64usize {
            let mut signature = [0u8; 64];
            noise(zeros as u64 + 5, &mut signature[zeros..]);
            let text = bs58::encode(signature).into_vec();
            let mut back = [0u8; 64];
            decode_64(&text, &mut back).expect("decode");
            assert_eq!(back, signature, "path {path}, {} characters", text.len());
        }
    }
    force(UNKNOWN);
}

// random values round trip on every path
#[test]
fn random_values() {
    let _guard = pinned();
    for path in paths() {
        force(path);
        for seed in 0..256u64 {
            let mut key = [0u8; 32];
            noise(seed * 2 + 1, &mut key);
            let text = encoded_32(&key);
            assert_eq!(
                text,
                bs58::encode(key).into_vec(),
                "path {path}, key {seed}"
            );
            let mut back = [0u8; 32];
            decode_32(&text, &mut back).expect("decode");
            assert_eq!(back, key, "path {path}, key {seed}");

            let mut signature = [0u8; 64];
            noise(seed * 2 + 2, &mut signature);
            let text = encoded_64(&signature);
            assert_eq!(
                text,
                bs58::encode(signature).into_vec(),
                "path {path}, signature {seed}"
            );
            let mut back = [0u8; 64];
            decode_64(&text, &mut back).expect("decode");
            assert_eq!(back, signature, "path {path}, signature {seed}");
        }
    }
    force(UNKNOWN);
}

// a character outside the alphabet is rejected and named on every path
#[test]
fn rejects_bad_bytes() {
    let _guard = pinned();
    let mut key = [0u8; 32];
    noise(99, &mut key);
    let key_text = bs58::encode(key).into_vec();
    let mut signature = [0u8; 64];
    noise(100, &mut signature);
    let signature_text = bs58::encode(signature).into_vec();

    for path in paths() {
        force(path);
        for bad in [b'0', b'O', b'I', b'l', b' ', b'/', 0x00, 0x80, 0xFF] {
            for at in 0..key_text.len() {
                let mut broken = key_text.clone();
                broken[at] = bad;
                let mut out = [0u8; 32];
                assert_eq!(
                    decode_32(&broken, &mut out),
                    Err(DecodeError::InvalidCharacter(bad)),
                    "path {path}, key byte {bad} at {at}"
                );
            }
            for at in 0..signature_text.len() {
                let mut broken = signature_text.clone();
                broken[at] = bad;
                let mut out = [0u8; 64];
                assert_eq!(
                    decode_64(&broken, &mut out),
                    Err(DecodeError::InvalidCharacter(bad)),
                    "path {path}, signature byte {bad} at {at}"
                );
            }
        }
    }
    force(UNKNOWN);
}

// the widths and values every path refuses are the same ones
#[test]
fn rejects_wrong_width() {
    let _guard = pinned();
    for path in paths() {
        force(path);
        let mut out = [0u8; 32];
        assert_eq!(
            decode_32(&vec![b'z'; MAX_ENCODED_32 + 1], &mut out),
            Err(DecodeError::TooLong),
            "path {path}"
        );
        assert_eq!(
            decode_32(&vec![b'1'; 33], &mut out),
            Err(DecodeError::OutputTooLong),
            "path {path}"
        );
        assert_eq!(
            decode_32(&vec![b'z'; MAX_ENCODED_32], &mut out),
            Err(DecodeError::ValueTooLarge),
            "path {path}"
        );

        let mut wide = [0u8; 64];
        assert_eq!(
            decode_64(&vec![b'z'; MAX_ENCODED_64 + 1], &mut wide),
            Err(DecodeError::TooLong),
            "path {path}"
        );
        assert_eq!(
            decode_64(&vec![b'1'; 65], &mut wide),
            Err(DecodeError::OutputTooLong),
            "path {path}"
        );
        assert_eq!(
            decode_64(&vec![b'z'; MAX_ENCODED_64], &mut wide),
            Err(DecodeError::ValueTooLarge),
            "path {path}"
        );
    }
    force(UNKNOWN);
}

// a batch decodes to what the single path does, and names a bad input
#[test]
fn batches_match() {
    let _guard = pinned();
    for path in paths() {
        force(path);
        let mut keys = Vec::new();
        let mut signatures = Vec::new();
        for seed in 0..11u64 {
            let mut key = [0u8; 32];
            noise(seed + 41, &mut key[(seed as usize % 4)..]);
            keys.push(bs58::encode(key).into_vec());
            let mut signature = [0u8; 64];
            noise(seed + 43, &mut signature[(seed as usize % 3)..]);
            signatures.push(bs58::encode(signature).into_vec());
        }

        let borrowed: Vec<&[u8]> = keys.iter().map(|text| text.as_slice()).collect();
        let mut out = vec![[0u8; 32]; borrowed.len()];
        decode_32_batch(&borrowed, &mut out).expect("batch");
        for (at, text) in borrowed.iter().enumerate() {
            let mut single = [0u8; 32];
            decode_32(text, &mut single).expect("single");
            assert_eq!(out[at], single, "path {path}, key {at}");
        }

        let borrowed: Vec<&[u8]> = signatures.iter().map(|text| text.as_slice()).collect();
        let mut wide = vec![[0u8; 64]; borrowed.len()];
        decode_64_batch(&borrowed, &mut wide).expect("batch");
        for (at, text) in borrowed.iter().enumerate() {
            let mut single = [0u8; 64];
            decode_64(text, &mut single).expect("single");
            assert_eq!(wide[at], single, "path {path}, signature {at}");
        }

        let mut broken = keys[6].clone();
        broken[3] = b'O';
        let mut mixed: Vec<&[u8]> = keys.iter().map(|text| text.as_slice()).collect();
        mixed[6] = &broken;
        let mut out = vec![[0u8; 32]; mixed.len()];
        assert_eq!(
            decode_32_batch(&mixed, &mut out),
            Err(BatchError::Input {
                at: 6,
                error: DecodeError::InvalidCharacter(b'O'),
            }),
            "path {path}"
        );
    }
    force(UNKNOWN);
}

/// The scalar walk the vector place kernel stands in for
fn place_multiply_add_scalar(word: u32, entries: &[u32], limbs: &mut [u64]) {
    for (limb, entry) in limbs.iter_mut().zip(entries.iter()) {
        *limb += word as u64 * *entry as u64;
    }
}

/// The scalar walk the vector shed stands in for
fn shed_upward_scalar(limbs: &mut [u64]) {
    for at in (1..limbs.len()).rev() {
        let carry = limbs[at - 1] / LIMB_BASE;
        limbs[at - 1] %= LIMB_BASE;
        limbs[at] += carry;
    }
}

/// What one limb counts in
const LIMB_BASE: u64 = 58 * 58 * 58 * 58 * 58;

// the vector place product matches the scalar one at every length
#[test]
fn place_product() {
    let widest = available();
    if widest != AVX2 && widest != WIDE {
        return;
    }
    let mut state = 12345u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for len in 0..40usize {
        let entries: Vec<u32> = (0..len).map(|_| next() as u32).collect();
        let start: Vec<u64> = (0..len).map(|_| next() >> 34).collect();
        let word = next() as u32;

        let mut wanted = start.clone();
        place_multiply_add_scalar(word, &entries, &mut wanted);
        let mut got = start.clone();
        // SAFETY: the machine reported AVX2 at least.
        unsafe { avx2_place_multiply_add(word, &entries, &mut got) };
        assert_eq!(got, wanted, "avx2, length {len}");
        if widest == WIDE {
            let mut got = start.clone();
            // SAFETY: the machine reported the wide permutes.
            unsafe { wide_place_multiply_add(word, &entries, &mut got) };
            assert_eq!(got, wanted, "avx512, length {len}");
        }
    }
}

/// Carry every limb up until each one is below the limb base, exactly
fn settle_upward_scalar(limbs: &mut [u64]) {
    let mut carry = 0u64;
    for limb in limbs.iter_mut() {
        let value = *limb + carry;
        carry = value / LIMB_BASE;
        *limb = value % LIMB_BASE;
    }
    assert_eq!(carry, 0, "the value ran past the limbs it was given");
}

// the vector shed leaves the same value the scalar one does
#[test]
fn shed_matches() {
    let widest = available();
    if widest != AVX2 && widest != WIDE {
        return;
    }
    let mut state = 999u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for len in 4..40usize {
        // Limbs the size the encode actually sheds. A limb this big settles
        // into three, so the top three are spare and the last carry has
        // somewhere to land, which is what the encode allocates for too.
        let mut start: Vec<u64> = (0..len).map(|_| next() >> 5).collect();
        for slot in start[len - 3..].iter_mut() {
            *slot = 0;
        }

        let mut wanted = start.clone();
        shed_upward_scalar(&mut wanted);
        settle_upward_scalar(&mut wanted);

        // The estimate may undershoot by two, so the limbs need not match
        // before settling. Settling is exact, and after it they must.
        let mut got = start.clone();
        // SAFETY: the machine reported AVX2 at least.
        unsafe { avx2_shed_upward(&mut got) };
        settle_upward_scalar(&mut got);
        assert_eq!(got, wanted, "avx2, length {len}");

        if widest == WIDE {
            let mut got = start.clone();
            // SAFETY: the machine reported the wide permutes.
            unsafe { wide_shed_upward(&mut got) };
            settle_upward_scalar(&mut got);
            assert_eq!(got, wanted, "avx512, length {len}");
        }
    }
}

/// The scalar walk the word shed stands in for
fn shed_words_upward_scalar(words: &mut [u64]) {
    for at in (1..words.len()).rev() {
        let carry = words[at - 1] >> 32;
        words[at - 1] &= 0xFFFF_FFFF;
        words[at] += carry;
    }
}

// the vector word shed matches the scalar one exactly
#[test]
fn shed_words_matches() {
    let widest = available();
    if widest != AVX2 && widest != WIDE {
        return;
    }
    let mut state = 4242u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for len in 2..40usize {
        let mut start: Vec<u64> = (0..len).map(|_| next() >> 4).collect();
        start[len - 1] = 0;

        let mut wanted = start.clone();
        shed_words_upward_scalar(&mut wanted);

        let mut got = start.clone();
        // SAFETY: the machine reported AVX2 at least.
        unsafe { avx2_shed_words_upward(&mut got) };
        assert_eq!(got, wanted, "avx2, length {len}");

        if widest == WIDE {
            let mut got = start.clone();
            // SAFETY: the machine reported the wide permutes.
            unsafe { wide_shed_words_upward(&mut got) };
            assert_eq!(got, wanted, "avx512, length {len}");
        }
    }
}
