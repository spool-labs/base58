//! Holds the portable codec to the same reference the dispatched one answers to
//!
//! On a machine with a vector path the crate's entry points route around parts
//! of `scalar`, so the rest of the suite can pass without ever running them.
//! These go straight at the portable functions, and also check that the two
//! paths agree character for character.

use tape_base58::testing::{
    scalar_decode_32, scalar_decode_64, scalar_encode_32, scalar_encode_64,
};
use tape_base58::{DecodeError, MAX_ENCODED_32, MAX_ENCODED_64};

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

fn portable_32(input: &[u8; 32]) -> Vec<u8> {
    let mut out = [0u8; MAX_ENCODED_32];
    let len = scalar_encode_32(input, &mut out);
    out[..len].to_vec()
}

fn portable_64(input: &[u8; 64]) -> Vec<u8> {
    let mut out = [0u8; MAX_ENCODED_64];
    let len = scalar_encode_64(input, &mut out);
    out[..len].to_vec()
}

// the portable path matches the reference at every count of leading zero bytes
#[test]
fn by_leading_zeros() {
    for zeros in 0..=32usize {
        let mut key = [0u8; 32];
        noise(zeros as u64 + 1, &mut key[zeros..]);
        for slot in key[zeros..].iter_mut() {
            *slot |= 1;
        }
        let expected = bs58::encode(key).into_vec();
        assert_eq!(portable_32(&key), expected, "key, {zeros} leading zeros");
        assert_eq!(portable_32(&key), {
            let mut out = [0u8; MAX_ENCODED_32];
            let len = tape_base58::encode_32(&key, &mut out) as usize;
            out[..len].to_vec()
        });

        let mut back = [0u8; 32];
        scalar_decode_32(&expected, &mut back).expect("decode");
        assert_eq!(back, key, "key, {zeros} leading zeros");
    }

    for zeros in 0..=64usize {
        let mut signature = [0u8; 64];
        noise(zeros as u64 + 7, &mut signature[zeros..]);
        for slot in signature[zeros..].iter_mut() {
            *slot |= 1;
        }
        let expected = bs58::encode(signature).into_vec();
        assert_eq!(
            portable_64(&signature),
            expected,
            "signature, {zeros} leading zeros"
        );
        assert_eq!(portable_64(&signature), {
            let mut out = [0u8; MAX_ENCODED_64];
            let len = tape_base58::encode_64(&signature, &mut out) as usize;
            out[..len].to_vec()
        });

        let mut back = [0u8; 64];
        scalar_decode_64(&expected, &mut back).expect("decode");
        assert_eq!(back, signature, "signature, {zeros} leading zeros");
    }
}

// random values round trip through the portable path and match the reference
#[test]
fn random_values() {
    for seed in 0..512u64 {
        let mut key = [0u8; 32];
        noise(seed * 2 + 1, &mut key);
        let encoded = bs58::encode(key).into_vec();
        assert_eq!(portable_32(&key), encoded, "key {seed}");
        let mut back = [0u8; 32];
        scalar_decode_32(&encoded, &mut back).expect("decode");
        assert_eq!(back, key, "key {seed}");

        let mut signature = [0u8; 64];
        noise(seed * 2 + 2, &mut signature);
        let encoded = bs58::encode(signature).into_vec();
        assert_eq!(portable_64(&signature), encoded, "signature {seed}");
        let mut back = [0u8; 64];
        scalar_decode_64(&encoded, &mut back).expect("decode");
        assert_eq!(back, signature, "signature {seed}");
    }
}

// a character outside the alphabet is rejected wherever it sits, at either width
#[test]
fn rejects_bad_bytes() {
    let mut key = [0u8; 32];
    noise(99, &mut key);
    let key_text = bs58::encode(key).into_vec();
    let mut signature = [0u8; 64];
    noise(100, &mut signature);
    let signature_text = bs58::encode(signature).into_vec();

    for bad in [b'0', b'O', b'I', b'l', b' ', b'/', 0x00, 0x80, 0xFF] {
        for at in 0..key_text.len() {
            let mut broken = key_text.clone();
            broken[at] = bad;
            let mut out = [0u8; 32];
            assert_eq!(
                scalar_decode_32(&broken, &mut out),
                Err(DecodeError::InvalidCharacter(bad)),
                "key, byte {bad} at {at}"
            );
        }
        for at in 0..signature_text.len() {
            let mut broken = signature_text.clone();
            broken[at] = bad;
            let mut out = [0u8; 64];
            assert_eq!(
                scalar_decode_64(&broken, &mut out),
                Err(DecodeError::InvalidCharacter(bad)),
                "signature, byte {bad} at {at}"
            );
        }
    }
}

// with several bad characters it is the first one the error names
#[test]
fn names_the_first_bad_byte() {
    let mut key = [0u8; 32];
    noise(11, &mut key);
    let key_text = bs58::encode(key).into_vec();

    for first in 0..key_text.len() - 1 {
        for second in first + 1..key_text.len() {
            let mut broken = key_text.clone();
            broken[first] = b'O';
            broken[second] = b'l';
            let mut out = [0u8; 32];
            assert_eq!(
                scalar_decode_32(&broken, &mut out),
                Err(DecodeError::InvalidCharacter(b'O')),
                "bad bytes at {first} and {second}"
            );
        }
    }
}

// the widths the fixed paths refuse are refused by the portable one too
#[test]
fn rejects_wrong_width() {
    let mut out = [0u8; 32];
    assert_eq!(
        scalar_decode_32(&[b'z'; MAX_ENCODED_32 + 1], &mut out),
        Err(DecodeError::TooLong)
    );
    assert_eq!(
        scalar_decode_32(&[b'1'; 33], &mut out),
        Err(DecodeError::OutputTooLong)
    );
    assert_eq!(
        scalar_decode_32(&[b'z'; MAX_ENCODED_32], &mut out),
        Err(DecodeError::ValueTooLarge)
    );

    let mut wide = [0u8; 64];
    assert_eq!(
        scalar_decode_64(&[b'z'; MAX_ENCODED_64 + 1], &mut wide),
        Err(DecodeError::TooLong)
    );
    assert_eq!(
        scalar_decode_64(&[b'1'; 65], &mut wide),
        Err(DecodeError::OutputTooLong)
    );
    assert_eq!(
        scalar_decode_64(&[b'z'; MAX_ENCODED_64], &mut wide),
        Err(DecodeError::ValueTooLarge)
    );
}

// an encoding shorter than the widest still decodes, which the padding covers
#[test]
fn short_encodings() {
    for zeros in 0..32usize {
        let mut key = [0u8; 32];
        noise(zeros as u64 + 3, &mut key[zeros..]);
        let text = bs58::encode(key).into_vec();
        let mut back = [0u8; 32];
        scalar_decode_32(&text, &mut back).expect("decode");
        assert_eq!(back, key, "{} characters", text.len());
    }
    for zeros in 0..64usize {
        let mut signature = [0u8; 64];
        noise(zeros as u64 + 5, &mut signature[zeros..]);
        let text = bs58::encode(signature).into_vec();
        let mut back = [0u8; 64];
        scalar_decode_64(&text, &mut back).expect("decode");
        assert_eq!(back, signature, "{} characters", text.len());
    }
}

// the all zero value, where every digit the buffer holds is a leading one
#[test]
fn all_zeros() {
    let key = [0u8; 32];
    assert_eq!(portable_32(&key), vec![b'1'; 32]);
    let mut back = [0u8; 32];
    scalar_decode_32(&[b'1'; 32], &mut back).expect("decode");
    assert_eq!(back, key);

    let signature = [0u8; 64];
    assert_eq!(portable_64(&signature), vec![b'1'; 64]);
    let mut wide = [0u8; 64];
    scalar_decode_64(&[b'1'; 64], &mut wide).expect("decode");
    assert_eq!(wide, signature);
}
