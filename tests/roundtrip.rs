//! Holds the codec to what a reference implementation produces

use tape_base58::{BatchError, DecodeError, MAX_ENCODED_32, MAX_ENCODED_64, MAX_VARIABLE_LEN};

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
    let len = tape_base58::encode_32(input, &mut out) as usize;
    out[..len].to_vec()
}

fn encoded_64(input: &[u8; 64]) -> Vec<u8> {
    let mut out = [0u8; MAX_ENCODED_64];
    let len = tape_base58::encode_64(input, &mut out) as usize;
    out[..len].to_vec()
}

// a key matches the reference at every count of leading zero bytes
#[test]
fn keys_by_zeros() {
    for zeros in 0..=32usize {
        let mut key = [0u8; 32];
        noise(zeros as u64 + 1, &mut key[zeros..]);
        for slot in key[zeros..].iter_mut() {
            *slot |= 1;
        }
        let expected = bs58::encode(key).into_vec();
        assert_eq!(encoded_32(&key), expected, "{zeros} leading zeros");

        let mut back = [0u8; 32];
        tape_base58::decode_32(&expected, &mut back).expect("decode");
        assert_eq!(back, key, "{zeros} leading zeros");
    }
}

// a signature matches the reference at every count of leading zero bytes
#[test]
fn signatures_by_zeros() {
    for zeros in 0..=64usize {
        let mut signature = [0u8; 64];
        noise(zeros as u64 + 7, &mut signature[zeros..]);
        for slot in signature[zeros..].iter_mut() {
            *slot |= 1;
        }
        let expected = bs58::encode(signature).into_vec();
        assert_eq!(encoded_64(&signature), expected, "{zeros} leading zeros");

        let mut back = [0u8; 64];
        tape_base58::decode_64(&expected, &mut back).expect("decode");
        assert_eq!(back, signature, "{zeros} leading zeros");
    }
}

// random keys and signatures match the reference
#[test]
fn random_fixed() {
    for seed in 0..512u64 {
        let mut key = [0u8; 32];
        noise(seed * 2 + 1, &mut key);
        assert_eq!(encoded_32(&key), bs58::encode(key).into_vec(), "key {seed}");

        let mut signature = [0u8; 64];
        noise(seed * 2 + 2, &mut signature);
        assert_eq!(
            encoded_64(&signature),
            bs58::encode(signature).into_vec(),
            "signature {seed}"
        );
    }
}

// the limit is a whole Solana packet, and one byte past it is refused
#[test]
fn variable_limit() {
    for len in [MAX_VARIABLE_LEN - 1, MAX_VARIABLE_LEN] {
        let mut input = vec![0u8; len];
        noise(len as u64 + 23, &mut input);
        let expected = bs58::encode(&input).into_vec();

        let mut out = vec![0u8; tape_base58::encoded_len(len)];
        let written = tape_base58::encode(&input, &mut out).expect("encode");
        assert_eq!(&out[..written], &expected[..], "length {len}");

        let mut back = vec![0u8; tape_base58::decoded_len(expected.len())];
        let read = tape_base58::decode(&expected, &mut back).expect("decode");
        assert_eq!(&back[..read], &input[..], "length {len}");
    }

    #[cfg(not(feature = "alloc"))]
    {
        let past = vec![7u8; MAX_VARIABLE_LEN + 1];
        let mut out = vec![0u8; tape_base58::encoded_len(past.len())];
        assert_eq!(
            tape_base58::encode(&past, &mut out),
            Err(tape_base58::EncodeError::InputTooLong)
        );
    }
}

// an encoding standing for more bytes than the stack path takes is refused
#[cfg(not(feature = "alloc"))]
#[test]
fn long_zeros() {
    let text = vec![b'1'; MAX_VARIABLE_LEN + 1];
    let mut out = vec![0u8; tape_base58::decoded_len(text.len())];
    assert!(matches!(
        tape_base58::decode(&text, &mut out),
        Err(DecodeError::TooLong)
    ));
}

// an allocator lifts the length limit, at widths well past it
#[cfg(feature = "alloc")]
#[test]
fn past_limit() {
    for len in [
        MAX_VARIABLE_LEN + 1,
        MAX_VARIABLE_LEN * 3,
        MAX_VARIABLE_LEN * 8 + 7,
    ] {
        let mut input = vec![0u8; len];
        noise(len as u64 + 5, &mut input);
        input[0] = 0;
        input[1] = 0;
        let expected = bs58::encode(&input).into_vec();

        let mut out = vec![0u8; tape_base58::encoded_len(len)];
        let written = tape_base58::encode(&input, &mut out).expect("encode");
        assert_eq!(&out[..written], &expected[..], "length {len}");

        let mut back = vec![0u8; tape_base58::decoded_len(written)];
        let read = tape_base58::decode(&out[..written], &mut back).expect("decode");
        assert_eq!(&back[..read], &input[..], "length {len}");
    }

    // A run of ones is a byte apiece, which is the widest a decode can grow.
    let text = vec![b'1'; MAX_VARIABLE_LEN * 4];
    let mut out = vec![0u8; tape_base58::decoded_len(text.len())];
    let read = tape_base58::decode(&text, &mut out).expect("decode");
    assert_eq!(read, text.len());
    assert!(out[..read].iter().all(|byte| *byte == 0));
}

// an all-zero input decodes into a buffer sized by decoded_len
#[test]
fn zeros_fit() {
    for len in 0..=MAX_VARIABLE_LEN {
        let input = vec![0u8; len];
        let mut text = vec![0u8; tape_base58::encoded_len(len)];
        let written = tape_base58::encode(&input, &mut text).expect("encode");

        let mut back = vec![0u8; tape_base58::decoded_len(written)];
        let read = tape_base58::decode(&text[..written], &mut back).expect("decode");
        assert_eq!(&back[..read], &input[..], "length {len}");
    }
}

// variable length input matches the reference at every length it accepts
#[test]
fn variable_lengths() {
    for len in 0..=MAX_VARIABLE_LEN {
        let mut input = vec![0u8; len];
        noise(len as u64 + 11, &mut input);
        if len > 3 {
            input[0] = 0;
            input[1] = 0;
        }
        let expected = bs58::encode(&input).into_vec();

        let mut out = vec![0u8; tape_base58::encoded_len(len)];
        let written = tape_base58::encode(&input, &mut out).expect("encode");
        assert_eq!(&out[..written], &expected[..], "length {len}");

        let mut back = vec![0u8; tape_base58::decoded_len(expected.len())];
        let read = tape_base58::decode(&expected, &mut back).expect("decode");
        assert_eq!(&back[..read], &input[..], "length {len}");
    }
}

// a batch encodes each input exactly as encoding it alone would
#[test]
fn batch_matches_single() {
    for count in [1usize, 2, 3, 4, 5, 7, 8, 33] {
        let mut keys = vec![[0u8; 32]; count];
        for (at, key) in keys.iter_mut().enumerate() {
            noise(at as u64 + 3, key);
        }
        keys[0] = [0u8; 32];

        let mut out = vec![0u8; count * MAX_ENCODED_32];
        let mut lengths = vec![0usize; count];
        tape_base58::encode_32_batch(&keys, &mut out, &mut lengths).expect("batch");

        for (at, key) in keys.iter().enumerate() {
            let slot = &out[at * MAX_ENCODED_32..][..lengths[at]];
            assert_eq!(slot, &encoded_32(key)[..], "count {count} index {at}");
        }
    }
}

// a batch of signatures spells what one at a time does, whatever the last group holds
#[test]
fn signature_batch_matches_single() {
    for count in [1usize, 2, 3, 4, 5, 7, 8, 33] {
        let mut signatures = vec![[0u8; 64]; count];
        for (at, signature) in signatures.iter_mut().enumerate() {
            noise(at as u64 + 11, signature);
        }
        signatures[0] = [0u8; 64];

        let mut out = vec![0u8; count * MAX_ENCODED_64];
        let mut lengths = vec![0usize; count];
        tape_base58::encode_64_batch(&signatures, &mut out, &mut lengths).expect("batch");

        for (at, signature) in signatures.iter().enumerate() {
            let slot = &out[at * MAX_ENCODED_64..][..lengths[at]];
            assert_eq!(slot, &encoded_64(signature)[..], "count {count} index {at}");
        }
    }
}

// a batch decodes each input exactly as decoding it alone would
#[test]
fn decode_batch_matches_single() {
    for count in [1usize, 2, 3, 4, 5, 7, 8, 33] {
        let mut keys = vec![[0u8; 32]; count];
        for (at, key) in keys.iter_mut().enumerate() {
            noise(at as u64 + 5, key);
        }
        keys[0] = [0u8; 32];
        if count > 1 {
            keys[1][0] = 0;
        }
        let texts: Vec<Vec<u8>> = keys.iter().map(encoded_32).collect();
        let inputs: Vec<&[u8]> = texts.iter().map(Vec::as_slice).collect();

        let mut out = vec![[0u8; 32]; count];
        tape_base58::decode_32_batch(&inputs, &mut out).expect("batch");
        assert_eq!(out, keys, "count {count}");
    }
}

// the same for signatures, whose limbs take two passes on the way out
#[test]
fn decode_signature_batch_matches_single() {
    for count in [1usize, 2, 3, 4, 5, 7, 8, 33] {
        let mut signatures = vec![[0u8; 64]; count];
        for (at, signature) in signatures.iter_mut().enumerate() {
            noise(at as u64 + 17, signature);
        }
        signatures[0] = [0u8; 64];
        if count > 1 {
            signatures[1][0] = 0;
        }
        let texts: Vec<Vec<u8>> = signatures.iter().map(encoded_64).collect();
        let inputs: Vec<&[u8]> = texts.iter().map(Vec::as_slice).collect();

        let mut out = vec![[0u8; 64]; count];
        tape_base58::decode_64_batch(&inputs, &mut out).expect("batch");
        assert_eq!(out, signatures, "count {count}");
    }
}

// a batch names the input it stopped on, not just that it stopped
#[test]
fn decode_batch_names_the_bad_input() {
    let mut keys = vec![[0u8; 32]; 7];
    for (at, key) in keys.iter_mut().enumerate() {
        noise(at as u64 + 13, key);
    }
    let mut texts: Vec<Vec<u8>> = keys.iter().map(encoded_32).collect();

    // in the second group of four, so the position has to carry the group with it
    texts[5][2] = b'O';
    let inputs: Vec<&[u8]> = texts.iter().map(Vec::as_slice).collect();
    let mut out = vec![[0u8; 32]; 7];
    assert_eq!(
        tape_base58::decode_32_batch(&inputs, &mut out),
        Err(BatchError::Input {
            at: 5,
            error: DecodeError::InvalidCharacter(b'O'),
        })
    );

    // a failure the reader cannot see, which only settling the words finds
    let too_large = vec![b'z'; MAX_ENCODED_32];
    let mut inputs: Vec<&[u8]> = texts.iter().map(Vec::as_slice).collect();
    inputs[5] = &texts[4];
    inputs[6] = &too_large;
    assert_eq!(
        tape_base58::decode_32_batch(&inputs, &mut out),
        Err(BatchError::Input {
            at: 6,
            error: DecodeError::ValueTooLarge,
        })
    );

    // and an output with fewer slots than the batch has inputs
    let mut short = vec![[0u8; 32]; 6];
    assert_eq!(
        tape_base58::decode_32_batch(&inputs, &mut short),
        Err(BatchError::OutputTooSmall)
    );
}

// a bad byte in any position is rejected and named
#[test]
fn rejects_bad_bytes() {
    let mut key = [0u8; 32];
    noise(99, &mut key);
    let encoded = bs58::encode(key).into_vec();
    for bad in [b'0', b'O', b'I', b'l', b' ', b'/', 0x00, 0x80, 0xFF] {
        for at in 0..encoded.len() {
            let mut broken = encoded.clone();
            broken[at] = bad;
            let mut out = [0u8; 32];
            assert_eq!(
                tape_base58::decode_32(&broken, &mut out),
                Err(DecodeError::InvalidCharacter(bad)),
                "byte {bad} at {at}"
            );
        }
    }
}

// an encoding of the wrong width is rejected rather than truncated
#[test]
fn rejects_wrong_width() {
    let mut out = [0u8; 32];
    let long = vec![b'z'; MAX_ENCODED_32 + 1];
    assert_eq!(
        tape_base58::decode_32(&long, &mut out),
        Err(DecodeError::TooLong)
    );

    let too_many_ones = vec![b'1'; 33];
    assert_eq!(
        tape_base58::decode_32(&too_many_ones, &mut out),
        Err(DecodeError::OutputTooLong)
    );

    let value_too_large = vec![b'z'; MAX_ENCODED_32];
    assert_eq!(
        tape_base58::decode_32(&value_too_large, &mut out),
        Err(DecodeError::ValueTooLarge)
    );
}
