//! Encoding bytes and reading them back
//!
//! Covers the widths the test suite sweeps and the shapes it does not think
//! to build: runs of leading zeros, values that sit right on a limb boundary,
//! and whatever else the fuzzer finds interesting.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tape_base58::{decoded_len, encoded_len, MAX_ENCODED_32, MAX_ENCODED_64, MAX_VARIABLE_LEN};

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_VARIABLE_LEN {
        return;
    }

    let mut text = vec![0u8; encoded_len(data.len())];
    let written = tape_base58::encode(data, &mut text).expect("encode");
    assert_eq!(
        &text[..written],
        &bs58::encode(data).into_vec()[..],
        "encode disagrees"
    );

    let mut back = vec![0u8; decoded_len(written)];
    let read = tape_base58::decode(&text[..written], &mut back).expect("decode");
    assert_eq!(&back[..read], data, "roundtrip disagrees");

    // A key or a signature has a fixed path of its own, and it has to spell
    // the same characters the any-length one does.
    if let Ok(key) = <&[u8; 32]>::try_from(data) {
        let mut out = [0u8; MAX_ENCODED_32];
        let len = tape_base58::encode_32(key, &mut out) as usize;
        assert_eq!(&out[..len], &text[..written], "encode_32 disagrees");
        let mut round = [0u8; 32];
        tape_base58::decode_32(&out[..len], &mut round).expect("decode_32");
        assert_eq!(&round, key, "decode_32 disagrees");
    }
    if let Ok(signature) = <&[u8; 64]>::try_from(data) {
        let mut out = [0u8; MAX_ENCODED_64];
        let len = tape_base58::encode_64(signature, &mut out) as usize;
        assert_eq!(&out[..len], &text[..written], "encode_64 disagrees");
        let mut round = [0u8; 64];
        tape_base58::decode_64(&out[..len], &mut round).expect("decode_64");
        assert_eq!(&round[..], &signature[..], "decode_64 disagrees");
    }
});
