//! Decoding characters nobody promised were well formed
//!
//! This is the untrusted side of the codec: an encoding arrives from an RPC
//! caller, not from our own encoder. Nothing in here may panic, and whatever
//! is accepted has to mean what the reference says it means.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tape_base58::{decoded_len, encoded_len, MAX_VARIABLE_LEN};

/// Pin the codec to one path, so each kernel gets a run of its own
///
/// Dispatch only ever picks the widest a machine has, so without this the
/// narrower kernels never see an input.
fn pin() {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            use tape_base58::testing::{available, force, AVX2, PORTABLE, WIDE};
            let widest = available();
            let asked = std::env::var("TAPE_PATH").unwrap_or_default();
            let marker = match asked.as_str() {
                "portable" => PORTABLE,
                "avx2" if widest == AVX2 || widest == WIDE => AVX2,
                "avx512" if widest == WIDE => WIDE,
                "" => return,
                other => panic!("this machine cannot run {other}"),
            };
            force(marker);
        });
    }
}

fuzz_target!(|data: &[u8]| {
    pin();
    if data.len() > encoded_len(MAX_VARIABLE_LEN) {
        return;
    }

    let mut out = vec![0u8; decoded_len(data.len())];
    if let Ok(len) = tape_base58::decode(data, &mut out) {
        let expected = bs58::decode(data)
            .into_vec()
            .expect("reference rejected what we accepted");
        assert_eq!(&out[..len], &expected[..], "decode disagrees");

        // Base58 has one spelling per byte string, so anything accepted has
        // to come back out exactly as it went in.
        let mut text = vec![0u8; encoded_len(len)];
        let written = tape_base58::encode(&out[..len], &mut text).expect("re-encode");
        assert_eq!(&text[..written], data, "re-encode disagrees");
    }

    // The fixed decoders take the same untrusted characters and answer for
    // their own widths.
    let mut key = [0u8; 32];
    if tape_base58::decode_32(data, &mut key).is_ok() {
        let expected = bs58::decode(data).into_vec().expect("reference decode");
        assert_eq!(&key[..], &expected[..], "decode_32 disagrees");
    }
    let mut signature = [0u8; 64];
    if tape_base58::decode_64(data, &mut signature).is_ok() {
        let expected = bs58::decode(data).into_vec().expect("reference decode");
        assert_eq!(&signature[..], &expected[..], "decode_64 disagrees");
    }
});
