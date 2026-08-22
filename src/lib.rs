//! Base58 encoding and decoding for Solana-shaped data
//!
//! Public keys and signatures get fixed-size paths that avoid the long division
//! a general base58 codec performs. Everything else goes through a limb-based
//! codec that is still far cheaper than the byte-at-a-time form. The widest
//! instruction set the running machine supports is chosen at first use, so a
//! build carrying no target flags still gets the fast path.

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]

mod backend;

#[cfg(target_arch = "x86_64")]
mod avx2;

#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "x86_64")]
mod dispatch;

mod batch;
mod error;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod wide;

#[cfg(feature = "variable")]
mod place_values;

mod scalar;
mod tables;
mod variable;
mod variable_simd;

pub use batch::{decode_32_batch, decode_64_batch, encode_32_batch, encode_64_batch};

/// Internals the test suite checks and the benchmarks time, not a public API
#[doc(hidden)]
pub mod testing {
    pub use crate::scalar::{
        decode_32 as scalar_decode_32, decode_64 as scalar_decode_64,
        encode_32 as scalar_encode_32, encode_64 as scalar_encode_64,
        leading_zero_bytes as scalar_leading_zero_bytes, sum_32 as scalar_sum_32,
        to_words as scalar_to_words, write_32 as scalar_write_32,
    };

    /// The vector 32-byte encoders, reachable without dispatch for their
    /// tests and the bench rows that compare them in one binary
    #[cfg(target_arch = "x86_64")]
    pub use crate::avx2::{encode_32 as avx2_encode_32, encode_32_entry as avx2_encode_32_entry};
    #[cfg(target_arch = "x86_64")]
    pub use crate::avx512::{encode_32 as wide_encode_32, encode_32_entry as wide_encode_32_entry};

    /// Both 64-byte decode readers, for the differential test that keeps the
    /// wide one honest while dispatch routes every machine to the other
    #[cfg(target_arch = "x86_64")]
    pub use crate::avx2::{
        read_64 as avx2_read_64, words_from_limbs_64 as avx2_words_from_limbs_64,
    };
    #[cfg(target_arch = "x86_64")]
    pub use crate::avx512::{
        read_64 as wide_read_64, words_from_limbs_64 as wide_words_from_limbs_64,
    };
    pub use crate::tables::{DECODE_32, DECODE_64, ENCODE_32, ENCODE_64};

    #[cfg(feature = "variable")]
    pub use crate::place_values::{LIMB_OFFSETS, LIMB_VALUES, PLACE_OFFSETS, PLACE_VALUES};

    /// Pin the codec to one path, so a test can reach the ones this machine
    /// would not have chosen for itself
    #[cfg(target_arch = "x86_64")]
    pub use crate::dispatch::{available, force, AVX2, PORTABLE, UNKNOWN, WIDE};

    /// The any-length kernels, for the differential test that checks them
    /// against the scalar walk they stand in for
    #[cfg(target_arch = "x86_64")]
    pub use crate::avx2::{
        place_multiply_add as avx2_place_multiply_add, shed_upward as avx2_shed_upward,
        shed_words_upward as avx2_shed_words_upward,
    };
    #[cfg(target_arch = "x86_64")]
    pub use crate::avx512::{
        place_multiply_add as wide_place_multiply_add, shed_upward as wide_shed_upward,
        shed_words_upward as wide_shed_words_upward,
    };
}
pub use error::{BatchError, DecodeError, EncodeError};
pub use variable::{decode, decoded_len, encode, encoded_len, MAX_VARIABLE_LEN};

/// Bytes in a public key
pub const KEY_LEN: usize = 32;

/// Bytes in a signature
pub const SIGNATURE_LEN: usize = 64;

/// Longest base58 encoding a 32-byte input can produce
pub const MAX_ENCODED_32: usize = 44;

/// Longest base58 encoding a 64-byte input can produce
pub const MAX_ENCODED_64: usize = 88;

/// Encode a public key, returning how many bytes of the output were written
///
/// The count is a `u8` because `MAX_ENCODED_32` is 44, so it always fits, and
/// callers holding it alongside a fixed-width buffer keep it byte-sized.
pub fn encode_32(input: &[u8; KEY_LEN], out: &mut [u8; MAX_ENCODED_32]) -> u8 {
    backend::encode_32(input, out) as u8
}

/// Encode a signature, returning how many bytes of the output were written
///
/// The count is a `u8` because `MAX_ENCODED_64` is 88, so it always fits.
pub fn encode_64(input: &[u8; SIGNATURE_LEN], out: &mut [u8; MAX_ENCODED_64]) -> u8 {
    backend::encode_64(input, out) as u8
}

/// Decode a public key, rejecting anything that is not exactly 32 bytes wide
pub fn decode_32(encoded: &[u8], out: &mut [u8; KEY_LEN]) -> Result<(), DecodeError> {
    backend::decode_32(encoded, out)
}

/// Decode a signature, rejecting anything that is not exactly 64 bytes wide
pub fn decode_64(encoded: &[u8], out: &mut [u8; SIGNATURE_LEN]) -> Result<(), DecodeError> {
    backend::decode_64(encoded, out)
}
