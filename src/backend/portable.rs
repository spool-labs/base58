//! The codec for machines with no vector path of their own

use crate::error::DecodeError;
use crate::scalar::{self, LIMBS_32, LIMBS_64, WORDS_32, WORDS_64};
use crate::tables::{DECODE_32, DECODE_64};

/// The portable ends, which are the only ones a machine with no vector path has
pub(crate) use crate::scalar::{read_32, read_64, write_32, write_64};

/// Sum several keys' limbs against the decode table
pub(crate) fn words_32_lanes<const LANES: usize>(
    limbs: &[[u32; LIMBS_32]; LANES],
    wide: &mut [[u64; WORDS_32]; LANES],
) {
    scalar::words_lanes(limbs, &DECODE_32, wide)
}

/// Sum several signatures' limbs against the decode table
pub(crate) fn words_64_lanes<const LANES: usize>(
    limbs: &[[u32; LIMBS_64]; LANES],
    wide: &mut [[u64; WORDS_64]; LANES],
) {
    scalar::words_lanes(limbs, &DECODE_64, wide)
}

pub(crate) fn encode_32(input: &[u8; 32], out: &mut [u8; crate::MAX_ENCODED_32]) -> usize {
    scalar::encode_32(input, out)
}

pub(crate) fn encode_64(input: &[u8; 64], out: &mut [u8; crate::MAX_ENCODED_64]) -> usize {
    scalar::encode_64(input, out)
}

pub(crate) fn decode_32(encoded: &[u8], out: &mut [u8; 32]) -> Result<(), DecodeError> {
    scalar::decode_32(encoded, out)
}

pub(crate) fn decode_64(encoded: &[u8], out: &mut [u8; 64]) -> Result<(), DecodeError> {
    scalar::decode_64(encoded, out)
}
