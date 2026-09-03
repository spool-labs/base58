//! The codec on aarch64, where every machine has the vector path

use crate::error::DecodeError;
use crate::neon;
use crate::scalar::{
    self, DIGITS_32, DIGITS_64, LIMBS_32, LIMBS_64, PADDED_32, PADDED_64, WORDS_32, WORDS_64,
};
use crate::tables::{DECODE_32, DECODE_64};
use crate::{MAX_ENCODED_32, MAX_ENCODED_64};

/// Characters below which a whole-register read would run past the input
const MIN_VECTOR_CHARS: usize = 16;

pub(crate) fn encode_32(input: &[u8; 32], out: &mut [u8; MAX_ENCODED_32]) -> usize {
    let words = scalar::to_words::<32, WORDS_32>(input);
    write_32(
        scalar::sum_32(&words),
        scalar::leading_zero_bytes(input),
        out,
    )
}

pub(crate) fn encode_64(input: &[u8; 64], out: &mut [u8; MAX_ENCODED_64]) -> usize {
    let words = scalar::to_words::<64, WORDS_64>(input);
    write_64(
        scalar::sum_64(&words),
        scalar::leading_zero_bytes(input),
        out,
    )
}

/// Spell a public key's limbs, as [`scalar::sum_32`] leaves them, into characters
///
/// The tail of the encoding, split out so that a caller holding limbs it
/// arrived at some other way reaches the vector lookup rather than the table
/// the portable path walks. Settling comes first here, because the digits the
/// lookup indexes are read out of settled limbs in one pass.
#[inline]
pub(crate) fn write_32(mut limbs: [u64; LIMBS_32], input_zeros: usize, out: &mut [u8]) -> usize {
    scalar::settle(&mut limbs);
    let digits = scalar::to_digits::<LIMBS_32, PADDED_32>(&limbs);
    neon::write_encoded::<PADDED_32, 3, 2, 1>(&digits, DIGITS_32, input_zeros, out)
}

/// Spell a signature's limbs, as [`scalar::sum_64`] leaves them, into characters
#[inline]
pub(crate) fn write_64(mut limbs: [u64; LIMBS_64], input_zeros: usize, out: &mut [u8]) -> usize {
    scalar::settle(&mut limbs);
    let digits = scalar::to_digits::<LIMBS_64, PADDED_64>(&limbs);
    neon::write_encoded::<PADDED_64, 6, 4, 2>(&digits, DIGITS_64, input_zeros, out)
}

pub(crate) fn decode_32(encoded: &[u8], out: &mut [u8; 32]) -> Result<(), DecodeError> {
    let mut limbs = [0u32; LIMBS_32];
    read_32(encoded, &mut limbs)?;
    let mut wide = [0u64; WORDS_32];
    neon::words_from_limbs::<LIMBS_32, WORDS_32, { WORDS_32 / 2 }>(&limbs, &DECODE_32, &mut wide);
    let words = scalar::settle_words(&mut wide)?;
    scalar::from_words::<32, WORDS_32>(&words, out);
    scalar::check_leading_ones(out, encoded)
}

pub(crate) fn decode_64(encoded: &[u8], out: &mut [u8; 64]) -> Result<(), DecodeError> {
    // Sixteen words is past what the vector form wins on, so this one stays flat.
    let mut limbs = [0u32; LIMBS_64];
    read_64(encoded, &mut limbs)?;
    let mut wide = [0u64; WORDS_64];
    scalar::accumulate_words(&limbs, &DECODE_64, &mut wide);
    let words = scalar::settle_words(&mut wide)?;
    scalar::from_words::<64, WORDS_64>(&words, out);
    scalar::check_leading_ones(out, encoded)
}

/// Sum several keys' limbs against the decode table
///
/// The vector product is already a whole register wide for one input, so the
/// lanes go through it one at a time rather than interleaving, which would
/// only add chains the processor finds on its own.
pub(crate) fn words_32_lanes<const LANES: usize>(
    limbs: &[[u32; LIMBS_32]; LANES],
    wide: &mut [[u64; WORDS_32]; LANES],
) {
    for (lane, slot) in wide.iter_mut().enumerate() {
        neon::words_from_limbs::<LIMBS_32, WORDS_32, { WORDS_32 / 2 }>(
            &limbs[lane],
            &DECODE_32,
            slot,
        );
    }
}

/// Sum several signatures' limbs against the decode table
///
/// Sixteen words is past what the vector form wins on, so these interleave.
pub(crate) fn words_64_lanes<const LANES: usize>(
    limbs: &[[u32; LIMBS_64]; LANES],
    wide: &mut [[u64; WORDS_64]; LANES],
) {
    scalar::words_lanes(limbs, &DECODE_64, wide)
}

/// Read a public key's encoding into the limbs it stands for
///
/// The vector reader works in whole registers, so an encoding shorter than one
/// goes to the portable reader instead. So does one it finds a character
/// outside the alphabet in: it reports only that there was one, and naming it
/// is the portable reader's work.
#[inline(always)]
pub(crate) fn read_32(encoded: &[u8], limbs: &mut [u32; LIMBS_32]) -> Result<(), DecodeError> {
    if encoded.len() < MIN_VECTOR_CHARS || encoded.len() > MAX_ENCODED_32 {
        return scalar::read_32(encoded, limbs);
    }
    let mut digits = [0u8; PADDED_32];
    if !neon::digits_from_encoded::<PADDED_32>(encoded, DIGITS_32, &mut digits) {
        return scalar::read_32(encoded, limbs);
    }
    *limbs = scalar::digits_to_limbs::<LIMBS_32, PADDED_32>(&digits);
    Ok(())
}

/// Read a signature's encoding into the limbs it stands for
#[inline(always)]
pub(crate) fn read_64(encoded: &[u8], limbs: &mut [u32; LIMBS_64]) -> Result<(), DecodeError> {
    if encoded.len() < MIN_VECTOR_CHARS || encoded.len() > MAX_ENCODED_64 {
        return scalar::read_64(encoded, limbs);
    }
    let mut digits = [0u8; PADDED_64];
    if !neon::digits_from_encoded::<PADDED_64>(encoded, DIGITS_64, &mut digits) {
        return scalar::read_64(encoded, limbs);
    }
    *limbs = scalar::digits_to_limbs::<LIMBS_64, PADDED_64>(&digits);
    Ok(())
}
