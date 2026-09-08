//! Converting several keys or signatures in one pass
//!
//! The conversion is a chain of multiply-accumulates per input, whichever way
//! it runs, and one chain leaves most of the multiplier idle. Four inputs share
//! the table read and the loop, and their chains are independent, so the work
//! interleaves into the gaps a single input leaves. A path whose own signature
//! encoder already fills the multiplier converts those one at a time instead.

use crate::backend;
use crate::error::{BatchError, DecodeError, EncodeError};
use crate::scalar::{self, LIMBS_32, LIMBS_64, WORDS_32, WORDS_64};
use crate::tables::{ENCODE_32, ENCODE_64};

/// Inputs converted together
const LANES: usize = 4;

/// Encode many public keys, one output slot of the maximum width each
///
/// Encoding number `at` lands in `out[at * MAX_ENCODED_32..]` and runs for
/// `lengths[at]` bytes. Fixed slots rather than packed, so a caller can address
/// any encoding without walking the ones before it.
pub fn encode_32_batch(
    inputs: &[[u8; 32]],
    out: &mut [u8],
    lengths: &mut [usize],
) -> Result<(), EncodeError> {
    if out.len() < inputs.len() * crate::MAX_ENCODED_32 || lengths.len() < inputs.len() {
        return Err(EncodeError::OutputTooSmall);
    }
    let (slots, _) = out.as_chunks_mut::<{ crate::MAX_ENCODED_32 }>();
    if backend::IS_ENCODE_32_DIRECT {
        for ((input, slot), length) in inputs.iter().zip(slots).zip(lengths.iter_mut()) {
            *length = backend::encode_32(input, slot);
        }
        return Ok(());
    }
    for (group, chunk) in inputs.chunks(LANES).enumerate() {
        let mut limbs = [[0u64; LIMBS_32]; LANES];
        let mut words = [[0u32; WORDS_32]; LANES];
        for (lane, input) in chunk.iter().enumerate() {
            words[lane] = scalar::to_words::<32, WORDS_32>(input);
        }
        interleave::<WORDS_32, LIMBS_32, { LIMBS_32 - 1 }, 0, WORDS_32>(
            &mut limbs, &words, &ENCODE_32,
        );
        for lane in 0..chunk.len() {
            let at = group * LANES + lane;
            lengths[at] = backend::write_32(
                limbs[lane],
                scalar::leading_zero_bytes(&chunk[lane]),
                &mut slots[at],
            );
        }
    }
    Ok(())
}

/// Encode many signatures, one output slot of the maximum width each
pub fn encode_64_batch(
    inputs: &[[u8; 64]],
    out: &mut [u8],
    lengths: &mut [usize],
) -> Result<(), EncodeError> {
    if out.len() < inputs.len() * crate::MAX_ENCODED_64 || lengths.len() < inputs.len() {
        return Err(EncodeError::OutputTooSmall);
    }
    let (slots, _) = out.as_chunks_mut::<{ crate::MAX_ENCODED_64 }>();
    if backend::is_encode_64_direct() {
        for ((input, slot), length) in inputs.iter().zip(slots.iter_mut()).zip(lengths.iter_mut()) {
            *length = backend::encode_64(input, slot);
        }
        return Ok(());
    }
    for (group, chunk) in inputs.chunks(LANES).enumerate() {
        let mut limbs = [[0u64; LIMBS_64]; LANES];
        let mut words = [[0u32; WORDS_64]; LANES];
        for (lane, input) in chunk.iter().enumerate() {
            words[lane] = scalar::to_words::<64, WORDS_64>(input);
        }
        interleave::<WORDS_64, LIMBS_64, { LIMBS_64 - 1 }, 0, 8>(&mut limbs, &words, &ENCODE_64);
        // A lane standing in for a missing input holds zeros, which sheds nothing.
        for lane in limbs.iter_mut() {
            lane[LIMBS_64 - 3] += lane[LIMBS_64 - 2] / scalar::LIMB_BASE;
            lane[LIMBS_64 - 2] %= scalar::LIMB_BASE;
        }
        interleave::<WORDS_64, LIMBS_64, { LIMBS_64 - 1 }, 8, WORDS_64>(
            &mut limbs, &words, &ENCODE_64,
        );
        for lane in 0..chunk.len() {
            let at = group * LANES + lane;
            lengths[at] = backend::write_64(
                limbs[lane],
                scalar::leading_zero_bytes(&chunk[lane]),
                &mut slots[at],
            );
        }
    }
    Ok(())
}

/// Decode many public keys, one output each
///
/// Decoding number `at` lands in `out[at]`. Stops at the first input it cannot
/// convert and names it, leaving the outputs it had already written in place
/// and the rest as they were.
pub fn decode_32_batch(encoded: &[&[u8]], out: &mut [[u8; 32]]) -> Result<(), BatchError> {
    if out.len() < encoded.len() {
        return Err(BatchError::OutputTooSmall);
    }
    if backend::IS_DECODE_32_DIRECT {
        for (at, (input, slot)) in encoded.iter().zip(out.iter_mut()).enumerate() {
            backend::decode_32(input, slot).map_err(blame(at))?;
        }
        return Ok(());
    }
    for (group, chunk) in encoded.chunks(LANES).enumerate() {
        let mut limbs = [[0u32; LIMBS_32]; LANES];
        for (lane, input) in chunk.iter().enumerate() {
            backend::read_32(input, &mut limbs[lane]).map_err(blame(group * LANES + lane))?;
        }
        let mut wide = [[0u64; WORDS_32]; LANES];
        backend::words_32_lanes(&limbs, &mut wide);
        for lane in 0..chunk.len() {
            let at = group * LANES + lane;
            let words = scalar::settle_words(&mut wide[lane]).map_err(blame(at))?;
            scalar::from_words::<32, WORDS_32>(&words, &mut out[at]);
            scalar::check_leading_ones(&out[at], chunk[lane]).map_err(blame(at))?;
        }
    }
    Ok(())
}

/// Decode many signatures, one output each
pub fn decode_64_batch(encoded: &[&[u8]], out: &mut [[u8; 64]]) -> Result<(), BatchError> {
    if out.len() < encoded.len() {
        return Err(BatchError::OutputTooSmall);
    }
    if backend::IS_DECODE_64_DIRECT {
        for (at, (input, slot)) in encoded.iter().zip(out.iter_mut()).enumerate() {
            backend::decode_64(input, slot).map_err(blame(at))?;
        }
        return Ok(());
    }
    for (group, chunk) in encoded.chunks(LANES).enumerate() {
        let mut limbs = [[0u32; LIMBS_64]; LANES];
        for (lane, input) in chunk.iter().enumerate() {
            backend::read_64(input, &mut limbs[lane]).map_err(blame(group * LANES + lane))?;
        }
        let mut wide = [[0u64; WORDS_64]; LANES];
        backend::words_64_lanes(&limbs, &mut wide);
        for lane in 0..chunk.len() {
            let at = group * LANES + lane;
            let words = scalar::settle_words(&mut wide[lane]).map_err(blame(at))?;
            scalar::from_words::<64, WORDS_64>(&words, &mut out[at]);
            scalar::check_leading_ones(&out[at], chunk[lane]).map_err(blame(at))?;
        }
    }
    Ok(())
}

/// Say which input an error came from
fn blame(at: usize) -> impl Fn(DecodeError) -> BatchError {
    move |error| BatchError::Input { at, error }
}

/// Run every lane's accumulation against one shared walk of the table
///
/// All four lanes run whether or not the last group is full. A lane the caller
/// left empty holds zero words, so it contributes nothing and its limbs stay
/// zero, and the caller reads only the lanes it filled. Counting the lanes at
/// runtime instead would put a loop of unknown length innermost, which is the
/// one place the accumulators have to stay in registers for the chains to
/// overlap at all.
fn interleave<
    const WORDS: usize,
    const LIMBS: usize,
    const COLUMNS: usize,
    const FROM: usize,
    const UNTIL: usize,
>(
    limbs: &mut [[u64; LIMBS]; LANES],
    words: &[[u32; WORDS]; LANES],
    table: &[[u32; COLUMNS]; WORDS],
) {
    for column in 0..COLUMNS {
        let mut totals = [0u64; LANES];
        for row in FROM..UNTIL {
            let factor = table[row][column] as u64;
            for lane in 0..LANES {
                totals[lane] += words[lane][row] as u64 * factor;
            }
        }
        for lane in 0..LANES {
            limbs[lane][column + 1] += totals[lane];
        }
    }
}
