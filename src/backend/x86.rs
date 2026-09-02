//! The codec on x86-64, chosen from what the machine reports it can run
//!
//! Which path runs each conversion is measured rather than assumed.

use crate::avx2;
use crate::avx512;
use crate::dispatch::{self, Path};
use crate::error::DecodeError;
use crate::scalar::{self, LIMBS_32, LIMBS_64, WORDS_32, WORDS_64};
use crate::tables::{DECODE_32, DECODE_64};
use crate::{MAX_ENCODED_32, MAX_ENCODED_64};

/// Encode a public key
///
/// Both vector paths run the product, the reduction and the digits in
/// registers together, and which one a wide machine should take depends on
/// its datapaths: at full width the wide encoder wins by 19% (Turin), at
/// half width the four-lane one wins by 14% (Genoa), so the wide arm asks.
pub(crate) fn encode_32(input: &[u8; 32], out: &mut [u8; MAX_ENCODED_32]) -> usize {
    // SAFETY: the path reported is the processor and the operating system both
    // agreeing to these instructions, and `out` is a whole slot. The vector
    // entries read and swap the words themselves: a `target_feature` call
    // does not inline, so words handed across it would be staged twice.
    match dispatch::path() {
        Path::Avx2 => unsafe { avx2::encode_32_entry(input, out) },
        Path::Wide => match dispatch::is_wide_half_width() {
            true => unsafe { avx2::encode_32_entry(input, out) },
            false => unsafe { avx512::encode_32_entry(input, out) },
        },
        Path::Portable => {
            let words = scalar::to_words::<32, WORDS_32>(input);
            let input_zeros = scalar::leading_zero_bytes(input);
            scalar::write_32(scalar::sum_32(&words), input_zeros, out)
        }
    }
}

/// Encode a signature
///
/// Both vector paths run this one in registers end to end. At eighteen limbs
/// the vector reduction pays for itself where at nine it does not.
pub(crate) fn encode_64(input: &[u8; 64], out: &mut [u8; MAX_ENCODED_64]) -> usize {
    let words = scalar::to_words::<64, WORDS_64>(input);
    let input_zeros = scalar::leading_zero_bytes(input);
    // SAFETY: as above.
    match dispatch::path() {
        Path::Wide => unsafe { avx512::encode_64(&words, input_zeros, out) },
        Path::Avx2 => unsafe { avx2::encode_64(&words, input_zeros, out) },
        Path::Portable => scalar::write_64(scalar::sum_64(&words), input_zeros, out),
    }
}

/// Whether a key decodes better alone than interleaved with its neighbours
pub(crate) fn is_decode_32_direct() -> bool {
    false
}

/// Whether a signature decodes better alone than interleaved with its neighbours
pub(crate) fn is_decode_64_direct() -> bool {
    false
}

/// Whether a key encodes better alone than interleaved with its neighbours
pub(crate) fn is_encode_32_direct() -> bool {
    false
}

/// Whether a signature encodes better alone than interleaved with its neighbours
///
/// The vector encoders run the whole conversion behind one `target_feature`
/// call, and a batch that interleaved first would re-enter that call per lane
/// with the limbs staged through memory.
pub(crate) fn is_encode_64_direct() -> bool {
    !matches!(dispatch::path(), Path::Portable)
}

/// Spell a public key's limbs, as [`scalar::sum_32`] leaves them, into characters
///
/// Portable on every path, including AVX2, and that is measured rather than
/// left over: routing the batch's lanes through the vector spell cost 31%,
/// because a `target_feature` call per lane neither inlines nor lets the
/// scalar chains of neighbouring lanes overlap the way the inlined walk does.
#[inline]
pub(crate) fn write_32(limbs: [u64; LIMBS_32], input_zeros: usize, out: &mut [u8]) -> usize {
    scalar::write_32(limbs, input_zeros, out)
}

/// Spell a signature's limbs, as [`scalar::sum_64`] leaves them, into characters
///
/// Portable for the same reason [`write_32`] is, and reached only from the
/// portable batch: a vector path encodes its signatures one at a time.
#[inline]
pub(crate) fn write_64(limbs: [u64; LIMBS_64], input_zeros: usize, out: &mut [u8]) -> usize {
    scalar::write_64(limbs, input_zeros, out)
}

pub(crate) fn decode_32(encoded: &[u8], out: &mut [u8; 32]) -> Result<(), DecodeError> {
    let mut limbs = [0u32; LIMBS_32];
    read_32(encoded, &mut limbs)?;
    let mut wide = [0u64; WORDS_32];
    // SAFETY: gated on what the machine reported.
    match dispatch::path() {
        Path::Wide | Path::Avx2 => unsafe { avx2::words_from_limbs_32(&limbs, &mut wide) },
        Path::Portable => scalar::accumulate_words(&limbs, &DECODE_32, &mut wide),
    }
    let words = scalar::settle_words(&mut wide)?;
    scalar::from_words::<32, WORDS_32>(&words, out);
    scalar::check_leading_ones(out, encoded)
}

pub(crate) fn decode_64(encoded: &[u8], out: &mut [u8; 64]) -> Result<(), DecodeError> {
    let mut limbs = [0u32; LIMBS_64];
    read_64(encoded, &mut limbs)?;
    let mut wide = [0u64; WORDS_64];
    words_from_limbs_64(&limbs, &mut wide);
    let words = scalar::settle_words(&mut wide)?;
    scalar::from_words::<64, WORDS_64>(&words, out);
    scalar::check_leading_ones(out, encoded)
}

/// Read a public key's encoding into the limbs it stands for
///
/// A machine without the vectors reads it portably, as does one handed an
/// encoding the vector reader has no room for. So does one the reader finds a
/// character outside the alphabet in: naming that character is the portable
/// reader's work.
#[inline(always)]
pub(crate) fn read_32(encoded: &[u8], limbs: &mut [u32; LIMBS_32]) -> Result<(), DecodeError> {
    // SAFETY: each path is gated on what the machine reported and on a length
    // the reader has room for.
    let is_read = match dispatch::path() {
        Path::Wide | Path::Avx2 => match avx2::fits_32(encoded.len()) {
            true => unsafe { avx2::read_32(encoded, limbs) },
            false => false,
        },
        Path::Portable => false,
    };
    match is_read {
        true => Ok(()),
        false => scalar::read_32(encoded, limbs),
    }
}

/// Read a signature's encoding into the limbs it stands for
///
/// The four-lane reader on both vector paths: level with the wide one at
/// full width (Turin, and the Zen 5 desktop before it) and 16% ahead at
/// half width (Genoa), so nothing routes to the wide reader any more.
#[inline(always)]
pub(crate) fn read_64(encoded: &[u8], limbs: &mut [u32; LIMBS_64]) -> Result<(), DecodeError> {
    // SAFETY: as above.
    let is_read = match dispatch::path() {
        Path::Wide | Path::Avx2 => match avx2::fits_64(encoded.len()) {
            true => unsafe { avx2::read_64(encoded, limbs) },
            false => false,
        },
        Path::Portable => false,
    };
    match is_read {
        true => Ok(()),
        false => scalar::read_64(encoded, limbs),
    }
}

/// Sum several keys' limbs against the decode table
///
/// The vector product is already a whole register wide for one input, so the
/// lanes go through it one at a time. Without one, they interleave instead,
/// which is what keeps the multiplier busy on the portable path.
pub(crate) fn words_32_lanes<const LANES: usize>(
    limbs: &[[u32; LIMBS_32]; LANES],
    wide: &mut [[u64; WORDS_32]; LANES],
) {
    match dispatch::path() {
        Path::Portable => scalar::words_lanes(limbs, &DECODE_32, wide),
        Path::Wide | Path::Avx2 => {
            for (lane, slot) in wide.iter_mut().enumerate() {
                unsafe { avx2::words_from_limbs_32(&limbs[lane], slot) }
            }
        }
    }
}

/// Sum several signatures' limbs against the decode table
pub(crate) fn words_64_lanes<const LANES: usize>(
    limbs: &[[u32; LIMBS_64]; LANES],
    wide: &mut [[u64; WORDS_64]; LANES],
) {
    match dispatch::path() {
        Path::Portable => scalar::words_lanes(limbs, &DECODE_64, wide),
        Path::Wide | Path::Avx2 => {
            for (lane, slot) in wide.iter_mut().enumerate() {
                words_from_limbs_64(&limbs[lane], slot);
            }
        }
    }
}

/// Sum one signature's limbs against the decode table
///
/// Four lanes on both vector paths; see [`read_64`] for the measurements.
fn words_from_limbs_64(limbs: &[u32; LIMBS_64], wide: &mut [u64; WORDS_64]) {
    // SAFETY: gated on what the machine reported.
    match dispatch::path() {
        Path::Wide | Path::Avx2 => unsafe { avx2::words_from_limbs_64(limbs, wide) },
        Path::Portable => scalar::accumulate_words(limbs, &DECODE_64, wide),
    }
}
