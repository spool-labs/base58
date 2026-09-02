//! The aarch64 codec
//!
//! Two stages are worth vectorising. Turning digits into characters is a table
//! lookup over 58 entries, and a four-register lookup reaches 64, so the whole
//! alphabet fits one instruction. Reading characters back is the same lookup
//! inverted, which replaces a table load and a branch for every character.

use core::arch::aarch64::{
    uint8x16_t, vaddq_u64, vceqzq_u8, vdupq_n_u64, vdupq_n_u8, vget_lane_u64, vld1_u32, vld1q_u8, vld1q_u8_x4,
    vminq_u8, vminvq_u8, vmlal_n_u32, vorrq_u8, vqtbl4q_u8, vreinterpret_u64_u8,
    vreinterpretq_u16_u8, vshrn_n_u16, vst1q_u64, vst1q_u8, vsubq_u8,
};

use crate::scalar::ALPHABET;

/// The alphabet padded to what a four-register lookup indexes
#[repr(align(16))]
struct Table([u8; 64]);

/// Digits are below 58, so the entries past the alphabet are never reached
static FORWARD: Table = {
    let mut table = [0u8; 64];
    let mut at = 0;
    while at < 58 {
        table[at] = ALPHABET[at];
        at += 1;
    }
    Table(table)
};

/// One plus the digit, so that a zero from either half means the byte was invalid
static REVERSE_LOW: Table = reverse_half(0);
static REVERSE_HIGH: Table = reverse_half(64);

const fn reverse_half(offset: usize) -> Table {
    let mut table = [0u8; 64];
    let mut at = 0;
    while at < 58 {
        let byte = ALPHABET[at] as usize;
        if byte >= offset && byte < offset + 64 {
            table[byte - offset] = at as u8 + 1;
        }
        at += 1;
    }
    Table(table)
}

/// Turn digits into characters and write the ones the value needs
///
/// `FULL` whole registers are always written, and `TAIL` more end exactly on the
/// last character. The two runs overlap rather than reach past the encoding, so
/// nothing outside the written length is touched.
pub(crate) fn write_encoded<
    const PADDED: usize,
    const CHUNKS: usize,
    const FULL: usize,
    const TAIL: usize,
>(
    digits: &[u8; PADDED],
    count: usize,
    input_zeros: usize,
    out: &mut [u8],
) -> usize {
    let mut characters = [0u8; PADDED];
    // SAFETY: every load and store below stays inside `digits` and
    // `characters`, both of which are `PADDED` bytes and a multiple of a
    // register wide.
    unsafe {
        let table = vld1q_u8_x4(FORWARD.0.as_ptr());
        for chunk in 0..CHUNKS {
            let held = vld1q_u8(digits.as_ptr().add(chunk * 16));
            vst1q_u8(
                characters.as_mut_ptr().add(chunk * 16),
                vqtbl4q_u8(table, held),
            );
        }
    }

    let skip = leading_zero_digits::<CHUNKS>(digits).saturating_sub(input_zeros);
    let len = count - skip;

    // SAFETY: `skip` is below `count`, so the reads start inside the characters
    // and the last of them ends at `count`, which is inside `PADDED`. The stores
    // cover `out[..len]` and the buffer is at least that wide.
    unsafe {
        let from = characters.as_ptr().add(skip);
        let to = out.as_mut_ptr();
        for chunk in 0..FULL {
            vst1q_u8(to.add(chunk * 16), vld1q_u8(from.add(chunk * 16)));
        }
        for tail in (1..=TAIL).rev() {
            let at = len - 16 * tail;
            vst1q_u8(to.add(at), vld1q_u8(from.add(at)));
        }
    }
    len
}

/// Digits that are zero before the value starts
fn leading_zero_digits<const CHUNKS: usize>(digits: &[u8]) -> usize {
    for chunk in 0..CHUNKS {
        // SAFETY: the caller's buffer is `CHUNKS` whole registers wide.
        let bits = unsafe { zero_lanes(vld1q_u8(digits.as_ptr().add(chunk * 16))) };
        if bits != u64::MAX {
            return chunk * 16 + (bits.trailing_ones() as usize >> 2);
        }
    }
    CHUNKS * 16
}

/// Four bits per lane saying whether that byte was zero, in lane order
unsafe fn zero_lanes(bytes: uint8x16_t) -> u64 {
    // SAFETY: a shift and narrow over a register the caller owns.
    unsafe {
        let zeros = vceqzq_u8(bytes);
        vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(
            zeros,
        ))))
    }
}

/// Read characters as digits, right aligned, or report that one was not base58
///
/// Reads run in whole registers, so the last starts sixteen back from the end
/// rather than on a boundary, and the caller must pass at least that many
/// characters. A rejected input is handed back to the portable path, which finds
/// the character to name.
pub(crate) fn digits_from_encoded<const PADDED: usize>(
    encoded: &[u8],
    count: usize,
    digits: &mut [u8; PADDED],
) -> bool {
    let len = encoded.len();
    *digits = [0u8; PADDED];

    // SAFETY: every read is a whole register inside `encoded`, whose length is
    // at least sixteen. Writes start at the offset a short input implies and the
    // last ends at `count`, which is inside `PADDED`.
    let seen = unsafe {
        let source = encoded.as_ptr();
        let target = digits.as_mut_ptr().add(count - len);
        let mut lowest = vdupq_n_u8(0xFF);
        let mut at = 0;
        while at + 16 <= len {
            let mapped = map_chunk(vld1q_u8(source.add(at)));
            lowest = vminq_u8(lowest, mapped);
            vst1q_u8(target.add(at), vsubq_u8(mapped, vdupq_n_u8(1)));
            at += 16;
        }
        if at < len {
            let tail = len - 16;
            let mapped = map_chunk(vld1q_u8(source.add(tail)));
            lowest = vminq_u8(lowest, mapped);
            vst1q_u8(target.add(tail), vsubq_u8(mapped, vdupq_n_u8(1)));
        }
        vminvq_u8(lowest)
    };
    seen != 0
}

/// Sixteen characters to sixteen digits, or a zero lane where one was invalid
unsafe fn map_chunk(characters: uint8x16_t) -> uint8x16_t {
    // SAFETY: both lookups read their own static table, and an index outside a
    // half falls out of range and yields zero, so the two never collide.
    unsafe {
        let low = vqtbl4q_u8(vld1q_u8_x4(REVERSE_LOW.0.as_ptr()), characters);
        let high = vqtbl4q_u8(
            vld1q_u8_x4(REVERSE_HIGH.0.as_ptr()),
            vsubq_u8(characters, vdupq_n_u8(64)),
        );
        vorrq_u8(low, high)
    }
}

/// Multiply the limbs by the decode table, two products at a time
///
/// A limb is below the limb base and a table entry is a word, so each product
/// widens from 32 bits and one instruction does two of them and accumulates.
/// Two accumulator sets alternate by limb, which halves how deep the dependent
/// chain runs. Which set a limb lands in is a branch rather than a reference
/// picked between two arrays: a picked reference is a pointer the optimizer
/// cannot see through, and it puts both sets in memory.
pub(crate) fn words_from_limbs<const LIMBS: usize, const WORDS: usize, const PAIRS: usize>(
    limbs: &[u32; LIMBS],
    table: &[[u32; WORDS]; LIMBS],
    wide: &mut [u64; WORDS],
) {
    // SAFETY: a pair load reads two entries of a row that holds `WORDS` of
    // them, and every store covers a pair of the words the caller owns.
    unsafe {
        let mut even = [vdupq_n_u64(0); PAIRS];
        let mut odd = [vdupq_n_u64(0); PAIRS];
        for (at, row) in table.iter().enumerate() {
            let limb = limbs[at];
            let entries = row.as_ptr();
            match at & 1 {
                0 => {
                    for (pair, lane) in even.iter_mut().enumerate() {
                        *lane = vmlal_n_u32(*lane, vld1_u32(entries.add(pair * 2)), limb);
                    }
                }
                _ => {
                    for (pair, lane) in odd.iter_mut().enumerate() {
                        *lane = vmlal_n_u32(*lane, vld1_u32(entries.add(pair * 2)), limb);
                    }
                }
            }
        }
        for pair in 0..PAIRS {
            vst1q_u64(
                wide.as_mut_ptr().add(pair * 2),
                vaddq_u64(even[pair], odd[pair]),
            );
        }
    }
}
