//! Tables and constants the two x86 paths share
//!
//! `vpmuludq` reads the low 32 bits of each 64-bit lane, so the generated
//! tables are widened to u64 at compile time rather than converted per use.

use crate::scalar::LIMB_BASE;
use crate::tables::{DECODE_32, DECODE_64};

/// A table laid out on a cache line, which the aligned loads require
#[repr(align(64))]
pub(crate) struct Aligned<Table>(pub(crate) Table);

/// Widen every entry to its own 64-bit lane
pub(crate) const fn widen<const ROWS: usize, const COLUMNS: usize, const WIDE: usize>(
    source: &[[u32; COLUMNS]; ROWS],
) -> [[u64; WIDE]; ROWS] {
    let kept = match COLUMNS < WIDE {
        true => COLUMNS,
        false => WIDE,
    };
    let mut widened = [[0u64; WIDE]; ROWS];
    let mut row = 0;
    while row < ROWS {
        let mut column = 0;
        while column < kept {
            widened[row][column] = source[row][column] as u64;
            column += 1;
        }
        row += 1;
    }
    widened
}

/// Widen and shift one lane, so lane `n` of an accumulator is limb `n` itself
///
/// Limb zero takes nothing from the product, and limb `column + 1` takes the
/// row's column, which is the offset the scalar accumulation applies by hand.
pub(crate) const fn widen_shift<const ROWS: usize, const COLUMNS: usize, const WIDE: usize>(
    source: &[[u32; COLUMNS]; ROWS],
) -> [[u64; WIDE]; ROWS] {
    let kept = match COLUMNS < WIDE - 1 {
        true => COLUMNS,
        false => WIDE - 1,
    };
    let mut widened = [[0u64; WIDE]; ROWS];
    let mut row = 0;
    while row < ROWS {
        let mut column = 0;
        while column < kept {
            widened[row][column + 1] = source[row][column] as u64;
            column += 1;
        }
        row += 1;
    }
    widened
}

/// The 32-byte encode table for the wide path, lane `n` holding limb `n`
pub(crate) static ENCODE_32_WIDE: Aligned<[[u64; 8]; 8]> =
    Aligned(widen_shift(&crate::tables::ENCODE_32));

/// The 32-byte table's ninth column, which no register lane covers
pub(crate) static ENCODE_32_TAIL: Aligned<[u64; 8]> = tail_column();

const fn tail_column() -> Aligned<[u64; 8]> {
    let mut column = [0u64; 8];
    let mut row = 0;
    while row < 8 {
        column[row] = crate::tables::ENCODE_32[row][7] as u64;
        row += 1;
    }
    Aligned(column)
}

pub(crate) static DECODE_32_WIDE: Aligned<[[u64; 8]; 9]> = Aligned(widen(&DECODE_32));
pub(crate) static DECODE_64_WIDE: Aligned<[[u64; 16]; 18]> = Aligned(widen(&DECODE_64));

/// Bytes 3, 2, 1, 0 of each doubleword, the byte swap `to_words` runs
pub(crate) static FLIP_WORDS: Aligned<[u8; 32]> = Aligned(flip_words());

const fn flip_words() -> [u8; 32] {
    let mut lanes = [0u8; 32];
    let mut at = 0;
    while at < 32 {
        lanes[at] = ((at + 3 - 2 * (at % 4)) % 16) as u8;
        at += 1;
    }
    lanes
}

/// The limb base with its five factors of two removed
///
/// `floor(x / 58^5)` is `floor((x >> 5) / LIMB_BASE_ODD)`, and the odd divisor
/// is what makes a reciprocal that fits the lanes.
const LIMB_BASE_ODD: u64 = LIMB_BASE >> 5;

/// `ceil(2^88 / LIMB_BASE_ODD)`, split at bit 32
const MAGIC: u64 = 15088623744157144426;

pub(crate) const MAGIC_LOW: i64 = (MAGIC & 0xFFFF_FFFF) as i64;
pub(crate) const MAGIC_HIGH: i64 = (MAGIC >> 32) as i64;

/// `ceil(2^55 / LIMB_BASE_ODD)`, exact for anything below 2^35
pub(crate) const MAGIC_SMALL: i64 = 1756546990;

/// Both reciprocals round up by at most the divisor, which is what makes the
/// estimates in `divide_base_*` undershoot rather than overshoot.
const _: () = {
    let overshoot = MAGIC as u128 * LIMB_BASE_ODD as u128 - (1u128 << 88);
    assert!(LIMB_BASE_ODD << 5 == LIMB_BASE);
    assert!(overshoot > 0 && overshoot <= LIMB_BASE_ODD as u128);
    let small = MAGIC_SMALL as u128 * LIMB_BASE_ODD as u128 - (1u128 << 55);
    assert!(small > 0 && small <= LIMB_BASE_ODD as u128);
};

/// Reciprocal of 58, exact for anything below 58^5
pub(crate) const MAGIC_58: i64 = 2369637129;

/// Reciprocal of 58 squared, exact over the same range
pub(crate) const MAGIC_3364: i64 = 1307386003;

/// Two digits per 16-bit lane against `[58, 1]`
pub(crate) const PAIR_58: i32 = 0x013A_013A;

/// Two pairs per 32-bit lane against `[3364, 1]`
pub(crate) const PAIR_3364: i32 = 0x0001_0D24;

/// What a limb's leading digit is worth
pub(crate) const HEAD_WEIGHT: i32 = 11_316_496;
