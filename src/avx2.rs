//! The x86-64 codec for machines with AVX2 but no wide byte permute
//!
//! `vpshufb` reaches sixteen entries within a 128-bit lane, so the alphabet
//! cannot be one lookup the way it is under VBMI or NEON: characters come
//! from a chain of compares and digits are compacted with half-register
//! shifts. The products are still written out in `vpmuludq` against widened
//! tables, and the carry reduction still runs every limb at once.

use core::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi32, _mm256_add_epi64, _mm256_add_epi8, _mm256_alignr_epi8,
    _mm256_and_si256, _mm256_blend_epi32, _mm256_castsi256_si128, _mm256_cmpeq_epi64,
    _mm256_cmpeq_epi8, _mm256_cmpgt_epi64, _mm256_cmpgt_epi8, _mm256_cvtepu32_epi64,
    _mm256_extract_epi64, _mm256_extractf128_si256, _mm256_insert_epi64, _mm256_load_si256,
    _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_maskstore_epi32,
    _mm256_maskstore_epi64, _mm256_movemask_epi8, _mm256_mul_epu32, _mm256_mullo_epi32,
    _mm256_or_si256, _mm256_permute2x128_si256, _mm256_permute4x64_epi64,
    _mm256_permutevar8x32_epi32, _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_set1_epi8,
    _mm256_set_m128i, _mm256_setr_epi32, _mm256_setr_epi64x, _mm256_setr_epi8,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_slli_epi64, _mm256_slli_si256,
    _mm256_srli_epi16, _mm256_srli_epi64, _mm256_srlv_epi64, _mm256_storeu_si256, _mm256_sub_epi64,
    _mm256_sub_epi8, _mm256_zextsi128_si256, _mm_add_epi8, _mm_alignr_epi8, _mm_and_si128,
    _mm_bslli_si128, _mm_cmpeq_epi8, _mm_cmpgt_epi8, _mm_cvtsi64_si128, _mm_load_si128,
    _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi8, _mm_setzero_si128,
    _mm_shuffle_epi8, _mm_slli_si128, _mm_srli_si128, _mm_storeu_si128, _mm_sub_epi8,
};

use crate::scalar::{
    ALPHABET, DIGITS_32, DIGITS_64, DIGITS_PER_LIMB, LIMBS_32, LIMBS_64, LIMB_BASE, WORDS_32,
    WORDS_64,
};
use crate::tables::ENCODE_64;
use crate::wide::{
    skip_32, skip_64, widen_shift, Aligned, DECODE_32_WIDE, DECODE_64_WIDE, ENCODE_32_TAIL,
    ENCODE_32_WIDE, FLIP_WORDS, HEAD_WEIGHT, MAGIC_3364, MAGIC_58, MAGIC_HIGH, MAGIC_LOW,
    MAGIC_SMALL, MIN_ENCODED_32, MIN_ENCODED_64, PAIR_3364, PAIR_58,
};
use crate::{MAX_ENCODED_32, MAX_ENCODED_64};

/// Lanes one register holds
const LANES: usize = 4;

/// Limb slots the 64-byte encode's five registers cover
const SLOTS_64: usize = 20;

/// The 64-byte encode table, lane `n` of the accumulators holding limb `n`
static ENCODE_64_SHIFTED: Aligned<[[u64; SLOTS_64]; 16]> = Aligned(widen_shift(&ENCODE_64));

/// `floor(x / 58^5)`, within two and never over, for any lane value
#[target_feature(enable = "avx2")]
unsafe fn divide_base_under(terms: __m256i) -> __m256i {
    let odd = _mm256_srli_epi64::<5>(terms);
    let high = _mm256_srli_epi64::<32>(odd);
    let low_high = _mm256_mul_epu32(odd, _mm256_set1_epi64x(MAGIC_HIGH));
    let high_low = _mm256_mul_epu32(high, _mm256_set1_epi64x(MAGIC_LOW));
    let high_high = _mm256_mul_epu32(high, _mm256_set1_epi64x(MAGIC_HIGH));
    let middle = _mm256_add_epi64(low_high, high_low);
    _mm256_add_epi64(
        _mm256_srli_epi64::<24>(high_high),
        _mm256_srli_epi64::<56>(middle),
    )
}

/// `floor(x / 58^5)` exactly, for lane values below 2^35
#[target_feature(enable = "avx2")]
unsafe fn divide_base_small(terms: __m256i) -> __m256i {
    _mm256_srli_epi64::<55>(_mm256_mul_epu32(
        _mm256_srli_epi64::<5>(terms),
        _mm256_set1_epi64x(MAGIC_SMALL),
    ))
}

/// `quotient * 58^5` for quotients that can exceed 32 bits
#[target_feature(enable = "avx2")]
unsafe fn multiply_base_wide(quotients: __m256i) -> __m256i {
    let base = _mm256_set1_epi64x(LIMB_BASE as i64);
    _mm256_add_epi64(
        _mm256_mul_epu32(quotients, base),
        _mm256_slli_epi64::<32>(_mm256_mul_epu32(_mm256_srli_epi64::<32>(quotients), base)),
    )
}

/// Each quotient one lane down, the next register's first lane entering on top
#[target_feature(enable = "avx2")]
unsafe fn shift_lane_down(current: __m256i, next: __m256i) -> __m256i {
    let rotated = _mm256_permute4x64_epi64::<0b11_11_10_01>(current);
    let first = _mm256_permute4x64_epi64::<0b00_00_00_00>(next);
    _mm256_blend_epi32::<0b1100_0000>(rotated, first)
}

/// Bring every limb below the limb base, carrying quotients left
///
/// The four-lane form of the reduction: every limb divides at once, the
/// quotients shift a lane down, a second cheaper pass catches the 35-bit
/// carries, and compare-and-carry rounds finish the stragglers.
#[target_feature(enable = "avx2")]
unsafe fn settle<const REGISTERS: usize>(terms: &mut [__m256i; REGISTERS]) {
    // SAFETY: lane arithmetic on registers the caller owns.
    unsafe {
        let zero = _mm256_setzero_si256();
        let base = _mm256_set1_epi64x(LIMB_BASE as i64);
        let mut quotients = [zero; REGISTERS];

        for at in 0..REGISTERS {
            quotients[at] = divide_base_under(terms[at]);
        }
        for at in 0..REGISTERS {
            let next = match at + 1 < REGISTERS {
                true => quotients[at + 1],
                false => zero,
            };
            terms[at] = _mm256_add_epi64(
                _mm256_sub_epi64(terms[at], multiply_base_wide(quotients[at])),
                shift_lane_down(quotients[at], next),
            );
        }

        for at in 0..REGISTERS {
            quotients[at] = divide_base_small(terms[at]);
        }
        for at in 0..REGISTERS {
            let next = match at + 1 < REGISTERS {
                true => quotients[at + 1],
                false => zero,
            };
            terms[at] = _mm256_add_epi64(
                _mm256_sub_epi64(terms[at], _mm256_mul_epu32(quotients[at], base)),
                shift_lane_down(quotients[at], next),
            );
        }

        // Values stay below 2^63 here, so the signed compare reads as unsigned.
        let highest = _mm256_set1_epi64x((LIMB_BASE - 1) as i64);
        loop {
            let mut over = [zero; REGISTERS];
            let mut any = zero;
            for at in 0..REGISTERS {
                over[at] = _mm256_cmpgt_epi64(terms[at], highest);
                any = _mm256_or_si256(any, over[at]);
            }
            if _mm256_movemask_epi8(any) == 0 {
                break;
            }
            for at in 0..REGISTERS {
                quotients[at] = _mm256_srli_epi64::<63>(over[at]);
            }
            for at in 0..REGISTERS {
                let next = match at + 1 < REGISTERS {
                    true => quotients[at + 1],
                    false => zero,
                };
                terms[at] = _mm256_add_epi64(
                    _mm256_sub_epi64(terms[at], _mm256_and_si256(over[at], base)),
                    shift_lane_down(quotients[at], next),
                );
            }
        }
    }
}

/// One table row onto the accumulators from register `START` up
///
/// `START` is a constant so both loops unroll and the accumulators stay in
/// registers; a data-dependent start put the whole array on the stack.
#[target_feature(enable = "avx2")]
unsafe fn encode_row<const START: usize, const REGISTERS: usize>(
    totals: &mut [__m256i; REGISTERS],
    word: u32,
    row: *const u64,
) {
    // SAFETY: the row is a whole aligned register of the widened table.
    unsafe {
        let broadcast = _mm256_set1_epi64x(word as i64);
        for (at, slot) in totals.iter_mut().enumerate().skip(START) {
            *slot = _mm256_add_epi64(
                *slot,
                _mm256_mul_epu32(
                    broadcast,
                    _mm256_load_si256(row.add(LANES * at) as *const __m256i),
                ),
            );
        }
    }
}

/// A signature's words to its settled limbs, without touching memory
#[target_feature(enable = "avx2")]
unsafe fn sum_and_settle_64(words: &[u32; WORDS_64]) -> [__m256i; 5] {
    // SAFETY: every row is a whole aligned span of the widened table.
    unsafe {
        let mut totals = [_mm256_setzero_si256(); 5];
        let row = |at: usize| ENCODE_64_SHIFTED.0[at].as_ptr();

        encode_row::<0, 5>(&mut totals, words[0], row(0));
        encode_row::<0, 5>(&mut totals, words[1], row(1));
        encode_row::<0, 5>(&mut totals, words[2], row(2));
        encode_row::<1, 5>(&mut totals, words[3], row(3));
        encode_row::<1, 5>(&mut totals, words[4], row(4));
        encode_row::<1, 5>(&mut totals, words[5], row(5));
        encode_row::<1, 5>(&mut totals, words[6], row(6));
        encode_row::<2, 5>(&mut totals, words[7], row(7));

        // Limbs 15 and 16 would overflow if the second half of the rows landed
        // on them untouched, so they shed once here.
        let mut limb_15 = _mm256_extract_epi64::<3>(totals[3]) as u64;
        let mut limb_16 = _mm256_extract_epi64::<0>(totals[4]) as u64;
        limb_15 += limb_16 / LIMB_BASE;
        limb_16 %= LIMB_BASE;
        totals[3] = _mm256_insert_epi64::<3>(totals[3], limb_15 as i64);
        totals[4] = _mm256_insert_epi64::<0>(totals[4], limb_16 as i64);

        encode_row::<2, 5>(&mut totals, words[8], row(8));
        encode_row::<2, 5>(&mut totals, words[9], row(9));
        encode_row::<2, 5>(&mut totals, words[10], row(10));
        encode_row::<3, 5>(&mut totals, words[11], row(11));
        encode_row::<3, 5>(&mut totals, words[12], row(12));
        encode_row::<3, 5>(&mut totals, words[13], row(13));
        encode_row::<3, 5>(&mut totals, words[14], row(14));
        encode_row::<4, 5>(&mut totals, words[15], row(15));

        settle(&mut totals);
        totals
    }
}

/// Four limbs to their five digits each, ten bytes per 128-bit half
///
/// `floor(x / 58^k) - 58 * floor(x / 58^(k+1))` per digit, so one division
/// feeds the next. Crossing the halves is expensive here, so the digits land
/// in an awkward layout that the packing undoes.
#[target_feature(enable = "avx2")]
unsafe fn to_digits(limbs: __m256i) -> __m256i {
    let by_58 = _mm256_set1_epi64x(MAGIC_58);
    let by_3364 = _mm256_set1_epi64x(MAGIC_3364);
    let fifty_eight = _mm256_set1_epi64x(58);
    let divide_3364 = |value: __m256i| {
        _mm256_srli_epi64::<40>(_mm256_mul_epu32(_mm256_srli_epi64::<2>(value), by_3364))
    };
    let first = _mm256_srli_epi64::<37>(_mm256_mul_epu32(limbs, by_58));
    let fifth = _mm256_sub_epi64(limbs, _mm256_mul_epu32(first, fifty_eight));
    let second = divide_3364(limbs);
    let fourth = _mm256_sub_epi64(first, _mm256_mul_epu32(second, fifty_eight));
    let third_divided = divide_3364(first);
    let third = _mm256_sub_epi64(second, _mm256_mul_epu32(third_divided, fifty_eight));
    let fourth_divided = divide_3364(second);
    let second_digit =
        _mm256_sub_epi64(third_divided, _mm256_mul_epu32(fourth_divided, fifty_eight));
    let first_digit = fourth_divided;

    // Byte zero of each lane to byte 0 and 5 of each half, so the five
    // shifted copies or together into ten contiguous bytes.
    let gather = _mm256_setr_epi8(
        0, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //
        0, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    );
    let at0 = _mm256_shuffle_epi8(first_digit, gather);
    let at1 = _mm256_slli_si256(_mm256_shuffle_epi8(second_digit, gather), 1);
    let at2 = _mm256_slli_si256(_mm256_shuffle_epi8(third, gather), 2);
    let at3 = _mm256_slli_si256(_mm256_shuffle_epi8(fourth, gather), 3);
    let at4 = _mm256_slli_si256(_mm256_shuffle_epi8(fifth, gather), 4);
    _mm256_or_si256(
        _mm256_or_si256(_mm256_or_si256(at0, at1), _mm256_or_si256(at2, at3)),
        at4,
    )
}

/// Digits to characters, as the compare chain the alphabet's gaps describe
///
/// `char = '1' + digit + 7[d>8] + [d>16] + [d>21] + 6[d>32] + [d>43]`, each
/// bracket a run the alphabet skips. A compare yields `0xFF` for true, so
/// adding one is subtracting the compare.
#[target_feature(enable = "avx2")]
unsafe fn to_chars(digits: __m256i) -> __m256i {
    let past_8 = _mm256_cmpgt_epi8(digits, _mm256_set1_epi8(8));
    let past_16 = _mm256_cmpgt_epi8(digits, _mm256_set1_epi8(16));
    let past_21 = _mm256_cmpgt_epi8(digits, _mm256_set1_epi8(21));
    let past_32 = _mm256_cmpgt_epi8(digits, _mm256_set1_epi8(32));
    let past_43 = _mm256_cmpgt_epi8(digits, _mm256_set1_epi8(43));
    let seven = _mm256_and_si256(past_8, _mm256_set1_epi8(-7));
    let six = _mm256_and_si256(past_32, _mm256_set1_epi8(-6));
    let total = _mm256_add_epi8(
        _mm256_add_epi8(
            _mm256_add_epi8(_mm256_set1_epi8(-(ALPHABET[0] as i8)), past_16),
            _mm256_add_epi8(past_21, past_43),
        ),
        _mm256_add_epi8(seven, six),
    );
    _mm256_sub_epi8(digits, total)
}

/// The 90 digits of a signature, ten per half, packed contiguous
#[target_feature(enable = "avx2")]
unsafe fn pack_90(
    first: __m256i,
    second: __m256i,
    third: __m256i,
    fourth: __m256i,
    fifth: __m256i,
) -> (__m256i, __m256i, __m256i) {
    let low0 = _mm256_extractf128_si256::<0>(first);
    let high0 = _mm256_extractf128_si256::<1>(first);
    let low1 = _mm256_extractf128_si256::<0>(second);
    let high1 = _mm256_extractf128_si256::<1>(second);
    let low2 = _mm256_extractf128_si256::<0>(third);
    let high2 = _mm256_extractf128_si256::<1>(third);
    let low3 = _mm256_extractf128_si256::<0>(fourth);
    let high3 = _mm256_extractf128_si256::<1>(fourth);
    let low4 = _mm256_extractf128_si256::<0>(fifth);

    let out0 = _mm_or_si128(low0, _mm_slli_si128::<10>(high0));
    let out1 = _mm_or_si128(
        _mm_or_si128(_mm_srli_si128::<6>(high0), _mm_slli_si128::<4>(low1)),
        _mm_slli_si128::<14>(high1),
    );
    let out2 = _mm_or_si128(_mm_srli_si128::<2>(high1), _mm_slli_si128::<8>(low2));
    let out3 = _mm_or_si128(
        _mm_or_si128(_mm_srli_si128::<8>(low2), _mm_slli_si128::<2>(high2)),
        _mm_slli_si128::<12>(low3),
    );
    let out4 = _mm_or_si128(_mm_srli_si128::<4>(low3), _mm_slli_si128::<6>(high3));
    (
        _mm256_set_m128i(out1, out0),
        _mm256_set_m128i(out3, out2),
        _mm256_set_m128i(low4, out4),
    )
}

/// A mask with the low `bits` set, and zero once there are none to set
const fn low_mask(bits: usize) -> u64 {
    match bits >= 64 {
        true => u64::MAX,
        false => (1u64 << bits) - 1,
    }
}

/// One bit per byte of the register, set where the byte is zero
#[target_feature(enable = "avx2")]
unsafe fn zero_bytes(digits: __m256i) -> u64 {
    // SAFETY: a compare and a move mask over a register the caller owns.
    _mm256_movemask_epi8(_mm256_cmpeq_epi8(digits, _mm256_setzero_si256())) as u32 as u64
}

/// Zero bytes before the first that is not, over the low `COUNT` of a register
///
/// Complementing puts a one on the first byte that is not zero, and the bit
/// just above the counted ones stops the scan at `COUNT` without a branch.
#[target_feature(enable = "avx2")]
unsafe fn leading_zero_digits<const COUNT: usize>(digits: __m256i) -> usize {
    // SAFETY: a compare and a move mask over a register the caller owns.
    unsafe {
        let stopped = low_mask(COUNT + 1) ^ (zero_bytes(digits) & low_mask(COUNT));
        stopped.trailing_zeros() as usize
    }
}

/// The same over a register pair, the second holding the top `HIGH` bytes
#[target_feature(enable = "avx2")]
unsafe fn leading_zero_digits_pair<const HIGH: usize>(low: __m256i, high: __m256i) -> usize {
    // SAFETY: compares and move masks over registers the caller owns.
    unsafe {
        let combined = ((zero_bytes(high) & low_mask(HIGH)) << 32) | zero_bytes(low);
        let total = 32 + HIGH;
        match total >= 64 {
            true => match !combined == 0 {
                true => total,
                false => (!combined).trailing_zeros() as usize,
            },
            false => (low_mask(total + 1) ^ combined).trailing_zeros() as usize,
        }
    }
}

/// Encode a signature, from its words to the string, all in registers
///
/// # Safety
///
/// As [`write_32`], with `out` at least `MAX_ENCODED_64` bytes.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn encode_64(
    words: &[u32; WORDS_64],
    input_zeros: usize,
    out: &mut [u8],
) -> usize {
    // SAFETY: `out` holds a whole slot, which the caller has checked.
    unsafe { spell_90(sum_and_settle_64(words), input_zeros, out) }
}

/// A signature's settled limbs to the characters they spell
#[target_feature(enable = "avx2")]
unsafe fn spell_90(terms: [__m256i; 5], input_zeros: usize, out: &mut [u8]) -> usize {
    // SAFETY: every store lands inside `out`; see the note on `store_90`.
    unsafe {
        let (packed_low, packed_middle, packed_high) = pack_90(
            to_digits(terms[0]),
            to_digits(terms[1]),
            to_digits(terms[2]),
            to_digits(terms[3]),
            to_digits(terms[4]),
        );
        let low = leading_zero_digits_pair::<32>(packed_low, packed_middle);
        let leading = match low < 64 {
            true => low,
            false => 64 + leading_zero_digits::<{ DIGITS_64 - 64 }>(packed_high),
        };

        let chars_low = to_chars(packed_low);
        let chars_middle = to_chars(packed_middle);
        let chars_high = to_chars(packed_high);
        let skip = skip_64(leading, input_zeros);
        store_90(out.as_mut_ptr(), chars_low, chars_middle, chars_high, skip);
        DIGITS_64 - skip
    }
}

/// Write 90 characters starting `skip` in, which is between 2 and 26
#[target_feature(enable = "avx2")]
unsafe fn store_90(out: *mut u8, low: __m256i, middle: __m256i, high: __m256i, skip: usize) {
    // SAFETY: skip is in [2, 26] because the encoding is 64 to 88 characters,
    // and every store below ends at or before `out + 88`.
    unsafe {
        let lanes = _mm256_setr_epi64x(0, 1, 2, 3);
        let whole = _mm256_set1_epi64x((skip / 8) as i64);
        let shifted = _mm256_srlv_epi64(
            low,
            _mm256_slli_epi64::<3>(_mm256_set1_epi64x((skip % 8) as i64)),
        );
        _mm256_maskstore_epi64(
            out.offset(-8 * (skip as isize / 8)) as *mut i64,
            _mm256_cmpeq_epi64(whole, lanes),
            shifted,
        );
        _mm256_maskstore_epi64(
            out.offset(-(skip as isize)) as *mut i64,
            _mm256_cmpgt_epi64(lanes, whole),
            low,
        );
        _mm256_storeu_si256(out.offset(32 - skip as isize) as *mut __m256i, middle);

        let tail = _mm_bslli_si128::<6>(_mm256_extractf128_si256::<1>(high));
        _mm_storeu_si128(out.offset(74 - skip as isize) as *mut __m128i, tail);
        _mm_storeu_si128(
            out.offset(64 - skip as isize) as *mut __m128i,
            _mm256_extractf128_si256::<0>(high),
        );
    }
}

/// Five digits per 64-bit lane down to ten contiguous bytes per half
static PACK_TEN: Aligned<[u8; 32]> = Aligned(pack_ten());

const fn pack_ten() -> [u8; 32] {
    let mut lanes = [0x80u8; 32];
    let mut at = 0;
    while at < 10 {
        let index = (8 * (at / DIGITS_PER_LIMB) + at % DIGITS_PER_LIMB) as u8;
        lanes[at] = index;
        lanes[at + 16] = index;
        at += 1;
    }
    lanes
}

/// For each skip, the two shuffles that slide a 32-byte window down by it
///
/// Byte `at` of the window is `chars[skip + at]`: the first shuffle reads it
/// from the register below when `skip + at` is inside it, the second from the
/// register above when it is not, and the high bit parks the other one.
static ALIGN_SKIP: Aligned<[[u8; 32]; 16]> = Aligned(skip_masks());

const fn skip_masks() -> [[u8; 32]; 16] {
    let mut table = [[0u8; 32]; 16];
    let mut skip = 0;
    while skip < 16 {
        let mut at = 0;
        while at < 16 {
            let from = skip + at;
            table[skip][at] = match from < 16 {
                true => from as u8,
                false => 0x80,
            };
            table[skip][at + 16] = match from >= 16 {
                true => (from - 16) as u8,
                false => 0x80,
            };
            at += 1;
        }
        skip += 1;
    }
    table
}

/// Each quotient one lane down, the next register's first lane entering on top
///
/// Two instructions where [`shift_lane_down`] takes three: the halves that
/// cross are one `vperm2i128`, and byte eight of each half is where the
/// shifted quads begin.
#[inline(always)]
unsafe fn shift_lane_down_pair(current: __m256i, next: __m256i) -> __m256i {
    // SAFETY: lane arithmetic on registers the caller owns.
    unsafe {
        let crossed = _mm256_permute2x128_si256::<0x21>(current, next);
        _mm256_alignr_epi8::<8>(crossed, current)
    }
}

/// A key's words summed into limbs 0..8 across two registers, limb 8 on its own
///
/// The table is upper triangular twice over: words three on reach only the
/// second register, and the last word's whole contribution is the tail limb.
/// The tail sums in a general register, off the vector ports, and the last
/// row of the widened table is all zeros so it is skipped rather than read.
#[inline(always)]
// The index pairs a word with its table row and carries the triangular
// structure the comments describe, so it stays.
#[allow(clippy::needless_range_loop)]
unsafe fn sum_32(words: &[u32; WORDS_32]) -> (__m256i, __m256i, u64) {
    // SAFETY: every row is a whole aligned span of the widened table, and
    // `vpmuludq` reads the word from the low half of each lane the broadcast
    // fills.
    unsafe {
        let mut low = _mm256_setzero_si256();
        let mut high = _mm256_setzero_si256();
        for at in 0..3 {
            let broadcast = _mm256_set1_epi32(words[at] as i32);
            let row = ENCODE_32_WIDE.0[at].as_ptr();
            low = _mm256_add_epi64(
                low,
                _mm256_mul_epu32(broadcast, _mm256_load_si256(row as *const __m256i)),
            );
            high = _mm256_add_epi64(
                high,
                _mm256_mul_epu32(
                    broadcast,
                    _mm256_load_si256(row.add(LANES) as *const __m256i),
                ),
            );
        }
        for at in 3..WORDS_32 - 1 {
            let broadcast = _mm256_set1_epi32(words[at] as i32);
            let row = ENCODE_32_WIDE.0[at].as_ptr();
            high = _mm256_add_epi64(
                high,
                _mm256_mul_epu32(
                    broadcast,
                    _mm256_load_si256(row.add(LANES) as *const __m256i),
                ),
            );
        }
        let mut tail = words[WORDS_32 - 1] as u64;
        for at in 0..WORDS_32 - 1 {
            tail += words[at] as u64 * ENCODE_32_TAIL.0[at];
        }
        (low, high, tail)
    }
}

/// Bring a key's nine limbs below the limb base
///
/// The two-register form of [`settle`], written straight through so it stays
/// in its caller. The ninth limb settles in a general register, where its one
/// division is exact and runs beside the vector work rather than costing a
/// third register's worth of it; settled, it neither sheds nor receives
/// again, so the vector passes only take its first quotient.
#[inline(always)]
unsafe fn settle_32(low: __m256i, high: __m256i, tail: u64) -> (__m256i, __m256i, u64) {
    // SAFETY: lane arithmetic on registers the caller owns.
    unsafe {
        let base = _mm256_set1_epi64x(LIMB_BASE as i64);
        let zero = _mm256_setzero_si256();

        // Every limb divides at once, the 35-bit quotients shift a lane down.
        let quotient_tail = tail / LIMB_BASE;
        let settled_tail = tail - quotient_tail * LIMB_BASE;
        let incoming = _mm256_zextsi128_si256(_mm_cvtsi64_si128(quotient_tail as i64));
        let quotients_low = divide_base_under(low);
        let quotients_high = divide_base_under(high);
        let low = _mm256_add_epi64(
            _mm256_sub_epi64(low, multiply_base_wide(quotients_low)),
            shift_lane_down_pair(quotients_low, quotients_high),
        );
        let high = _mm256_add_epi64(
            _mm256_sub_epi64(high, multiply_base_wide(quotients_high)),
            shift_lane_down_pair(quotients_high, incoming),
        );

        // A second cheaper pass catches those carries, all below 2^35.
        let quotients_low = divide_base_small(low);
        let quotients_high = divide_base_small(high);
        let mut low = _mm256_add_epi64(
            _mm256_sub_epi64(low, _mm256_mul_epu32(quotients_low, base)),
            shift_lane_down_pair(quotients_low, quotients_high),
        );
        let mut high = _mm256_add_epi64(
            _mm256_sub_epi64(high, _mm256_mul_epu32(quotients_high, base)),
            shift_lane_down_pair(quotients_high, zero),
        );

        // Compare-and-carry rounds finish the stragglers. Values stay below
        // 2^63 here, so the signed compare reads as unsigned.
        let highest = _mm256_set1_epi64x((LIMB_BASE - 1) as i64);
        loop {
            let over_low = _mm256_cmpgt_epi64(low, highest);
            let over_high = _mm256_cmpgt_epi64(high, highest);
            if _mm256_movemask_epi8(_mm256_or_si256(over_low, over_high)) == 0 {
                break;
            }
            let carries_low = _mm256_srli_epi64::<63>(over_low);
            let carries_high = _mm256_srli_epi64::<63>(over_high);
            low = _mm256_add_epi64(
                _mm256_sub_epi64(low, _mm256_and_si256(over_low, base)),
                shift_lane_down_pair(carries_low, carries_high),
            );
            high = _mm256_add_epi64(
                _mm256_sub_epi64(high, _mm256_and_si256(over_high, base)),
                shift_lane_down_pair(carries_high, zero),
            );
        }
        (low, high, settled_tail)
    }
}

/// Four limbs to their five digits each, ten contiguous bytes per half
///
/// The same division chain as [`to_digits`], but each digit lands in a byte
/// of its own 64-bit lane, so one shuffle packs a register where shuffling
/// each digit into place takes five.
#[inline(always)]
unsafe fn to_digits_packed(limbs: __m256i, pack: __m256i) -> __m256i {
    // SAFETY: lane arithmetic on a register the caller owns.
    unsafe {
        let by_58 = _mm256_set1_epi64x(MAGIC_58);
        let by_3364 = _mm256_set1_epi64x(MAGIC_3364);
        let fifty_eight = _mm256_set1_epi64x(58);
        let divide_3364 = |value: __m256i| {
            _mm256_srli_epi64::<40>(_mm256_mul_epu32(_mm256_srli_epi64::<2>(value), by_3364))
        };
        let first = _mm256_srli_epi64::<37>(_mm256_mul_epu32(limbs, by_58));
        let fifth = _mm256_sub_epi64(limbs, _mm256_mul_epu32(first, fifty_eight));
        let second = divide_3364(limbs);
        let fourth = _mm256_sub_epi64(first, _mm256_mul_epu32(second, fifty_eight));
        let third_divided = divide_3364(first);
        let third = _mm256_sub_epi64(second, _mm256_mul_epu32(third_divided, fifty_eight));
        let fourth_divided = divide_3364(second);
        let second_digit =
            _mm256_sub_epi64(third_divided, _mm256_mul_epu32(fourth_divided, fifty_eight));
        let first_digit = fourth_divided;

        let merged = _mm256_or_si256(
            _mm256_or_si256(first_digit, _mm256_slli_epi64::<8>(second_digit)),
            _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_slli_epi64::<16>(third),
                    _mm256_slli_epi64::<24>(fourth),
                ),
                _mm256_slli_epi64::<32>(fifth),
            ),
        );
        _mm256_shuffle_epi8(merged, pack)
    }
}

/// [`to_chars`] over one sixteen-byte register
#[inline(always)]
unsafe fn to_chars_half(digits: __m128i) -> __m128i {
    // SAFETY: lane arithmetic on a register the caller owns.
    unsafe {
        let past_8 = _mm_cmpgt_epi8(digits, _mm_set1_epi8(8));
        let past_16 = _mm_cmpgt_epi8(digits, _mm_set1_epi8(16));
        let past_21 = _mm_cmpgt_epi8(digits, _mm_set1_epi8(21));
        let past_32 = _mm_cmpgt_epi8(digits, _mm_set1_epi8(32));
        let past_43 = _mm_cmpgt_epi8(digits, _mm_set1_epi8(43));
        let seven = _mm_and_si128(past_8, _mm_set1_epi8(-7));
        let six = _mm_and_si128(past_32, _mm_set1_epi8(-6));
        let total = _mm_add_epi8(
            _mm_add_epi8(
                _mm_add_epi8(_mm_set1_epi8(-(ALPHABET[0] as i8)), past_16),
                _mm_add_epi8(past_21, past_43),
            ),
            _mm_add_epi8(seven, six),
        );
        _mm_sub_epi8(digits, total)
    }
}

/// A key's settled limbs to the characters they spell
///
/// The 45 digits pack into two and three-quarter sixteen-byte registers, the
/// tail limb splitting in a general register beside the vector chains. Three
/// plain stores spell them: two shuffled windows cover the first 32 bytes,
/// and the third lands flush against the end, overlapping the second on
/// bytes they agree about. Every store stays inside `written`, which is at
/// least 32.
#[inline(always)]
unsafe fn spell_45(
    low: __m256i,
    high: __m256i,
    tail: u64,
    input_zeros: usize,
    out: &mut [u8],
) -> usize {
    // SAFETY: the loads are whole aligned tables, and the stores end at or
    // before `out + 44`, which the caller has checked room for.
    unsafe {
        let pack = _mm256_load_si256(PACK_TEN.0.as_ptr() as *const __m256i);
        let digits_low = to_digits_packed(low, pack);
        let digits_high = to_digits_packed(high, pack);

        let value = tail as u32;
        let first = value / 58;
        let second = first / 58;
        let third = second / 58;
        let fourth = third / 58;
        let packed_tail = fourth as u64
            | ((third - fourth * 58) as u64) << 8
            | ((second - third * 58) as u64) << 16
            | ((first - second * 58) as u64) << 24
            | ((value - first * 58) as u64) << 32;

        let a0 = _mm256_castsi256_si128(digits_low);
        let a1 = _mm256_extractf128_si256::<1>(digits_low);
        let b0 = _mm256_castsi256_si128(digits_high);
        let b1 = _mm256_extractf128_si256::<1>(digits_high);
        let tail_digits = _mm_cvtsi64_si128(packed_tail as i64);

        let first16 = _mm_or_si128(a0, _mm_slli_si128::<10>(a1));
        let middle16 = _mm_or_si128(
            _mm_or_si128(_mm_srli_si128::<6>(a1), _mm_slli_si128::<4>(b0)),
            _mm_slli_si128::<14>(b1),
        );
        let last16 = _mm_or_si128(_mm_srli_si128::<2>(b1), _mm_slli_si128::<8>(tail_digits));

        let empty = _mm_setzero_si128();
        let zeros_first = _mm_movemask_epi8(_mm_cmpeq_epi8(first16, empty)) as u32 as u64;
        let zeros_middle = _mm_movemask_epi8(_mm_cmpeq_epi8(middle16, empty)) as u32 as u64;
        let zeros_last = _mm_movemask_epi8(_mm_cmpeq_epi8(last16, empty)) as u32 as u64;
        // The bit past digit 44 is clear, so the scan cannot leave the digits
        // even when every one of them is zero.
        let combined = zeros_first | zeros_middle << 16 | (zeros_last & 0x1FFF) << 32;
        let leading = (!combined).trailing_zeros() as usize;

        let chars_low = to_chars(_mm256_set_m128i(middle16, first16));
        let chars_last = to_chars_half(last16);
        let low16 = _mm256_castsi256_si128(chars_low);
        let high16 = _mm256_extractf128_si256::<1>(chars_low);

        let skip = skip_32(leading, input_zeros);
        let masks = ALIGN_SKIP.0[skip].as_ptr();
        let select_low = _mm_load_si128(masks as *const __m128i);
        let select_high = _mm_load_si128(masks.add(16) as *const __m128i);

        let out_ptr = out.as_mut_ptr();
        _mm_storeu_si128(
            out_ptr as *mut __m128i,
            _mm_or_si128(
                _mm_shuffle_epi8(low16, select_low),
                _mm_shuffle_epi8(high16, select_high),
            ),
        );
        _mm_storeu_si128(
            out_ptr.add(16) as *mut __m128i,
            _mm_or_si128(
                _mm_shuffle_epi8(high16, select_low),
                _mm_shuffle_epi8(chars_last, select_high),
            ),
        );
        _mm_storeu_si128(
            out_ptr.add(29 - skip) as *mut __m128i,
            _mm_alignr_epi8::<13>(chars_last, high16),
        );
        DIGITS_32 - skip
    }
}

/// Encode a public key, from its words to the string, all in registers
///
/// # Safety
///
/// The caller must have established AVX2, and that `out` is at least
/// `MAX_ENCODED_32` bytes.
#[target_feature(enable = "avx2")]
pub unsafe fn encode_32(words: &[u32; WORDS_32], input_zeros: usize, out: &mut [u8]) -> usize {
    // SAFETY: the stores stay inside `out`, which the caller has checked.
    unsafe {
        let (low, high, tail) = sum_32(words);
        let (low, high, tail) = settle_32(low, high, tail);
        spell_45(low, high, tail, input_zeros, out)
    }
}

/// Encode a public key from its bytes, the whole call behind one boundary
///
/// A `target_feature` function does not inline into the dispatcher, so work
/// left outside it is paid for twice: once done, and once re-staged across
/// the call. This entry takes the input instead of the words, reads it once,
/// and swaps and counts in registers on the way in.
///
/// # Safety
///
/// As [`encode_32`].
#[target_feature(enable = "avx2")]
pub unsafe fn encode_32_entry(input: &[u8; 32], out: &mut [u8]) -> usize {
    // SAFETY: the input is exactly one register wide, and the words land in a
    // buffer their size.
    unsafe {
        let bytes = _mm256_loadu_si256(input.as_ptr() as *const __m256i);
        let zero_bytes =
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(bytes, _mm256_setzero_si256())) as u32;
        let input_zeros = (!zero_bytes).trailing_zeros() as usize;

        let flip = _mm256_load_si256(FLIP_WORDS.0.as_ptr() as *const __m256i);
        let mut words = [0u32; WORDS_32];
        _mm256_storeu_si256(
            words.as_mut_ptr() as *mut __m256i,
            _mm256_shuffle_epi8(bytes, flip),
        );

        let (low, high, tail) = sum_32(&words);
        let (low, high, tail) = settle_32(low, high, tail);
        spell_45(low, high, tail, input_zeros, out)
    }
}

// any length

/// Each quotient one lane up, the previous register's last lane entering below
#[target_feature(enable = "avx2")]
unsafe fn shift_lane_up(current: __m256i, previous: __m256i) -> __m256i {
    let rotated = _mm256_permute4x64_epi64::<0b10_01_00_11>(current);
    let last = _mm256_permute4x64_epi64::<0b11_11_11_11>(previous);
    _mm256_blend_epi32::<0b0000_0011>(rotated, last)
}

/// Add one word's contribution across every limb its place reaches
///
/// The four-lane form of the wide kernel of the same name. Whole chunks load
/// and store unmasked and the remainder goes scalar, because AVX2's masked
/// move is microcoded on AMD, where these kernels have no wider path to fall
/// back on.
///
/// # Safety
///
/// The caller must have established that this machine has AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn place_multiply_add(word: u32, entries: &[u32], limbs: &mut [u64]) {
    // SAFETY: the vector part stops a whole chunk short of the end, and the
    // remainder is walked by index.
    unsafe {
        let reach = entries.len().min(limbs.len());
        let broadcast = _mm256_set1_epi32(word as i32);
        let row = entries.as_ptr();
        let total = limbs.as_mut_ptr();

        let mut at = 0;
        while at + LANES <= reach {
            let places = _mm256_cvtepu32_epi64(_mm_loadu_si128(row.add(at) as *const __m128i));
            let carried = _mm256_loadu_si256(total.add(at) as *const __m256i);
            _mm256_storeu_si256(
                total.add(at) as *mut __m256i,
                _mm256_add_epi64(carried, _mm256_mul_epu32(broadcast, places)),
            );
            at += LANES;
        }
        while at < reach {
            *total.add(at) += word as u64 * *row.add(at) as u64;
            at += 1;
        }
    }
}

/// Hand each limb's overflow to the limb above it, once, without settling
///
/// The four-lane form of the wide kernel of the same name, with the same
/// undershooting estimate and the same rule that the topmost limb only
/// receives.
///
/// # Safety
///
/// The caller must have established that this machine has AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn shed_upward(limbs: &mut [u64]) {
    // SAFETY: the vector part stops a whole chunk short of the limb that only
    // receives, and the remainder is walked by index.
    unsafe {
        let len = limbs.len();
        if len < 2 {
            return;
        }
        let shedding = len - 1;
        let total = limbs.as_mut_ptr();
        let mut carried = _mm256_setzero_si256();

        let mut at = 0;
        while at + LANES <= shedding {
            let value = _mm256_loadu_si256(total.add(at) as *const __m256i);
            let quotient = divide_base_under(value);
            let residue = _mm256_sub_epi64(value, multiply_base_wide(quotient));
            _mm256_storeu_si256(
                total.add(at) as *mut __m256i,
                _mm256_add_epi64(residue, shift_lane_up(quotient, carried)),
            );
            carried = quotient;
            at += LANES;
        }

        let mut incoming = _mm256_extract_epi64::<3>(carried) as u64;
        while at < len {
            let value = *total.add(at);
            let quotient = match at < shedding {
                true => value / LIMB_BASE,
                false => 0,
            };
            *total.add(at) = value - quotient * LIMB_BASE + incoming;
            incoming = quotient;
            at += 1;
        }
    }
}

/// Hand each word's high half to the word above it, once
///
/// The four-lane form of the wide kernel of the same name, exact.
///
/// # Safety
///
/// The caller must have established that this machine has AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn shed_words_upward(words: &mut [u64]) {
    // SAFETY: as [`shed_upward`].
    unsafe {
        let len = words.len();
        if len < 2 {
            return;
        }
        let shedding = len - 1;
        let total = words.as_mut_ptr();
        let mut carried = _mm256_setzero_si256();

        let mut at = 0;
        while at + LANES <= shedding {
            let value = _mm256_loadu_si256(total.add(at) as *const __m256i);
            let quotient = _mm256_srli_epi64::<32>(value);
            let residue = _mm256_sub_epi64(value, _mm256_slli_epi64::<32>(quotient));
            _mm256_storeu_si256(
                total.add(at) as *mut __m256i,
                _mm256_add_epi64(residue, shift_lane_up(quotient, carried)),
            );
            carried = quotient;
            at += LANES;
        }

        let mut incoming = _mm256_extract_epi64::<3>(carried) as u64;
        while at < len {
            let value = *total.add(at);
            let quotient = match at < shedding {
                true => value >> 32,
                false => 0,
            };
            *total.add(at) = (value - (quotient << 32)) + incoming;
            incoming = quotient;
            at += 1;
        }
    }
}

/// Whether every byte of the register is a base58 character
///
/// A bitset product: each high nibble owns a bit, each low nibble holds the
/// set of high nibbles it is valid under, one `vpshufb` each. A set high bit
/// zeroes the lookup, so bytes past ASCII fail on their own.
#[target_feature(enable = "avx2")]
unsafe fn is_all_base58(chars: __m256i) -> bool {
    let high_bits = _mm256_setr_epi8(
        0, 0, 0, 0x01, 0x02, 0x04, 0x08, 0x10, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0x01, 0x02, 0x04, 0x08, 0x10, 0, 0, 0, 0, 0, 0, 0, 0,
    );
    let low_sets = _mm256_setr_epi8(
        0x14, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1D, 0x1E, 0x0A, 0x02, 0x0A, 0x0A,
        0x08, //
        0x14, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1D, 0x1E, 0x0A, 0x02, 0x0A, 0x0A,
        0x08,
    );
    let high = _mm256_shuffle_epi8(
        high_bits,
        _mm256_and_si256(_mm256_srli_epi16::<4>(chars), _mm256_set1_epi8(0x0F)),
    );
    let low = _mm256_shuffle_epi8(low_sets, chars);
    let invalid = _mm256_cmpeq_epi8(_mm256_and_si256(high, low), _mm256_setzero_si256());
    _mm256_movemask_epi8(invalid) == 0
}

/// Characters to digits for one window, validity already established
///
/// The encoder's compare chain run in reverse: subtract the runs of ASCII the
/// alphabet skips.
#[target_feature(enable = "avx2")]
unsafe fn digits_of(chars: __m128i) -> __m128i {
    let past_at = _mm_and_si128(_mm_cmpgt_epi8(chars, _mm_set1_epi8(64)), _mm_set1_epi8(-7));
    let past_i = _mm_cmpgt_epi8(chars, _mm_set1_epi8(73));
    let past_o = _mm_cmpgt_epi8(chars, _mm_set1_epi8(79));
    let past_tick = _mm_and_si128(_mm_cmpgt_epi8(chars, _mm_set1_epi8(96)), _mm_set1_epi8(-6));
    let past_l = _mm_cmpgt_epi8(chars, _mm_set1_epi8(108));
    _mm_add_epi8(
        chars,
        _mm_add_epi8(
            _mm_add_epi8(
                _mm_set1_epi8(-(ALPHABET[0] as i8)),
                _mm_add_epi8(past_at, past_tick),
            ),
            _mm_add_epi8(past_i, _mm_add_epi8(past_o, past_l)),
        ),
    )
}

/// Prepends a window can have to cross into the implied leading zeros
const PREPENDS: usize = 27;

/// The alignment shuffle for a window that overlaps the implied leading zeros
///
/// A window at `base` wants the digit at `base + at - prepend`; when that runs
/// before the input the load is clamped to its start, and this shuffle both
/// shifts the digits into place and zeroes the positions the prepend owns.
const fn shuffle_align(base: usize, prepend: usize) -> [u8; 16] {
    let loaded_at = match base >= prepend {
        true => base - prepend,
        false => 0,
    };
    let mut lanes = [0x80u8; 16];
    let mut at = 0;
    while at < 16 {
        if base + at >= prepend {
            lanes[at] = (base + at - prepend - loaded_at) as u8;
        }
        at += 1;
    }
    lanes
}

const fn shuffle_table(base: usize) -> [[u8; 16]; PREPENDS] {
    let mut table = [[0u8; 16]; PREPENDS];
    let mut prepend = 0;
    while prepend < PREPENDS {
        table[prepend] = shuffle_align(base, prepend);
        prepend += 1;
    }
    table
}

static ALIGN_AT_0: Aligned<[[u8; 16]; PREPENDS]> = Aligned(shuffle_table(0));
static ALIGN_AT_15: Aligned<[[u8; 16]; PREPENDS]> = Aligned(shuffle_table(15));

/// A 16-byte window holds exactly three limbs' worth of digits
const fn window_tail(offset: usize) -> [u8; 16] {
    let mut lanes = [0x80u8; 16];
    let mut limb = 0;
    while limb < 3 {
        let mut at = 0;
        while at < 4 {
            lanes[4 * limb + at] = (offset + DIGITS_PER_LIMB * limb + 1 + at) as u8;
            at += 1;
        }
        limb += 1;
    }
    lanes
}

const fn window_head(offset: usize) -> [u8; 16] {
    let mut lanes = [0x80u8; 16];
    let mut limb = 0;
    while limb < 3 {
        lanes[4 * limb] = (offset + DIGITS_PER_LIMB * limb) as u8;
        limb += 1;
    }
    lanes
}

const fn both_halves(low: [u8; 16], high: [u8; 16]) -> [u8; 32] {
    let mut lanes = [0u8; 32];
    let mut at = 0;
    while at < 16 {
        lanes[at] = low[at];
        lanes[at + 16] = high[at];
        at += 1;
    }
    lanes
}

static TAIL_BOTH: Aligned<[u8; 32]> = Aligned(both_halves(window_tail(0), window_tail(0)));
static HEAD_BOTH: Aligned<[u8; 32]> = Aligned(both_halves(window_head(0), window_head(0)));
static TAIL_LAST: Aligned<[u8; 32]> = Aligned(both_halves(window_tail(0), window_tail(1)));
static HEAD_LAST: Aligned<[u8; 32]> = Aligned(both_halves(window_head(0), window_head(1)));
static TAIL_EARLY: Aligned<[u8; 32]> = Aligned(both_halves(window_tail(1), window_tail(1)));
static HEAD_EARLY: Aligned<[u8; 32]> = Aligned(both_halves(window_head(1), window_head(1)));

/// Six limbs left-packed out of the two window lanes
static COMPACT: Aligned<[u32; 8]> = Aligned([0, 1, 2, 4, 5, 6, 3, 3]);

/// A fold writes eight lanes where the limbs it carries are fewer, and the
/// last one has no following store to correct its overrun, so it is masked to
/// the lanes it owns. Without this it writes past the caller's array, which is
/// harmless between the lanes of a batch and not harmless past the last of
/// them.
#[target_feature(enable = "avx2")]
unsafe fn store_lanes<const LANES_KEPT: usize>(into: *mut u32, value: __m256i) {
    let kept = match LANES_KEPT {
        3 => _mm256_setr_epi32(-1, -1, -1, 0, 0, 0, 0, 0),
        _ => _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0),
    };
    // SAFETY: the mask names only the lanes the caller's array holds.
    unsafe { _mm256_maskstore_epi32(into as *mut i32, kept, value) };
}

/// Two windows of digits to six limbs, three per lane
///
/// Trailing pairs via `vpmaddubsw` against `[58, 1]`, folded by `vpmaddwd`
/// against `[3364, 1]`, and the leading digit's weight added with a 32-bit
/// multiply. The junk lane shuffles from zeroed bytes and folds to zero.
#[target_feature(enable = "avx2")]
unsafe fn fold_windows(windows: __m256i, tail_index: __m256i, head_index: __m256i) -> __m256i {
    let tail = _mm256_shuffle_epi8(windows, tail_index);
    let head = _mm256_shuffle_epi8(windows, head_index);
    let pairs = _mm256_maddubs_epi16(tail, _mm256_set1_epi32(PAIR_58));
    let folded = _mm256_madd_epi16(pairs, _mm256_set1_epi32(PAIR_3364));
    _mm256_add_epi32(
        folded,
        _mm256_mullo_epi32(head, _mm256_set1_epi32(HEAD_WEIGHT)),
    )
}

/// The limbs against the widened table, four columns per register
#[target_feature(enable = "avx2")]
unsafe fn accumulate_words<const LIMBS: usize, const WORDS: usize, const REGISTERS: usize>(
    limbs: &[u32; LIMBS],
    table: &[[u64; WORDS]; LIMBS],
    wide: &mut [u64; WORDS],
) {
    // SAFETY: the rows are whole and the stores cover the columns the
    // registers hold.
    unsafe {
        let mut totals = [_mm256_setzero_si256(); REGISTERS];
        for (at, limb) in limbs.iter().enumerate() {
            let broadcast = _mm256_set1_epi32(*limb as i32);
            let row = table[at].as_ptr();
            for (register, slot) in totals.iter_mut().enumerate() {
                *slot = _mm256_add_epi64(
                    *slot,
                    _mm256_mul_epu32(
                        broadcast,
                        _mm256_loadu_si256(row.add(register * LANES) as *const __m256i),
                    ),
                );
            }
        }
        for (at, total) in totals.iter().enumerate() {
            _mm256_storeu_si256(wide.as_mut_ptr().add(at * LANES) as *mut __m256i, *total);
        }
    }
}

/// Sum a public key's limbs against the decode table
///
/// # Safety
///
/// The caller must have established that this machine has AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn words_from_limbs_32(limbs: &[u32; LIMBS_32], wide: &mut [u64; WORDS_32]) {
    // SAFETY: the widened table has a whole register per group of columns.
    unsafe {
        accumulate_words::<LIMBS_32, WORDS_32, { WORDS_32 / LANES }>(limbs, &DECODE_32_WIDE.0, wide)
    }
}

/// Sum a signature's limbs against the decode table
///
/// # Safety
///
/// As [`words_from_limbs_32`].
#[target_feature(enable = "avx2")]
pub unsafe fn words_from_limbs_64(limbs: &[u32; LIMBS_64], wide: &mut [u64; WORDS_64]) {
    // SAFETY: as above.
    unsafe {
        accumulate_words::<LIMBS_64, WORDS_64, { WORDS_64 / LANES }>(limbs, &DECODE_64_WIDE.0, wide)
    }
}

/// Decode a public key's characters to its words, before the carries settle
///
/// `None` means a character outside the alphabet, which sends the input back
/// to the portable path so the error names the right byte.
///
/// # Safety
///
/// The caller must have established AVX2 and an input of at least 32 bytes.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn read_32(encoded: &[u8], limbs: &mut [u32; LIMBS_32]) -> bool {
    // SAFETY: every load stays inside the input, which the caller has checked
    // is at least 32 bytes and at most 44.
    unsafe {
        let len = encoded.len();
        let prepend = DIGITS_32 - len;
        let start = encoded.as_ptr();
        if !(is_all_base58(_mm256_loadu_si256(start as *const __m256i))
            & is_all_base58(_mm256_loadu_si256(start.add(len - 32) as *const __m256i)))
        {
            return false;
        }

        let window_0 = _mm_shuffle_epi8(
            digits_of(_mm_loadu_si128(start as *const __m128i)),
            _mm_load_si128(ALIGN_AT_0.0[prepend].as_ptr() as *const __m128i),
        );
        let window_1 = digits_of(_mm_loadu_si128(start.add(15 - prepend) as *const __m128i));
        let window_2 = digits_of(_mm_loadu_si128(start.add(len - 16) as *const __m128i));

        let first_six = _mm256_permutevar8x32_epi32(
            fold_windows(
                _mm256_set_m128i(window_1, window_0),
                _mm256_loadu_si256(TAIL_BOTH.0.as_ptr() as *const __m256i),
                _mm256_loadu_si256(HEAD_BOTH.0.as_ptr() as *const __m256i),
            ),
            _mm256_loadu_si256(COMPACT.0.as_ptr() as *const __m256i),
        );
        let last_three = fold_windows(
            _mm256_set_m128i(window_2, window_2),
            _mm256_loadu_si256(TAIL_EARLY.0.as_ptr() as *const __m256i),
            _mm256_loadu_si256(HEAD_EARLY.0.as_ptr() as *const __m256i),
        );

        let spill = limbs.as_mut_ptr();
        _mm256_storeu_si256(spill as *mut __m256i, first_six);
        store_lanes::<3>(spill.add(LIMBS_32 - 3), last_three);
        true
    }
}

/// Decode a signature's characters to its words, before the carries settle
///
/// # Safety
///
/// The caller must have established AVX2 and an input of at least 64 bytes.
#[target_feature(enable = "avx2")]
pub unsafe fn read_64(encoded: &[u8], limbs: &mut [u32; LIMBS_64]) -> bool {
    // SAFETY: every load stays inside the input, which the caller has checked
    // is at least 64 bytes and at most 88.
    unsafe {
        let len = encoded.len();
        let prepend = DIGITS_64 - len;
        let start = encoded.as_ptr();
        if !(is_all_base58(_mm256_loadu_si256(start as *const __m256i))
            & is_all_base58(_mm256_loadu_si256(start.add(32) as *const __m256i))
            & is_all_base58(_mm256_loadu_si256(start.add(len - 32) as *const __m256i)))
        {
            return false;
        }

        let aligned = |base: usize, table: &Aligned<[[u8; 16]; PREPENDS]>| {
            _mm_shuffle_epi8(
                digits_of(_mm_loadu_si128(
                    start.add(base.saturating_sub(prepend)) as *const __m128i
                )),
                _mm_load_si128(table.0[prepend].as_ptr() as *const __m128i),
            )
        };
        let plain =
            |at: usize| digits_of(_mm_loadu_si128(start.add(at - prepend) as *const __m128i));

        let tail_both = _mm256_loadu_si256(TAIL_BOTH.0.as_ptr() as *const __m256i);
        let head_both = _mm256_loadu_si256(HEAD_BOTH.0.as_ptr() as *const __m256i);
        let compact = _mm256_loadu_si256(COMPACT.0.as_ptr() as *const __m256i);

        let spill = limbs.as_mut_ptr();

        let first = fold_windows(
            _mm256_set_m128i(aligned(15, &ALIGN_AT_15), aligned(0, &ALIGN_AT_0)),
            tail_both,
            head_both,
        );
        _mm256_storeu_si256(
            spill as *mut __m256i,
            _mm256_permutevar8x32_epi32(first, compact),
        );

        let second = fold_windows(_mm256_set_m128i(plain(45), plain(30)), tail_both, head_both);
        _mm256_storeu_si256(
            spill.add(6) as *mut __m256i,
            _mm256_permutevar8x32_epi32(second, compact),
        );

        let last = digits_of(_mm_loadu_si128(start.add(len - 16) as *const __m128i));
        let third = fold_windows(
            _mm256_set_m128i(last, plain(60)),
            _mm256_loadu_si256(TAIL_LAST.0.as_ptr() as *const __m256i),
            _mm256_loadu_si256(HEAD_LAST.0.as_ptr() as *const __m256i),
        );
        store_lanes::<6>(spill.add(12), _mm256_permutevar8x32_epi32(third, compact));
        true
    }
}

/// Whether an encoding is wide enough for the whole-register reads
pub(crate) fn fits_32(len: usize) -> bool {
    (MIN_ENCODED_32..=MAX_ENCODED_32).contains(&len)
}

/// The same for a signature
pub(crate) fn fits_64(len: usize) -> bool {
    (MIN_ENCODED_64..=MAX_ENCODED_64).contains(&len)
}
