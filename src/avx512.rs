//! The x86-64 codec for machines with byte permutes across a whole register
//!
//! `vpermb` indexes 64 bytes and `vpermi2b` 128 across two registers, so a
//! table lookup, a compaction or a variable byte shift is one instruction
//! each. Everything else follows from keeping the work in registers: the
//! decoder never writes its digits to memory, and the encoder's only stores
//! are the ones that produce the string.
//!
//! The matrix products are written out in `vpmuludq` against the widened
//! tables rather than left to the auto-vectorizer, which reaches for
//! `vpgatherdd` or `vpmullq` depending on the compiler version.

use core::arch::x86_64::{
    __m128i, __m512i, _mm512_add_epi32, _mm512_add_epi64, _mm512_add_epi8, _mm512_alignr_epi64,
    _mm512_castsi512_si128, _mm512_cmpeq_epi8_mask, _mm512_cmpge_epu64_mask,
    _mm512_extracti32x4_epi32, _mm512_inserti32x4, _mm512_load_si512, _mm512_loadu_si512,
    _mm512_madd_epi16, _mm512_maddubs_epi16, _mm512_mask_storeu_epi8, _mm512_mask_sub_epi64,
    _mm512_maskz_loadu_epi64, _mm512_maskz_loadu_epi8, _mm512_maskz_permutex2var_epi8,
    _mm512_maskz_permutexvar_epi8, _mm512_maskz_set1_epi64, _mm512_movepi8_mask, _mm512_mul_epu32,
    _mm512_mullo_epi32, _mm512_permutex2var_epi8, _mm512_permutexvar_epi8, _mm512_set1_epi32,
    _mm512_set1_epi64, _mm512_set1_epi8, _mm512_setzero_si512, _mm512_slli_epi64,
    _mm512_srli_epi64, _mm512_storeu_si512, _mm512_sub_epi64, _mm512_sub_epi8,
    _mm512_ternarylogic_epi64, _mm512_zextsi128_si512, _mm_add_epi64, _mm_cvtsi128_si64,
    _mm_cvtsi64_si128, _mm_extract_epi64, _mm_insert_epi64, _mm_loadl_epi64, _mm_loadu_si128,
    _mm_mul_epu32, _mm_set_epi64x, _mm_setzero_si128, _mm_storel_epi64,
};
use core::arch::x86_64::{
    __m256i, _mm256_cmpeq_epi8_mask, _mm256_load_si256, _mm256_loadu_si256,
    _mm256_maskz_loadu_epi32, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_storeu_si256,
    _mm512_cvtepu32_epi64, _mm512_mask_storeu_epi64, _mm512_maskz_mov_epi64,
};

use crate::scalar::{
    ALPHABET, DIGITS_32, DIGITS_64, DIGITS_PER_LIMB, LIMBS_64, LIMB_BASE, WORDS_32, WORDS_64,
};
use crate::tables::ENCODE_64;
use crate::wide::{
    widen_shift, Aligned, DECODE_64_WIDE, ENCODE_32_TAIL, ENCODE_32_WIDE, FLIP_WORDS, HEAD_WEIGHT,
    MAGIC_3364, MAGIC_58, MAGIC_HIGH, MAGIC_LOW, MAGIC_SMALL, PAIR_3364, PAIR_58,
};

/// One plus the digit, so that zero marks a byte outside the alphabet
///
/// Two registers covering ASCII, indexed by `vpermi2b` with the low seven
/// bits. A high bit would alias onto the table, so those lanes are masked
/// out before the lookup.
static INVERSE: Aligned<[u8; 128]> = Aligned([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, //
    0, 10, 11, 12, 13, 14, 15, 16, 17, 0, 18, 19, 20, 21, 22, 0, //
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 0, 0, 0, 0, 0, //
    0, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 0, 45, 46, 47, //
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 0, 0, 0, 0, 0,
]);

/// The alphabet padded to the 64 bytes a whole-register permute indexes
static CHARS: Aligned<[u8; 64]> = Aligned(build_chars());

const fn build_chars() -> [u8; 64] {
    let mut table = [0u8; 64];
    let mut at = 0;
    while at < 58 {
        table[at] = ALPHABET[at];
        at += 1;
    }
    table
}

/// Byte `base + n` in lane `n`, the identity a shifted store offsets from
const fn iota(base: u8) -> [u8; 64] {
    let mut lanes = [0u8; 64];
    let mut at = 0;
    while at < 64 {
        lanes[at] = base + at as u8;
        at += 1;
    }
    lanes
}

static IOTA: Aligned<[u8; 64]> = Aligned(iota(0));
static IOTA_HIGH: Aligned<[u8; 64]> = Aligned(iota(64));

/// Five digits per 64-bit lane down to five contiguous bytes per limb
const fn pack_index() -> [u8; 64] {
    let mut lanes = [0u8; 64];
    let mut at = 0;
    while at < 40 {
        lanes[at] = (8 * (at / DIGITS_PER_LIMB) + at % DIGITS_PER_LIMB) as u8;
        at += 1;
    }
    lanes
}

static PACK: Aligned<[u8; 64]> = Aligned(pack_index());

/// The 90 digits of a signature, gathered from three packed registers
///
/// Each packed register carries forty real bytes, so the first output takes
/// A's forty then B's first twenty-four, and the second B's remaining sixteen
/// then C's ten. An index at 64 or above selects the second operand.
const fn gather_90(is_first: bool) -> [u8; 64] {
    let mut lanes = [0u8; 64];
    let mut at = 0;
    while at < 64 {
        if is_first {
            lanes[at] = (if at < 40 { at } else { at + 24 }) as u8;
        } else if at < 16 {
            lanes[at] = (24 + at) as u8;
        } else if at < 26 {
            lanes[at] = (48 + at) as u8;
        }
        at += 1;
    }
    lanes
}

/// The 45 digits of a key: A's forty and the ninth limb's five
const fn gather_45() -> [u8; 64] {
    let mut lanes = [0u8; 64];
    let mut at = 0;
    while at < 64 {
        lanes[at] = (if at < 40 { at } else { at + 24 }) as u8;
        at += 1;
    }
    lanes
}

static GATHER_45: Aligned<[u8; 64]> = Aligned(gather_45());

static GATHER_90_LOW: Aligned<[u8; 64]> = Aligned(gather_90(true));
static GATHER_90_HIGH: Aligned<[u8; 64]> = Aligned(gather_90(false));

/// For each 32-bit lane, that limb's four trailing digits
const fn limb_tail(base: i64, limbs: usize, offset: i64) -> [u8; 64] {
    let mut lanes = [0u8; 64];
    let mut limb = 0;
    while limb < limbs {
        let mut at = 0;
        while at < 4 {
            lanes[4 * limb + at] =
                (base + DIGITS_PER_LIMB as i64 * limb as i64 + 1 + at as i64 + offset) as u8;
            at += 1;
        }
        limb += 1;
    }
    lanes
}

/// For each 32-bit lane, that limb's leading digit in byte zero
const fn limb_head(base: i64, limbs: usize, offset: i64) -> [u8; 64] {
    let mut lanes = [0u8; 64];
    let mut limb = 0;
    while limb < limbs {
        lanes[4 * limb] = (base + DIGITS_PER_LIMB as i64 * limb as i64 + offset) as u8;
        limb += 1;
    }
    lanes
}

static TAIL_64: Aligned<[u8; 64]> = Aligned(limb_tail(0, 16, 0));
static HEAD_64: Aligned<[u8; 64]> = Aligned(limb_head(0, 16, 0));

// Limbs 16 and 17 read digits 80..90, which sit at bytes 16..26 of the second
// digit register.
static TAIL_64_TOP: Aligned<[u8; 64]> = Aligned(limb_tail(80, 2, -64));
static HEAD_64_TOP: Aligned<[u8; 64]> = Aligned(limb_head(80, 2, -64));

/// Byte zero of every 32-bit lane, the head-digit lanes a permute keeps
const HEAD_LANES: u64 = 0x1111_1111_1111_1111;

/// The lanes limbs 16 and 17 occupy
const HEAD_LANES_TOP: u64 = 0x11;

/// Limbs the last register of a 64-byte encode holds
const TOP_LIMBS: usize = 2;

/// Limbs one register holds
const LANES: usize = 8;

/// The 64-byte encode's table, three registers' worth per row
static ENCODE_64_SHIFTED: Aligned<[[u64; 24]; 16]> = Aligned(widen_shift(&ENCODE_64));

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn load(from: *const u8) -> __m512i {
    // SAFETY: every caller passes a cache-line aligned table of 64 bytes.
    unsafe { _mm512_load_si512(from as *const __m512i) }
}

/// A mask with the low `len` bits set, for a load or store of exactly that many
fn low_mask(len: usize) -> u64 {
    match len >= 64 {
        true => u64::MAX,
        false => (1u64 << len) - 1,
    }
}

/// `floor(x / 58^5)`, within two and never over, for any lane value
///
/// Three multiplies estimating the 128-bit product's high bits. The dropped
/// low-half carries can undershoot by two, which later passes absorb.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn divide_base_under(terms: __m512i) -> __m512i {
    let odd = _mm512_srli_epi64::<5>(terms);
    let high = _mm512_srli_epi64::<32>(odd);
    let low_high = _mm512_mul_epu32(odd, _mm512_set1_epi64(MAGIC_HIGH));
    let high_low = _mm512_mul_epu32(high, _mm512_set1_epi64(MAGIC_LOW));
    let high_high = _mm512_mul_epu32(high, _mm512_set1_epi64(MAGIC_HIGH));
    let middle = _mm512_add_epi64(low_high, high_low);
    _mm512_add_epi64(
        _mm512_srli_epi64::<24>(high_high),
        _mm512_srli_epi64::<56>(middle),
    )
}

/// `floor(x / 58^5)` exactly, for lane values below 2^35
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn divide_base_small(terms: __m512i) -> __m512i {
    _mm512_srli_epi64::<55>(_mm512_mul_epu32(
        _mm512_srli_epi64::<5>(terms),
        _mm512_set1_epi64(MAGIC_SMALL),
    ))
}

/// `quotient * 58^5` for quotients that can exceed 32 bits
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn multiply_base_wide(quotients: __m512i) -> __m512i {
    let base = _mm512_set1_epi64(LIMB_BASE as i64);
    _mm512_add_epi64(
        _mm512_mul_epu32(quotients, base),
        _mm512_slli_epi64::<32>(_mm512_mul_epu32(_mm512_srli_epi64::<32>(quotients), base)),
    )
}

/// Bring every limb below the limb base, carrying quotients left
///
/// The scalar chain waits on the limb above at every step. Here every limb
/// divides at once and the quotients shift a lane down, leaving a 35-bit
/// carry; a second cheaper pass and then compare-and-carry rounds finish it.
/// Limb zero only ever receives.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn settle<const REGISTERS: usize>(terms: &mut [__m512i; REGISTERS]) {
    // SAFETY: lane arithmetic on registers the caller owns.
    unsafe {
        let zero = _mm512_setzero_si512();
        let base = _mm512_set1_epi64(LIMB_BASE as i64);
        let mut quotients = [zero; REGISTERS];

        for at in 0..REGISTERS {
            quotients[at] = divide_base_under(terms[at]);
        }
        for at in 0..REGISTERS {
            let next = match at + 1 < REGISTERS {
                true => quotients[at + 1],
                false => zero,
            };
            terms[at] = _mm512_add_epi64(
                _mm512_sub_epi64(terms[at], multiply_base_wide(quotients[at])),
                _mm512_alignr_epi64::<1>(next, quotients[at]),
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
            terms[at] = _mm512_add_epi64(
                _mm512_sub_epi64(terms[at], _mm512_mul_epu32(quotients[at], base)),
                _mm512_alignr_epi64::<1>(next, quotients[at]),
            );
        }

        loop {
            let mut over = [0u8; REGISTERS];
            let mut any = 0u8;
            for at in 0..REGISTERS {
                over[at] = _mm512_cmpge_epu64_mask(terms[at], base);
                any |= over[at];
            }
            if any == 0 {
                break;
            }
            for at in 0..REGISTERS {
                quotients[at] = _mm512_maskz_set1_epi64(over[at], 1);
            }
            for at in 0..REGISTERS {
                let next = match at + 1 < REGISTERS {
                    true => quotients[at + 1],
                    false => zero,
                };
                terms[at] = _mm512_add_epi64(
                    _mm512_mask_sub_epi64(terms[at], over[at], terms[at], base),
                    _mm512_alignr_epi64::<1>(next, quotients[at]),
                );
            }
        }
    }
}

/// Eight limbs to their five digits each, five bytes per 64-bit lane
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn to_digits(limbs: __m512i) -> __m512i {
    // SAFETY: lane arithmetic on a register the caller owns.
    unsafe {
        let by_58 = _mm512_set1_epi64(MAGIC_58);
        let by_3364 = _mm512_set1_epi64(MAGIC_3364);
        let fifty_eight = _mm512_set1_epi64(58);
        let divide_3364 = |value: __m512i| {
            _mm512_srli_epi64::<40>(_mm512_mul_epu32(_mm512_srli_epi64::<2>(value), by_3364))
        };
        let first = _mm512_srli_epi64::<37>(_mm512_mul_epu32(limbs, by_58));
        let fifth = _mm512_sub_epi64(limbs, _mm512_mul_epu32(first, fifty_eight));
        let second = divide_3364(limbs);
        let fourth = _mm512_sub_epi64(first, _mm512_mul_epu32(second, fifty_eight));
        let third_divided = divide_3364(first);
        let third = _mm512_sub_epi64(second, _mm512_mul_epu32(third_divided, fifty_eight));
        let fourth_divided = divide_3364(second);
        let second_digit =
            _mm512_sub_epi64(third_divided, _mm512_mul_epu32(fourth_divided, fifty_eight));
        let first_digit = fourth_divided;

        let low = _mm512_ternarylogic_epi64::<0xFE>(
            first_digit,
            _mm512_slli_epi64::<8>(second_digit),
            _mm512_slli_epi64::<16>(third),
        );
        let merged = _mm512_ternarylogic_epi64::<0xFE>(
            low,
            _mm512_slli_epi64::<24>(fourth),
            _mm512_slli_epi64::<32>(fifth),
        );
        _mm512_permutexvar_epi8(load(PACK.0.as_ptr()), merged)
    }
}

/// One row of the words-to-limbs product, eight limbs at once
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn encode_multiply_add(total: __m512i, word: __m512i, row: *const u64) -> __m512i {
    // SAFETY: the row is a whole aligned register of the widened table.
    unsafe { _mm512_add_epi64(total, _mm512_mul_epu32(word, load(row as *const u8))) }
}

/// A signature's words to its settled limbs, without touching memory
///
/// Limbs 0..16 in two vector accumulators and the last two in an xmm pair.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn sum_and_settle_64(words: &[u32; WORDS_64]) -> [__m512i; 3] {
    // SAFETY: every row is a whole aligned span of the widened table.
    unsafe {
        let mut low = _mm512_setzero_si512();
        let mut high = _mm512_setzero_si512();
        let mut top = _mm_cvtsi64_si128(0);

        for at in 0..WORDS_64 / 2 {
            let word = _mm512_set1_epi64(words[at] as i64);
            let row = ENCODE_64_SHIFTED.0[at].as_ptr();
            low = encode_multiply_add(low, word, row);
            high = encode_multiply_add(high, word, row.add(8));
            top = _mm_add_epi64(
                top,
                _mm_mul_epu32(
                    _mm512_castsi512_si128(word),
                    _mm_loadu_si128(row.add(16) as *const __m128i),
                ),
            );
        }

        // Limbs 15 and 16 would overflow if the second half of the rows landed
        // on them untouched, so they shed once here.
        let top_lanes = _mm512_extracti32x4_epi32::<3>(high);
        let mut limb_15 = _mm_extract_epi64::<1>(top_lanes) as u64;
        let mut limb_16 = _mm_cvtsi128_si64(top) as u64;
        limb_15 += limb_16 / LIMB_BASE;
        limb_16 %= LIMB_BASE;
        high = _mm512_inserti32x4::<3>(
            high,
            _mm_set_epi64x(limb_15 as i64, _mm_cvtsi128_si64(top_lanes)),
        );
        top = _mm_insert_epi64::<0>(top, limb_16 as i64);

        for at in WORDS_64 / 2..WORDS_64 {
            let word = _mm512_set1_epi64(words[at] as i64);
            let row = ENCODE_64_SHIFTED.0[at].as_ptr();
            // Upper triangular again: these rows start past everything the
            // first register holds, and the last lives only in the pair.
            if at < WORDS_64 - 1 {
                high = encode_multiply_add(high, word, row.add(8));
            }
            top = _mm_add_epi64(
                top,
                _mm_mul_epu32(
                    _mm512_castsi512_si128(word),
                    _mm_loadu_si128(row.add(16) as *const __m128i),
                ),
            );
        }

        let mut terms = [low, high, _mm512_zextsi128_si512(top)];
        settle(&mut terms);
        terms
    }
}

/// A key's words to its settled limbs, without touching memory
///
/// The ninth column has no lane of its own, so it rides an xmm rather than a
/// general register: keeping it in the vector domain makes the widen a cast
/// instead of a `movq` the whole reduction then waits on. Two accumulators
/// rather than one halve the depth of the add chain behind it.
///
/// No `target_feature` of its own: left one, a build on Genoa kept it out of
/// line and returned its three registers through the stack, where the cost
/// model on Zen 5 had always folded it in. `inline(always)` removes the
/// choice.
#[inline(always)]
unsafe fn sum_and_settle_32(words: &[u32; WORDS_32]) -> [__m512i; 2] {
    // SAFETY: every row is a whole aligned register of the widened table.
    unsafe {
        let mut even = _mm512_setzero_si512();
        let mut odd = _mm512_setzero_si512();
        let mut tail_even = _mm_setzero_si128();
        let mut tail_odd = _mm_setzero_si128();

        for (at, word) in words.iter().enumerate() {
            let spread = _mm512_set1_epi64(*word as i64);
            let column = _mm_loadl_epi64(ENCODE_32_TAIL.0.as_ptr().add(at) as *const __m128i);
            let tail = _mm_mul_epu32(_mm512_castsi512_si128(spread), column);
            // The table is upper triangular, so the last row reaches only the
            // column with no lane.
            let row = match at < WORDS_32 - 1 {
                true => Some(ENCODE_32_WIDE.0[at].as_ptr()),
                false => None,
            };
            match at % 2 == 0 {
                true => {
                    if let Some(row) = row {
                        even = encode_multiply_add(even, spread, row);
                    }
                    tail_even = _mm_add_epi64(tail_even, tail);
                }
                false => {
                    if let Some(row) = row {
                        odd = encode_multiply_add(odd, spread, row);
                    }
                    tail_odd = _mm_add_epi64(tail_odd, tail);
                }
            }
        }

        let mut terms = [
            _mm512_add_epi64(even, odd),
            _mm512_zextsi128_si512(_mm_add_epi64(tail_even, tail_odd)),
        ];
        settle(&mut terms);
        terms
    }
}

/// The words-to-string body, carrying no `target_feature` so either
/// boundary composes it in whole
#[inline(always)]
unsafe fn encode_32_body(words: &[u32; WORDS_32], input_zeros: usize, out: &mut [u8]) -> usize {
    // SAFETY: the one store is masked to a length no encoding exceeds.
    unsafe {
        let terms = sum_and_settle_32(words);
        let digits = _mm512_permutex2var_epi8(
            to_digits(terms[0]),
            load(GATHER_45.0.as_ptr()),
            to_digits(terms[1]),
        );
        let present = !_mm512_cmpeq_epi8_mask(digits, _mm512_setzero_si512()) & low_mask(DIGITS_32);
        let leading = match present != 0 {
            true => present.trailing_zeros() as usize,
            false => DIGITS_32,
        };
        let chars = _mm512_permutexvar_epi8(digits, load(CHARS.0.as_ptr()));

        // Storing the characters unshifted against a base that far back, with
        // the mask moved up instead, reads as the cheaper form and measured
        // sixty percent worse: the masked-off lanes reach behind the buffer
        // and the core takes an assist for it.
        let skip = leading - input_zeros;
        let shift = _mm512_add_epi8(load(IOTA.0.as_ptr()), _mm512_set1_epi8(skip as i8));
        let written = DIGITS_32 - skip;
        _mm512_mask_storeu_epi8(
            out.as_mut_ptr() as *mut i8,
            low_mask(written),
            _mm512_permutexvar_epi8(shift, chars),
        );
        written
    }
}

/// Encode a public key, from its words to the string, all in registers
///
/// # Safety
///
/// The caller must have established the wide permutes, and that `out` is at
/// least `MAX_ENCODED_32` bytes.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
pub unsafe fn encode_32(words: &[u32; WORDS_32], input_zeros: usize, out: &mut [u8]) -> usize {
    // SAFETY: as the body.
    unsafe { encode_32_body(words, input_zeros, out) }
}

/// Encode a public key from its bytes, the whole call behind one boundary
///
/// As the AVX2 entry: a `target_feature` call does not inline into the
/// dispatcher, so the byte swap and the zero count move inside it rather
/// than staging across it.
///
/// # Safety
///
/// As [`encode_32`].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
pub unsafe fn encode_32_entry(input: &[u8; 32], out: &mut [u8]) -> usize {
    // SAFETY: the input is exactly one register wide, and the words land in a
    // buffer their size.
    unsafe {
        let bytes = _mm256_loadu_si256(input.as_ptr() as *const __m256i);
        let zero_bytes = _mm256_cmpeq_epi8_mask(bytes, _mm256_setzero_si256());
        let input_zeros = (!zero_bytes).trailing_zeros() as usize;

        let flip = _mm256_load_si256(FLIP_WORDS.0.as_ptr() as *const __m256i);
        let mut words = [0u32; WORDS_32];
        _mm256_storeu_si256(
            words.as_mut_ptr() as *mut __m256i,
            _mm256_shuffle_epi8(bytes, flip),
        );
        encode_32_body(&words, input_zeros, out)
    }
}

/// Encode a signature, from its words to the string, all in registers
///
/// # Safety
///
/// As [`encode_32`], with `out` at least `MAX_ENCODED_64` bytes.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
pub(crate) unsafe fn encode_64(
    words: &[u32; WORDS_64],
    input_zeros: usize,
    out: &mut [u8],
) -> usize {
    // SAFETY: as above.
    unsafe { spell_90(sum_and_settle_64(words), input_zeros, out) }
}

/// Spell a signature's limbs, as `sum_64` leaves them, into characters
///
/// # Safety
///
/// As [`write_32`], with `out` at least `MAX_ENCODED_64` bytes.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
pub(crate) unsafe fn write_64(limbs: [u64; LIMBS_64], input_zeros: usize, out: &mut [u8]) -> usize {
    // SAFETY: the loads read the caller's limbs.
    unsafe {
        let source = limbs.as_ptr() as *const __m512i;
        let mut terms = [
            _mm512_loadu_si512(source),
            _mm512_loadu_si512(source.add(1)),
            _mm512_maskz_loadu_epi64(
                low_mask(TOP_LIMBS) as u8,
                limbs.as_ptr().add(LIMBS_64 - TOP_LIMBS) as *const i64,
            ),
        ];
        settle(&mut terms);
        spell_90(terms, input_zeros, out)
    }
}

/// A signature's settled limbs to the characters they spell
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn spell_90(terms: [__m512i; 3], input_zeros: usize, out: &mut [u8]) -> usize {
    // SAFETY: the unmasked store covers 64 bytes, which no encoding of a
    // 64-byte value is shorter than, and the second is masked to the rest.
    unsafe {
        let packed_low = to_digits(terms[0]);
        let packed_high = to_digits(terms[1]);
        let packed_top = to_digits(terms[2]);
        let digits_low =
            _mm512_permutex2var_epi8(packed_low, load(GATHER_90_LOW.0.as_ptr()), packed_high);
        let digits_high =
            _mm512_permutex2var_epi8(packed_high, load(GATHER_90_HIGH.0.as_ptr()), packed_top);

        let zero = _mm512_setzero_si512();
        let present_low = !_mm512_cmpeq_epi8_mask(digits_low, zero);
        let leading = match present_low != 0 {
            true => present_low.trailing_zeros() as usize,
            false => {
                let tail = DIGITS_64 - 64;
                let present_high = !_mm512_cmpeq_epi8_mask(digits_high, zero) & low_mask(tail);
                64 + (present_high.trailing_zeros() as usize).min(tail)
            }
        };

        let alphabet = load(CHARS.0.as_ptr());
        let chars_low = _mm512_permutexvar_epi8(digits_low, alphabet);
        let chars_high = _mm512_permutexvar_epi8(digits_high, alphabet);

        let skip = leading - input_zeros;
        let shift = _mm512_add_epi8(load(IOTA.0.as_ptr()), _mm512_set1_epi8(skip as i8));
        let out_ptr = out.as_mut_ptr();
        _mm512_storeu_si512(
            out_ptr as *mut __m512i,
            _mm512_permutex2var_epi8(chars_low, shift, chars_high),
        );
        _mm512_mask_storeu_epi8(
            out_ptr.add(64) as *mut i8,
            low_mask(DIGITS_64 - 64 - skip),
            _mm512_permutexvar_epi8(shift, chars_high),
        );
        DIGITS_64 - skip
    }
}

/// Add one word's contribution across every limb its place reaches
///
/// Eight multiply-accumulates an instruction, with the table's words widened
/// on the way in rather than in rodata. `entries` is one row of the place
/// table, least significant first, and `limbs` the accumulators it lands on.
///
/// # Safety
///
/// The caller must have established that this machine has the wide permutes.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
pub unsafe fn place_multiply_add(word: u32, entries: &[u32], limbs: &mut [u64]) {
    // SAFETY: every access is bounded by the shorter of the two slices, and
    // the tail is masked to what is left of it.
    unsafe {
        let reach = entries.len().min(limbs.len());
        let broadcast = _mm512_set1_epi32(word as i32);
        let row = entries.as_ptr();
        let total = limbs.as_mut_ptr();

        let mut at = 0;
        while at + LANES <= reach {
            let places = _mm512_cvtepu32_epi64(_mm256_loadu_si256(row.add(at) as *const __m256i));
            let carried = _mm512_loadu_si512(total.add(at) as *const __m512i);
            _mm512_storeu_si512(
                total.add(at) as *mut __m512i,
                _mm512_add_epi64(carried, _mm512_mul_epu32(broadcast, places)),
            );
            at += LANES;
        }

        let left = low_mask(reach - at) as u8;
        if left != 0 {
            let places =
                _mm512_cvtepu32_epi64(_mm256_maskz_loadu_epi32(left, row.add(at) as *const i32));
            let carried = _mm512_maskz_loadu_epi64(left, total.add(at) as *const i64);
            _mm512_mask_storeu_epi64(
                total.add(at) as *mut i64,
                left,
                _mm512_add_epi64(carried, _mm512_mul_epu32(broadcast, places)),
            );
        }
    }
}

/// Hand each limb's overflow to the limb above it, once, without settling
///
/// Every quotient comes from the limb's own value, so no lane waits on
/// another. The estimate undershoots by at most two, leaving a limb below
/// three times the base rather than one, which is still far enough under what
/// a `u64` holds for the words this makes room for; the settling pass that
/// closes the encode is exact regardless.
///
/// The topmost limb only receives, since shedding it would carry off the end.
///
/// # Safety
///
/// The caller must have established that this machine has the wide permutes.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
pub unsafe fn shed_upward(limbs: &mut [u64]) {
    // SAFETY: every access is masked to what is left of the slice.
    unsafe {
        let len = limbs.len();
        if len < 2 {
            return;
        }
        let shedding = len - 1;
        let total = limbs.as_mut_ptr();
        let mut carried = _mm512_setzero_si512();

        let mut at = 0;
        while at < len {
            let here = low_mask((len - at).min(LANES)) as u8;
            let sheds = low_mask(shedding.saturating_sub(at).min(LANES)) as u8;
            let value = _mm512_maskz_loadu_epi64(here, total.add(at) as *const i64);
            let quotient = _mm512_maskz_mov_epi64(sheds, divide_base_under(value));
            let residue = _mm512_sub_epi64(value, multiply_base_wide(quotient));
            let incoming = _mm512_alignr_epi64::<7>(quotient, carried);
            _mm512_mask_storeu_epi64(
                total.add(at) as *mut i64,
                here,
                _mm512_add_epi64(residue, incoming),
            );
            carried = quotient;
            at += LANES;
        }
    }
}

/// Hand each word's high half to the word above it, once
///
/// The same shape as [`shed_upward`], against a base of two to the thirty-two
/// rather than the limb base, which makes the quotient a shift and the whole
/// pass exact. The decode side wants this one.
///
/// The topmost word only receives.
///
/// # Safety
///
/// The caller must have established that this machine has the wide permutes.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
pub unsafe fn shed_words_upward(words: &mut [u64]) {
    // SAFETY: every access is masked to what is left of the slice.
    unsafe {
        let len = words.len();
        if len < 2 {
            return;
        }
        let shedding = len - 1;
        let total = words.as_mut_ptr();
        let mut carried = _mm512_setzero_si512();

        let mut at = 0;
        while at < len {
            let here = low_mask((len - at).min(LANES)) as u8;
            let sheds = low_mask(shedding.saturating_sub(at).min(LANES)) as u8;
            let value = _mm512_maskz_loadu_epi64(here, total.add(at) as *const i64);
            let quotient = _mm512_maskz_mov_epi64(sheds, _mm512_srli_epi64::<32>(value));
            let residue = _mm512_sub_epi64(value, _mm512_slli_epi64::<32>(quotient));
            let incoming = _mm512_alignr_epi64::<7>(quotient, carried);
            _mm512_mask_storeu_epi64(
                total.add(at) as *mut i64,
                here,
                _mm512_add_epi64(residue, incoming),
            );
            carried = quotient;
            at += LANES;
        }
    }
}

/// Characters to digits, with a mask of the lanes that were not base58
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn translate(chars: __m512i, present: u64) -> (__m512i, u64) {
    // SAFETY: both halves of the table are aligned and whole.
    unsafe {
        let table_low = load(INVERSE.0.as_ptr());
        let table_high = load(INVERSE.0.as_ptr().add(64));
        let inside = present & !_mm512_movepi8_mask(chars);
        let mapped = _mm512_maskz_permutex2var_epi8(inside, table_low, chars, table_high);
        let bad = _mm512_cmpeq_epi8_mask(mapped, _mm512_setzero_si512()) & present;
        (_mm512_sub_epi8(mapped, _mm512_set1_epi8(1)), bad)
    }
}

/// Four trailing digits per limb folded with the leading one
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn fold_limbs(tail: __m512i, head: __m512i) -> __m512i {
    let pairs = _mm512_maddubs_epi16(tail, _mm512_set1_epi32(PAIR_58));
    let folded = _mm512_madd_epi16(pairs, _mm512_set1_epi32(PAIR_3364));
    _mm512_add_epi32(
        folded,
        _mm512_mullo_epi32(head, _mm512_set1_epi32(HEAD_WEIGHT)),
    )
}

/// Read a signature's encoding into the limbs it stands for
///
/// # Safety
///
/// As [`read_32`], with an encoding no longer than `MAX_ENCODED_64`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
pub unsafe fn read_64(encoded: &[u8], limbs: &mut [u32; LIMBS_64]) -> bool {
    // SAFETY: both loads are masked to the input's own length. The second
    // pointer is past the slice for a short input, but its mask is empty so
    // the lanes it would cover are never touched.
    unsafe {
        let len = encoded.len();
        let start = encoded.as_ptr();
        let present_low = low_mask(len);
        let present_high = low_mask(len.saturating_sub(64));
        let chars_low = _mm512_maskz_loadu_epi8(present_low, start as *const i8);
        let chars_high = _mm512_maskz_loadu_epi8(present_high, start.wrapping_add(64) as *const i8);
        let (loaded_low, bad_low) = translate(chars_low, present_low);
        let (loaded_high, bad_high) = translate(chars_high, present_high);
        if bad_low | bad_high != 0 {
            return false;
        }

        let prepend = DIGITS_64 - len;
        let shift = _mm512_set1_epi8(prepend as i8);
        let align_low = _mm512_sub_epi8(load(IOTA.0.as_ptr()), shift);
        let keep_low = match prepend >= 64 {
            true => 0,
            false => !0u64 << prepend,
        };
        let digits_low = _mm512_maskz_permutexvar_epi8(keep_low, align_low, loaded_low);
        let align_high = _mm512_sub_epi8(load(IOTA_HIGH.0.as_ptr()), shift);
        let keep_high = match prepend > 64 {
            true => !0u64 << (prepend - 64),
            false => !0u64,
        };
        let digits_high =
            _mm512_maskz_permutex2var_epi8(keep_high, loaded_low, align_high, loaded_high);

        let tail = _mm512_permutex2var_epi8(digits_low, load(TAIL_64.0.as_ptr()), digits_high);
        let head = _mm512_maskz_permutex2var_epi8(
            HEAD_LANES,
            digits_low,
            load(HEAD_64.0.as_ptr()),
            digits_high,
        );
        let tail_top = _mm512_permutexvar_epi8(load(TAIL_64_TOP.0.as_ptr()), digits_high);
        let head_top = _mm512_maskz_permutexvar_epi8(
            HEAD_LANES_TOP,
            load(HEAD_64_TOP.0.as_ptr()),
            digits_high,
        );

        let target = limbs.as_mut_ptr();
        _mm512_storeu_si512(target as *mut __m512i, fold_limbs(tail, head));
        _mm_storel_epi64(
            target.add(LIMBS_64 - TOP_LIMBS) as *mut __m128i,
            _mm512_castsi512_si128(fold_limbs(tail_top, head_top)),
        );
        true
    }
}

/// One limb broadcast against one widened table row, onto eight columns
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn column_multiply_add(total: __m512i, limb: u32, row: *const u64) -> __m512i {
    // SAFETY: `vpmuludq` reads the low half of each lane, which the broadcast
    // fills, and the row is a whole aligned register of the widened table.
    unsafe {
        _mm512_add_epi64(
            total,
            _mm512_mul_epu32(_mm512_set1_epi32(limb as i32), load(row as *const u8)),
        )
    }
}

/// Sum a signature's limbs against the decode table
///
/// # Safety
///
/// As [`words_from_limbs_32`].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
pub unsafe fn words_from_limbs_64(limbs: &[u32; LIMBS_64], wide: &mut [u64; WORDS_64]) {
    // SAFETY: as above, over two column registers.
    unsafe {
        let mut low = _mm512_setzero_si512();
        let mut high = _mm512_setzero_si512();
        for (at, limb) in limbs.iter().enumerate() {
            let row = DECODE_64_WIDE.0[at].as_ptr();
            low = column_multiply_add(low, *limb, row);
            high = column_multiply_add(high, *limb, row.add(8));
        }
        let target = wide.as_mut_ptr();
        _mm512_storeu_si512(target as *mut __m512i, low);
        _mm512_storeu_si512(target.add(8) as *mut __m512i, high);
    }
}
