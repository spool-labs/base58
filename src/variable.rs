//! The codec for inputs that are neither a key nor a signature
//!
//! There is no table for an arbitrary width, so the value is divided down
//! instead. What keeps it cheap is the size of the step: each division pulls out
//! a whole base 58^5 limb across 32-bit words, where the textbook form pulls out
//! one digit across bytes.

use crate::error::{DecodeError, EncodeError};
use crate::fold::fold_in_place;
use crate::scalar::{
    check_leading_ones, digit_of, leading_zero_bytes, ALPHABET, DIGITS_PER_LIMB, LIMB_BASE,
};

/// Longest input this codec converts in one piece
///
/// One Solana packet, which is what the largest thing anyone base58s here is:
/// a serialized transaction on its way through `sendTransaction`.
pub const MAX_VARIABLE_LEN: usize = 1232;

/// Longest input this codec accepts at all
///
/// The conversion is quadratic, so the ceiling is what bounds the work one
/// call can buy with a length it chose. Without an allocator the stack scratch
/// is the tighter of the two.
#[cfg(feature = "alloc")]
pub const MAX_ACCEPTED_LEN: usize = 4096;

/// Longest input this codec accepts at all, which is what the stack holds
#[cfg(not(feature = "alloc"))]
pub const MAX_ACCEPTED_LEN: usize = MAX_VARIABLE_LEN;

/// Words the widest input occupies
const MAX_WORDS: usize = MAX_VARIABLE_LEN.div_ceil(4);

// The tables are generated against the limit above, so a limit that moved
// without them says so here rather than by converting something wrongly.

/// Longest encoding an input of this many bytes can produce
pub const fn encoded_len(input_len: usize) -> usize {
    input_len * 138 / 100 + 2
}

/// Most bytes an encoding of this many characters can produce
///
/// A leading one stands for a zero byte apiece, so an encoding that is all
/// ones is as many bytes as it is characters. Everything else is shorter.
pub const fn decoded_len(encoded_len: usize) -> usize {
    let scaled = encoded_len * 733 / 1000 + 2;
    match scaled > encoded_len {
        true => scaled,
        false => encoded_len,
    }
}

/// Encode bytes of any length up to the codec's limit
pub fn encode(input: &[u8], out: &mut [u8]) -> Result<usize, EncodeError> {
    if input.len() > MAX_ACCEPTED_LEN {
        return Err(EncodeError::InputTooLong);
    }
    let zeros = leading_zero_bytes(input);
    if out.len() < encoded_len(input.len()) {
        return Err(EncodeError::OutputTooSmall);
    }
    if let Some(written) = spell_fixed(input, out) {
        return Ok(written);
    }

    let count = spell(&input[zeros..], &mut out[zeros..]);

    for slot in out.iter_mut().take(zeros) {
        *slot = ALPHABET[0];
    }
    Ok(zeros + count)
}

/// Spell a key or a signature through its fixed path
///
/// The same value for a fifth the time, and most of what anyone converts is
/// one of the two. Decoding cannot take this road: an encoding's width does
/// not say which width the caller meant.
fn spell_fixed(input: &[u8], out: &mut [u8]) -> Option<usize> {
    match input.len() {
        crate::KEY_LEN => {
            let key = input.try_into().ok()?;
            let slot = out.get_mut(..crate::MAX_ENCODED_32)?.try_into().ok()?;
            Some(crate::backend::encode_32(key, slot))
        }
        crate::SIGNATURE_LEN => {
            let signature = input.try_into().ok()?;
            let slot = out.get_mut(..crate::MAX_ENCODED_64)?.try_into().ok()?;
            Some(crate::backend::encode_64(signature, slot))
        }
        _ => None,
    }
}

/// Spell the value as characters, returning how many
fn spell(src: &[u8], out: &mut [u8]) -> usize {
    if src.len() >= FOLD_BYTES {
        return spell_by_folding(src, out);
    }
    let mut words = [0u32; MAX_WORDS];
    let used = load_words(src, &mut words);
    spell_by_dividing(&mut words, used, out)
}

/// Bytes below which dividing the value down beats folding it
const FOLD_BYTES: usize = 128;

/// Bytes folded into the value in one step
const BLOCK: usize = 64;

/// Limbs 2^512 occupies in limb base
pub(crate) const BLOCK_LIMBS: usize = 18;

/// Limbs a block's own value occupies, with a slot of headroom
pub(crate) const BLOCK_DIGITS: usize = 19;

/// Limbs the value and its window need for an input this long
const fn fold_limbs(input_len: usize) -> usize {
    (input_len * 138 / 100 + 2).div_ceil(DIGITS_PER_LIMB) + BLOCK_DIGITS + 2
}

/// Limbs the widest input the stack path takes needs
const FOLD_LIMBS: usize = fold_limbs(MAX_VARIABLE_LEN);

/// Multiply a little-endian bignum by two, in limb base
const fn doubled(mut limbs: [u32; BLOCK_LIMBS]) -> [u32; BLOCK_LIMBS] {
    let mut carry = 0u64;
    let mut at = 0;
    while at < BLOCK_LIMBS {
        let value = limbs[at] as u64 * 2 + carry;
        limbs[at] = (value % LIMB_BASE) as u32;
        carry = value / LIMB_BASE;
        at += 1;
    }
    assert!(carry == 0, "a power of two ran past the limbs it was given");
    limbs
}

/// 2^exponent as little-endian limbs
const fn power_of_two(exponent: usize) -> [u32; BLOCK_LIMBS] {
    let mut limbs = [0u32; BLOCK_LIMBS];
    limbs[0] = 1;
    let mut done = 0;
    while done < exponent {
        limbs = doubled(limbs);
        done += 1;
    }
    limbs
}

/// What each of a block's sixteen words is worth
///
/// Shifted one place along rather than raised from scratch each time, which
/// keeps the compile-time cost down.
const fn block_places() -> [[u32; BLOCK_LIMBS]; 16] {
    let mut table = [[0u32; BLOCK_LIMBS]; 16];
    let mut value = [0u32; BLOCK_LIMBS];
    value[0] = 1;
    let mut at = 0;
    while at < 16 {
        table[15 - at] = value;
        let mut done = 0;
        while done < 32 {
            value = doubled(value);
            done += 1;
        }
        at += 1;
    }
    table
}

/// What a whole block is worth, which is what a fold multiplies by
pub(crate) const POWER: [u32; BLOCK_LIMBS] = power_of_two(512);

/// The same, highest place first and a lane apiece
///
/// A column reads its window forward against this, and a vector multiply
/// takes the low half of each 64-bit lane, so the places are widened once
/// here rather than on every load.
#[cfg(target_arch = "x86_64")]
pub(crate) const POWER_REV: [u64; BLOCK_LIMBS] = reversed(POWER);

/// Turn the places around and widen them
#[cfg(target_arch = "x86_64")]
const fn reversed(limbs: [u32; BLOCK_LIMBS]) -> [u64; BLOCK_LIMBS] {
    let mut lanes = [0u64; BLOCK_LIMBS];
    let mut at = 0;
    while at < BLOCK_LIMBS {
        lanes[at] = limbs[BLOCK_LIMBS - 1 - at] as u64;
        at += 1;
    }
    lanes
}

/// What each word of a block is worth
const PLACE: [[u32; BLOCK_LIMBS]; 16] = block_places();

/// Spell the value by folding it a block at a time
///
/// Each step is value = value * 2^512 + block, so the value is reduced once
/// per limb per 64 bytes rather than once per limb per word.
fn spell_by_folding(src: &[u8], out: &mut [u8]) -> usize {
    let needed = fold_limbs(src.len());
    if needed > FOLD_LIMBS {
        #[cfg(feature = "alloc")]
        {
            let mut value = alloc::vec![0u32; needed];
            return fold_and_spell(src, &mut value, out);
        }
        #[cfg(not(feature = "alloc"))]
        unreachable!("encode turns away anything the stack path cannot hold");
    }
    let mut value = [0u32; FOLD_LIMBS];
    fold_and_spell(src, &mut value, out)
}

/// Fold the input into the limbs it is handed, then spell them
fn fold_and_spell(src: &[u8], value: &mut [u32], out: &mut [u8]) -> usize {
    // Whatever is not a whole block, taken while the value is still short.
    let (head, blocks) = src.split_at(src.len() % BLOCK);
    let ragged = head.len() % 4;
    let mut count = 0;
    if ragged > 0 {
        let mut word = 0u64;
        for byte in &head[..ragged] {
            word = (word << 8) | *byte as u64;
        }
        raise(value, &mut count, 1u64 << (8 * ragged), word);
    }
    for chunk in head[ragged..].as_chunks::<4>().0 {
        let word = u32::from_be_bytes(*chunk);
        raise(value, &mut count, 1u64 << 32, word as u64);
    }

    for block in blocks.as_chunks::<BLOCK>().0 {
        let digits = block_digits(block);
        count = fold_in_place(value, count, &digits);
    }

    write_limbs(&value[..count], out)
}

/// `value = value * scale + word`, for the head of an input
fn raise(value: &mut [u32], count: &mut usize, scale: u64, word: u64) {
    let mut carry = word;
    for limb in value[..*count].iter_mut() {
        let wide = *limb as u64 * scale + carry;
        *limb = (wide % LIMB_BASE) as u32;
        carry = wide / LIMB_BASE;
    }
    while carry > 0 {
        value[*count] = (carry % LIMB_BASE) as u32;
        carry /= LIMB_BASE;
        *count += 1;
    }
}

/// Read a block's sixteen words as limbs
///
/// The halves are reduced apart so a column stays inside a u64.
fn block_digits(block: &[u8]) -> [u32; BLOCK_DIGITS] {
    let mut words = [0u32; 16];
    for (slot, chunk) in words.iter_mut().zip(block.as_chunks::<4>().0) {
        *slot = u32::from_be_bytes(*chunk);
    }

    let mut digits = [0u32; BLOCK_DIGITS];
    let mut wide = [0u64; BLOCK_LIMBS];
    for half in [0usize, 8] {
        for (word, row) in words[half..half + 8].iter().zip(PLACE[half..].iter()) {
            let word = *word as u64;
            for (slot, place) in wide.iter_mut().zip(row.iter()) {
                *slot += word * *place as u64;
            }
        }
        let mut carry = 0u64;
        for (slot, digit) in wide.iter_mut().zip(digits.iter_mut()) {
            let value = *slot + carry;
            *digit = (value % LIMB_BASE) as u32;
            carry = value / LIMB_BASE;
            *slot = 0;
        }
        digits[BLOCK_LIMBS] = carry as u32;
        if half == 0 {
            for (slot, digit) in wide.iter_mut().zip(digits.iter()) {
                *slot = *digit as u64;
            }
            digits[BLOCK_LIMBS] = 0;
        }
    }
    digits
}

/// Spell the value by dividing it down by the limb base a limb at a time
///
/// Each division waits on the one before it, which is what the table path
/// exists to avoid.
fn spell_by_dividing(words: &mut [u32; MAX_WORDS], used: usize, out: &mut [u8]) -> usize {
    // Each division hands back the least significant limb still there, so the
    // characters arrive backwards. They go straight into the output anyway and
    // are turned around at the end, which is one pass over what was written
    // rather than a buffer as wide as the widest input this codec takes.
    let mut count = 0;
    let mut highest = 0;
    while highest < used {
        let limb = divide_by_limb_base(&mut words[highest..used]);
        let mut digits = [0u8; DIGITS_PER_LIMB];
        for (offset, digit) in digits.iter_mut().enumerate() {
            *digit = ((limb / pow58(offset)) % 58) as u8;
        }
        while highest < used && words[highest] == 0 {
            highest += 1;
        }
        // Only the last limb can hold digits the value does not need, and they
        // are the ones above it, which is where this stops writing.
        let mut take = DIGITS_PER_LIMB;
        if highest == used {
            while take > 0 && digits[take - 1] == 0 {
                take -= 1;
            }
        }
        for digit in &digits[..take] {
            out[count] = ALPHABET[*digit as usize];
            count += 1;
        }
    }
    out[..count].reverse();
    count
}

/// Spell settled limbs as characters, least significant first, then turn
/// them around
fn write_limbs(limbs: &[u32], out: &mut [u8]) -> usize {
    let mut top = limbs.len();
    while top > 0 && limbs[top - 1] == 0 {
        top -= 1;
    }
    let mut written = 0;
    for (at, limb) in limbs[..top].iter().enumerate() {
        let mut value = *limb as u64;
        // The highest limb spells only the digits the value reaches into.
        let mut take = DIGITS_PER_LIMB;
        if at + 1 == top {
            take = 0;
            let mut left = value;
            while left > 0 {
                take += 1;
                left /= 58;
            }
        }
        for _ in 0..take {
            out[written] = ALPHABET[(value % 58) as usize];
            value /= 58;
            written += 1;
        }
    }
    out[..written].reverse();
    written
}

/// Decode characters of any length up to the codec's limit
pub fn decode(encoded: &[u8], out: &mut [u8]) -> Result<usize, DecodeError> {
    if encoded.len() > encoded_len(MAX_ACCEPTED_LEN) {
        return Err(DecodeError::TooLong);
    }
    let ones = leading_ones(encoded);
    let needed = value_words(encoded.len());
    if needed > MAX_VALUE_WORDS {
        #[cfg(feature = "alloc")]
        {
            let mut words = alloc::vec![0u64; needed];
            return lay_out(encoded, ones, &mut words, out);
        }
        #[cfg(not(feature = "alloc"))]
        unreachable!("decode turns away anything the stack path cannot hold");
    }
    let mut words = [0u64; MAX_VALUE_WORDS];
    lay_out(encoded, ones, &mut words, out)
}

/// Read the characters into the words handed over, then lay them out as bytes
fn lay_out(
    encoded: &[u8],
    ones: usize,
    words: &mut [u64],
    out: &mut [u8],
) -> Result<usize, DecodeError> {
    let used = gather(&encoded[ones..], words)?;

    // Only the topmost word can carry leading zero bytes, and `gather` never
    // leaves it empty, so the whole value's leading run is that word's.
    let skip = match used {
        0 => 0,
        _ => words[used - 1].leading_zeros() as usize / 8,
    };
    let body = used * 8 - skip;
    // A run of ones is one byte apiece, so an encoding short enough to accept
    // can still stand for a value this codec would refuse to encode.
    if ones + body > MAX_ACCEPTED_LEN {
        return Err(DecodeError::TooLong);
    }
    if out.len() < ones + body {
        return Err(DecodeError::OutputTooLong);
    }
    for slot in out.iter_mut().take(ones) {
        *slot = 0;
    }
    let mut at = ones;
    for (index, word) in words[..used].iter().rev().enumerate() {
        let bytes = word.to_be_bytes();
        let from = match index {
            0 => skip,
            _ => 0,
        };
        out[at..at + 8 - from].copy_from_slice(&bytes[from..]);
        at += 8 - from;
    }
    check_leading_ones(&out[..ones + body], encoded)?;
    Ok(ones + body)
}

/// Base58 digits read in one pass over the value
///
/// Ten fits because 58^10 still multiplies a 64-bit word inside a u128.
const CHARS_PER_PASS: usize = 10;

/// `58^10`, what a whole pass is worth
const PASS_SCALE: u64 = 430_804_206_899_405_824;

/// Words an encoding this long can occupy, least significant first
const fn value_words(encoded_len: usize) -> usize {
    encoded_len.div_ceil(8) + 1
}

/// Words the widest encoding the stack path takes occupies
const MAX_VALUE_WORDS: usize = value_words(encoded_len(MAX_VARIABLE_LEN));

/// Read the characters into words, returning how many they filled
///
/// Each scaling waits on the one before it, so a pass is made as wide as a
/// u128 product allows.
fn gather(encoded: &[u8], words: &mut [u64]) -> Result<usize, DecodeError> {
    let mut used = 0;
    // The short run leads, so every other pass reads the same scale. Either
    // way the characters go front to back, which names a bad one by position.
    let ragged = encoded.len() % CHARS_PER_PASS;
    if ragged > 0 {
        let mut value = 0u64;
        let mut scale = 1u64;
        for byte in &encoded[..ragged] {
            value = value * 58 + digit_of(*byte)? as u64;
            scale *= 58;
        }
        used = multiply_add(words, used, scale, value)?;
    }
    for run in encoded[ragged..].as_chunks::<CHARS_PER_PASS>().0 {
        let mut value = 0u64;
        for byte in run {
            value = value * 58 + digit_of(*byte)? as u64;
        }
        used = multiply_add(words, used, PASS_SCALE, value)?;
    }
    Ok(used)
}

/// Leading base58 zeros, which stand for leading zero bytes
fn leading_ones(encoded: &[u8]) -> usize {
    let mut count = 0;
    while count < encoded.len() && encoded[count] == b'1' {
        count += 1;
    }
    count
}

/// Read bytes as big-endian words, left padded into the first word
///
/// The padding is counted rather than written, so the input is read where it
/// already is instead of through a copy of itself as wide as the codec's limit.
fn load_words(input: &[u8], words: &mut [u32; MAX_WORDS]) -> usize {
    let used = input.len().div_ceil(4);
    let pad = used * 4 - input.len();
    let mut read = 0;
    for (index, word) in words[..used].iter_mut().enumerate() {
        let mut value = 0u32;
        for byte in 0..4 {
            value <<= 8;
            if index * 4 + byte >= pad {
                value |= input[read] as u32;
                read += 1;
            }
        }
        *word = value;
    }
    used
}

/// Divide the value in place by the limb base, handing back the remainder
fn divide_by_limb_base(words: &mut [u32]) -> u64 {
    let mut carry = 0u64;
    for word in words.iter_mut() {
        let acc = (carry << 32) | *word as u64;
        *word = (acc / LIMB_BASE) as u32;
        carry = acc % LIMB_BASE;
    }
    carry
}

/// `value = value * scale + add`, over the words the value occupies
fn multiply_add(
    words: &mut [u64],
    used: usize,
    scale: u64,
    add: u64,
) -> Result<usize, DecodeError> {
    let mut carry = add as u128;
    let scale = scale as u128;
    for word in words[..used].iter_mut() {
        let wide = *word as u128 * scale + carry;
        *word = wide as u64;
        carry = wide >> 64;
    }
    // The words grow upward, so an oversized value is caught here.
    let mut grown = used;
    while carry > 0 {
        if grown == words.len() {
            return Err(DecodeError::TooLong);
        }
        words[grown] = carry as u64;
        carry >>= 64;
        grown += 1;
    }
    Ok(grown)
}

/// Powers of 58 below one limb
fn pow58(exponent: usize) -> u64 {
    const POWERS: [u64; DIGITS_PER_LIMB] = [1, 58, 3364, 195112, 11316496];
    POWERS[exponent]
}
