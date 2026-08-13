//! The codec for inputs that are neither a key nor a signature
//!
//! There is no table for an arbitrary width, so the value is divided down
//! instead. What keeps it cheap is the size of the step: each division pulls out
//! a whole base 58^5 limb across 32-bit words, where the textbook form pulls out
//! one digit across bytes.

#[cfg(feature = "variable")]
use crate::place_values::{
    LIMB_OFFSETS, LIMB_ROWS, LIMB_VALUES, LIMB_WIDTH, PLACE_OFFSETS, PLACE_ROWS, PLACE_VALUES,
    PLACE_WIDTH,
};

use crate::error::{DecodeError, EncodeError};
use crate::scalar::{
    check_leading_ones, digit_of, leading_zero_bytes, ALPHABET, DIGITS_PER_LIMB, LIMB_BASE,
};

/// Longest input this codec converts in one piece
///
/// One Solana packet, which is what the largest thing anyone base58s here is:
/// a serialized transaction on its way through `sendTransaction`.
pub const MAX_VARIABLE_LEN: usize = 1232;

/// Words the widest input occupies
const MAX_WORDS: usize = MAX_VARIABLE_LEN.div_ceil(4);

// The tables are generated against the limit above, so a limit that moved
// without them says so here rather than by converting something wrongly.
#[cfg(feature = "variable")]
const _: () = assert!(
    PLACE_ROWS >= MAX_WORDS,
    "place values do not reach MAX_VARIABLE_LEN, regenerate src/place_values.rs"
);
#[cfg(feature = "variable")]
const _: () = assert!(
    LIMB_ROWS >= (MAX_VARIABLE_LEN * 138 / 100 + 2).div_ceil(DIGITS_PER_LIMB),
    "limb places do not reach MAX_VARIABLE_LEN, regenerate src/place_values.rs"
);

/// Longest encoding an input of this many bytes can produce
pub fn encoded_len(input_len: usize) -> usize {
    input_len * 138 / 100 + 2
}

/// Most bytes an encoding of this many characters can produce
pub fn decoded_len(encoded_len: usize) -> usize {
    encoded_len * 733 / 1000 + 2
}

/// Encode bytes of any length up to the codec's limit
pub fn encode(input: &[u8], out: &mut [u8]) -> Result<usize, EncodeError> {
    if input.len() > MAX_VARIABLE_LEN {
        return Err(EncodeError::InputTooLong);
    }
    let zeros = leading_zero_bytes(input);
    if out.len() < encoded_len(input.len()) {
        return Err(EncodeError::OutputTooSmall);
    }
    if let Some(written) = spell_fixed(input, out) {
        return Ok(written);
    }

    let mut words = [0u32; MAX_WORDS];
    let used = load_words(&input[zeros..], &mut words);
    let count = spell(&mut words, used, &mut out[zeros..]);

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

/// Spell the value in `words[..used]` as characters, returning how many
///
/// A short value divides down; past [`HORNER_WORDS`] the Horner walk takes
/// over, which costs no table and no serial chain.
#[cfg(not(feature = "variable"))]
fn spell(words: &mut [u32; MAX_WORDS], used: usize, out: &mut [u8]) -> usize {
    match used < HORNER_WORDS {
        true => spell_by_dividing(words, used, out),
        false => spell_by_horner(&words[..used], out),
    }
}

/// Words above which Horner's walk beats dividing the value down
///
/// The two are level here and the walk pulls away past it, several times
/// over by a whole packet.
#[cfg(not(feature = "variable"))]
const HORNER_WORDS: usize = 32;

/// Limbs the widest input's encoding reaches, with the sheds' margin
#[cfg(not(feature = "variable"))]
const HORNER_LIMBS: usize = (MAX_VARIABLE_LEN * 138 / 100 + 2).div_ceil(DIGITS_PER_LIMB) + 2;

/// Spell the value by Horner's walk, without tables
///
/// Each word shifts the whole value 32 bits and lands on the lowest limb.
/// The shift and the sheds run every limb independently, which is the lane
/// shape the dividing chain lacks; [`crate::variable_simd`] runs them. The
/// tables replace this walk entirely, so it only exists without them.
#[cfg(not(feature = "variable"))]
fn spell_by_horner(words: &[u32], out: &mut [u8]) -> usize {
    let mut limbs = [0u64; HORNER_LIMBS];
    let count = crate::variable_simd::horner_places(words, &mut limbs);
    settle_upward(&mut limbs[..count]);
    write_limbs(&limbs[..count], out)
}

/// Spell the value in `words[..used]` as characters, returning how many
///
/// A short value is still quicker divided down: the table path clears limbs
/// enough for the widest input this codec takes whatever it is handed, and
/// below this many words that costs more than the divisions do. Measured; the
/// two are within a percent of each other at the word this turns on.
#[cfg(feature = "variable")]
fn spell(words: &mut [u32; MAX_WORDS], used: usize, out: &mut [u8]) -> usize {
    match used < TABLE_WORDS {
        true => spell_by_dividing(words, used, out),
        false => spell_by_table(words, used, out),
    }
}

/// Words below which dividing the value down beats reading the table
#[cfg(feature = "variable")]
const TABLE_WORDS: usize = 12;

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

/// Spell the value against the table of place values
///
/// Every word's contribution to every limb is a multiply-accumulate, and none
/// of them wait on each other. Carries are left to build up and shed in passes
/// of their own, so the only ordered work left is one settling at the end. This
/// is the same shape the fixed widths use, and it costs the table it reads:
/// see [`crate::place_values`].
#[cfg(feature = "variable")]
fn spell_by_table(words: &mut [u32; MAX_WORDS], used: usize, out: &mut [u8]) -> usize {
    if used == 0 {
        return 0;
    }
    // A word at the highest place carries two limbs past the place itself.
    let count = place_row(used - 1).len() + 2;
    let mut limbs = [0u64; PLACE_WIDTH];
    accumulate_places(&words[..used], count, &mut limbs);
    settle_upward(&mut limbs[..count]);
    write_limbs(&limbs[..count], out)
}

/// Spell settled limbs as characters, least significant first, then turn
/// them around
fn write_limbs(limbs: &[u64], out: &mut [u8]) -> usize {
    let mut top = limbs.len();
    while top > 0 && limbs[top - 1] == 0 {
        top -= 1;
    }
    let mut written = 0;
    for (at, limb) in limbs[..top].iter().enumerate() {
        let mut value = *limb;
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

/// Add every word's contribution to every limb it reaches
///
/// The accumulation is a row of multiply-accumulates against nothing else,
/// taken through whatever kernels the machine has: see
/// [`crate::variable_simd`].
#[cfg(feature = "variable")]
fn accumulate_places(words: &[u32], count: usize, limbs: &mut [u64; PLACE_WIDTH]) {
    crate::variable_simd::accumulate_places(words, count, limbs);
}

/// Words accumulated before the limbs are brought back within a word
///
/// A word is below 2^32 and a table entry below the limb base, so six of their
/// products is the most a limb can hold without passing what a `u64` counts.
#[cfg(feature = "variable")]
pub(crate) const SHED: usize = 6;

/// The limbs a power of two occupies, least significant first
#[cfg(feature = "variable")]
pub(crate) fn place_row(power: usize) -> &'static [u32] {
    let from = PLACE_OFFSETS[power] as usize;
    let until = PLACE_OFFSETS[power + 1] as usize;
    &PLACE_VALUES[from..until]
}

/// Hand each limb's overflow to the limb above it, once, without settling
///
/// Every quotient is taken from the limb's value before anything lands on
/// it, and the one in flight rides a register, so no division queues behind
/// another the way a settling pass makes them — and none of it round-trips
/// through the stores, which compiled to a chain three times the cost when
/// this walked downward in place. What is left is not below the limb base,
/// only far enough below what a `u64` holds that another [`SHED`] words can
/// land on it. The topmost limb only receives: shedding it would carry off
/// the end.
pub(crate) fn shed(limbs: &mut [u64]) {
    let Some(last) = limbs.len().checked_sub(1) else {
        return;
    };
    let mut carry = 0;
    for limb in limbs[..last].iter_mut() {
        let value = *limb;
        *limb = value % LIMB_BASE + carry;
        carry = value / LIMB_BASE;
    }
    limbs[last] += carry;
}

/// Carry every limb up until each one is below the limb base
fn settle_upward(limbs: &mut [u64]) {
    let mut carry = 0u64;
    for limb in limbs.iter_mut() {
        let value = *limb + carry;
        carry = value / LIMB_BASE;
        *limb = value % LIMB_BASE;
    }
    debug_assert_eq!(carry, 0, "the value ran past the limbs it was given");
}

/// Decode characters of any length up to the codec's limit
pub fn decode(encoded: &[u8], out: &mut [u8]) -> Result<usize, DecodeError> {
    if encoded.len() > encoded_len(MAX_VARIABLE_LEN) {
        return Err(DecodeError::TooLong);
    }
    let ones = leading_ones(encoded);
    let mut words = [0u32; MAX_WORDS];
    let used = gather(&encoded[ones..], &mut words)?;

    let value_len = used * 4;
    let mut buffer = [0u8; MAX_WORDS * 4];
    for (word, chunk) in words[..used]
        .iter()
        .zip(buffer[..value_len].chunks_exact_mut(4))
    {
        chunk.copy_from_slice(&word.to_be_bytes());
    }
    let skip = leading_zero_bytes(&buffer[..value_len]);
    let body = value_len - skip;
    if out.len() < ones + body {
        return Err(DecodeError::OutputTooLong);
    }
    for slot in out.iter_mut().take(ones) {
        *slot = 0;
    }
    out[ones..ones + body].copy_from_slice(&buffer[skip..value_len]);
    check_leading_ones(&out[..ones + body], encoded)?;
    Ok(ones + body)
}

/// Read the characters into `words`, most significant first, returning how many
#[cfg(not(feature = "variable"))]
fn gather(encoded: &[u8], words: &mut [u32; MAX_WORDS]) -> Result<usize, DecodeError> {
    gather_by_scaling(encoded, words)
}

/// Read the characters by scaling the value a run at a time
///
/// Each scaling waits on the one before it, which is what the table path
/// avoids — wherever the machine gives the tables lanes to pay for it.
fn gather_by_scaling(encoded: &[u8], words: &mut [u32; MAX_WORDS]) -> Result<usize, DecodeError> {
    let mut used = 0;
    let mut at = 0;
    while at < encoded.len() {
        let take = (encoded.len() - at).min(DIGITS_PER_LIMB);
        let mut limb = 0u64;
        let mut scale = 1u64;
        for byte in &encoded[at..at + take] {
            limb = limb * 58 + digit_of(*byte)? as u64;
            scale *= 58;
        }
        used = multiply_add(words, used, scale, limb)?;
        at += take;
    }
    Ok(used)
}

/// Read the characters into `words`, most significant first, returning how many
///
/// Every limb's contribution to every word is a multiply-accumulate against a
/// table of the limb base's powers, and none of them wait on each other. The
/// mirror of the table path in [`spell`], and it costs the same kind of table:
/// see [`crate::place_values`].
#[cfg(feature = "variable")]
fn gather(encoded: &[u8], words: &mut [u32; MAX_WORDS]) -> Result<usize, DecodeError> {
    // A machine without vector kernels reads faster by scaling: the table
    // product costs more than the chain it replaces without lanes behind it.
    if !crate::variable_simd::tables_win_decoding() {
        return gather_by_scaling(encoded, words);
    }
    if encoded.is_empty() {
        return Ok(0);
    }
    let count = encoded.len().div_ceil(DIGITS_PER_LIMB);

    // The run that is short is the leading one, so every other run sits at a
    // whole power of the limb base. Either reader takes them front to back
    // regardless, which names a bad character by where it is rather than by
    // which run holds it.
    let first = encoded.len() - (count - 1) * DIGITS_PER_LIMB;
    // A limb at the highest place carries two words past the place itself.
    let width = limb_row(count - 1).len() + 2;
    let mut wide = [0u64; LIMB_WIDTH];
    match count <= FUSED_LIMBS {
        true => read_into_words(encoded, count, first, width, &mut wide)?,
        false => read_through_buffer(encoded, count, first, width, &mut wide)?,
    }
    settle_words_upward(&mut wide[..width]);

    let mut top = width;
    while top > 0 && wide[top - 1] == 0 {
        top -= 1;
    }
    if top > MAX_WORDS {
        return Err(DecodeError::TooLong);
    }
    for (at, word) in wide[..top].iter().rev().enumerate() {
        words[at] = *word as u32;
    }
    Ok(top)
}

/// Limbs above which reading them into a buffer of their own pays for itself
///
/// The buffer costs the same to clear however short the input is, and below
/// this many limbs that fixed cost is more than the accumulation gives up by
/// having the reading of a run sitting in the middle of it. Measured; the two
/// are within a few percent of each other on either side of it.
#[cfg(feature = "variable")]
const FUSED_LIMBS: usize = 28;

/// Read each run of characters and add it in where it is read
#[cfg(feature = "variable")]
fn read_into_words(
    encoded: &[u8],
    count: usize,
    first: usize,
    width: usize,
    wide: &mut [u64; LIMB_WIDTH],
) -> Result<(), DecodeError> {
    let mut at = 0;
    for (read, index) in (0..count).rev().enumerate() {
        let take = if at == 0 { first } else { DIGITS_PER_LIMB };
        let mut limb = 0u64;
        for byte in &encoded[at..at + take] {
            limb = limb * 58 + digit_of(*byte)? as u64;
        }
        at += take;

        for (word, entry) in wide.iter_mut().zip(limb_row(index).iter()) {
            *word += limb * *entry as u64;
        }
        if (read + 1) % SHED == 0 {
            shed_words(&mut wide[..width]);
        }
    }
    Ok(())
}

/// Read every run of characters first, then add them all in
///
/// Reading is a chain per run and carries a branch for a character outside the
/// alphabet. Kept away from the accumulation, the accumulation is a row of
/// multiply-accumulates against nothing else, which is what lets it be taken
/// several at a time.
#[cfg(feature = "variable")]
fn read_through_buffer(
    encoded: &[u8],
    count: usize,
    first: usize,
    width: usize,
    wide: &mut [u64; LIMB_WIDTH],
) -> Result<(), DecodeError> {
    let mut limbs = [0u32; LIMB_ROWS];
    let mut at = 0;
    for index in (0..count).rev() {
        let take = if at == 0 { first } else { DIGITS_PER_LIMB };
        let mut value = 0u32;
        for byte in &encoded[at..at + take] {
            value = value * 58 + digit_of(*byte)? as u32;
        }
        limbs[index] = value;
        at += take;
    }

    crate::variable_simd::accumulate_limbs(&limbs[..count], width, wide);
    Ok(())
}

/// The words a power of the limb base occupies, least significant first
#[cfg(feature = "variable")]
pub(crate) fn limb_row(power: usize) -> &'static [u32] {
    let from = LIMB_OFFSETS[power] as usize;
    let until = LIMB_OFFSETS[power + 1] as usize;
    &LIMB_VALUES[from..until]
}

/// Hand each word's overflow to the word above it, once, without settling
///
/// The same register-riding walk as [`shed`], for the same reason.
#[cfg(feature = "variable")]
pub(crate) fn shed_words(wide: &mut [u64]) {
    let Some(last) = wide.len().checked_sub(1) else {
        return;
    };
    let mut carry = 0;
    for word in wide[..last].iter_mut() {
        let value = *word;
        *word = (value & 0xFFFF_FFFF) + carry;
        carry = value >> 32;
    }
    wide[last] += carry;
}

/// Carry every word up until each one is below 2^32
#[cfg(feature = "variable")]
fn settle_words_upward(wide: &mut [u64]) {
    let mut carry = 0u64;
    for word in wide.iter_mut() {
        let value = *word + carry;
        carry = value >> 32;
        *word = value & 0xFFFF_FFFF;
    }
    debug_assert_eq!(carry, 0, "the value ran past the words it was given");
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

/// Scale the value in place and add a limb, growing it by a word if it carries
fn multiply_add(
    words: &mut [u32; MAX_WORDS],
    used: usize,
    scale: u64,
    add: u64,
) -> Result<usize, DecodeError> {
    let mut carry = add;
    for word in words[..used].iter_mut().rev() {
        let acc = *word as u64 * scale + carry;
        *word = acc as u32;
        carry = acc >> 32;
    }
    let mut grown = used;
    while carry != 0 {
        if grown == MAX_WORDS {
            return Err(DecodeError::TooLong);
        }
        words.copy_within(..grown, 1);
        words[0] = carry as u32;
        carry >>= 32;
        grown += 1;
    }
    Ok(grown)
}

/// Powers of 58 below one limb
fn pow58(exponent: usize) -> u64 {
    const POWERS: [u64; DIGITS_PER_LIMB] = [1, 58, 3364, 195112, 11316496];
    POWERS[exponent]
}
