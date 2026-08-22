//! The portable codec every architecture falls back to
//!
//! A value is carried in limbs of base 58^5 rather than converted a digit at a
//! time. Going in, a fixed table turns the input's 32-bit words straight into
//! limbs; coming out, another turns limbs back into words. Both directions then
//! only have to settle carries, which is what replaces the long division a
//! general base58 codec runs per digit.

use crate::error::DecodeError;
use crate::tables::{DECODE_32, DECODE_64, ENCODE_32, ENCODE_64};

/// One limb holds five base58 digits
pub(crate) const DIGITS_PER_LIMB: usize = 5;

/// The base a limb counts in
pub(crate) const LIMB_BASE: u64 = 58 * 58 * 58 * 58 * 58;

/// Limbs a 32-byte value needs
pub(crate) const LIMBS_32: usize = 9;

/// Limbs a 64-byte value needs
pub(crate) const LIMBS_64: usize = 18;

/// Digits a 32-byte value expands into before leading zeros are dropped
pub(crate) const DIGITS_32: usize = LIMBS_32 * DIGITS_PER_LIMB;

/// Digits a 64-byte value expands into before leading zeros are dropped
pub(crate) const DIGITS_64: usize = LIMBS_64 * DIGITS_PER_LIMB;

/// Digit buffer for a key, rounded up so a vector path works in whole registers
pub(crate) const PADDED_32: usize = 48;

/// Digit buffer for a signature, rounded up the same way
pub(crate) const PADDED_64: usize = 96;

/// 32-bit words a 32-byte value holds
pub(crate) const WORDS_32: usize = 8;

/// 32-bit words a 64-byte value holds
pub(crate) const WORDS_64: usize = 16;

/// The base58 alphabet, in digit order
pub(crate) const ALPHABET: [u8; 58] =
    *b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// The alphabet padded to a power of two
///
/// A digit derived by dividing is a value the compiler cannot see the range of,
/// so indexing the alphabet with one costs a bounds check. Masking the index
/// against a table this size cannot leave it, and the mask folds away.
const CHARS: [u8; 64] = build_chars();

/// Index into [`CHARS`] that cannot address a character
const CHARS_MASK: u32 = 63;

const fn build_chars() -> [u8; 64] {
    let mut table = [ALPHABET[0]; 64];
    let mut at = 0;
    while at < 58 {
        table[at] = ALPHABET[at];
        at += 1;
    }
    table
}

/// Values a pair of digits counts through
#[cfg(not(target_os = "solana"))]
const PAIR_BASE: u32 = 58 * 58;

/// Every value below [`PAIR_BASE`] as the two characters it spells
///
/// Halves the work of splitting a limb: two divisions instead of four, and
/// three lookups instead of five.
#[cfg(not(target_os = "solana"))]
const PAIRS: [u16; 4096] = build_pairs();

/// Index into [`PAIRS`] that cannot address a pair
#[cfg(not(target_os = "solana"))]
const PAIRS_MASK: u32 = 4095;

#[cfg(not(target_os = "solana"))]
const fn build_pairs() -> [u16; 4096] {
    let mut table = [0u16; 4096];
    let mut at = 0;
    while at < PAIR_BASE as usize {
        // Little end first, so the pair reads out in the order it is written.
        table[at] = ALPHABET[at / 58] as u16 | (ALPHABET[at % 58] as u16) << 8;
        at += 1;
    }
    table
}

/// What fills a buffer past the characters a value spells
///
/// Anything but the character a zero digit spells, so a scan for the leading
/// ones stops inside the real characters even when the value is zero.
const PAD_CHAR: u8 = ALPHABET[1];

/// Encode a public key
pub fn encode_32(input: &[u8; 32], out: &mut [u8; crate::MAX_ENCODED_32]) -> usize {
    let words = to_words::<32, WORDS_32>(input);
    write_32(sum_32(&words), leading_zero_bytes(input), out)
}

/// Encode a signature
pub fn encode_64(input: &[u8; 64], out: &mut [u8; crate::MAX_ENCODED_64]) -> usize {
    let words = to_words::<64, WORDS_64>(input);
    write_64(sum_64(&words), leading_zero_bytes(input), out)
}

/// Spell a public key's limbs, as [`sum_32`] leaves them, into characters
///
/// The tail of the encoding, split out so that a caller holding limbs it
/// arrived at some other way reaches the same characters. Settling is folded
/// into the walk that spells them, which is what [`to_chars`] is for.
#[inline]
pub fn write_32(limbs: [u64; LIMBS_32], input_zeros: usize, out: &mut [u8]) -> usize {
    let chars = to_chars::<LIMBS_32, PADDED_32>(&limbs);
    write_chars(&chars, DIGITS_32, input_zeros, out)
}

/// Spell a signature's limbs, as [`sum_64`] leaves them, into characters
#[inline]
pub(crate) fn write_64(limbs: [u64; LIMBS_64], input_zeros: usize, out: &mut [u8]) -> usize {
    let chars = to_chars::<LIMBS_64, PADDED_64>(&limbs);
    write_chars(&chars, DIGITS_64, input_zeros, out)
}

/// Decode a public key
pub fn decode_32(encoded: &[u8], out: &mut [u8; 32]) -> Result<(), DecodeError> {
    let mut limbs = [0u32; LIMBS_32];
    read_32(encoded, &mut limbs)?;
    let words = words_from_limbs::<LIMBS_32, WORDS_32>(&limbs, &DECODE_32)?;
    from_words::<32, WORDS_32>(&words, out);
    check_leading_ones(out, encoded)
}

/// Decode a signature
pub fn decode_64(encoded: &[u8], out: &mut [u8; 64]) -> Result<(), DecodeError> {
    let mut limbs = [0u32; LIMBS_64];
    read_64(encoded, &mut limbs)?;
    let words = words_from_limbs::<LIMBS_64, WORDS_64>(&limbs, &DECODE_64)?;
    from_words::<64, WORDS_64>(&words, out);
    check_leading_ones(out, encoded)
}

/// Read a public key's encoding into the limbs it stands for
///
/// The front of the decoding, split out so that a caller converting several at
/// once reaches the same reader. The mirror of [`write_32`]. The limbs are
/// filled in rather than handed back, which keeps a buffer their width out of
/// the result the caller unwraps.
#[inline(always)]
pub(crate) fn read_32(encoded: &[u8], limbs: &mut [u32; LIMBS_32]) -> Result<(), DecodeError> {
    *limbs = limbs_from_encoded::<LIMBS_32, DIGITS_32>(encoded, crate::MAX_ENCODED_32)?;
    Ok(())
}

/// Read a signature's encoding into the limbs it stands for
#[inline(always)]
pub(crate) fn read_64(encoded: &[u8], limbs: &mut [u32; LIMBS_64]) -> Result<(), DecodeError> {
    *limbs = limbs_from_encoded::<LIMBS_64, DIGITS_64>(encoded, crate::MAX_ENCODED_64)?;
    Ok(())
}

/// Read the input as big-endian 32-bit words
pub fn to_words<const BYTES: usize, const WORDS: usize>(input: &[u8; BYTES]) -> [u32; WORDS] {
    let mut words = [0u32; WORDS];
    for (word, chunk) in words.iter_mut().zip(input.chunks_exact(4)) {
        *word = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    words
}

/// Write big-endian 32-bit words back out as bytes
pub(crate) fn from_words<const BYTES: usize, const WORDS: usize>(
    words: &[u32; WORDS],
    out: &mut [u8; BYTES],
) {
    for (word, chunk) in words.iter().zip(out.chunks_exact_mut(4)) {
        chunk.copy_from_slice(&word.to_be_bytes());
    }
}

/// Leading zero bytes, which become leading ones in the encoding
///
/// A value whose first byte is not zero is answered from that byte alone, which
/// is the ordinary case at both ends of the codec and the one the decoder waits
/// on. A value that does start with zeros is walked a word at a time: reading
/// the word little end first puts the earliest byte lowest, where counting the
/// zero bits below it says which byte that is.
pub fn leading_zero_bytes(input: &[u8]) -> usize {
    if !matches!(input.first(), Some(0)) {
        return 0;
    }
    let mut count = 0;
    let mut words = input.chunks_exact(8);
    for word in words.by_ref() {
        let value = u64::from_le_bytes(unwrap_eight(word));
        if value != 0 {
            return count + (value.trailing_zeros() / 8) as usize;
        }
        count += 8;
    }
    for byte in words.remainder() {
        if *byte != 0 {
            break;
        }
        count += 1;
    }
    count
}

/// Eight bytes as an array, which the caller has already sized the slice for
#[inline(always)]
fn unwrap_eight(bytes: &[u8]) -> [u8; 8] {
    let mut word = [0u8; 8];
    word.copy_from_slice(bytes);
    word
}

/// Marks a byte that is not a base58 character
const INVALID: u8 = 255;

/// Byte value back to digit, derived from the alphabet so the two cannot drift
///
/// One entry per byte value rather than per ASCII code, so a lookup keyed on a
/// byte is in bounds by construction and needs nothing checked.
const INVERSE: [u8; 256] = build_inverse();

const fn build_inverse() -> [u8; 256] {
    let mut table = [INVALID; 256];
    let mut at = 0;
    while at < 58 {
        table[ALPHABET[at] as usize] = at as u8;
        at += 1;
    }
    table
}

/// Sum a public key's words into limbs, before their carries are settled
///
/// Settling is left out because not every caller wants it on its own: the
/// portable path folds the carries into the walk that spells the characters,
/// where they cost the walk nothing, while a vector path settles first and
/// spells them from the settled limbs.
pub fn sum_32(words: &[u32; WORDS_32]) -> [u64; LIMBS_32] {
    let mut limbs = [0u64; LIMBS_32];
    accumulate::<WORDS_32, LIMBS_32, { LIMBS_32 - 1 }, 0, WORDS_32>(&mut limbs, words, &ENCODE_32);
    limbs
}

/// Sum a signature's words into limbs, before their carries are settled
///
/// The last limb would overflow if all sixteen words landed on it untouched, so
/// the sum pauses halfway and sheds what has already built up there.
pub(crate) fn sum_64(words: &[u32; WORDS_64]) -> [u64; LIMBS_64] {
    let mut limbs = [0u64; LIMBS_64];
    accumulate::<WORDS_64, LIMBS_64, { LIMBS_64 - 1 }, 0, 8>(&mut limbs, words, &ENCODE_64);
    limbs[LIMBS_64 - 3] += limbs[LIMBS_64 - 2] / LIMB_BASE;
    limbs[LIMBS_64 - 2] %= LIMB_BASE;
    accumulate::<WORDS_64, LIMBS_64, { LIMBS_64 - 1 }, 8, WORDS_64>(&mut limbs, words, &ENCODE_64);
    limbs
}

/// Add every input word's contribution to the limbs it lands on
///
/// One limb at a time, summed over the input. aarch64 has registers enough to
/// hold the running total for a limb while it walks the whole input.
#[cfg(target_arch = "aarch64")]
fn accumulate<
    const WORDS: usize,
    const LIMBS: usize,
    const COLUMNS: usize,
    const FROM: usize,
    const UNTIL: usize,
>(
    limbs: &mut [u64; LIMBS],
    words: &[u32; WORDS],
    table: &[[u32; COLUMNS]; WORDS],
) {
    for column in 0..COLUMNS {
        let mut total = 0u64;
        for row in FROM..UNTIL {
            total += words[row] as u64 * table[row][column] as u64;
        }
        limbs[column + 1] += total;
    }
}

/// Add every input word's contribution to the limbs it lands on
///
/// One input word at a time, spread across the limbs. x86-64 does not have the
/// registers to hold a limb open, and reaches the accumulators in memory more
/// cheaply than it keeps them live.
#[cfg(not(target_arch = "aarch64"))]
fn accumulate<
    const WORDS: usize,
    const LIMBS: usize,
    const COLUMNS: usize,
    const FROM: usize,
    const UNTIL: usize,
>(
    limbs: &mut [u64; LIMBS],
    words: &[u32; WORDS],
    table: &[[u32; COLUMNS]; WORDS],
) {
    for row in FROM..UNTIL {
        for column in 0..COLUMNS {
            limbs[column + 1] += words[row] as u64 * table[row][column] as u64;
        }
    }
}

/// Carry each limb down until every one is below the limb base
///
/// For the vector encode path, which spells from settled limbs. The portable
/// path folds the carries into [`to_chars`], and on x86 that is what every
/// path uses for a key.
#[cfg(target_arch = "aarch64")]
pub(crate) fn settle<const LIMBS: usize>(limbs: &mut [u64; LIMBS]) {
    for at in (1..LIMBS).rev() {
        limbs[at - 1] += limbs[at] / LIMB_BASE;
        limbs[at] %= LIMB_BASE;
    }
}

/// Expand every limb into the five digits it holds
///
/// The buffer runs past the digits so a vector path can work in whole
/// registers. What follows them is non-zero, which keeps a scan for leading
/// zeros inside the real digits even when the value is zero.
#[cfg(target_arch = "aarch64")]
pub(crate) fn to_digits<const LIMBS: usize, const PADDED: usize>(
    limbs: &[u64; LIMBS],
) -> [u8; PADDED] {
    let mut digits = [1u8; PADDED];
    for (at, limb) in limbs.iter().enumerate() {
        let value = *limb as u32;
        let first = value / 58;
        let second = first / 58;
        let third = second / 58;
        let fourth = third / 58;
        let base = at * DIGITS_PER_LIMB;
        digits[base] = fourth as u8;
        digits[base + 1] = (third - fourth * 58) as u8;
        digits[base + 2] = (second - third * 58) as u8;
        digits[base + 3] = (first - second * 58) as u8;
        digits[base + 4] = (value - first * 58) as u8;
    }
    digits
}

/// Settle the limbs and spell each one as the five characters it holds
///
/// Takes the limbs as [`sum_32`] and [`sum_64`] leave them, carries and all.
/// Settling is a chain: a limb is not final until the one above it has
/// handed down what it owes, and nothing breaks that. Splitting a limb into
/// characters is not part of the chain, though, so it runs in the room the
/// chain leaves rather than in a pass of its own over the whole buffer. The
/// split goes two digits at a time through [`PAIRS`], leaving the odd one out
/// to [`CHARS`].
pub(crate) fn to_chars<const LIMBS: usize, const PADDED: usize>(
    limbs: &[u64; LIMBS],
) -> [u8; PADDED] {
    let mut chars = [PAD_CHAR; PADDED];
    let spelled = LIMBS * DIGITS_PER_LIMB;
    let mut carry = 0u64;
    for (limb, slot) in limbs
        .iter()
        .zip(chars[..spelled].chunks_exact_mut(DIGITS_PER_LIMB))
        .rev()
    {
        let settled = limb + carry;
        carry = settled / LIMB_BASE;
        let value = (settled - carry * LIMB_BASE) as u32;
        #[cfg(not(target_os = "solana"))]
        {
            let high = value / PAIR_BASE;
            let low = value - high * PAIR_BASE;
            let first = high / PAIR_BASE;
            let middle = high - first * PAIR_BASE;
            slot[0] = CHARS[(first & CHARS_MASK) as usize];
            slot[1..3].copy_from_slice(&PAIRS[(middle & PAIRS_MASK) as usize].to_le_bytes());
            slot[3..5].copy_from_slice(&PAIRS[(low & PAIRS_MASK) as usize].to_le_bytes());
        }
        // Digit at a time through [`CHARS`]: two more divides per limb, no
        // eight-kilobyte table in the binary
        #[cfg(target_os = "solana")]
        {
            let first = value / 58;
            let second = first / 58;
            let third = second / 58;
            let fourth = third / 58;
            slot[0] = CHARS[(fourth & CHARS_MASK) as usize];
            slot[1] = CHARS[((third - fourth * 58) & CHARS_MASK) as usize];
            slot[2] = CHARS[((second - third * 58) & CHARS_MASK) as usize];
            slot[3] = CHARS[((first - second * 58) & CHARS_MASK) as usize];
            slot[4] = CHARS[((value - first * 58) & CHARS_MASK) as usize];
        }
    }
    chars
}

/// Characters standing for a zero digit before the value starts
///
/// Eight at a time, against a word of the character a zero digit spells. The
/// buffer's padding is not that character, so the scan stops on its own.
fn leading_ones<const PADDED: usize>(chars: &[u8; PADDED]) -> usize {
    const ONES: u64 = u64::from_le_bytes([ALPHABET[0]; 8]);
    let mut count = 0;
    while count + 8 <= PADDED {
        let word = u64::from_le_bytes(unwrap_eight(&chars[count..count + 8]));
        let differs = word ^ ONES;
        if differs != 0 {
            return count + (differs.trailing_zeros() / 8) as usize;
        }
        count += 8;
    }
    count
}

/// Drop the characters the value does not need and hand back the rest
///
/// A value spells at least as many leading ones as the input had zero bytes, so
/// dropping the difference leaves exactly one character per leading zero byte.
pub(crate) fn write_chars<const PADDED: usize>(
    chars: &[u8; PADDED],
    count: usize,
    input_zeros: usize,
    out: &mut [u8],
) -> usize {
    let skip = leading_ones(chars).saturating_sub(input_zeros);
    let written = (count - skip).min(out.len());
    out[..written].copy_from_slice(&chars[skip..skip + written]);
    written
}

/// One character's digit value
pub(crate) fn digit_of(byte: u8) -> Result<u8, DecodeError> {
    match INVERSE[byte as usize] {
        INVALID => Err(DecodeError::InvalidCharacter(byte)),
        digit => Ok(digit),
    }
}

/// Read the characters straight into limbs, right aligned against the widest value
///
/// Left padding with the character a zero digit spells is what right aligning
/// comes to, and it costs one fixed-width fill. Every position then holds a
/// character, so five of them fold into a limb as they are read and the buffer
/// of digits that would sit between the two is never written at all.
pub(crate) fn limbs_from_encoded<const LIMBS: usize, const DIGITS: usize>(
    encoded: &[u8],
    max_encoded: usize,
) -> Result<[u32; LIMBS], DecodeError> {
    if encoded.len() > max_encoded {
        return Err(DecodeError::TooLong);
    }
    let mut chars = [0u8; DIGITS];
    let offset = DIGITS - encoded.len();
    chars[..offset].fill(ALPHABET[0]);
    chars[offset..].copy_from_slice(encoded);

    let mut limbs = [0u32; LIMBS];
    let mut seen = 0u8;
    for (limb, group) in limbs.iter_mut().zip(chars.chunks_exact(DIGITS_PER_LIMB)) {
        let mut value = 0u32;
        for byte in group {
            let digit = INVERSE[*byte as usize];
            seen |= digit;
            value = value * 58 + digit as u32;
        }
        *limb = value;
    }
    // A digit is at most 57, so only the marker can set either of the top two
    // bits, and one test over their union stands in for a test per character.
    if seen & 0xC0 != 0 {
        return Err(invalid_character(encoded));
    }
    Ok(limbs)
}

/// Name the character the alphabet does not hold, once one is known to be there
#[cold]
#[inline(never)]
fn invalid_character(encoded: &[u8]) -> DecodeError {
    match encoded
        .iter()
        .find(|byte| INVERSE[**byte as usize] == INVALID)
    {
        Some(byte) => DecodeError::InvalidCharacter(*byte),
        None => DecodeError::InvalidCharacter(0),
    }
}

/// Gather each run of five digits back into the limb it came from
///
/// For a vector path, which reads the characters into digits with a permute.
/// The portable path folds them into limbs as it reads them, in
/// [`limbs_from_encoded`].
#[cfg(target_arch = "aarch64")]
pub(crate) fn digits_to_limbs<const LIMBS: usize, const DIGITS: usize>(
    digits: &[u8; DIGITS],
) -> [u32; LIMBS] {
    let mut limbs = [0u32; LIMBS];
    for (at, limb) in limbs.iter_mut().enumerate() {
        let mut value = 0u32;
        for offset in 0..DIGITS_PER_LIMB {
            value = value * 58 + digits[at * DIGITS_PER_LIMB + offset] as u32;
        }
        *limb = value;
    }
    limbs
}

/// Turn limbs back into 32-bit words, rejecting a value too wide to fit
pub(crate) fn words_from_limbs<const LIMBS: usize, const WORDS: usize>(
    limbs: &[u32; LIMBS],
    table: &[[u32; WORDS]; LIMBS],
) -> Result<[u32; WORDS], DecodeError> {
    let mut wide = [0u64; WORDS];
    accumulate_words(limbs, table, &mut wide);
    settle_words(&mut wide)
}

/// Sum every limb's contribution to each word
pub(crate) fn accumulate_words<const LIMBS: usize, const WORDS: usize>(
    limbs: &[u32; LIMBS],
    table: &[[u32; WORDS]; LIMBS],
    wide: &mut [u64; WORDS],
) {
    for (at, slot) in wide.iter_mut().enumerate() {
        let mut total = 0u64;
        for (limb, row) in limbs.iter().zip(table.iter()) {
            total += *limb as u64 * row[at] as u64;
        }
        *slot = total;
    }
}

/// Sum several inputs' limbs against one shared walk of the decode table
///
/// The mirror of the encode side's interleave: one chain per input leaves most
/// of the multiplier idle, and the lanes are independent, so their work fills
/// the gaps each other leaves. All the lanes run whether or not the caller
/// filled them, since a lane of zeros contributes nothing and counting them at
/// runtime would put a loop of unknown length innermost.
pub(crate) fn words_lanes<const LIMBS: usize, const WORDS: usize, const LANES: usize>(
    limbs: &[[u32; LIMBS]; LANES],
    table: &[[u32; WORDS]; LIMBS],
    wide: &mut [[u64; WORDS]; LANES],
) {
    for word in 0..WORDS {
        let mut totals = [0u64; LANES];
        for (row, entries) in table.iter().enumerate() {
            let factor = entries[word] as u64;
            for lane in 0..LANES {
                totals[lane] += limbs[lane][row] as u64 * factor;
            }
        }
        for lane in 0..LANES {
            wide[lane][word] = totals[lane];
        }
    }
}

/// Carry the words down and reject a value wider than the output
pub(crate) fn settle_words<const WORDS: usize>(
    wide: &mut [u64; WORDS],
) -> Result<[u32; WORDS], DecodeError> {
    for at in (1..WORDS).rev() {
        wide[at - 1] += wide[at] >> 32;
        wide[at] &= 0xFFFF_FFFF;
    }
    if wide[0] > 0xFFFF_FFFF {
        return Err(DecodeError::ValueTooLarge);
    }
    let mut words = [0u32; WORDS];
    for (word, value) in words.iter_mut().zip(wide.iter()) {
        *word = *value as u32;
    }
    Ok(words)
}

/// Hold the encoding's leading ones to the value's leading zero bytes
///
/// Base58 has no other way to say how many zero bytes a value starts with, so an
/// encoding that disagrees decodes to something the caller did not ask for.
pub(crate) fn check_leading_ones(out: &[u8], encoded: &[u8]) -> Result<(), DecodeError> {
    let zeros = leading_zero_bytes(out);
    if encoded.len() < zeros {
        return Err(DecodeError::TooShort);
    }
    for byte in &encoded[..zeros] {
        if *byte != b'1' {
            return Err(DecodeError::TooShort);
        }
    }
    match encoded.get(zeros) {
        Some(&b'1') => Err(DecodeError::OutputTooLong),
        Some(_) | None => Ok(()),
    }
}
