//! The variable codec's accumulation, taken a register at a time
//!
//! Two walks route through here. The table product is the same matrix the
//! fixed widths compute, only ragged: every scale-times-row product lands on
//! its own limb and none of them wait on each other. Horner's walk carries
//! no table at all: each word shifts the whole value and sheds, and the
//! shift and the sheds are every limb at once too. Both keep the scalar
//! codec's cadence and leave the exact settling pass unchanged at the end.
//!
//! The x86-64 kernels live beside the fixed-width ones in [`crate::avx512`]
//! and [`crate::avx2`], bounded by the slices they are given. Their
//! `shed_upward` undershoots each quotient by at most two, so limbs land
//! below three times the base rather than one — still room for what the next
//! rows add, and the settle at the end is exact either way. A build with no
//! target flags reaches them anyway: the baseline the compiler vectorises
//! against is SSE2, and the dispatch is what buys the wider registers.
//!
//! On aarch64 both walks stay as the scalar codec writes them. NEON is the
//! baseline there and the compiler already takes the loops in `umlal` pairs
//! with the accumulators in registers, which a hand-written kernel only
//! undoes by spilling them to memory.

use crate::variable::shed;

#[cfg(feature = "variable")]
use crate::variable::{limb_row, place_row, shed_words, SHED};

/// Words below which a wide machine still runs the 256-bit table kernels
///
/// The same too-few-lanes story as the fixed 32-byte path: the wider
/// registers only pay once there are enough limbs to fill them.
#[cfg(all(target_arch = "x86_64", feature = "variable"))]
const WIDE_WORDS: usize = 72;

/// Words below which Horner's walk stays scalar
///
/// The walk has less to vectorise than the table product, only the sheds, so
/// its dispatch clears a higher bar than [`WIDE_WORDS`].
#[cfg(all(target_arch = "x86_64", not(feature = "variable")))]
const HORNER_LANES: usize = 48;

/// Words past which a wide machine's Horner walk takes the 512-bit sheds
#[cfg(all(target_arch = "x86_64", not(feature = "variable")))]
const HORNER_WIDE: usize = 64;

/// Limbs below which a wide machine still decodes through the 256-bit kernels
///
/// The decode crossover sits higher than the encode one.
#[cfg(all(target_arch = "x86_64", feature = "variable"))]
const WIDE_LIMBS: usize = 112;

/// Every word shifted into the value and shed, without tables
///
/// The accumulation half of Horner's walk, returning how many limbs the
/// value reaches. The caller settles and spells them.
#[cfg(not(feature = "variable"))]
pub(crate) fn horner_places(words: &[u32], limbs: &mut [u64]) -> usize {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: gated on what the machine reported.
    return match crate::dispatch::path() {
        crate::dispatch::Path::Wide if words.len() >= HORNER_WIDE => unsafe {
            wide::horner(words, limbs)
        },
        crate::dispatch::Path::Wide | crate::dispatch::Path::Avx2
            if words.len() >= HORNER_LANES =>
        unsafe { avx2::horner(words, limbs) },
        _ => horner_scalar(words, limbs),
    };
    #[cfg(not(target_arch = "x86_64"))]
    horner_scalar(words, limbs)
}

/// Add every word's contribution to every limb it reaches
///
/// The vector form of the accumulation in [`crate::variable::spell`], same
/// cadence, same bounds.
#[cfg(feature = "variable")]
pub(crate) fn accumulate_places(words: &[u32], count: usize, limbs: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: gated on what the machine reported.
    match crate::dispatch::path() {
        crate::dispatch::Path::Wide if words.len() >= WIDE_WORDS => unsafe {
            wide::places(words, count, limbs)
        },
        crate::dispatch::Path::Wide | crate::dispatch::Path::Avx2 => unsafe {
            avx2::places(words, count, limbs)
        },
        crate::dispatch::Path::Portable => places_scalar(words, count, limbs),
    }
    #[cfg(not(target_arch = "x86_64"))]
    places_scalar(words, count, limbs);
}

/// Add every limb's contribution to every word it reaches
///
/// The vector form of the accumulation in the buffered reader, same cadence,
/// same bounds.
#[cfg(feature = "variable")]
pub(crate) fn accumulate_limbs(parsed: &[u32], width: usize, wide: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: gated on what the machine reported.
    match crate::dispatch::path() {
        crate::dispatch::Path::Wide if parsed.len() >= WIDE_LIMBS => unsafe {
            wide::limbs_in(parsed, width, wide)
        },
        crate::dispatch::Path::Wide | crate::dispatch::Path::Avx2 => unsafe {
            avx2::limbs_in(parsed, width, wide)
        },
        crate::dispatch::Path::Portable => limbs_scalar(parsed, width, wide),
    }
    #[cfg(not(target_arch = "x86_64"))]
    limbs_scalar(parsed, width, wide);
}

/// Whether the decode tables pay on this machine
///
/// Without the vector kernels the table product loses to scaling the value
/// directly, and with them it wins several times over, so the path with no
/// kernels answers no. The encode tables win everywhere, so only decode asks.
#[cfg(feature = "variable")]
pub(crate) fn tables_win_decoding() -> bool {
    #[cfg(target_arch = "x86_64")]
    return crate::dispatch::path() != crate::dispatch::Path::Portable;
    #[cfg(not(target_arch = "x86_64"))]
    true
}

/// Horner's walk as the scalar codec writes it
///
/// The shift rides inside the first shed: a limb below the base shifts to
/// under 2^63, divides once, and the word enters as the lowest limb's low
/// half — one pass where shift-then-shed is two. What that pass leaves
/// carries a few bits past 2^33, so a second, near-empty shed brings every
/// limb back low enough for the next shift. A word is 32 bits and a limb
/// holds a shade over 29, so the value needs twelve limbs for every eleven
/// words, and two more for the carries.
#[cfg(not(feature = "variable"))]
fn horner_scalar(words: &[u32], limbs: &mut [u64]) -> usize {
    use crate::scalar::LIMB_BASE;

    let mut count = 2;
    for (done, word) in words.iter().enumerate() {
        let first = (limbs[0] << 32) | *word as u64;
        limbs[0] = first % LIMB_BASE;
        let mut carry = first / LIMB_BASE;
        for limb in limbs[1..count - 1].iter_mut() {
            let value = *limb << 32;
            *limb = value % LIMB_BASE + carry;
            carry = value / LIMB_BASE;
        }
        // The topmost limb only receives: shedding it would carry off the
        // end, and the limbs stay ahead of the value, so it has the room.
        limbs[count - 1] = (limbs[count - 1] << 32) + carry;
        shed(&mut limbs[..count]);
        count = limbs.len().min(12 * (done + 2) / 11 + 2);
    }
    count
}

/// The walk the kernels stand in for, for a machine that has none
#[cfg(feature = "variable")]
fn places_scalar(words: &[u32], count: usize, limbs: &mut [u64]) {
    for (index, word) in words.iter().enumerate() {
        let entries = place_row(words.len() - 1 - index);
        let word = *word as u64;
        for (limb, entry) in limbs.iter_mut().zip(entries.iter()) {
            *limb += word * *entry as u64;
        }
        if (index + 1) % SHED == 0 {
            shed(&mut limbs[..count]);
        }
    }
}

/// As [`places_scalar`], reading the decode table
#[cfg(feature = "variable")]
fn limbs_scalar(parsed: &[u32], width: usize, wide: &mut [u64]) {
    for (index, limb) in parsed.iter().enumerate() {
        let entries = limb_row(index);
        let limb = *limb as u64;
        for (word, entry) in wide.iter_mut().zip(entries.iter()) {
            *word += limb * *entry as u64;
        }
        if (index + 1) % SHED == 0 {
            shed_words(&mut wide[..width]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod wide {
    use crate::avx512::shed_upward;
    #[cfg(feature = "variable")]
    use crate::avx512::{place_multiply_add, shed_words_upward};
    #[cfg(feature = "variable")]
    use crate::variable::{limb_row, place_row, SHED};

    /// Horner's walk with the sheds eight limbs at an instruction
    ///
    /// # Safety
    ///
    /// The caller must have established the wide permutes.
    #[cfg(not(feature = "variable"))]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
    pub(super) unsafe fn horner(words: &[u32], limbs: &mut [u64]) -> usize {
        let mut count = 2;
        for (done, word) in words.iter().enumerate() {
            for limb in limbs[..count].iter_mut() {
                *limb <<= 32;
            }
            limbs[0] += *word as u64;
            // SAFETY: the kernel bounds itself by the slice it is given.
            unsafe {
                shed_upward(&mut limbs[..count]);
                shed_upward(&mut limbs[..count]);
            }
            count = limbs.len().min(12 * (done + 2) / 11 + 2);
        }
        count
    }

    /// Every word against its row, eight products at an instruction
    ///
    /// # Safety
    ///
    /// As [`horner`].
    #[cfg(feature = "variable")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
    pub(super) unsafe fn places(words: &[u32], count: usize, limbs: &mut [u64]) {
        // SAFETY: the kernels bound themselves by the slices they are given.
        unsafe {
            for (index, word) in words.iter().enumerate() {
                place_multiply_add(*word, place_row(words.len() - 1 - index), limbs);
                if (index + 1) % SHED == 0 {
                    shed_upward(&mut limbs[..count]);
                }
            }
        }
    }

    /// Every limb against its row, eight products at an instruction
    ///
    /// # Safety
    ///
    /// As [`horner`].
    #[cfg(feature = "variable")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
    pub(super) unsafe fn limbs_in(parsed: &[u32], width: usize, wide: &mut [u64]) {
        // SAFETY: as above.
        unsafe {
            for (index, limb) in parsed.iter().enumerate() {
                place_multiply_add(*limb, limb_row(index), wide);
                if (index + 1) % SHED == 0 {
                    shed_words_upward(&mut wide[..width]);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use crate::avx2::shed_upward;
    #[cfg(feature = "variable")]
    use crate::avx2::{place_multiply_add, shed_words_upward};
    #[cfg(feature = "variable")]
    use crate::variable::{limb_row, place_row, SHED};

    /// Horner's walk with the sheds four limbs at an instruction
    ///
    /// # Safety
    ///
    /// The caller must have established AVX2.
    #[cfg(not(feature = "variable"))]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn horner(words: &[u32], limbs: &mut [u64]) -> usize {
        let mut count = 2;
        for (done, word) in words.iter().enumerate() {
            for limb in limbs[..count].iter_mut() {
                *limb <<= 32;
            }
            limbs[0] += *word as u64;
            // SAFETY: the kernel bounds itself by the slice it is given.
            unsafe {
                shed_upward(&mut limbs[..count]);
                shed_upward(&mut limbs[..count]);
            }
            count = limbs.len().min(12 * (done + 2) / 11 + 2);
        }
        count
    }

    /// Every word against its row, four products at an instruction
    ///
    /// # Safety
    ///
    /// As [`horner`].
    #[cfg(feature = "variable")]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn places(words: &[u32], count: usize, limbs: &mut [u64]) {
        // SAFETY: the kernels bound themselves by the slices they are given.
        unsafe {
            for (index, word) in words.iter().enumerate() {
                place_multiply_add(*word, place_row(words.len() - 1 - index), limbs);
                if (index + 1) % SHED == 0 {
                    shed_upward(&mut limbs[..count]);
                }
            }
        }
    }

    /// Every limb against its row, four products at an instruction
    ///
    /// # Safety
    ///
    /// As [`horner`].
    #[cfg(feature = "variable")]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn limbs_in(parsed: &[u32], width: usize, wide: &mut [u64]) {
        // SAFETY: as above.
        unsafe {
            for (index, limb) in parsed.iter().enumerate() {
                place_multiply_add(*limb, limb_row(index), wide);
                if (index + 1) % SHED == 0 {
                    shed_words_upward(&mut wide[..width]);
                }
            }
        }
    }
}
