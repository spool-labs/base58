//! The fold's arithmetic

use crate::scalar::LIMB_BASE;
use crate::variable::{BLOCK_DIGITS, BLOCK_LIMBS, POWER};

/// Columns settled in one pass over the window
///
/// Their sums do not wait on each other, so the group is what keeps the
/// multiply running ahead of the reduction that follows it.
const GROUP: usize = 32;

/// Limbs a column reaches back over
const SPAN: usize = BLOCK_LIMBS - 1;

/// The limb at this place, or nought once the value runs out
#[inline(always)]
fn limb_at(value: &[u32], count: usize, at: usize) -> u32 {
    match at < count {
        true => value[at],
        false => 0,
    }
}

/// Fold a block into the value in place, returning the limbs left
///
/// Each column is eighteen products against the block power. Columns run
/// upward so the carries can, which means a column is written over a limb the
/// next seventeen still need, so a group carries those in a window rather
/// than the whole value carrying a second buffer.
pub(crate) fn fold_in_place(
    value: &mut [u32],
    count: usize,
    digits: &[u32; BLOCK_DIGITS],
) -> usize {
    let reach = count + BLOCK_LIMBS + 1;
    let mut held = [0u32; GROUP + SPAN];
    let mut sums = [0u64; GROUP];
    let mut carry = 0u64;
    let mut top = 0;

    // The window opens on nothing, which is what the short columns at the
    // bottom want, and takes zeros again once the limbs run out at the top.
    for (at, slot) in held[SPAN..].iter_mut().enumerate() {
        *slot = limb_at(value, count, at);
    }

    let mut base = 0;
    while base < reach {
        columns(&held, &mut sums);

        // Slide before the settling below overwrites what it reads.
        held.copy_within(GROUP.., 0);
        for (at, slot) in held[SPAN..].iter_mut().enumerate() {
            *slot = limb_at(value, count, base + GROUP + at);
        }

        for (at, sum) in sums.iter().enumerate() {
            let place = base + at;
            if place >= reach {
                break;
            }
            let mut total = *sum + carry;
            if place < BLOCK_DIGITS {
                total += digits[place] as u64;
            }
            let limb = (total % LIMB_BASE) as u32;
            carry = total / LIMB_BASE;
            value[place] = limb;
            if limb != 0 {
                top = place + 1;
            }
        }
        base += GROUP;
    }

    let mut at = reach;
    while carry > 0 {
        value[at] = (carry % LIMB_BASE) as u32;
        carry /= LIMB_BASE;
        at += 1;
        top = at;
    }
    top
}

/// Sum every column in the group from the window it reads
fn columns(held: &[u32; GROUP + SPAN], sums: &mut [u64; GROUP]) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: gated on what the machine reported.
        match crate::dispatch::path() {
            crate::dispatch::Path::Wide => return unsafe { wide::columns(held, sums) },
            crate::dispatch::Path::Avx2 => return unsafe { avx2::columns(held, sums) },
            crate::dispatch::Path::Portable => {}
        }
    }
    scalar(held, sums);
}

/// The walk the kernels stand in for
///
/// Fixed-width windows, so the inner trip count is constant and the places
/// are read in order. A computed index carries a bounds check apiece and does
/// not vectorise.
fn scalar(held: &[u32; GROUP + SPAN], sums: &mut [u64; GROUP]) {
    for (sum, window) in sums.iter_mut().zip(held.windows(BLOCK_LIMBS)) {
        let mut total = 0u64;
        for (limb, weight) in window.iter().zip(POWER.iter().rev()) {
            total += *limb as u64 * *weight as u64;
        }
        *sum = total;
    }
}

#[cfg(target_arch = "x86_64")]
mod wide {
    use super::{GROUP, SPAN};
    use crate::variable::POWER_REV;
    use core::arch::x86_64::*;

    /// Columns a register carries
    const LANES: usize = 8;

    /// As [`super::scalar`], eight columns to a register
    ///
    /// A vector multiply reads the low half of each 64-bit lane, so the
    /// window is widened once for the whole group rather than on each of the
    /// eighteen loads a column takes. That widening is what sank three
    /// earlier kernels. Eighteen products of limbs below the base leave a
    /// column under 2^62.8, so nothing is reduced in here.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn columns(held: &[u32; GROUP + SPAN], sums: &mut [u64; GROUP]) {
        unsafe {
            let mut lanes = [0u64; GROUP + SPAN];
            for (lane, limb) in lanes.iter_mut().zip(held.iter()) {
                *lane = *limb as u64;
            }

            let mut at = 0;
            while at < GROUP {
                let mut running = _mm512_setzero_si512();
                for (place, weight) in POWER_REV.iter().enumerate() {
                    let window = _mm512_loadu_si512(lanes[at + place..].as_ptr().cast());
                    let held = _mm512_set1_epi64(*weight as i64);
                    running = _mm512_add_epi64(running, _mm512_mul_epu32(window, held));
                }
                _mm512_storeu_si512(sums[at..].as_mut_ptr().cast(), running);
                at += LANES;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{GROUP, SPAN};
    use crate::variable::POWER_REV;
    use core::arch::x86_64::*;

    /// Columns a register carries here, half what the wide path takes
    const LANES: usize = 4;

    /// As [`wide::columns`], four columns to a register
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn columns(held: &[u32; GROUP + SPAN], sums: &mut [u64; GROUP]) {
        unsafe {
            let mut lanes = [0u64; GROUP + SPAN];
            for (lane, limb) in lanes.iter_mut().zip(held.iter()) {
                *lane = *limb as u64;
            }

            let mut at = 0;
            while at < GROUP {
                let mut running = _mm256_setzero_si256();
                for (place, weight) in POWER_REV.iter().enumerate() {
                    let window = _mm256_loadu_si256(lanes[at + place..].as_ptr().cast());
                    let held = _mm256_set1_epi64x(*weight as i64);
                    running = _mm256_add_epi64(running, _mm256_mul_epu32(window, held));
                }
                _mm256_storeu_si256(sums[at..].as_mut_ptr().cast(), running);
                at += LANES;
            }
        }
    }
}
