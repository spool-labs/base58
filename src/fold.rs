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
/// upward so the carry can, which means a column is written over a limb the
/// next seventeen still need, so a group carries those in a window rather
/// than the whole value carrying a second buffer. Eighteen products of limbs
/// below the base come to 2^62.8, and the carry and the block's own digits
/// fit above that.
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
        // Fixed-width windows, so the inner trip count is constant and the
        // places are read in order. A computed index carries a bounds check
        // apiece and does not vectorise.
        for (sum, window) in sums.iter_mut().zip(held.windows(BLOCK_LIMBS)) {
            let mut total = 0u64;
            for (limb, weight) in window.iter().zip(POWER.iter().rev()) {
                total += *limb as u64 * *weight as u64;
            }
            *sum = total;
        }

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
