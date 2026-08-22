//! The fold's multiply
//!
//! Every limb's products against the block power land on eighteen columns and
//! none of them wait on each other, so the compiler vectorises this on its
//! own. Hand-written x86 kernels were tried and lost to it.

use crate::variable::{BLOCK_LIMBS, COMBA_LIMBS, POWER};

/// Add every limb's products to the columns they land in, overwriting them
pub(crate) fn multiply(value: &[u32], count: usize, loose: &mut [u64]) {
    match count < COMBA_LIMBS {
        true => spread(value, count, loose),
        false => columns(value, count, loose),
    }
}

/// Spread each limb's products across the columns they land in
fn spread(value: &[u32], count: usize, loose: &mut [u64]) {
    for slot in loose.iter_mut() {
        *slot = 0;
    }
    for (at, limb) in value[..count].iter().enumerate() {
        let limb = *limb as u64;
        for (slot, place) in loose[at..at + BLOCK_LIMBS].iter_mut().zip(POWER.iter()) {
            *slot += limb * *place as u64;
        }
    }
}

/// Gather each column's products, one store per column
///
/// The middle walks fixed-width windows so the inner trip count is constant
/// and unrolls. Eighteen products of limbs below the base stay inside a u64.
fn columns(value: &[u32], count: usize, loose: &mut [u64]) {
    const SPAN: usize = BLOCK_LIMBS - 1;
    let ramp = SPAN.min(loose.len());
    let middle = count.max(ramp);

    for (at, slot) in loose[..ramp].iter_mut().enumerate() {
        let highest = at.min(count - 1);
        let mut sum = 0u64;
        for (limb, place) in value[..=highest]
            .iter()
            .zip(POWER[at - highest..=at].iter().rev())
        {
            sum += *limb as u64 * *place as u64;
        }
        *slot = sum;
    }
    for (slot, window) in loose[ramp..middle]
        .iter_mut()
        .zip(value[ramp - SPAN..count].windows(BLOCK_LIMBS))
    {
        let mut sum = 0u64;
        for (limb, place) in window.iter().zip(POWER.iter().rev()) {
            sum += *limb as u64 * *place as u64;
        }
        *slot = sum;
    }
    for (at, slot) in loose.iter_mut().enumerate().skip(middle) {
        // Past the top limb once the ramp clears it.
        let lowest = at - SPAN;
        if lowest >= count {
            *slot = 0;
            continue;
        }
        let mut sum = 0u64;
        for (limb, place) in value[lowest..count]
            .iter()
            .zip(POWER[at + 1 - count..].iter().rev())
        {
            sum += *limb as u64 * *place as u64;
        }
        *slot = sum;
    }
}
