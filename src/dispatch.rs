//! Which codec the running machine can execute
//!
//! Asked once and remembered. A crate built with no target flags is built for
//! the baseline of its architecture, so without this a machine with wide
//! vector units would still run the portable path.

use core::sync::atomic::{AtomicU8, Ordering};

/// Not yet asked
pub const UNKNOWN: u8 = 0;

/// Nothing beyond the architecture baseline
pub const PORTABLE: u8 = 1;

/// Vectors of four 64-bit lanes
pub const AVX2: u8 = 2;

/// Byte permutes across a whole 64-byte register
pub const WIDE: u8 = 3;

static CHOSEN: AtomicU8 = AtomicU8::new(UNKNOWN);

/// Wide datapaths not yet probed
const WIDTH_UNKNOWN: u8 = 0;

/// 512-bit operations issue at full width
const FULL_WIDTH: u8 = 1;

/// 512-bit operations issue as two 256-bit halves
const HALF_WIDTH: u8 = 2;

static WIDTH: AtomicU8 = AtomicU8::new(WIDTH_UNKNOWN);

/// The widest codec this machine can run
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Path {
    Portable,
    Avx2,
    Wide,
}

/// The path for this machine, probing the first time it is asked
pub(crate) fn path() -> Path {
    match CHOSEN.load(Ordering::Relaxed) {
        WIDE => Path::Wide,
        AVX2 => Path::Avx2,
        PORTABLE => Path::Portable,
        UNKNOWN => settle(),
        _ => Path::Portable,
    }
}

fn settle() -> Path {
    let chosen = available();
    CHOSEN.store(chosen, Ordering::Relaxed);
    match chosen {
        WIDE => Path::Wide,
        AVX2 => Path::Avx2,
        PORTABLE => Path::Portable,
        _ => Path::Portable,
    }
}

/// The widest path this machine can run, without recording it
pub fn available() -> u8 {
    // SAFETY: cpuid and xgetbv are always present on x86-64, and each leaf is
    // checked against the highest one the processor reports before it is read.
    // The wide path calls the AVX2 code for the widths AVX-512 loses at, so
    // it may only be reported when both are present.
    match unsafe { probe() } {
        (true, true) => WIDE,
        (false, true) => AVX2,
        (_, false) => PORTABLE,
    }
}

/// Pin the path, for a test that wants one the machine would not choose
pub fn force(chosen: u8) {
    CHOSEN.store(chosen, Ordering::Relaxed);
}

/// Whether this machine's 512-bit operations issue as two 256-bit halves
///
/// Zen 4 reports every feature the wide path asks for and runs each 512-bit
/// operation at half width, which flips the shorter conversions to the
/// four-lane kernels. AMD family 0x19 is that implementation; later families
/// and the Intel parts carry full-width datapaths, and so does anything
/// unreleased, which leaves an unknown part with the wide path it has today.
pub(crate) fn is_wide_half_width() -> bool {
    match WIDTH.load(Ordering::Relaxed) {
        HALF_WIDTH => true,
        FULL_WIDTH => false,
        _ => {
            let width = match probe_half_width() {
                true => HALF_WIDTH,
                false => FULL_WIDTH,
            };
            WIDTH.store(width, Ordering::Relaxed);
            width == HALF_WIDTH
        }
    }
}

// `__cpuid` is safe to call from 1.90 on, and this crate still builds on 1.89,
// so the blocks below stay and the lint that flags them is turned off.
#[allow(unused_unsafe)]
fn probe_half_width() -> bool {
    use core::arch::x86_64::__cpuid;

    // SAFETY: cpuid is always present on x86-64, and both leaves are below
    // the lowest highest-leaf any processor reports.
    let vendor = unsafe { __cpuid(0) };
    let is_amd =
        vendor.ebx == 0x6874_7541 && vendor.edx == 0x6974_6e65 && vendor.ecx == 0x444d_4163;
    if !is_amd {
        return false;
    }
    let signature = unsafe { __cpuid(1) }.eax;
    let base = (signature >> 8) & 0xF;
    let family = match base == 0xF {
        true => base + ((signature >> 20) & 0xFF),
        false => base,
    };
    family == 0x19
}

/// Whether the processor and the operating system offer the wide permutes, and
/// separately whether they offer AVX2
#[allow(unused_unsafe)]
unsafe fn probe() -> (bool, bool) {
    use core::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};

    // SAFETY: leaf zero is always readable and reports the highest leaf there is.
    let highest = unsafe { __cpuid(0) }.eax;
    if highest < 7 {
        return (false, false);
    }

    // The operating system has to have agreed to save the vector registers, or
    // using them faults however capable the processor is.
    let baseline = unsafe { __cpuid(1) };
    let has_saved_state = baseline.ecx & (1 << 27) != 0;
    if !has_saved_state {
        return (false, false);
    }
    let saved = unsafe { _xgetbv(0) };
    const SAVED_YMM: u64 = (1 << 1) | (1 << 2);
    const SAVED_ZMM: u64 = SAVED_YMM | (1 << 5) | (1 << 6) | (1 << 7);

    let extended = unsafe { __cpuid_count(7, 0) };
    let has_avx2 = saved & SAVED_YMM == SAVED_YMM && extended.ebx & (1 << 5) != 0;
    if saved & SAVED_ZMM != SAVED_ZMM {
        return (false, has_avx2);
    }

    let has_foundation = extended.ebx & (1 << 16) != 0;
    let has_byte_word = extended.ebx & (1 << 30) != 0;
    let has_short_vectors = extended.ebx & (1 << 31) != 0;
    let has_byte_permute = extended.ecx & (1 << 1) != 0;
    let has_wide = has_foundation && has_byte_word && has_short_vectors && has_byte_permute;
    (has_wide, has_avx2)
}
