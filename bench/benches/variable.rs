//! Times the variable length codec, which is what account data and whole
//! transactions go through
//!
//! Build with `--features variable` to time the table path against the
//! same inputs; the entry points are the same either way.
//!
//! Set `TAPE_PATH` to `portable`, `avx2` or `avx512` to pin the instruction
//! set the codec dispatches to, which only `variable-simd` reads. Unset, or
//! wider than the machine can run, and it picks for itself as usual.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Sizes worth knowing: a memcmp filter, RPC's base58 account data ceiling, a
/// middling transaction, and a full packet.
const SIZES: [usize; 5] = [32, 128, 256, 512, 1232];

/// The widths to time, which `TAPE_SIZES` can replace with its own comma
/// separated list when a run is hunting for a crossover rather than reporting
fn sizes() -> Vec<usize> {
    match std::env::var("TAPE_SIZES") {
        Ok(asked) => asked
            .split(',')
            .filter_map(|width| width.trim().parse().ok())
            .collect(),
        Err(_) => SIZES.to_vec(),
    }
}

fn noise(seed: u64, into: &mut [u8]) {
    let mut state = seed | 1;
    for slot in into.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *slot = (state >> 24) as u8;
    }
}

/// Pin the codec to one instruction set, so a run can name which it timed
#[cfg(target_arch = "x86_64")]
fn pin() -> String {
    use tape_base58::testing::{available, force, AVX2, PORTABLE, WIDE};
    let widest = available();
    let asked = std::env::var("TAPE_PATH").unwrap_or_default();
    let (marker, name) = match asked.as_str() {
        "portable" => (PORTABLE, "portable"),
        "avx2" if widest == AVX2 || widest == WIDE => (AVX2, "avx2"),
        "avx512" if widest == WIDE => (WIDE, "avx512"),
        "" => return "native".to_string(),
        other => panic!("this machine cannot run {other}"),
    };
    force(marker);
    name.to_string()
}

#[cfg(not(target_arch = "x86_64"))]
fn pin() -> String {
    "native".to_string()
}

fn variable(criterion: &mut Criterion) {
    let path = pin();
    let mut group = criterion.benchmark_group(format!("variable_{path}"));
    for len in sizes() {
        let mut input = vec![0u8; len];
        noise(len as u64 + 1, &mut input);

        let mut out = vec![0u8; tape_base58::encoded_len(len)];
        let written = tape_base58::encode(&input, &mut out).expect("encode");
        let text = out[..written].to_vec();
        let mut back = vec![0u8; tape_base58::decoded_len(text.len())];

        group.bench_function(format!("encode_{len}"), |bencher| {
            bencher.iter(|| tape_base58::encode(black_box(&input), black_box(&mut out)))
        });
        group.bench_function(format!("decode_{len}"), |bencher| {
            bencher.iter(|| tape_base58::decode(black_box(&text), black_box(&mut back)))
        });
        group.bench_function(format!("bs58_encode_{len}"), |bencher| {
            let mut theirs = vec![0u8; text.len()];
            bencher.iter(|| bs58::encode(black_box(&input)).onto(black_box(&mut theirs[..])))
        });
        group.bench_function(format!("bs58_decode_{len}"), |bencher| {
            let mut theirs = vec![0u8; len + 2];
            bencher.iter(|| bs58::decode(black_box(&text)).onto(black_box(&mut theirs[..])))
        });
    }
    group.finish();
}

criterion_group!(benches, variable);
criterion_main!(benches);
