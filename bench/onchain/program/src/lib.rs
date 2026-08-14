//! CU measurement program: op byte, iteration byte, then the payload.
//!
//! Every op loops `iters` times over a black-boxed input so the codec runs
//! once per iteration. Op 0 is the empty loop the runner subtracts.

#![no_std]

use core::hint::black_box;

#[cfg(target_os = "solana")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// # Safety
/// Called by the loader with its serialized input; expects zero accounts.
#[no_mangle]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    let data_len = unsafe { core::ptr::read_unaligned(input.add(8) as *const u64) } as usize;
    let data = unsafe { core::slice::from_raw_parts(input.add(16), data_len) };
    let op = data[0];
    let iters = data[1] as usize;
    let payload = &data[2..];

    match op {
        0 => {
            for _ in 0..iters {
                black_box(payload);
            }
        }
        #[cfg(feature = "tape")]
        1 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&payload[..32]);
            let mut out = [0u8; 44];
            for _ in 0..iters {
                let n = tape_base58::encode_32(black_box(&arr), &mut out);
                black_box((n, &out));
            }
        }
        #[cfg(feature = "tape")]
        2 => {
            let mut out = [0u8; 32];
            for _ in 0..iters {
                if tape_base58::decode_32(black_box(payload), &mut out).is_err() {
                    return 1;
                }
                black_box(&out);
            }
        }
        #[cfg(feature = "tape")]
        3 => {
            let mut arr = [0u8; 64];
            arr.copy_from_slice(&payload[..64]);
            let mut out = [0u8; 88];
            for _ in 0..iters {
                let n = tape_base58::encode_64(black_box(&arr), &mut out);
                black_box((n, &out));
            }
        }
        #[cfg(feature = "tape")]
        4 => {
            let mut out = [0u8; 64];
            for _ in 0..iters {
                if tape_base58::decode_64(black_box(payload), &mut out).is_err() {
                    return 1;
                }
                black_box(&out);
            }
        }
        #[cfg(feature = "five8")]
        5 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&payload[..32]);
            let mut out = [0u8; 44];
            for _ in 0..iters {
                let n = five8::encode_32(black_box(&arr), &mut out);
                black_box((n, &out));
            }
        }
        #[cfg(feature = "five8")]
        6 => {
            let mut out = [0u8; 32];
            for _ in 0..iters {
                if five8::decode_32(black_box(payload), &mut out).is_err() {
                    return 1;
                }
                black_box(&out);
            }
        }
        #[cfg(feature = "five8")]
        7 => {
            let mut arr = [0u8; 64];
            arr.copy_from_slice(&payload[..64]);
            let mut out = [0u8; 88];
            for _ in 0..iters {
                let n = five8::encode_64(black_box(&arr), &mut out);
                black_box((n, &out));
            }
        }
        #[cfg(feature = "five8")]
        8 => {
            let mut out = [0u8; 64];
            for _ in 0..iters {
                if five8::decode_64(black_box(payload), &mut out).is_err() {
                    return 1;
                }
                black_box(&out);
            }
        }
        #[cfg(feature = "bs58")]
        9 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&payload[..32]);
            let mut out = [0u8; 44];
            for _ in 0..iters {
                let n = bs58::encode(black_box(&arr[..])).onto(&mut out[..]).unwrap_or(0);
                black_box((n, &out));
            }
        }
        #[cfg(feature = "bs58")]
        10 => {
            let mut out = [0u8; 32];
            for _ in 0..iters {
                if bs58::decode(black_box(payload)).onto(&mut out[..]).is_err() {
                    return 1;
                }
                black_box(&out);
            }
        }
        #[cfg(feature = "bs58")]
        11 => {
            let mut arr = [0u8; 64];
            arr.copy_from_slice(&payload[..64]);
            let mut out = [0u8; 88];
            for _ in 0..iters {
                let n = bs58::encode(black_box(&arr[..])).onto(&mut out[..]).unwrap_or(0);
                black_box((n, &out));
            }
        }
        #[cfg(feature = "bs58")]
        12 => {
            let mut out = [0u8; 64];
            for _ in 0..iters {
                if bs58::decode(black_box(payload)).onto(&mut out[..]).is_err() {
                    return 1;
                }
                black_box(&out);
            }
        }
        #[cfg(all(feature = "tape", feature = "five8"))]
        13 => {
            let mut key = [0u8; 32];
            key.copy_from_slice(&payload[..32]);
            let mut sig = [0u8; 64];
            sig.copy_from_slice(&payload[32..96]);

            let mut state = 0x9E37_79B9_7F4A_7C15u64;
            let mut step = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            };

            for round in 0..iters {
                for b in key.iter_mut() {
                    *b = step();
                }
                for b in sig.iter_mut() {
                    *b = step();
                }
                // Zero prefixes on a cycle, so leading-one spelling gets hit
                let zeros = round % 7;
                key[..zeros].fill(0);
                sig[..zeros].fill(0);

                let mut a32 = [0u8; 44];
                let mut b32 = [0u8; 44];
                let na = tape_base58::encode_32(&key, &mut a32);
                let nb = five8::encode_32(&key, &mut b32) as usize;
                if na != nb || a32[..na] != b32[..nb] {
                    return 10;
                }
                let mut back = [0u8; 32];
                if tape_base58::decode_32(&b32[..nb], &mut back).is_err() || back != key {
                    return 11;
                }
                if five8::decode_32(&a32[..na], &mut back).is_err() || back != key {
                    return 12;
                }

                let mut a64 = [0u8; 88];
                let mut b64 = [0u8; 88];
                let na = tape_base58::encode_64(&sig, &mut a64);
                let nb = five8::encode_64(&sig, &mut b64) as usize;
                if na != nb || a64[..na] != b64[..nb] {
                    return 13;
                }
                let mut back = [0u8; 64];
                if tape_base58::decode_64(&b64[..nb], &mut back).is_err() || back != sig {
                    return 14;
                }
                if five8::decode_64(&a64[..na], &mut back).is_err() || back != sig {
                    return 15;
                }
            }
        }
        _ => return 2,
    }
    0
}
