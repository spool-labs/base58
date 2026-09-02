//! Picks the widest codec the running machine can execute

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
mod portable;

#[cfg(target_arch = "aarch64")]
pub(crate) use aarch64::{
    decode_32, decode_64, encode_32, encode_64, is_decode_32_direct, is_decode_64_direct,
    is_encode_32_direct, is_encode_64_direct,
    read_32, read_64,
    words_32_lanes, words_64_lanes, write_32, write_64,
};

#[cfg(target_arch = "x86_64")]
pub(crate) use x86::{
    decode_32, decode_64, encode_32, encode_64, is_decode_32_direct, is_decode_64_direct,
    is_encode_32_direct, is_encode_64_direct,
    read_32, read_64,
    words_32_lanes, words_64_lanes, write_32, write_64,
};

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) use portable::{
    decode_32, decode_64, encode_32, encode_64, is_decode_32_direct, is_decode_64_direct,
    is_encode_32_direct, is_encode_64_direct,
    read_32, read_64,
    words_32_lanes, words_64_lanes, write_32, write_64,
};
