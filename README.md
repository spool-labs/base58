# tape-base58

Fast base58 for Solana-shaped data. Fixed-size paths for public keys and signatures,
a limb-based codec for everything else, and the widest instruction set the
running machine supports chosen at first use.

## API

```rust
// Fixed widths, no allocation.
encode_32(&[u8; 32], &mut [u8; MAX_ENCODED_32]) -> usize
encode_64(&[u8; 64], &mut [u8; MAX_ENCODED_64]) -> usize
decode_32(&[u8], &mut [u8; 32]) -> Result<(), DecodeError>
decode_64(&[u8], &mut [u8; 64]) -> Result<(), DecodeError>

// Any length up to MAX_VARIABLE_LEN, which is one Solana packet.
encode(&[u8], &mut [u8]) -> Result<usize, EncodeError>
decode(&[u8], &mut [u8]) -> Result<usize, DecodeError>
encoded_len(usize) -> usize
decoded_len(usize) -> usize

// Four inputs at a time, sharing one walk of the table.
encode_32_batch(&[[u8; 32]], &mut [u8], &mut [usize]) -> Result<(), EncodeError>
encode_64_batch(&[[u8; 64]], &mut [u8], &mut [usize]) -> Result<(), EncodeError>
decode_32_batch(&[&[u8]], &mut [[u8; 32]]) -> Result<(), BatchError>
decode_64_batch(&[&[u8]], &mut [[u8; 64]]) -> Result<(), BatchError>
```

`no_std`, no allocation, no dependencies.

## Paths

| Architecture | Encode | Decode |
|---|---|---|
| aarch64 | alphabet through one four-register lookup | characters through the same lookup inverted |
| x86-64 with AVX512VBMI | signatures in registers end to end | characters through a two-register permute |
| x86-64 with AVX2 | the same, with the alphabet as a compare chain | characters through a per-window shuffle |
| anything else | portable | portable |

Which path runs is settled by measurement rather than by width. A public key
encodes through the portable walk everywhere, because nine limbs are too few
to pay for a vector carry reduction; a signature has eighteen and does not
have that problem. The reduction the vector paths share divides every limb at
once and shifts the quotients a lane down, rather than walking them in a chain
of dependent divisions.

## Any length

Input that is neither a key nor a signature is folded rather than walked. A
whole 64-byte block enters the value at once, so the value is reduced once per
limb per block instead of once per limb per word, and the multiply that
replaces the rest has no serial chain in it. It folds in place, against a
window rather than a second buffer. The constants are about 1.3 KiB, not a
table. Decoding runs the same trade the other way, ten characters a pass
into 64-bit words.

Below 128 bytes there is no whole block to fold and the value divides down
instead, which is cheaper while it is short.

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md), which carries the numbers, how to
reproduce them, and the four ways this crate will mislead a naive benchmark.

## Attribution

The conversion algorithm is Firedancer's `fd_base58`, Apache-2.0. See NOTICE.
