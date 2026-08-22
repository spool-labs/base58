# On-chain CU comparison

Measures what the fixed-width codecs cost inside the SBF VM, against
five8 and bs58. five8 is not an arbitrary rival: the SDK's address type
spells `Display` through `five8::encode_32` and parses `FromStr` through
`five8::decode_32` (solana-address 2.7.0), so the five8 column is what a
program on the current SDK already pays whenever it logs or parses a
pubkey. The bs58 column is the same story one SDK generation back. The
tape column is the same operations through this crate.

`program/` is a raw no_std entrypoint with one op per codec path.
`runner/` drives it through Mollusk and prints CU per call as the slope
between two iteration counts, so entrypoint cost cancels and a
dead-code-eliminated loop would read as zero. A verification op runs 200
fuzzed inputs through both codecs on chain, requiring byte-identical
encodings and mutual round-trips, before anything is timed.

```sh
rustup run stable ./run.sh          # CU table
rustup run stable ./run.sh sizes    # per-codec .so sizes
```

Both steps need `cargo-build-sbf`. The stable toolchain is for cargo >=
1.85, which the runner's dependency tree requires to parse.

## Numbers (2026-08-14, Mollusk 0.15 / Agave 4.2, platform-tools v1.54)

CU per call, typical inputs (43/87-char encodings, no leading zeros):

| op        | tape-base58 | five8 | bs58   |
|-----------|------------:|------:|-------:|
| encode_32 |         744 | 1,231 |  9,520 |
| decode_32 |         770 | 1,609 |  7,658 |
| encode_64 |       1,741 | 3,015 | 35,704 |
| decode_64 |       2,382 | 4,589 | 27,814 |

Every `msg!("{}", pubkey)` pays one encode_32 before the log syscall
sees a byte. Through the SDK that is 1,231 CU today. Through this crate
it is 744, so a program saves 487 CU per logged address, and against the
older SDK's bs58 Display it saves 8,776.

Binary size added over a 1.1 KB harness, all four entry points linked,
fat LTO: tape +23.1 KB, five8 +21.4 KB, bs58 +3.7 KB. The SBF build
spells digits without the eight-kilobyte pair table the host encoders
use (see `to_chars` in src/scalar.rs). The table bought 124 CU on
encode_32 and 320 on encode_64. Size is deploy rent at about 6,960
lamports per byte, not execution CU. Linking one entry point alone
costs less: encode_32 is 4.3 KB, decode_64 6.4 KB.
