# On-chain CU comparison

Measures what the fixed-width codecs cost inside the SBF VM, against five8
and bs58, with PDA operations as a scale anchor. `program/` is a raw no_std
entrypoint with one op per codec path. `runner/` drives it through Mollusk
and prints CU per call as the slope between two iteration counts, so
entrypoint cost cancels and a dead-code-eliminated loop would read as zero.
A verification op checks both codecs agree byte-for-byte on-chain before
anything is timed. `pda-program/` holds the PDA ops on solana-program,
the stack tape's program uses.

```sh
rustup run stable ./run.sh          # CU table
rustup run stable ./run.sh sizes    # per-codec .so sizes
```

Both steps need `cargo-build-sbf`. The stable toolchain is for cargo >= 1.85,
which the solana-program dependency tree requires to parse.

## Numbers (2026-08-14, Mollusk 0.15 / Agave 4.2, platform-tools v1.54)

CU per call, typical inputs (43/87-char encodings, no leading zeros):

| op        | tape-base58 | five8 | bs58   |
|-----------|------------:|------:|-------:|
| encode_32 |         744 | 1,231 |  9,520 |
| decode_32 |         770 | 1,609 |  7,658 |
| encode_64 |       1,741 | 3,015 | 35,704 |
| decode_64 |       2,382 | 4,589 | 27,814 |

PDA anchors on the same runtime: create_program_address 1,584 CU,
find_program_address ~1,500 per bump tried (4,546 at bump 253),
create-account CPI via invoke_signed 1,772 CU net of entrypoint.

Binary size added over a 1.1 KB harness, all four entry points linked,
fat LTO: tape +23.1 KB, five8 +21.4 KB, bs58 +3.7 KB. The SBF build
spells digits without the eight-kilobyte pair table the host encoders
use (see `to_chars` in src/scalar.rs). The table bought 124 CU on
encode_32 and 320 on encode_64. Size is deploy rent at about 6,960
lamports per byte, not execution CU. Linking one entry point alone
costs less: encode_32 is 4.3 KB, decode_64 6.4 KB.
