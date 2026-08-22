# Benchmarks

Every number here is stamped with the machine, the date and the build flags
that produced it. Treat these as evidence that the paths do what they claim, not as a
specification: one machine per class, and a class the routing has never seen
can behave differently — Genoa already did.

Read [Reproducing](#reproducing) before comparing anything against your own
run. Four things about this crate will otherwise mislead you.

## Machines

| | | vector width |
|---|---|---|
| **Zen 5** | AMD Ryzen 5 9600X, 6c/12t, 32 GB, Linux 7.0.0-28 | AVX-512 with VBMI |
| **Turin** | AMD EPYC 9B45 (GCP c4d-standard-4), 4 vCPU | AVX-512 with VBMI, full width |
| **Genoa** | AMD EPYC 9B14 (GCP c3d-standard-4), 4 vCPU | AVX-512 with VBMI, double-pumped |
| **Zen 3** | AMD EPYC Milan, 8 dedicated vCPU, 32 GB, Linux 6.8 | AVX2 only |
| **Zen 2** | AMD EPYC Rome (Hetzner cx33), 4 shared vCPU | AVX2 only |
| **Emerald Rapids** | Intel Xeon Platinum 8581C (GCP c4-standard-4), 4 vCPU | AVX-512 with VBMI, full width |
| **Skylake** | Intel Xeon Skylake-SP (Hetzner cx33), 4 vCPU | AVX-512 without VBMI |
| **Neoverse V2** | Google Axion (GCP c4a-standard-4), 4 vCPU | NEON, server part |
| **M4** | Apple M4 Max, macOS 26.5 | NEON, laptop part |

All measured 2026-08-13 on rustc 1.97.1 (1.93.1 on the M4), governor at
`performance` where the platform has one, one benchmark at a time.

Skylake is there for one reason: it has `avx512f`, `avx512bw` and `avx512vl`
but not `avx512vbmi`, so it is the case the cpuid guard exists for. It
declines the wide path and runs AVX2, which is what should happen.

## Reproducing

The benchmarks are their own crate, so that the published one carries neither
them nor the dependencies they need.

```sh
cd bench
cargo bench --bench paths                          # each x86 path, pinned in turn
cargo bench --bench against                        # against five8 and bs58
cargo bench --bench variable                       # any-length, by size
cargo bench --bench codec                          # the dispatched entry points
cargo bench --bench scalar                         # the portable path alone
```

`TAPE_PATH=portable|avx2|avx512` pins the instruction set, and
`paths` sweeps all three itself. Pinning goes through `testing::force`, which
is `#[doc(hidden)]` and exists for this.

### Four traps

**1. This crate dispatches at run time; five8 dispatches at compile time.**
So a stock build is not a like-for-like comparison — it is the deployment
comparison, which is a different and equally fair question. Build twice:

```sh
cargo bench --bench against                        # as each crate ships
RUSTFLAGS="-C target-cpu=native" cargo bench --bench against
```

**2. Agave ships `-Ctarget-cpu=x86-64-v2`**, which does *not* include AVX2 —
that is v3. In the validator's own build five8 compiles to scalar and this
crate still reaches AVX-512 through cpuid. Benchmarking with `target-cpu=native`
therefore flatters five8 relative to what operators actually run. The
deployment-accurate column is the stock one, or v2 explicitly.

**3. These benches move 7–13% between builds of identical source.**
`[profile.bench]` uses fat LTO with one codegen unit, and function layout
shifts between builds. Demonstrated here: two builds of the same source, 18
minutes apart, measured 8.62 µs and 7.51 µs on the same input. **Do not
believe a sub-15% delta measured across two builds** — reproduce each side
twice first. Same-binary comparisons (pinning a path at run time) have a floor
around 2.6% and are safe.

Adding or removing a bench target relinks everything, so never compare a saved
criterion baseline across a change to `Cargo.toml`.

**4. `-C target-cpu=native` auto-vectorises the portable path too**, so the
`portable` rows stop meaning portable under that flag. Do not mix baselines
between the two builds.

## Fixed width

Zen 5, `-Ctarget-cpu=x86-64-v2` — the flags agave builds with.

| | tape | five8 | bs58 | vs five8 | vs bs58 |
|---|---|---|---|---|---|
| `encode_32` | 19.1 ns | 39.2 | 676 | 2.1× | 35× |
| `encode_64` | 22.9 ns | 106.3 | 2,914 | 4.6× | 127× |
| `decode_32` | 15.4 ns | 36.1 | 248 | 2.3× | 16× |
| `decode_64` | 21.5 ns | 94.1 | 944 | 4.4× | 44× |

M4, stock flags. five8 has no aarch64 vector path, so this is our NEON against
their scalar — which is what a Mac user gets today, not a claim about the
algorithms.

| | tape | five8 | bs58 | vs five8 | vs bs58 |
|---|---|---|---|---|---|
| `encode_32` | 13.7 ns | 32.0 | 1,024 | 2.3× | 74× |
| `encode_64` | 45.8 ns | 110.9 | 3,339 | 2.4× | 73× |
| `decode_32` | 12.8 ns | 32.8 | 409 | 2.6× | 32× |
| `decode_64` | 27.0 ns | 99.4 | 1,339 | 3.7× | 50× |

Zen 3 under the same agave flags, where there is no AVX-512 at all — the
common case in validator fleets.

| | tape | five8 | bs58 | vs five8 | vs bs58 |
|---|---|---|---|---|---|
| `encode_32` | 30.3 ns | 58.5 | 1,014 | 1.9× | 33× |
| `encode_64` | 47.1 ns | 160.0 | 4,384 | 3.4× | 93× |
| `decode_32` | 23.2 ns | 54.9 | 378 | 2.4× | 16× |
| `decode_64` | 32.1 ns | 141.9 | 1,430 | 4.4× | 45× |

`decode_64` is 4.4× on both AMD parts, three microarchitectures apart in the
case of the encode, which suggests the margins are the algorithm rather than
one machine.

Emerald Rapids, agave flags again. The gap against five8 is widest here,
because five8 compiles to scalar under those flags while this crate still
reaches the wide path through cpuid.

| | tape | five8 | bs58 | vs five8 | vs bs58 |
|---|---|---|---|---|---|
| `encode_32` | 34.5 ns | 69.9 | 929 | 2.0× | 27× |
| `encode_64` | 49.7 ns | 218.4 | 4,292 | 4.4× | 86× |
| `decode_32` | 23.8 ns | 73.0 | 337 | 3.1× | 14× |
| `decode_64` | 39.5 ns | 218.5 | 1,367 | 5.5× | 35× |

Built `-C target-cpu=native`, where five8 reaches its AVX2 path, the Zen 5 gap
narrows to 1.0× on `encode_32` and 1.5–2.6× elsewhere. Both framings are
honest; they answer different questions.

### By path

Zen 5, stock flags, one binary with each path pinned in turn.

| | portable | AVX2 | AVX-512 |
|---|---|---|---|
| `encode_32` | 19.0 ns | 19.0 | see below |
| `encode_64` | 45.5 ns | 36.1 | **25.1** |
| `decode_32` | 27.0 ns | **14.5** | 14.5 |
| `decode_64` | 86.6 ns | **20.6** | 21.0 |
| `encode_32_batch` | 1.34 µs | 1.34 | 1.34 |
| `encode_64_batch` | 3.40 µs | 3.12 | **2.65** |
| `decode_32_batch` | 1.47 µs | **549 ns** | 549 |
| `decode_64_batch` | 5.12 µs | **1.27 µs** | 1.29 |

Batches are 64 inputs. The `encode_32` row predates the wide encoder landing;
the section below replaces it and the rest of the table still holds.

**On Zen 5, AVX-512 wins the encodes and ties the decodes.** That sentence
used to carry no qualifier; Zen 4 broke it — see "Route by datapath" below.
AVX2 carries the decode side, where it is within a few percent of the wider
path and ahead of it below 512 bytes on the any-length codec.

### The 32-byte encoders

A key encodes in registers on both vector paths: through AVX-512 where that
exists, and through a specialised AVX2 encoder elsewhere. Getting there took
three rounds — the first AVX2 encoder was deleted on a wrong measurement, the
second was rebuilt and lost, and the third won — so the evidence is here
rather than summarised.

Every variant below takes the same input — the words and the leading-zero
count — so the portable row is the tail of `encode_32` rather than the whole
entry point. Timing the entry point against variants handed pre-read words
would charge the read to one side only. One binary, all rows. The second
round measured:

| | Zen 5 | Zen 3 |
|---|---|---|
| `entry_point` (whole call) | 22.09 ns | 33.32 |
| **portable tail** (the bar) | **21.88** | **33.10** |
| AVX2 | 24.17 | 36.24 |
| AVX2, split accumulators | 24.16 | 36.19 |
| AVX-512 | 14.24 | — |
| AVX-512, vector ninth column | 13.83 | — |
| **AVX-512, and split accumulators** | **13.80** | — |
| AVX-512, store in place | 22.62 | — |
| AVX-512, all three | 22.05 | — |

Four things fell out:

- **AVX-512 wins by 37%**, and almost all of that is the plain version. The
  ninth column moving from a general register to an xmm is worth 3%; splitting
  the accumulator chain is worth 0.2%, which is noise. An earlier measurement
  of this same code at 28 ns was wrong, and the deletion it justified with it.
- **Storing the characters unshifted** at a base behind the buffer, with the
  mask moved up instead, reads as strictly cheaper and costs 60%. The
  masked-off lanes reach behind the allocation and the core takes an assist.
- **That round's AVX2 lost by 10% on both AMD parts.** Not for want of lanes:
  its `settle` compiled to 161 instructions and stayed a separate symbol even
  under fat LTO, where the AVX-512 one inlines into its callers and the whole
  AVX-512 `encode_32` is 112 instructions. Splitting the rare straggler loop
  out to get under the inliner's threshold made it larger, because the two
  divide passes are the function. Rearranging the call could not fix it;
  removing the call did, which is the third round.
- **The portable walk is good.** It settles and spells in one backward pass
  out of a two-digit table, so the digit split hides behind the carry chain
  rather than adding to it.

The third round rewrote the encoder around that disassembly, as helpers with
no `target_feature` of their own marked `#[inline(always)]` and composed
inside one `target_feature` function, which is what the attribute ban on
`inline(always)` leaves open. The reduction is specialised to two registers,
with the ninth limb in a general register where its one division is exact and
runs beside the vector work; the lane shift is `vperm2i128` and `vpalignr`
rather than two cross-lane permutes and a blend; the digits merge in the
64-bit domain so one shuffle packs a register where the old path shuffled
each digit vector; and the characters go out through three plain sixteen-byte
stores — two windows shuffled down by the skip, one flush store against the
end that agrees with its neighbour on the bytes they share — with no masked
store anywhere. `nm` on the bench binary shows no symbol survives out of
line. Zen 3, same method, stable within half a nanosecond across four builds:

| | Zen 3 | Zen 2 |
|---|---|---|
| `entry_point` (whole call, dispatched to AVX2) | 29.11 ns | 38.99 |
| **portable tail** (the bar) | 32.93 | 60.45 |
| **AVX2, third round** | **28.76** | **38.27** |

The Zen 2 column is a shared vCPU (Hetzner cx33, EPYC Rome), so read its
ratio rather than its speed: the win grows from 13% to 37% as the scalar
core weakens, which is the right direction for the machines that actually
take this path.

Two traps surfaced on the way to that entry-point row:

- **The boundary migrates.** Wired naively — words and zero count computed in
  the dispatcher, handed across the `target_feature` call — the whole call
  measured 42.7 ns against 33.7 before wiring: the tail won by 13% while the
  entry point lost by 27%, because the call does not inline, so the words are
  staged twice and nothing schedules across the boundary. The entry now takes
  the input bytes themselves and does the byte swap and the zero count in
  registers behind the boundary, one thin call deep. The same 8 ns gap
  between `entry_point` and the AVX-512 tail in the second-round table is
  this cost on the wide path; it has since moved its entry the same way,
  and the Genoa table below carries the result.
- **The batch spell stays portable.** Routing `write_32`'s lanes through the
  vector spell read as free reuse and cost 31% on `encode_32_batch`: a
  `target_feature` call per lane neither inlines nor lets neighbouring
  lanes' scalar chains overlap the way the inlined walk does.

Genoa — GCP c3d-standard-4, EPYC 9B14, Zen 4 with VBMI, the first Zen 4 this
crate has seen — then ran the finished shape and moved the picture twice
more. One binary, all rows:

| | Genoa |
|---|---|
| `entry_point` (whole call, dispatched to AVX-512) | 33.51 ns |
| portable tail | 44.98 |
| **AVX2, third round** | **27.87** |
| AVX-512 tail, handed words | 48.47 |
| AVX-512 entry, handed bytes | 33.63 |

- **The inliner's mercy is per-build.** This build kept `sum_and_settle_32`
  out of line, returning three registers through the stack on every call,
  where the Zen 5 builds had always folded it in. It is `inline(always)`
  with no `target_feature` now, which removes the choice.
- **Handing words across the boundary costs ~15 ns here by itself.** The
  words-taking row and the bytes-taking row run the same inlined body; the
  difference is eight opaque-pointer loads staged into zmm broadcasts
  against a swap the entry does in one register. The 8 ns Zen 5 gap was
  never just the call.
- **Zen 4 runs the AVX2 encoder faster than the AVX-512 one.** Genoa
  double-pumps 512-bit operations, so the zmm encoder pays twice per op
  while the four-lane one does not. A second session reproduced this
  independently on its own Genoa box, pinned paths, whole calls: encode_32
  28.62 AVX2 against 38.85 AVX-512, and decode_64 41.88 against 49.67. What
  that did to dispatch is the next section.

### Route by datapath

Turin — GCP c4d-standard-4, EPYC 9B45, Zen 5 with full-width 512-bit units —
supplied the missing column. One binary per machine, whole calls; the Genoa
column is a second session's box of the same type, run after the routing
below landed, and it agrees with the first Genoa box to within tenths:

| | Turin (Zen 5) | Genoa (Zen 4) | Emerald Rapids |
|---|---|---|---|
| AVX-512 entry, handed bytes | **20.66 ns** | 33.57 | **33.96** |
| AVX2, handed words | 24.84 | 27.89 | 36.03 |
| AVX2 entry, handed bytes | 25.64 | **28.40** | 36.38 |
| portable tail | 28.44 | 44.66 | 34.66 |
| `entry_point`, through dispatch | 20.67 | 28.41 | 34.45 |

The wide encoder wins by 19% at full width and loses by 15% at half width,
so no uniform routing is right and `encode_32` asks the machine: AMD family
0x19 — Zen 4, the double-pumped implementation — takes the four-lane entry,
and every other wide machine takes the wide one. The check ages forward
rather than badly: an unreleased family keeps the wide path it would have
today, and the one family that needs the exception is a closed, historical
set. The `entry_point` row is the check at work: on all three machines it
matches whichever entry won, confirmed on family 0x19 silicon and on Intel.

Intel is the case the check assumes rather than tests, so it was worth
measuring. Emerald Rapids is full width and the wide encoder wins there by
7%, which is the smaller margin of the two full-width parts but the same
direction — the gate lets it through and that is right. Its portable tail is
also unusually close to its wide path, 34.66 against 33.96, the narrowest
such margin on any machine here.

Handing words across the boundary instead of bytes costs the same on both
implementations — 14.5 ns on Turin, 14.9 on Genoa — so the staging cost is
the argument shape, not a microarchitecture.

The 64-byte decode needed no split. Four-lane and wide were level on Turin
(27.27 against 27.28, as on the Zen 5 desktop before it), level again on
Emerald Rapids (39.81 against 39.50) and four-lane wins by 16% on Genoa, so
every machine now decodes signatures through the
four-lane kernels and the wide reader survives under a differential test,
in case an unmeasured part ever wants it back.

`encode_64` stays wide everywhere, the one conversion where the wide
registers win on both implementations: 27.7 against 48.8 on Turin, 49.7
against 64.7 on Genoa, 49.7 against 66.6 on Emerald Rapids.

## Against FluxRPC

`fluxrpc/base58` is a Go implementation with hand-written AVX2, AVX-512 and
arm64 assembly, and it dispatches at run time as this crate does — the closest
comparison here in design as well as speed. Stock build on both sides, whole
calls, one session per machine.

| | Genoa tape | FluxRPC | | Emerald tape | FluxRPC | |
|---|---|---|---|---|---|---|
| `encode_32` | 28.4 ns | 65.2 | 2.3× | 34.5 ns | 53.3 | 1.5× |
| `encode_64` | 49.7 | 93.8 | 1.9× | 49.7 | 90.2 | 1.8× |
| `decode_32` | 21.7 | 39.0 | 1.8× | 23.8 | 34.1 | 1.4× |
| `decode_64` | 41.9 | 85.5 | 2.0× | 39.5 | 64.7 | 1.6× |
| `encode_32_batch`, per item | 32.6 | 38.6 | 1.2× | — | — | |

It is about twice five8 on either box, so these are margins against a strong
implementation rather than a weak one, and it is the narrowest field here.
Two things worth saying rather than burying: the batch row is the closest of
all, so whatever they do there recovers most of the gap; and they do
relatively better on Intel than on AMD, 1.4–1.8× against 1.8–2.3×, which
suggests their assembly is tuned for Intel's pipeline.

Any length is the wider difference, because they divide the value down where
this crate has place tables.

| bytes | tape encode | FluxRPC | tape decode | FluxRPC |
|---|---|---|---|---|
| ~100 | 0.75 µs | 0.69 µs | 0.30 µs | 0.20 µs |
| 1000 | 5.5 µs | **49.3 µs** | 4.0 µs | 8.9 µs |

They are ahead at a hundred bytes and nine times behind at a thousand, which
is the shape of a quadratic against a table walk. The crossover is somewhere
in the low hundreds.

## Server ARM

Every other aarch64 number here is from a laptop, so the question was whether
the NEON path is Apple-specific. It is not. Neoverse V2, stock flags:

| | tape | five8 | bs58 | vs five8 | vs bs58 |
|---|---|---|---|---|---|
| `encode_32` | 37.9 ns | 60.5 | 1,226 | 1.6× | 32× |
| `encode_64` | 74.0 ns | 178.8 | 5,562 | 2.4× | 75× |
| `decode_32` | 35.4 ns | 69.1 | 386 | 2.0× | 11× |
| `decode_64` | 55.2 ns | 217.4 | 1,869 | 3.9× | 34× |

Same shape as the M4's 2.3–3.7×, with `decode_64` strongest on both. The
absolute times are around twice the M4's, which is clock and core width rather
than anything about the code.

## Any length

Anything that is neither a key nor a signature folds a 64-byte block at a
time. There is no feature flag and no table, only about 1.3 KiB of constants.
Below 128 bytes the value divides down instead.

M4 Max, stock flags:

| bytes | encode | bs58 | decode | bs58 |
|---|---|---|---|---|
| 128 | 344 ns | 14.8 us | 147 ns | 5.12 us |
| 512 | 1.89 us | 240 us | 1.46 us | 82.3 us |
| 1232 | 7.31 us | 1.41 ms | 7.56 us | 489 us |

Zen 5, stock flags:

| bytes | encode | decode |
|---|---|---|
| 128 | 581 ns | 156 ns |
| 512 | 3.56 us | 1.51 us |
| 1232 | 14.6 us | 8.00 us |

The in place fold is worth 1.32x on encode at a packet on aarch64 and nothing
on x86, where it measures level against the two buffer form it replaced. It
is worth taking anyway for the frame it saves.

Past a packet there is no length limit when the crate is built with `alloc`,
which is on by default. The cost is quadratic and measured so: four times the
input is about fifteen times the work.

| bytes | encode | decode |
|---|---|---|
| 4096 | 127 us | 84.0 us |
| 16384 | 1.86 ms | 1.28 ms |
| 65536 | 29.1 ms | 20.3 ms |

Programs build with `default-features = false` and keep the stack path, which
tops out at `MAX_VARIABLE_LEN`.

1232 bytes is one Solana packet. five8 has no any-length API, so bs58 is the
whole field here, and this is the path a transaction crosses on submission.

The fold replaced a walk that took 45.9 us to encode a packet on Zen 5 and a
table path that took 8.10. It beats the walk everywhere and beats the tables
on aarch64. Whether it now beats them on x86 is open. Hand-written x86 kernels
for the fold were tried and lost to what the compiler emits on its own.

The value is folded in place. A column is written over a limb the next
seventeen still need, so a group of thirty-two carries those in a window
rather than the whole value carrying a second buffer: 1900 bytes of frame
against 4344, which is what brings it inside an SBF frame. Settling one column
at a time instead of a group measured 2.2x worse, because a group is what
keeps the multiply running ahead of the reduction.

## On chain

The SBF target compiles none of the vector modules. What a program links is
the portable codec, so these rows are that path against five8's scalar and
bs58 inside the VM. Measured 2026-08-14 through Mollusk 0.15 on the Agave
4.2 runtime, platform-tools v1.54, from its own harness:

```sh
cd bench/onchain
rustup run stable ./run.sh          # CU table, needs cargo-build-sbf
rustup run stable ./run.sh sizes    # one binary per codec
```

Unlike every table above, these rows are exact. The VM meters instructions,
not time, so there is no machine column, no layout variance and no error
bar. A re-run returns the same integers.

Each op loops its codec over a black-boxed input, and the cost per call is
the slope between a 10-iteration run and a 60-iteration run. The slope
cancels the entrypoint, and a loop the optimizer hoisted would read as
zero. Before anything is timed, a verification op feeds 200 fuzzed inputs
through both crates on chain and requires identical encodings and mutual
round-trips. Inputs are the ordinary kind: no leading zeros, 43 and 87
character encodings.

| CU per call | tape | five8 | bs58 | vs five8 | vs bs58 |
|---|---|---|---|---|---|
| `encode_32` | 744 | 1,231 | 9,520 | 1.7× | 13× |
| `decode_32` | 770 | 1,609 | 7,658 | 2.1× | 9.9× |
| `encode_64` | 1,741 | 3,015 | 35,704 | 1.7× | 21× |
| `decode_64` | 2,382 | 4,589 | 27,814 | 1.9× | 12× |

The lead is narrower than in the host tables because dispatch has nothing
to reach here. Both crates run scalar, and what remains is the limb walk
against five8's table walk.

The five8 column is not a hypothetical. The SDK's address type spells
`Display` through `five8::encode_32` and parses `FromStr` through
`five8::decode_32` (solana-address 2.7.0), so a program on the current
SDK pays that column whenever it logs or parses a pubkey, and the older
SDK generations paid the bs58 column. Every `msg!("{}", pubkey)` runs one
encode_32 before the log syscall sees a byte: 1,231 CU through the SDK
today, 744 through this crate.

### Program size

A program linking all four entry points carries 23.1 KB of codec, against
21.4 KB for five8 and 3.7 KB for bs58. One entry point alone is smaller:
4.3 KB for `encode_32`, 9.3 for `decode_32`, 8.5 for `encode_64`, 6.4 for
`decode_64`, each measured as a minimal program deploying only that op.
Size is deploy rent at about 6,960 lamports per byte, a one-time 0.16 SOL
for the whole crate and 0.012 more than five8. It is not execution cost.
The transaction cost model reads 8 CU per 32 KiB page of loaded program
bytes, at most one page of difference here.

The host encoders spell two digits at a time through an eight-kilobyte
pair table. On SBF that table is rent, and the divisions it saves are a
few ALU ops, so `to_chars` spells digit by digit there instead. The swap
took the crate from 33.8 KB to 23.1 and costs 124 CU on `encode_32` and
320 on `encode_64`. Decode never touched the table.

Do not reach for `opt-level = "z"` to shrink further. The speed of this
path is its unrolling, and `z` rolls the loops back up: the tape rows rise
about four-fold while the binary only loses a quarter of its bytes.

## Known gaps

- One machine per class. Both aarch64 parts agree on shape, so the NEON
  numbers are not Apple-specific, but neither is a Graviton.
- The on-chain rows come from one runtime build, Mollusk 0.15 over Agave
  4.2. CU prices are protocol constants, but they do move across major
  releases, so a re-run should stamp its runtime the way the host rows
  stamp the machine.
- The Skylake part is a shared vCPU and its absolute numbers are three to four
  times the others; read it for the guard's behaviour and the AVX2-to-portable
  ratios, not for speed.
- `decode_64` on AVX-512 measured 27.3 ns under `-Ctarget-cpu=x86-64-v2`
  against 21.5 for AVX2, where under stock flags the two are level. It survived
  a disassembly review with no semantic cause and falls inside the layout
  variance band of trap 3, so no workaround is routed for it. Unresolved.
- The `By path` table's `encode_32` row was measured before the wide encoder
  landed and has not been re-run through that bench.
- No Intel AVX2-only part has run the portable-to-AVX2 flip. Both Intel parts
  measured here have some AVX-512: Emerald Rapids takes the wide path and
  Skylake-SP declines it for want of the byte permute, so neither exercises a
  machine whose best is AVX2. Nothing older than Skylake has run at all.
- Hetzner's cx line hands out different parts on different days — a Skylake-SP
  one morning, an EPYC Rome that afternoon — so a cx row must name the part it
  actually drew, not the line.
- The Zen 5 desktop rows predate the third-round encoders and the entry
  rebuild; Turin stands in for that class now. A 9600X re-run of the
  encode32 bench would tie the old tables to the new ones.
- The any-length path at 32 bytes is ~67 ns of wrapper on the decode side,
  against 15 ns for the fixed-width decode of the same value. Encode delegates;
  decode cannot, because the character count does not determine the byte width
  — 44 characters decode to 32 or 33 bytes depending on the value.
