# The 27B's dense trellis kernels on the 5090

- Kind: experiment
- Status: current
- Observed: 2026-10-23
- Last verified: 2026-10-23
- Scope: kernel / the dense trellis linear (`kernel/include/ignis_trellis_linear.h`): decode, verify and prefill
  routes at every matrix shape of the 27B's 16 GB artifact, with and without the debug SM cap
- Related: [GitHub #337](https://github.com/gpillon/ignis/issues/337),
  [spec 27b-16gb/02](../specs/27b-16gb/02-the-dense-trellis-kernels.md) (Acceptance 4 and 5),
  [ADR 0060](../adr/0060-the-27b-has-a-trellis-body-for-small-cards.md),
  [Flash-Next's MoE decode](2026-10-05-moe-decode-is-structure-bound.md)
- Superseded by: none

## Question

Spec 02 asks the decode route to read the body at one token at 65% or more of the 5090's DRAM roofline, with eight
tokens costing at most 1.35 times one, and the prefill route to reach 60% of the FP16-accumulate tensor peak at 1,024
tokens. Where do the kernels land, what limits them, and what do the SM caps (36, 70, 84: a 5060 Ti, 5070 Ti, 5080)
say about small cards?

## Evidence

`kernel/build/tests/ignis_kernel_trellis_linear_bench.exe` (source `kernel/tests/bench_trellis_linear.cu`), RTX 5090,
2026-10-23, one run on a cool card: the decode leg first (`--caps 0,36,70,84 --no-prefill`), then the prefill leg (all
caps). Device time per call from a CUDA graph of calls cycling over enough copies of each weight (> 512 MB) that none
is in the 96 MB L2; every time includes the call's activation rotation launch. Bytes are the trellis plus `suh`/`svh`.
Roofline 1,792 GB/s; a streaming read measured 1,662-1,713 GB/s. `nvidia-smi` sampled every 500 ms beside the
prefill leg: SM clock median 2,887 MHz (1,965-2,917), power median 430 W, peak 584 W, at most 76 °C. The bench's
register-only fp16-input, fp16-accumulate `mma.sync` loop measured **502 TFLOPS at 2,890 MHz**, against the
whitepaper's 419 TFLOPS (rated at the 2.41 GHz boost). Logs: `.scratch/final_{decode,prefill}.{log,smi}` of the
#337 worktree (untracked).

**Decode, uncapped** (K as the public `SC_3.00bpw_H4` quant gives each matrix):

| matrix | in x out | K | 1 tok µs | % roofline | 8 tok | 16 tok | 64 tok | 8/1 | 64/1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| GDN qkv | 5120 x 10240 | 3 | 17.7 | 62.0% | 18.4 | 20.3 | 45.8 | 1.04 | 2.58 |
| GDN z | 5120 x 6144 | 3 | 13.1 | 50.5% | 13.4 | 15.2 | 33.7 | 1.02 | 2.58 |
| GDN out | 6144 x 5120 | 3 | 12.8 | 51.7% | 13.4 | 14.8 | 31.3 | 1.05 | 2.45 |
| GQA q | 5120 x 12288 | 3 | 21.1 | 62.6% | 21.6 | 24.7 | 73.2 | 1.03 | 3.48 |
| GQA k, v | 5120 x 1024 | 4 | 5.9 | 24.8% | 7.0 | 11.6 | 24.5 | 1.18 | 4.1 |
| GQA o | 6144 x 5120 | 3 | 12.7 | 51.9% | 13.4 | 14.9 | 31.3 | 1.06 | 2.47 |
| MLP gate + up (one SwiGLU call) | 5120 x 2 x 17408 | 3 | 50.7 | 73.7% | 53.1 | 62.2 | 168.7 | 1.05 | 3.33 |
| MLP down | 17408 x 5120 | 3 | 27.0 | 69.3% | 27.8 | 31.4 | 73.6 | 1.03 | 2.73 |
| fused gate_up | 5120 x 34816 | 3 | 49.6 | 75.3% | 51.6 | 60.7 | 153.5 | 1.04 | 3.10 |
| head | 5120 x 248320 | 4 | 385.7 | 92.0% | 403.4 | 471.8 | 1074 | 1.05 | 2.79 |
| a call's floor | 128 x 128 | 3 | 3.3 | — | 3.4 | 3.8 | 6.1 | | |

**The body at one token** (48 GDN and 16 GQA layers, 9,157 MB): **7,788 µs = 1,176 GB/s, 65.6% of the roofline**;
eight tokens **1.044x**. The same figure was 64.1-66.4% over the day's other runs: lower when the card was hot from a
prefill leg (64.7% in the prefill leg's own decode pass).

**Decode under the SM cap** (the body at one token; eight tokens):

| cap | body µs | GB/s | per SM | % of the 5090's roofline | 8/1 |
|---|---:|---:|---:|---:|---:|
| none (170) | 7,788 | 1,176 | 6.9 | 65.6% | 1.04 |
| 84 | 12,308 | 744 | 8.9 | 41.5% | 1.00 |
| 70 | 14,607 | 627 | 9.0 | 35.0% | 1.02 |
| 36 | 29,222 | 313 | 8.7 | 17.5% | 1.02 |

**Prefill**, TFLOP/s (2 x tokens x in x out per call), against the whitepaper's 419 and the measured 502:

| matrix | 1,024 tok | 8,192 tok | cap 84, 1K | cap 70, 1K | cap 36, 1K |
|---|---:|---:|---:|---:|---:|
| GDN qkv | 231.7 | 205.0 | 136.6 | 111.1 | 62.0 |
| GDN z | 196.7 | 197.2 | 130.4 | 109.8 | 60.7 |
| GDN out | 226.9 | 199.2 | 137.0 | 111.1 | 62.3 |
| GQA q | 225.9 | 198.5 | 132.5 | 120.6 | 61.0 |
| GQA k, v | 101-102 | 160 | 84-91 | 76-83 | 54 |
| GQA o | 233.9 | 191.9 | 136.6 | 110.6 | 62.3 |
| MLP gate + up | 212.4 | 208.3 | 137.0 | 113.6 | 59.2 |
| MLP down | 238.1 | 207.6 | 116.1 | 103.2 | 55.8 |
| head | 201.7 | 200.9 | 137.6 | 116.0 | 61.5 |

At 1,024 tokens the body's large matrices reach **212-238 TFLOPS: 51-57% of 419, 42-47% of the measured 502**.

**What limits the prefill**, Nsight Compute on `prefill_kernel<6>` at GDN qkv, 1,024 tokens (ncu's base clock,
1.89 GHz): tensor pipe active 57%, issue slots busy 39%, 2 warps per scheduler (1 CTA of 8 warps per SM, 213-235
registers), top stalls `wait` (fixed-latency dependencies) and `math_pipe_throttle`. The main loop's SASS, per 64
inputs (64 HMMA): 128 HADD2 + 128 FFMA (the fp32 fold every 32 inputs: two converts and two FMAs per MMA), ~190 for
the four tile decodes, 32 LDSM, ~100 addressing and control: 10 instructions per MMA. SM120 has no packed f32 FMA:
`fma.rn.f32x2` compiles to two FFMAs.

Variants measured on the way (each against the same day's baseline, uncapped):

| change | effect |
|---|---|
| decode: split for the card's resident CTAs (one unit each) instead of ~1,024 units | body 53.5% → 60.2% |
| decode: programmatic dependent launch after the rotation | +2.2 points (off: 57.6%) |
| decode: weights as the MMA's A, tokens as N (1 MMA per tile up to 8 tokens), fp16 accumulation folded every 32 | body 60.2% → 65.6%; fold every 128 inputs exceeded the tolerance (worst 1.12 of the bound) |
| decode: activation rows padded to 272 B in shared memory | 16 and 64 tokens 1.4-2x faster, one token unchanged; the same layout with a runtime stride lost 4 points at one token |
| prefill: 4-stage, 64-input pipeline, next tile row decoded under this row's MMAs | 185-226 → 200-236 TFLOPS |
| prefill: 64-token tiles, 2 CTAs per SM | ~195, slower |
| prefill: A fragments prefetched / decode plan hoisted | within run-to-run noise (±5%) |

## Finding

- **Decode meets Acceptance 4, by a thin margin.** The body at one token reads at 65.6% of the roofline on a cool
  card (64-66% across runs), and eight tokens cost 1.04x one — the weight tile is the MMA's A operand and the tokens
  its N, so up to eight tokens are the same instructions. The large matrices stream at 69-75% and the head at 92%;
  the body is held back by the small ones (GDN z/out, GQA o at ~51%, k/v at 25%), which pay a call's fixed ~3.3 µs
  (two launches, the split reduction) on 4-12 µs of bytes.
- **Prefill misses Acceptance 5.** 212-238 TFLOPS at 1,024 tokens is 51-57% of the whitepaper's 419 and 42-47% of
  the 502 this card's MMA loop reaches at its real ~2.9 GHz clock. The decided fold every 32 inputs costs four
  instructions per MMA with no packed alternative on SM120, and one 8-warp CTA per SM cannot hide the decode's and the
  fold's fixed latencies. Four restructurings landed on the same plateau.
- **Small cards are near compute balance.** Capped, the decode streams ~9 GB/s per SM whatever the cap (the uncapped
  5090 is DRAM-bound at 6.9): 313 / 627 / 744 GB/s at 36 / 70 / 84 SMs, 70% of a 5060 Ti's 448 GB/s, 70% of a
  5070 Ti's 896 and 77% of a 5080's 960 — so on those cards decode speed is set by the tile decode's instruction
  count, as spec 02's Further Notes expected. These are the 5090's clocks and L2; a real card's own clock moves them.
- **Numerics.** Both routes accumulate in fp16 and fold into fp32 every 32 inputs, with a power-of-two activation
  scale per token and 128-input block that bounds every fold below 56,576 (< 65,504). Against the fp64 product of the
  decoded weights every route and shape stays inside |y − Y| ≤ 2^-8 |Y| + 2^-9 s (worst 0.92 of the bound, of which
  the BF16 output's rounding alone can take 2^-8 |Y|), including an adversarial input at the scale's bound at
  in = 17,408 (worst 0.67). The spec's fallback (reconstruct and BF16 GEMM) is not needed.

## Implications

- Acceptance 5 is the owner's call: accept the prefill as built (spec 05's end-to-end prefill on the 16 GB artifact is
  the real gate), allow a fold every 64 inputs (halves the fold's cost, moves a decided spec item, needs the numerics
  re-checked), or fund a deeper prefill restructure.
- Decode levers left: fewer calls (q/k/v and qkv/z as one launch: ~3 µs each, ~0.25 ms a token), and a split rule
  that counts waves rather than units (at a 70-SM cap GDN qkv runs 80 units on 140 CTAs).
- The plan line at 16 decode and 2,048 prefill tokens is 72.4 MB, almost all the down projection's rotated
  activations (2,048 x 17,408 fp16); a 16 GB card's prefill chunk sets it.

## Limits and unknowns

One card, one day; the cool/hot spread of the body figure is 2 points. Synthetic weights (counter-hash trellis
words; the quant's K mix, not its bytes). The caps model a small card's SM count only — its DRAM, L2 and clocks are
the 5090's. ncu profiles at base clock.
