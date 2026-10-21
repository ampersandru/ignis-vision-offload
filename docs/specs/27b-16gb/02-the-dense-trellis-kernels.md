# 02 - the dense trellis kernels: decode, verify, prefill and the head

GitHub: #337 (master #334).

Decided 2026-10-21: the owner approved this spec and vetoed none of the *(proposed)* items, so every one is decided.

The 27B's 16 GB artifact (spec 01) stores its body and output head in trellis (EXL3 `mul1` code, a K per matrix). These
are the kernels that serve it, every matrix of the 64 layers and the head, in our own code:
- decode and verify rounds, from 1 to `decode_lanes × (draft + 1)` columns;
- prefill chunks.

ADRs: 0060 (the body), 0044 (the trellis format, on Flash-Next), 0051 (ours, adapted or vendored), 0009 (step ABI),
0030 (workspaces as plan lines).

## Problem Statement

ignis's trellis kernels are Flash-Next's MoE kernels (`kernel/src/moe_decode*.cu`, `moe_prefill.cu`):
- geometry fixed at hidden 2560, 512 experts, top-10, intermediate 640 (`kernel/include/ignis_moe.h:41-47`);
- routing mandatory;
- output only into an int64 fixed-point accumulator.

What they reached on the 5090:
- **decode:** 33-56% of the DRAM roofline. They are bound by instruction issue, not memory: ~48 instructions per
  16×16 tile per warp (finding 2026-10-05-moe-decode-is-structure-bound);
- **prefill:** 32% of the FP16 tensor peak.

The dense 27B needs the shapes below, which the container already accepts (dims % 128,
`crates/artifact/src/lib.rs:488-491`):

| matrix | shape [out, in] | × layers |
|---|---|---:|
| MLP gate_up (fused) | [34816, 5120] | 64 |
| MLP down | [5120, 17408] | 64 |
| GDN in-proj qkvz (fused) | [16384, 5120] | 48 |
| GDN out-proj | [5120, 6144] | 48 |
| GQA qkgv (fused) | [14336, 5120] | 16 |
| GQA out-proj | [5120, 6144] | 16 |
| output head | [248320, 5120] | 1 |

If spec 01 repacks a public quant, the fused rows arrive split: `in_proj_qkv` [10240, 5120] + `in_proj_z`
[6144, 5120], and q / k / v separately, each with its own scales.

## Solution

1. **A trellis linear op**, called by the layer code where today it calls `ninfer::ops::linear` / `linear_add` on NVFP4,
   dispatched by the tensor's format (`TRELLIS_MUL1_K*`). One op family serves every shape above with a runtime
   in/out and a K ∈ {2, 2.5, 3, 4, 5, 6}. K5 and K6 are new format codes in `crates/artifact/src/lib.rs`.
2. **Decode / verify (1..64 columns).**
   - Each weight fragment is decoded once from packed words into registers (`kernel/src/trellis_decode.cuh`, reused)
     and fed to `mma.sync m16n8k16` for every column.
   - So 1 and 8 columns cost the same weight bytes and decode instructions.
   - Hadamard and `suh` / `svh` are applied on the activation side and on the output side, as the MoE kernels do.
   - Split-K reduction for in = 17,408.
   - BF16 out, with the epilogues the call sites need: plain, residual add, SwiGLU on gate_up.
3. **Prefill (≥ 65 columns).**
   - Decode tiles into shared memory and run FP16 MMA.
   - *(proposed)* FP16 accumulate folded into FP32 every 32 K-elements. GeForce halves HMMA throughput with FP32
     accumulate; exllamav3 ships the same fold as `hgemm_f16acc`.
   - The numerics are checked at K = 17,408.
   - The fallback, if they fail: reconstruct to scratch and run a BF16 GEMM.
4. **The head** is the same op at [248320, 5120]: the verify round's `k+1` columns and the drafter's proposal pass.
   *(proposed)* Columns blocked so that one decode of a tile serves all 64.
5. **A debug cap on SMs.** *(proposed)* An environment variable read at load, never a serving setting: the decode
   kernels' persistent grid is limited to N SMs (36, 70, 84). The 5090 then runs the kernels as a 5060 Ti / 5070 Ti /
   5080 would issue them. This is the proxy for the compute side of a small card (owner, Q18).

## Implementation Decisions

- The ops are our own (ADR 0051). The format reference is exllamav3 1.5.3's `mul1` codebook and tile layout (MIT).
  QTIP (GPL-3.0) is not read.
- Every op is graph-capturable, allocates nothing while serving, and declares its workspace as plan lines (ADR 0030).
- Deterministic for a fixed input and chunk split: no atomics in the reduction order.
- The 32 GB NVFP4 path is untouched: dispatch is per tensor format.

## Testing Decisions

- Tile decode bit-exact against exllamav3 `reconstruct` for every K. Reuse `kernel/tests/test_trellis_decode.cu`'s
  approach, with K5 and K6 added.
- Linear ops against an fp64 product of the decoded weights:
  - at every shape above (machine-local full shapes from the artifact; committed reduced shapes from exllamav3
    fixtures);
  - columns 1, 2, 8, 16, 64 and 1,024.
- The prefill FP16-accumulate path's error at K = 17,408 against fp64, reported. It must sit inside the stated
  tolerance, or the fallback is used.
- A microbenchmark report:
  - achieved DRAM bandwidth at 1, 8, 16 and 64 columns, every shape;
  - the width-cost ratio (64 columns over 1);
  - prefill TFLOP/s at 1,024 and 8,192 columns;
  - all of it with and without the SM cap at 36 / 70 / 84.

## Acceptance

1. Tile decode bit-exact against exllamav3 for K = 2, 2.5, 3, 4, 5, 6.
2. Decode/verify and prefill linear ops match fp64 within the stated tolerance at every shape and column count tested,
   with the epilogues the call sites need.
3. Every op is graph-capturable, allocation-free while serving, deterministic run to run, and its workspace is a plan
   line.
4. *(proposed)* Decode at 1 column reaches ≥ 65% of the 5090's DRAM roofline for its bytes, averaged over the body.
   8 columns cost ≤ 1.35× one column (today's NVFP4 is 1.30×). The report states the same figures under the SM caps.
5. *(proposed)* Prefill reaches ≥ 60% of the FP16-accumulate tensor peak at 1,024 columns on the 5090.
6. No exllamav3 or QTIP kernel source is copied. Every op is labelled ours (adapted from our own Flash-Next decode
   where it is).
7. `cargo test` workspace-wide; kernel build clean; the leaf's op tests green on a free 5090.

## Out of Scope

- Wiring into the forward and serving the artifact end to end: spec 05's integration, with spec 01's artifact.
- Flash-Next's MoE kernels: unchanged.
- Native sub-4-bit tensor-core formats: SM120 has none below FP4 (PTX ISA 9.4).

## Further Notes

- Ceilings (lit-sec3, whitepaper Tables 3-6): to keep pace with DRAM at 2.5-3.0 bpw, a trellis GEMV needs ~85-110% of
  the BF16-FP32-accumulate peak on the 5070 Ti / 5070, and 60-80% at 3.5 bpw. FP16 accumulate halves that term. On
  small cards the decode is near compute balance, hence the SM-cap proxy.
- exllamav3's own FP16 kernel sits at ~65-78% of DRAM peak on the 5090 (its source comments). The MoE kernel's lesson
  applies: the time is in the tile decode's instruction count, not in memory.
