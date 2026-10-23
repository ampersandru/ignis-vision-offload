# 06 - the dense trellis prefill at the tensor peak

GitHub: #345 (master #334). Follow-up of 02 (#337): its Acceptance 5, missed, moved here by the owner on 2026-10-23
(option c, "a deeper restructure of the prefill").

Spec 02's prefill route (`kernel/src/trellis_linear_prefill.cu`) reaches 212-238 TFLOPS at 1,024 tokens on the
27B's large matrices: 51-57% of the 5090's 419 TFLOPS fp16-accumulate peak (whitepaper), against the 60% spec 02 asked.
This ticket restructures that route until it reaches 60%, or measures why it cannot with the fold spec 02 decided.

ADRs: 0060 (the body), 0051 (our code). Evidence: [the 2026-10-23 finding](../../findings/2026-10-23-dense-trellis-kernels-on-the-5090.md).

## Problem Statement

What the 2026-10-23 measurements say limits the prefill:
- **The fold.** fp16 accumulation is folded into fp32, times the activations' inverse scale, every 32 inputs (spec 02,
  decided; it stays). Per m16n8k16 that is two fp16 → fp32 converts and two FFMA: ~4 instructions per MMA. SM120 has
  no packed f32 FMA (`fma.rn.f32x2` compiles to two FFMAs).
- **The decode** of a tile row, ~2.5 instructions per MMA at 128-token tiles, in the same warps as the MMAs.
- **Latency.** One CTA of 8 warps per SM (213-235 registers): 2 warps per scheduler. Nsight Compute: tensor pipe active
  57%, issue slots busy 39%, top stalls `wait` and `math_pipe_throttle`.
- **The wave tail.** Equal 128 x 128 tiles: GDN qkv at 1,024 tokens is 640 tiles on 170 SMs (3.76 waves), the
  MLP gate pass 1,088 (6.4).

Four variants landed on the same plateau (64-token tiles at 2 CTAs per SM; the current 4-stage pipeline with the
next row decoded under this row's MMAs; A fragments prefetched; the decode plan hoisted).

## Solution

Experiment first, measure each lever against the same day's baseline in the bench, keep what pays:

1. **Warp specialization:** producer warps that decode each stage's trellis tiles once into fp16 B tiles in shared
   memory (cp.async in, decoded out), consumer warps that only `ldmatrix` A and B, multiply and fold. The decode leaves
   the MMA warps' instruction stream, and the extra warps add latency hiding. The consumers' warp tile is then free
   (for example 2 x 4 warps of 64 x 32), which `ldmatrix` B allows.
2. **The fold's own cost:** keep the interval at 32, but cut instructions around it: the inverse scale held in
   registers for a 128-input block instead of re-read per fold, and the fold interleaved with the next row's MMAs.
3. **The wave tail:** stream-K or a smaller last wave, so the final wave of tiles does not run half empty.
4. **Tile to tile:** the next tile's prologue (its first stages' loads) issued under the current tile's epilogue.

## Implementation Decisions

- Unchanged from spec 02: the fold every 32 inputs and the activation scale's bound (2^9), the tolerance, BF16 out,
  the epilogues (store, residual add, SwiGLU's two passes), determinism with no floating-point atomics in a sum,
  graph capture, no allocation, the workspace as a plan line, the debug SM cap.
- The decode route (`trellis_linear_decode.cu`) is not this ticket's.
- Our own code (ADR 0051): no exllamav3, QTIP or CUTLASS kernel source is copied.

## Testing Decisions

- `test_trellis_linear` unchanged and green: every shape at 1,024 tokens on the prefill route, the epilogues, the
  adversarial input at the fp16 bound at in = 17,408, determinism and graph replay.
- The bench (`ignis_kernel_trellis_linear_bench`) is the measurement: one run on a cool card, its SM clock and power
  sampled beside it, as in the 2026-10-23 finding.

## Acceptance

1. Prefill reaches ≥ 60% of the 419 TFLOPS fp16-accumulate peak (≥ 251 TFLOPS) at 1,024 tokens on each of the body's
   large matrices (GDN qkv, z, out; GQA q, o; MLP gate + up; MLP down) on the 5090. The report also states the share of
   the card's measured register-loop peak.
2. No regression elsewhere: 8,192 tokens not slower than 2026-10-23's figures, the decode route's body still ≥ 65% at
   one token, the 16 GB caps (36 / 70 / 84 SMs) reported.
3. Numerics unchanged: `test_trellis_linear` green under the same tolerance, with the fold every 32 inputs.
4. Deterministic, graph-capturable and allocation-free as before; the plan line updated if the workspace changes.
5. The finding records every lever measured, kept or not. If 60% is not reachable with the fold at 32, the ticket
   ends with that measured result and the owner decides.
6. `cargo test` workspace-wide; the leaf's CTests green on a free 5090.

## Out of Scope

- The fold interval (spec 02's decision; changing it is the owner's option b, not taken).
- The decode / verify route, the layer wiring (spec 05), Flash-Next's MoE kernels.
