# 01 - the 27B's prefill on the 5090: the fixed part

GitHub: #341. Companion of #335 (the attention term).

The 27B's cold prefill per token, from the 2026-10-13 measurements (8K / 46K / 105K), is two terms:
- a **fixed part of ~97 µs**: GEMMs, GDN, norms, quantization, everything but attention;
- **~1.84 ns per key already seen**: attention.

Spec 27b-16gb/03 (#335) takes on the second term: no prompt planes, FP16-accumulate / FP8 attention.

This spec takes on the first. It is the other half of the gap to the fastest same-card engine we know of,
[mkyt/qwen3-27b-5090](https://github.com/mkyt/qwen3-27b-5090) (MIT). At a 400 W limit it measured ~80 µs fixed and
0.93 ns per key: 11,300 tok/s at 19K, 8,800 at 73K, 5,900 at 194K. ignis is at 7,600 tok/s at 46K and 5,200 at 105K.

Decided 2026-10-21 by the owner: open this as its own ticket, all levers below in scope. The KV format stays
hq-e8-2b.

ADRs: 0005 (performance first), 0010 / 0031 / 0037 / 0051 (vendored, patched, replaced, adapted), 0030 (workspace is
a plan line).

## Problem Statement

Where the fixed part goes, estimated from measured kernels:

| item | cost | source |
|---|---:|---|
| NVFP4 W4A4 backbone GEMMs | ~45 µs/token | `nvfp4_w4a4_tma` 16384×5120 takes 160 µs per 1,024-token chunk, 64% of the FP4 peak (finding 2026-09-29-gpu-resources-prefill-vs-decode) |
| 9 BF16 exception matrices | ~5 µs | attention qkgv in layers 3/7/11/15/19/23, attention out in 3/7, GDN out in 4. `bf16_gemm_mma` takes 781 µs per chunk for one qkgv, against 160 µs for an NVFP4 16384×5120 |
| GDN, norms, activation quantization, gates, copies | ~45 µs | the remainder (finding 2026-09-29: "other" ~11% of device time at 5K, plus the GDN kernels) |

The chunk is 1,024 tokens. A 4,096 chunk measured +4% at 32K (finding 2026-10-07-27b-prefill-chunk-width).

## Solution

Each lever is an A/B on the 5090, kept only if it is faster and passes the quality check:

1. **The 9 BF16 exception matrices in FP8 W8A8**, through the block-scaled MXFP8 MMA with unit scales. mkyt found it
   bit-identical to the plain FP8 GEMM and twice its rate on SM120.
   - *(proposed)* Quantized at load time from the artifact's BF16, per-row scales. The 32 GB artifact does not change.
   - mkyt keeps every attention projection FP8 because NVFP4 there cost parallel tool calls.
2. **A wider prefill chunk**: 2,048, then 4,096. The workspace grows ~400 KB per chunk token; spec 27b-16gb/03 frees
   ~2.4 GB on the 5090 to pay for it. TTFT and decode share under interleaving (ADR 0018) are both measured.
3. **The backbone GEMMs** from 64% toward the power-limited peak.
   - Profile the vendored `nvfp4_w4a4_tma` / `nvfp4_linear_swiglu_w4a4_tma` against CUTLASS's SM120 block-scaled GEMMs
     (128×128×128 tiles; BSD-3), which mkyt runs "at the power limit".
   - A measured bottleneck may be replaced by our own or an adapted op (ADR 0031/0051).
4. **The GDN prefill.**
   - Profile its share at 8K / 46K.
   - Compare with a chunked delta rule in WY form, chunks of 64: per-chunk matrices in FP32, the inter-chunk state
     update on BF16 MMAs (mkyt's design).
5. **The NVFP4 encoding of the GDN and MLP weights.** A quality lever first, measured before any artifact change.
   - The KLD of `juzbox/Qwen3.8-27B-NVFP4-GDN` (GPTQ from BF16), dequantized into the M1 harness
     (`.scratch/27b-16gb-study-2026-10-20/m1/`), against the served artifact's weights on the same windows.
   - A new 32 GB artifact only if the gain is clear.

## Testing Decisions

- Cold prefill TTFT at 8,192 / 45,605 / 105,276 tokens (the 2026-10-13 prompts), two launches per side, for each
  lever and for all kept levers together.
- Decode tok/s at 1 and 8 lanes, unchanged within noise.
- Quality for levers 1-4: serving KLD on the 2026-09-24 suite within noise of today's. Dumps go to `Y:`.
- Lever 5's KLD comparison weights-only, same windows.

## Acceptance

1. Each lever's A/B is recorded in a finding: kept or dropped, with the numbers.
2. *(proposed)* With the kept levers, the fixed part is ≤ 85 µs/token: cold prefill at 8K ≥ +12% against today's.
   Together with #335's attention as measured, 0.86-0.92 ns per key·token (its 8-bit band), the estimate at a fixed
   part of 85 µs is (recomputed 2026-10-21, finding 2026-10-21-the-27b-prefill-chunk-at-2048):
   - **~8,400-9,000 tok/s at 60-73K**, ~10,700 at 19K. The first estimate, ~9,000-10,500 and ~11,000-12,000,
     assumed ~0.65 ns per key (60-70% of the FP8 peak; mkyt's 0.93 ns at 400 W scaled to default clocks).
   - Measured with the 2,048 chunk: the fixed part is already ~82-85 µs at 46K-105K and ~106 µs at the 8K cell,
     whose excess is per request (~80 ms) and whose request is the first after the load. The 8K criterion above
     measures that, not the per-token term.
   - 12,500 at 60K would need a fixed part of ~53 µs at the measured attention (~66 at 0.65 ns). That push past
     ≤ 85 µs is **phase 2** (owner, 2026-10-21; `docs/specs/27b-16gb/phase2-notes.md`). This ticket stops at AC 2's
     target.
3. Serving KLD within noise of today's for every kept speed lever. Decode unchanged.
4. Lever 5: the KLD of the GPTQ-encoded weights against today's encoding, reported, with a recommendation for or
   against a new 32 GB artifact.
5. `cargo test` workspace-wide; GPU tests green on a free 5090.

## Out of Scope

- The KV format (hq-e8-2b stays) and prefill attention (#335).
- The 16 GB artifact: its attention projections are trellis (spec 27b-16gb/01).
- Power limits: ignis runs at the card's default; mkyt's numbers are at 400 W.
