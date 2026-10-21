# 03 - VRAM recovered on every card: no prompt planes, the token embedding in host memory

GitHub: #335 (master #334).

Decided 2026-10-21: the owner approved this spec and vetoed none of the *(proposed)* items, so every one is decided.

Two changes that give the 27B ~2.4 GB of VRAM back on every card, and that the 16 GB card class needs to reach 256K:
- **the prompt planes go**: 1.07 GB at ≥ 262K of context (ADR 0061);
- **the token embedding moves to pinned host memory**: 1.35 GB (ADR 0062).

Then the 5090's built-in profile is re-measured to spend what they free.

ADRs: 0061, 0062; 0030 (reservations at load), 0018 (prefill/decode interleaving), 0022 (BF16 KV as oracle), 0031 /
0051 (vendored, adapted, own), 0037 (recorded patches).

## Problem Statement

Under hq-e8-2b, the vendored GQA prompt route decodes a layer's whole visible history into BF16 prompt planes before a
prefill chunk attends:
- 4,096 bytes per context token, up to 262,144 keys;
- `kernel/vendor/src/ops/wrapper/gqa_attention.cpp` ~355-370 (`allocate_hq_prompt_scratch`) and ~527-575 (capacity).

The arena holds them for the whole load, though only a prefill chunk uses them: decode and verify read the codec.

The token embedding, `text/token_embedding`, W8G32 `[248320, 5120]`:
- is bound on the device like every text tensor (`crates/artifact/src/inventory.rs:174, 415-426`), 1.35 GB;
- each token reads one 5,440-byte row (`kernel/vendor/src/ops/wrapper/embedding.cpp:197-231`; callers
  `kernel/src/step.cu:345,462,984`, `decode_graph.cu:94,188`, `dflash2_drafter.cu:150`).

## Solution

1. **An hq prefill attention that reads the paged codec.** It is our own adapted op (ADR 0051).
   - Each CTA decodes a K/V tile of the visible history into shared memory once, exactly as the decode route decodes a
     key, and attends its block of the chunk's queries against it.
   - Online softmax across tiles.
   - The residual window (sinks, ring, the chunk itself) is served exact, as today (ADR 0037's ring-before-append
     included).
   - `kernel/src/gqa_layer.cu` calls it instead of the vendored `Prompt` route for hq-e8-2b chunks.
   - The workspace sizing drops the planes.
   - BF16 KV keeps its route.
2. **The attention readout** (#260: `/v1/decide`, `locate`, `kernel/src/gqa_layer.cu:258-283`) reads through the same
   op. The single-band condition goes.
3. **The token embedding in mapped pinned host memory.**
   - The binder retains `text/token_embedding` on the host (`Binder::retain_tensor_on_host`,
     `crates/artifact/src/binder.rs:255-266`), in an allocation that is mapped (`cudaHostAllocMapped`, or
     `cudaHostRegister(...Mapped)`).
   - The gather is handed the mapped device pointer. The vendored gather takes a raw pointer and needs no change.
   - The drafter embeds through the same pointer.
   - The VRAM plan drops the table; the 27B's host plan adds it and refuses a start that cannot pin it, as
     Flash-Next's does (`model.host_margin_bytes`).
4. **Re-measure the `rtx5090` profile**, raising until the plan refuses, as on 2026-10-10
   (`crates/server/src/config/profile.rs:81-97`). Then write the new values, and the `rtxpro6000` ones if they move.

6. **The new prefill attention computes for speed, not only for memory** (owner, 2026-10-21).
   - The growing term of the 27B's prefill is its attention: ~1.84 ns per key already seen, per token, from the
     2026-10-13 cold-prefill measurements. That is BF16 MMA with FP32 accumulate at about the card's peak, which
     GeForce runs at half rate.
   - The kernel accumulates **QK and PV in FP16 on the tensor cores**, with the softmax running max and sum in FP32
     and a fold into FP32 per key block.
   - **An FP8 (E4M3) variant** of Q·K and P·V is built behind a load option and measured. It becomes the default
     only if it passes the same agreement and KLD criteria.
   - Estimate (no measurement), cold prefill on the 5090:

     | prompt | today | FP16 accumulate | FP8 |
     |---|---:|---:|---:|
     | 46K | ~7.6K tok/s | ~9.1K | ~10.0K |
     | 60K | ~6.6K | ~8.0K | ~9.0K |
     | 105K | ~5.2K | ~6.9K | ~8.3K |

## Implementation Decisions

- *(proposed)* The new op lives in `kernel/src/` as an adapted op. Its header names what it adapts:
  - the vendored hq decode for a key;
  - the FA2 tiling of the vendored prompt kernel.
  The vendored route stays in the tree as the agreement oracle.
- *(proposed)* Tile: 64 keys × 256 dims per K/V tile, 64-128 queries per CTA (the chunk is 1,024). It is tuned by
  measurement against today's prompt route on the 5090, which is the floor.
- *(proposed)* The mapped table uses `cudaHostAllocMapped | cudaHostAllocPortable`, written once at load from the
  artifact stream. It is never registered from a pageable buffer.
- *(proposed)* Under WDDM a large pinned mapped allocation is part of the A/B: one run with the desktop under load.

## Testing Decisions

- Route agreement: the new op against ignis's BF16 route, on the fixtures and contexts of `test_hq_route_agreement`.
  At least as close as today's prompt route, which the test also runs, so both numbers are in the same report.
- `test_gqa_workspace_capacity`: the hq prompt scratch is gone for every envelope; BF16 unchanged.
- KLD on the 2026-09-24 suite, serving configuration (hq-e8-2b, chunk 1024), against BF16: within noise of 0.0513.
  The engine dumps go to `Y:` (2.03 GB per window).
- A/B on the 5090: today's main against the branch, same artifact, `IDLE_UNLOAD=0`:
  - cold prefill TTFT at 8,192 / 45,605 / 105,276 tokens (the 2026-10-13 prompts);
  - decode tok/s at 1 and 8 lanes (the 2026-09-24 coding prompts);
  - two launches per side.
- The decide/locate test suites green, with the readout on the new path.

## Acceptance

1. Under hq-e8-2b no prefill chunk materializes prompt planes. The VRAM plan's workspace no longer depends on
   `max_context` through them. At 524,288 the workspace drops by ≥ 1,073,741,824 bytes against today's plan for the
   same configuration.
2. Route agreement (median relative L2 and cosine on the test's contexts) is at least as close as today's prompt
   route's.
3. Serving KLD on the 2026-09-24 suite within noise of today's (0.0513 ± the bootstrap interval of the finding).
4. Decode tok/s at 1 and 8 lanes no slower than today's beyond the run's noise. Cold prefill at 46K and 105K faster
   than today's by at least half the estimate of Solution 6: ≥ +10% at 46K and ≥ +15% at 105K with FP16 accumulate.
   The FP8 variant's figures are reported. *(The owner set the equivalence and the FP16-accumulate design; the
   thresholds are the agent's.)*
5. `/v1/decide` and `locate` answer as before on their test suites. A locate past 262,144 keys no longer needs a single
   band.
6. The 27B's token embedding is in mapped pinned host memory. The VRAM plan's `weights_bytes` drops by its 1.35 GB;
   the host plan lists it. A start that cannot pin it is refused with the host plan's message.
7. Decode and prefill with the host embedding pass the same A/B as 4.
8. The `rtx5090` profile is re-measured and its new values are written with the measurement
   (`docs/findings/`). The `rtxpro6000` values too, if they move.
9. `cargo test` workspace-wide; kernel build clean; leaf tests green on a free 5090.

## Out of Scope

- Flash-Next: its QSA prefill never used the prompt planes. Its embedding and n-gram tables already have their own
  placement.
- BF16 KV's prefill route.
- Any change to the 32 GB artifact.

## Further Notes

- If AC 4 fails and cannot be fixed, the fallback recorded in ADR 0061 is the vendored band at 65,536 keys:
  - a recorded patch to `kGqaHqPromptScratchBandKeys`;
  - −0.81 GB at 256K;
  - its prefill cost measured the same way.
- Expected effect on the 5090: ~2.4 GB more for the KV pool, or ~260K tokens of hq KV. Whether that becomes context
  (the envelope is 1,048,576), retained slots or vision is AC 8's measurement.
