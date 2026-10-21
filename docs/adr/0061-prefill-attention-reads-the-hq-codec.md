# ADR 0061 — prefill attention reads the hq-e8-2b codec: no prompt planes, on every card

## Status

Accepted (2026-10-21, after the owner's review: "accetto tutto"). Proposed the same day. Owner decision in the 16 GB grill ("niente area mi piace; questo ce lo porteremo come vantaggio
anche per la 5090"):
- the prompt planes go;
- on every card class;
- with numerical, not bit-exact, equivalence;
- the attention readout moves to the same path.

Related:
- [ADR 0022](0022-two-kv-formats-bf16-as-oracle.md): BF16 KV is the oracle;
- [ADR 0030](0030-device-memory-reserved-at-load.md): device memory is reserved at load;
- [ADR 0031](0031-vendored-kernel-bottleneck-exemption.md) and [ADR 0051](0051-kernel-code-is-ours-adapted-or-vendored.md):
  vendored, adapted and own kernel code;
- [ADR 0060](0060-the-27b-has-a-trellis-body-for-small-cards.md): the 16 GB body.

Plan: `.scratch/27b-16gb-study-2026-10-20/PIANO-VRAM.md` §9 (untracked).

## Context

Under hq-e8-2b the vendored prompt route (`gqa_attention`, route `Prompt`) works in two steps:
1. before a prefill chunk attends, it decodes the layer's whole visible history into BF16 **prompt planes**,
   `[256, 4 KV heads, keys]` for K and for V;
2. it runs an FA2 kernel over them.

The planes cost 4,096 bytes per context token, up to one band of 262,144 keys, so 1.07 GB at 256K or more. They are
prefill-only:
- decode and verify rounds read the codec in place (the small-T routes);
- the planes die with the layer's scope.

Prefill interleaves with decode at chunk level (ADR 0018), and every device byte is carved at load (ADR 0030). So the
arena holds them for the whole load, on every card.

On a 16 GB card that 1.07 GB decides whether 256K fits. With the 3.0 bpw body and two lanes, the pool KV left beside
one full 256K sequence would be:
- **+82K** with the planes;
- **+169K** with a 64K band;
- **+199K** with no planes.

On the 5090 the same gigabyte is ~116K more KV tokens.

The options costed:

| option | bytes freed at 256K | why not |
|---|---:|---|
| planes in host memory | — | a chunk rereads the history in all 16 layers: ~16 GB per chunk at 256K over PCIe |
| lend them to the KV pool between prefills | — | pages live long; taking them back means evicting a sequence, and it breaks ADR 0030 |
| a smaller band (the vendored constant `kGqaHqPromptScratchBandKeys`, 262,144 → 65,536) | 0.81 | the banded path exists and is tested, but its prefill cost is unmeasured, and the attention readout needs a single band |
| one arena for prefill scratch and round buffers | 0.1-0.2 | small; decode graphs need fixed addresses |

## Decision

1. **A GQA prefill chunk under hq-e8-2b attends straight from the paged codec.**
   - An attention kernel of our own (an **adapted op**, ADR 0051) decodes each K/V tile into shared memory once per
     CTA, as the decode route decodes per key, and runs the chunk's 64-128 queries against it.
   - No prompt planes are materialized.
   - The workspace no longer grows with `max_context`.
2. **Every card class: the 27B's GQA layers, whichever artifact they run.** The 32 GB NVFP4 load gets the same bytes
   back. Flash-Next is not concerned: its QSA layers have their own prefill (dense up to 2,051 keys, sparse past it)
   and never call the vendored prompt route.
3. **Equivalence is numerical, not bit-exact** (owner). The decoded values are the same; the reduction order is not.
   The acceptance criteria below are the agent's proposal:
   - **route agreement** against ignis's BF16 route, at least as close as today's prompt route (`test_hq_route_agreement`);
   - **KLD unchanged** within noise on the 2026-09-24 suite (serving 0.0513);
   - **cold prefill TTFT** on the 5090 at 8K, 46K and 105K **no slower** beyond the run's noise. Performance-first: the
     owner has not set this one.
4. **The attention readout** (`/v1/decide`, `locate`, #260) reads from the same path. It loses its single-band
   restriction.

## Considered options

- **Keep the planes and shrink the band to 64K.** It is the fallback if the gate in point 3 fails: a recorded patch to
  a vendored constant (ADR 0037/0031), once its prefill cost is measured.
- **Bit-exact with the vendored route.** Rejected: it would force the vendored kernel's reduction order onto the new
  one and cancel the design.

## Consequences

- The vendored prompt route stays in the tree as the reference the agreement test compares against, but serving never
  calls it. Its kernels are no longer the prefill bottleneck to optimize.
- The VRAM plan's workspace line drops the 4,096 bytes per context token.
- The 5090's profile is re-measured ("raise until refused", as on 2026-10-10) once this and ADR 0062 land.
