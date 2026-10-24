# Phase 2 of the 27B on 16 GB cards — notes, not for now

The owner's phase 2, recorded on 2026-10-21. Nothing here is specified; each item becomes a spec when phase 1 has
shipped. After phase 2 comes the image model on 16 GB.

- **Flash-Next on 16 GB.**
  - The expert-cache floor (`EXPERT_CACHE_FLOOR_BYTES`, 12 GiB, `crates/core/src/residency/plan.rs:28-31`) becomes a
    config field with a 12 GiB default, by the owner's rule on thresholds.
  - The `rtx50-16gb` profile sets its own value: a first estimate of ~5 GiB, tuned on hardware.
    - At 1 lane with a 262K pool, Flash-Next's fixed lines are ~9.65 GB.
    - That leaves 5.4-7.1 GB of cache on a 16 GB card.
  - Hit rate and decode at that cache are not estimated. Decode will be bound by the user's PCIe link.
  - Its catalog entry then states `card_class: 16`.
- **The abliterated variant** at the same bpw: same converter (spec 01), its own catalog entry and alias.
- **A speed gate on a real card**, ignis against llama.cpp on the same card (Q8). The card comes from renting by the
  hour, from users running spec 05's bench, or from a purchase; decided when phase 2 starts.
- **The headroom on a real card** (GitHub #346; inferred, not measured).
  - Phase 1's capacity runs named their budget with `--vram-budget-bytes`, which skips the profile's 1,200 MiB
    headroom. On a real card the budget is the free memory less that headroom, from a total of about 15.9 GiB.
  - From the capacity finding's plan lines (1,820 pages a GiB; 262,144 needs a pool of 327,680 tokens):
    - the context stays at 262,144 only with about 0.3 GiB or less held by anything else;
    - with 1 GiB of desktop the load adapts to 131,072;
    - with 3 GiB the pool is ~11.5K tokens, ~37K at one lane, short of the 131,072 that 65,536 needs, so the load
      would be refused.
  - The first real card's `ignis-bench small-card` record settles it. If it holds, `rtx50-16gb` gets its own
    `vram.headroom_bytes` or `vram.adapt_spare_tokens`, measured on that card.
- **Per-card 16 GB profiles** (`rtx5060ti`, `rtx5070ti`, `rtx5080`), only where measurements show the lanes or draft
  tokens should differ.
  - The 5060 Ti is the likely one: it turns compute-bound past 2-3 lanes in the estimates.
- **12 GB:**
  - the 27B at ~2.5 bpw, 1 lane, with allocation (the coverage-model literature: uniform collapses near 2.9 bits, a
    measured allocation holds);
  - MTP instead of DFlash2 where the bytes are short.
- **MTP as the drafter fallback** (ADR 0060). Use it if M2 or a long-context measurement shows DFlash2 collapsing
  past 100K, or where VRAM is short.
  - ~1.1 GB fixed saved.
  - −5-15% on agent traffic and −30-35% on greedy coding at 8K (estimates).
  - The MTP layer is already in the 16 GB artifact.
- **A trellis drafter** (DFlash2 at ~4 bpw, −0.23 GB), if space is short.
- **BF16 embedding on the host**, if the W8 table is ever shown to cost quality.
- **Prefill to ~12,500 tok/s at 60K on the 5090** (owner, 2026-10-21). After #335 (FP8 attention) and #341 (fixed part
  ≤ 85 µs/token), the fixed part must come down to ~66 µs/token. That means:
  - GEMMs near 90% of the FP4 peak;
  - GDN, norms and quantization fused and lean.

  It is planned from #341's measurements, not before.
