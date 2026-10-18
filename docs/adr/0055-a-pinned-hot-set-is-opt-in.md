# ADR 0055 — a pinned hot set of experts, opt-in and off by default

## Status

Proposed (2026-10-18), for the owner. The owner asked for the feature as an
opt-in (#329 item 12, 2026-10-18: "build it as an opt-in feature; with the
field at its default the behaviour, plans, logs and outputs are byte-for-byte
today's"); the measurement is
[`docs/findings/2026-10-18-pinned-hot-set-measured-on-the-5090.md`](../findings/2026-10-18-pinned-hot-set-measured-on-the-5090.md),
after the study [`2026-10-18-partial-exclusive-expert-pool.md`](../findings/2026-10-18-partial-exclusive-expert-pool.md).
Amends spec flash-next/03's "every expert in pinned RAM" for an offloaded load
that asks for a pin. ADR 0054's device-only load is untouched.

## Context

An offloaded Flash-Next load (the 5090: the VRAM expert cache holds about half
the experts) keeps every projection in the pinned host pool, ~38 GB, so that a
miss copies it in. A projection that never leaves VRAM needs no host copy. The
study found that today's LRU keeps no such set for long, but that pinning the
calibration-hottest 1-2 GB anyway, outside the LRU, costs under 1% of decode
in replay. On a 64 GB host the pool is most of the RAM; the KV-RAM arena and
the n-gram hot rows compete for the rest.

## Decision

**`vram.pinned_expert_bytes` (`--vram-pinned-expert-bytes`, Flash-Next only,
changed through a reload) pins the calibration-hottest whole experts until
their bytes reach it. 0, the default, pins none.**

- **The set.** Whole experts in the warm start's order (the sidecar's
  calibration selections), taken out of the pools the cache split already
  made, each class keeping its decode minimum for the LRU
  (`pin_hot_experts`), as the study measured it. Not a new split of the rest.
- **Host.** The pool is laid out without the pinned records, which have no copy
  in RAM; they stream from the artifact to their VRAM slots before the pool is
  filled. The host plan's `expert_pool` line drops their bytes; the buffers
  they stream through are its `load_staging` line, held only while the experts
  load.
- **Device.** A pinned key has a slot past its class's LRU slots. A selected
  pinned key is a hit and nothing else: never stamped, evicted, staged,
  prefetched or copied. The CPU policy model pins the same way
  (`ResidencyModel::pin`), and the device equals it on random traces (spec
  flash-next/03 acceptance 4). With nothing pinned the resolve decides as
  before.
- **Device-only loads** (ADR 0054) ignore the field: they keep no expert in
  RAM already.
- **Observability.** A load that asked for a pin logs
  `ignis.runtime.flash_next_pinned_experts` (bytes, experts, per class, the
  LRU's slots left). At 0 no event and no plan line changes.

## Considered options

- **On by default at 1-2 GiB.** Measured free at 1 GiB and ~1% of three-lane
  decode at 2 GiB on the 5090. Rejected for now: the owner asked for an opt-in,
  and the RAM pays only where something else needs it.
- **Pin what the LRU never evicts.** Small (0.4-0.5 GB) and still shrinking
  with the length of the trace, and not known at load.
- **A fully exclusive pool** (RAM holds only what VRAM does not, victims
  written back). ~18 GB back for 20-70% of a round on this Gen 3 link (the
  study's inference). Not decided here.

## Consequences

- The host pool and the process's RAM are smaller by exactly the pinned bytes
  (1.075 GB at `1G`, 2.149 GB at `2G` on the served artifact); the plan's
  residency-phase peak adds 0.28-0.50 GB of stream buffers.
- Greedy output is unchanged; decode measured within noise at `1G` and 1.1%
  slower at three lanes at `2G`. The pin takes K = 4 slots mostly, so the cost
  climbs past 2 GiB.
- A load with a pin takes ~2 s longer to load its experts.
- No live resize of the expert cache may evict a pinned projection: it has no
  copy to come back from. Pools are fixed per load (ADR 0030), so today none
  does.
