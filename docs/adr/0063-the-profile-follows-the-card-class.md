# ADR 0063 — the profile follows the card class, and its defaults adapt to the budget

## Status

Accepted (2026-10-21, after the owner's review: "accetto tutto"). Proposed the same day. Owner decisions in the 16 GB grill:
- the profile is `auto` by card class;
- one 16 GB profile in phase 1;
- per-card profiles only if real-card measurements justify them;
- no display detection: "con i parametri di default si adatterà al caso migliore, come fa adesso espandendo il pool
  KV, o con gli esperti";
- architectures other than SM120 are out of scope.

The adaptation order and its 65,536-token floor (point 3), and the `rtx50-16gb` name, are the agent's proposals. The
owner approved them as a whole ("ok tutto") without changing them.

Related:
- [ADR 0030](0030-device-memory-reserved-at-load.md): device memory is reserved at load;
- [ADR 0046](0046-one-declaration-per-config-field.md): one declaration per config field;
- [ADR 0057](0057-rope-scaling-is-auto-by-default.md): `auto` by default, as for rope;
- [ADR 0060](0060-the-27b-has-a-trellis-body-for-small-cards.md): the 16 GB body;
- [ADR 0064](0064-a-catalog-entry-states-its-card-class.md): the catalog entry's card class.

## Context

Today the profile is chosen by name only (`--profile`, `IGNIS_PROFILE`, the config file), and its default is
`rtx5090`. Nothing reads the card:
- the total memory NVML reports is thrown away;
- there is no GPU name and no compute capability.

So a 16 GB card would start under the 5090's profile, without a word.

The budget a load gets is already the free memory at start minus the headroom: whatever the desktop holds is counted.
After that:
- the KV pool takes what is left (27B);
- the expert cache takes what is left (Flash-Next).

`max_context`, the lanes and speculation are fixed, so a short budget is refused with the byte deficit. On a 16 GB card
the free memory at start varies with what else holds the card, and 256K with DFlash2 fits only when little else does.

## Decision

1. **`--profile` defaults to `auto`.**
   - It picks the built-in profile by the card's total memory: ~32 GB → `rtx5090`, ~96 GB → `rtxpro6000`, ~16 GB → the
     new `rtx50-16gb`.
   - On any other total the start is refused, with a message that names `--profile`.
   - A named profile always wins, `none` included.
   - On today's cards `auto` resolves to today's profiles.
2. **`rtx50-16gb`** defaults to:
   - the 27B's 16 GB catalog entry (ADR 0064);
   - `decode_lanes` 2;
   - `max_context` 262,144;
   - DFlash2 on, at the family's default draft tokens (to be tuned on a real card: speculation yields less on a
     compute-poor card);
   - the drafter's proposal head `auto`;
   - vision off.
   - `decode_lanes` becomes one field for both families, 1..8.
3. **A profile's defaults adapt to the budget at load, in a fixed order.** An explicit value from any source never
   adapts and is refused as today. When the plan does not fit, the load tries, in turn:
   1. the drafter's proposal head as its `auto` rule chooses;
   2. `max_context` down in steps, 262,144 → 196,608 → 131,072 → 65,536. It keeps the largest at which one full
      sequence and one more 64K reservation fit the pool;
   3. `decode_lanes` 2 → 1.
   - DFlash2 is never switched off by adaptation.
   - Below 65,536 the start is refused with the deficit, as today.
   - Each step taken is named in the VRAM plan's log event (`adapted: max_context 262144 -> 131072`).

## Considered options

- **Detect whether the card drives a display** (NVML `display_active`) and choose the context from it. Rejected by
  the owner. The free memory at start already says what the desktop holds, and adaptation answers it.
- **A profile per 16 GB card** (5060 Ti, 5070 Ti, 5080). Deferred to phase 2. Memory-shaped fields are the same on all
  three; bandwidth-shaped ones (lanes, draft tokens) need a real card to tune.
- **Refuse instead of adapting.** Rejected: a 16 GB card's free memory varies run to run, and refusing what a smaller
  context would serve helps no one.

## Consequences

- A load may run a smaller context than its profile names, and says so in the log.
- The order (drafter head, then context, then lanes) puts speed before context length, on purpose.
- The plan's refusal message must say which adaptations it already tried.
