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

**As built** (2026-10-23, GitHub #336), adaptation and the drafter's head, where the decision left a choice or rested
on a wrong premise:

- **`draft_head: auto` is the full head on every card class.** Spec 04 proposed `shortlist` "when the plan needs the
  bytes it frees, or on the 16 GB class with the pool short of two full reservations". It frees none: the shortlist
  head is bound beside the full head (`ProposalHead::Shortlist`, "costs its 356 MB"). The measured plans: text +
  DFlash2 18,289,864,964 B, with the shortlist 18,646,905,344 B (+357,040,380). On the one card that measured it, it
  was slower too: −7.4% tokens/round, −5.5% tok/s (finding 2026-09-24-upstream-quick-wins-ab). Both clauses of the
  rule were about bytes, so with the facts right `auto` is `full`. The rule is revisited in phase 2, when a real
  16 GB card measures the trade.
  - `auto` is a third spelling of the field (`full | shortlist | auto`), not a third head:
    `ignis_core::ProposalHeadSetting` beside `ProposalHead`, as `RopeSetting` is beside `RopeScaling` (ADR 0057).
  - It resolves where the load's speculation is built (`derive_speculation`). `GET /v1/config` keeps `auto`.
  - It is valid with `dflash2` and with `mtp`, whose only head is `full`. Without a backend it is refused, as any head
    is.
  - A load whose head is `auto` logs `ignis.runtime.draft_head`: `draft_head auto -> full`.
- **Step 1 drops a `shortlist` only.** A head the profile or the default gave falls to `full` when it is `shortlist`.
  Since `auto` is `full`, it never fires on today's built-in profiles; a test profile that sets `shortlist`
  exercises it.
- **The context ladder generalizes §3's.** 262,144 → 196,608 → 131,072 → 65,536 is "down 65,536 at a time". From any
  value `v` the candidates are `v`, then every multiple of 65,536 below it, down to 65,536: 917,504 → 851,968 → … on
  the 5090, 1,048,576 → … on the RTX PRO 6000. The owner, 2026-10-21: with `auto` the 5090's context finds the
  largest that fits.
- **The criterion, at more than one lane:** a context `c` is kept when the plan fits and its KV pool holds `c` +
  65,536 tokens: one full sequence and one more 65,536-token reservation. The plan's own floor, a tail page per
  retained slot, still applies. At one lane a second reservation has no lane to serve, so the plan's floor alone
  decides.
- **Only a plan refused below its minimum adapts** (`VramPlanError::BelowMinimum`).
  - A start whose plan fits as configured takes no step, and its `adapted` is empty. On the 5090, a start that fits
    today is byte-identical.
  - A budget above free memory, or a named KV pool below its floor, is refused as before.
  - A budget with `--vram-allow-oversubscription` never refuses, so it never adapts.
- **Step 3 runs the ladder again.** When the ladder reaches 65,536 without meeting the criterion, the lanes drop to 1
  (if not explicit and above 1). The ladder then runs again from the profile's value, against the one-lane
  criterion. The lines that follow the lanes are planned again at one lane. So is the 27B's default of host
  retained slots (two a lane), unless the operator named it.
  - When the lanes cannot drop, the load keeps the largest context the plan itself holds, and the step says so:
    `(one sequence only: …)`. Below 262,144 tokens the pool only grows as the context falls, so that context is
    65,536 whenever it happens.
- **The refusal** carries the last, smallest attempt's deficit, in the plan's own words. After it come every
  adaptation tried, in order: `; adaptation tried, in order: draft_head shortlist -> full, max_context 262144 ->
  196608 -> 131072 -> 65536, decode_lanes 2 -> 1, max_context 262144 -> …`. With nothing adaptable it reads exactly
  as before.
- **The 27B only.** Flash-Next's plan sizes an expert cache from the rest; its 16 GB story is phase 2.
- **The 65,536 step, its floor and the 65,536-token second reservation are constants beside the ladder**
  (`CONTEXT_STEP_TOKENS`, `SECOND_RESERVATION_TOKENS`, `crates/server/src/adaptation.rs`). Neither refuses a start
  that code without adaptation accepted: each such start was refused already.
- **Where it lives.**
  - The ladder is a pure function, `adaptation::adapt`. Its inputs are the values, which of them may adapt, and a
    closure that plans at an attempt.
  - The closure lives in the planner's caller, `runtime::qwen27b_load`. At each attempt it plans the leaf's lines
    again: the hq prompt planes follow the context up to 262,144 keys, the rope table, the lanes, and the head's
    bound scope. It then calls `plan_vram`, unchanged.
  - Which values may adapt is decided by source alone (`Resolution::adapts`, `Config::adaptable`, carried as
    `EngineShape::adaptable`). The profile's and the hardcoded default's adapt. A flag, env var, file value or live
    patch never does, a file's value equal to the default included: a file `config generate` wrote names the
    numbers of the card it was generated on.
- **The values the load kept are the load's everywhere after the plan.**
  - They reach the leaf, the scheduler (admission, `max_model_len` in `GET /v1/models` and the tokenize route, the
    lanes Decide fans out to) and KV-disk's identity (its rope table).
  - An `auto` rope table is resolved again for an adapted context (`EngineShape::at`); a spelled-out table stays.
  - `ignis.model.loaded` names the kept values, and `/metrics` the adapted plan's lines.
  - `GET /v1/config` keeps the values configured.
- **`ignis.runtime.vram_plan` gains `adapted`**: the steps taken, in order, comma-separated; empty when none.
- **`rtx50-16gb`** gains `spec.qwen38.decode_lanes: 2` and `spec.qwen38.draft_head: auto`, scoped to the 27B. The
  other profiles leave the head unset, which is `full`.
- **Measured on the 5090** (`crates/server/tests/adaptation_gpu.rs`, explicit budgets, 2026-10-23). Each budget was
  derived from the plan's own refusal at 1 GiB, which names what the plan needs. The desktop held 2.6 GB of the card.
  - Under `rtx5090` the 917,504-token plan needs 30,607,308,032 bytes, more than the 30.41 GB that was free.
  - A 30,143,774,720-byte budget loads at `max_context 917504 -> 786432` (pool 868,224 tokens), serves
    `max_model_len` 786,432 and decodes.
  - A 23,349,097,728-byte budget, where no eight-lane context keeps a second reservation, loads at `decode_lanes 8
    -> 1, max_context 917504 -> 327680` with 2 host retained slots (pool 334,528 tokens) and decodes.
  - The 27B with 2 and with 1 lane named, at 262,144 tokens inside a budget 256 MB above its plan, loads with
    `adapted` empty and decodes.
  - The spec's 17,179,869,184-byte budget cannot hold today's NVFP4 27B: its weights line alone is 17,234,723,840
    bytes. The 16 GB body is #337/#338.

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
