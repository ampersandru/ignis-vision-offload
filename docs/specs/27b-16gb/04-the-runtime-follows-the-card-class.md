# 04 - the runtime follows the card class: decode lanes per load, profile `auto`, adaptation, catalog card class

GitHub: #336 (master #334).

Decided 2026-10-21: the owner approved this spec and vetoed none of the *(proposed)* items, so every one is decided.

What makes one binary serve the 16 GB card class correctly:
- the 27B's decode lanes become a load setting;
- the profile is chosen from the card;
- its defaults adapt to the budget the load finds;
- the catalog knows which card class each model needs, so a switch is refused before anything is torn down.

ADRs: 0063, 0064; 0030 (the VRAM plan), 0045 (KV pool policy, fixed reservations), 0046 (one declaration per field),
0047 (catalog), 0049 (idle unload), spec model-switch/01.

## Problem Statement

- **The 27B's lane count is a compile-time 8.** It is fixed by:
  - `N_DECODE_LANES` (`crates/core/src/types.rs:33`);
  - `IGNIS_DECODE_MAX_BATCH` (`kernel/include/ignis_step.h:153`);
  - `slot_count: N_DECODE_LANES` (`crates/runtime/src/cuda_leaf.rs:128`);
  - the leaf, which refuses a nonzero `decode_lanes` on the 27B (`kernel/src/model.cu:1138-1142`).

  Lane state (233 MB a lane), sampling, decode graphs and the verify/drafter round buffers are all sized for 8. On a
  16 GB card two lanes leave room for 256K; eight do not.
- **`spec.decode_lanes` is `FLASH_NEXT_ONLY`** (`crates/server/src/config/schema.rs:532-536`). Flash-Next already
  passes it at load (`crates/core/src/model_load.rs:818-843` → `IgnisModelLoadOptions.decode_lanes`).
- **The profile is chosen by name only, default `rtx5090`** (`crates/server/src/config.rs:784-793`,
  `config/profile.rs:20`). NVML's total memory is read and thrown away (`crates/server/src/runtime.rs:693, 958`). A
  16 GB card would start on the 5090's profile.
- **A short budget is refused.** The free memory at start varies with what else holds the card. On a 16 GB card,
  256K with DFlash2 fits only when little else does.
- **Served ids are unique**, and clients name `qwen3.8-27b`. A switch checks VRAM only after teardown
  (`crates/server/src/model_switch.rs:1172-1232, 1614-1626`), so a model that does not fit costs a full reload of the
  current one.

## Solution

1. **One `decode_lanes` field for both families, 1..8.**
   - It is family-scoped: `qwen38` / `qwen38flashnext`.
   - The 27B leaf takes it at load and sizes, from the load's lanes and never from the constant:
     - lane state and the hq residual window;
     - sampling buffers;
     - decode and verify graphs (widths 1..lanes);
     - the verify and drafter round buffers;
     - the scheduler's slots.
   - `N_DECODE_LANES` / `IGNIS_DECODE_MAX_BATCH` remain the maximum.
   - The 27B's default stays 8 on the 32 GB profile; Flash-Next's stays 3.
2. **`--profile auto`, the new default.** It picks the built-in profile from NVML's total memory:
   - ~32 GB → `rtx5090`;
   - ~96 GB → `rtxpro6000`;
   - ~16 GB → `rtx50-16gb`.

   Any other total refuses the start with a message naming `--profile`. A named profile, or `none`, always wins. The
   chosen profile and the total it was chosen by go in the start's log.
3. **The `rtx50-16gb` profile.** For the 27B:
   - the 16 GB catalog entry as the default model;
   - `decode_lanes` 2;
   - `max_context` 262,144;
   - DFlash2 on;
   - `draft_head: auto`;
   - vision off.

   Headroom: *(proposed)* the 5090's 1200M until a real card says otherwise.
4. **Adaptation at load.** Only for fields whose source is the profile or the hardcoded default; the resolved
   configuration already records each field's source (`crates/server/src/config/source.rs:45-79`). When the VRAM plan
   does not fit, the load tries, in this order:
   1. `draft_head` per its `auto` rule;
   2. `max_context` 262,144 → 196,608 → 131,072 → 65,536, keeping the largest at which one full sequence and one more
      65,536-token reservation fit the pool;
   3. `decode_lanes` → 1.

   - DFlash2 is never switched off.
   - Below 65,536 the start is refused with the deficit and the adaptations tried.
   - Each step taken is named in the `ignis.runtime.vram_plan` event (`adapted: [...]`).
5. **`draft_head: auto`.** The field exists (`full` | `shortlist`, `schema.rs:540`); `auto` is added.
   - *(proposed)* `auto` resolves to `shortlist` when the plan needs the bytes it frees, or when the card class is
     16 GB with the pool short of two full reservations; else `full`.
   - The shortlist is the artifact's existing `text/draft_head`: 131,072 rows of Q4G64, 357 MB. It measured
     −7.4% tokens/round and −5.5% tok/s on the 5090 (finding 2026-09-24-upstream-quick-wins-ab).
   - On a lower-bandwidth card the trade differs. The rule is revisited when a real card measures it (phase 2).
6. **Catalog card class, alias, refusal before teardown** (ADR 0064):
   - each entry in `crates/server/src/download/catalog.yaml` gains `card_class` (the smallest it loads on: 16 / 32 /
     96 GB) and may list `aliases`;
   - an alias resolves only within one family, and only to an entry the card class can load;
   - `POST /v1/models/switch`, the implicit switch and `model download` refuse an entry above the card's class before
     teardown or fetch, with the class in the message;
   - `GET /v1/models` marks such entries `available: false` with the reason.

## Implementation Decisions

- *(proposed)* Card class boundaries by total memory: below 20 GiB → 16 GB; 28-40 GiB → 32 GB; 88-104 GiB → 96 GB.
  Anything else is unknown. The boundaries are config-visible constants of the profile table, not magic numbers in
  the planner.
- *(proposed)* The alias table is part of the catalog entry, not of the profile. A profile only picks the default
  model; the alias makes old clients work on any card.
- *(proposed)* The adaptation lives in the VRAM planner's caller (`runtime.rs`, the 27B's `plan_vram` site), not in
  the planner: the planner stays a pure function of its inputs, called again with the next step's values.
- The 8-lane path on the 5090 must produce byte-identical plans and outputs to today's when `decode_lanes` is 8
  (graph widths, buffer sizes, sampling).

## Testing Decisions

- Plan tests (CPU): every line that scales with lanes, at 1, 2 and 8 lanes, against the formulas in
  `PIANO-VRAM.md`, which were fitted to 39 measured plans. 8 lanes equals today's plan to the byte.
- Profile tests (CPU):
  - `auto` with injected totals: 16.0, 31.8, 95.5 GiB and two unknowns;
  - an explicit `--profile` beats `auto`.
- Adaptation tests (CPU, injected budgets):
  - each step in order;
  - explicit values never adapted;
  - the refusal message listing what was tried;
  - the log event's `adapted` list.
- Catalog/switch tests:
  - an entry above the class is refused before teardown;
  - the old model is still loaded and serving afterwards;
  - aliases within and across families;
  - `GET /v1/models` availability.
- GPU, on the 5090 with an explicit budget of 17,179,869,184 bytes minus a desktop allowance:
  - the 27B at 2 lanes, 1 lane and 8 lanes (on the full card) loads and decodes;
  - outputs at 8 lanes bit-identical to today's on the greedy coding prompts.

## Acceptance

1. `decode_lanes` applies to both families (1..8), family-scoped. The 27B loads with 1, 2 and 8 lanes. Its plan's
   lane-scaled lines match the formulas. At 8 lanes, plans and greedy outputs are identical to today's.
2. `--profile auto` is the default. It resolves `rtx5090` and `rtxpro6000` on their cards exactly as today, and
   `rtx50-16gb` on a ~16 GB total. An unknown total is refused with a message naming `--profile`.
3. `rtx50-16gb` exists with the values in Solution 3.
4. Adaptation follows the order above, touches only profile- or default-sourced fields, and logs each step. A budget
   below the last step is refused with the deficit and the steps tried.
5. `draft_head: auto` exists and resolves by the rule in Solution 5, logged.
6. Catalog entries carry `card_class` and optional `aliases`. A switch, implicit or explicit, or a download of an
   entry above the card class is refused before teardown or fetch. The loaded model keeps serving. `GET /v1/models`
   shows the entry as unavailable with the reason.
7. `cargo test` workspace-wide; GPU tests green on a free 5090.

## Out of Scope

- Flash-Next on 16 GB: phase 2. Its expert-cache floor becomes a config field there, with the profile's own value.
- Detecting a display: the owner rejected it. Adaptation answers what the free memory says.
- Per-card 16 GB profiles: phase 2.

## Further Notes

- The 16 GB catalog entry's file pins (repo, revision, sha256) come from spec 01. Until then a test fixture entry
  stands in.
- On the 5090, `auto` must not change anything a user sees. AC 2 is the regression test for that.

## As built: adaptation and `draft_head: auto` (2026-10-23)

ADR 0063's **As built** has the decisions and the measurements. In short:

- **Solution 5's proposed rule did not hold.** The shortlist is bound beside the full head: it costs ~357 MB and
  frees none, and it measured slower. So `auto` is `full` on every card class. It is a third spelling of the field
  (`ProposalHeadSetting`), resolved where the speculation is built, and the load logs `draft_head auto -> full`.
  The rule is revisited in phase 2, on a real 16 GB card.
- **Solution 4's ladder** runs from any profile value `v` down every multiple of 65,536 to 65,536. It keeps the
  largest context whose pool holds it and one more 65,536-token reservation (at one lane, the plan's floor alone).
  It runs only when the plan refuses the configured values below its minimum, so a start that fits changes nothing.
  - When no context meets that at the load's lanes, the lanes drop to one and the ladder runs again from `v`.
  - Explicit lanes keep the largest context the plan holds, which the step names.
  - Step 1 drops a profile's `shortlist` to `full`; it never fires on the built-in profiles.
  - A start that still does not fit is refused with the last attempt's deficit and every step tried.
- **Code:**
  - the ladder: `crates/server/src/adaptation.rs`;
  - its closure: `runtime::qwen27b_load`;
  - what may adapt: `Config::adaptable` → `EngineShape::adaptable`;
  - a shape at an attempt: `EngineShape::at`;
  - the log: the `adapted` field of `ignis.runtime.vram_plan`.
- **Not adapted here:** Flash-Next (phase 2), and the context `make` passes. `mk/config.mk`'s `MAX_CONTEXT ?=
  524288` reaches the 27B as `--model-max-context`, an explicit value; `MAX_CONTEXT=` leaves it to the profile.
- **AC 4 and AC 5:** CPU tests in `adaptation.rs`, `runtime.rs` and `config/tests.rs`; GPU test
  `crates/server/tests/adaptation_gpu.rs`, green on the 5090 (adapted context, lanes to one, 2 and 1 named lanes).
