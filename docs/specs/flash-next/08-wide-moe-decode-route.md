# 08 - A native wide MoE decode route: Flash-Next verify rounds past 8 rows

GitHub: #331 (origin: #329 item 5).

**Status: proposed (2026-10-18), awaiting the owner's triage.** Phase 0 is a
measurement with a pre-registered go / no-go; nothing past it is built
before the owner reads its result.

ADRs:
- 0043 / 0051: every op this spec touches is OURS (the MoE ops, the FP8 and
  BF16 linears, the hyper-connection mix, the Flash-Next program); nothing
  vendored changes (0010, 0037).
- 0044: experts in trellis, a K per expert.
- 0030: plan lines; a load that names no budget plans as today.
- 0056 (accepted 2026-10-18; on branch `verify-rows` until it merges): a
  verify budget past 8 rows is held numerically equivalent to speculation
  off, not to its text. This spec's new routes are held to the same rule.

Sources:
- `.scratch/rows-card/results.jsonl` (render with `table.py` in the same
  directory) and the server log `.scratch/rows-card/server-full.jsonl`: the
  RTX PRO 6000 run of the verify-rows binary (`ignis-server` at `33c15274`)
  that motivates this spec;
- `docs/findings/2026-10-18-flash-next-verify-rows-past-8.md` and ADR 0056
  (branch `verify-rows`): what switches at row 9, the drift, the 5090 cost;
- `docs/findings/2026-10-18-rtxpro6000-profile-measured.md`: the card, MTP
  ratios, the 27B's DFlash2 at 8 requests;
- `docs/findings/2026-10-05-moe-decode-is-structure-bound.md` and
  `2026-10-08-flash-next-routed-experts-staged.md`: the MoE decode kernel;
- the kernel sources named under Implementation Decisions;
- the literature at the end, each number with its URL and date.

How to read the numbers: **measured** (ours, with the file),
**literature** (URL), **estimate** (method stated beside it).

## Problem Statement

A verify round carries lanes x (drafts + 1) rows. At the default budget of
8 rows, five or more lanes draft nothing. #329 item 5 raised the budget to
32 (`--spec-draft-rows`, branch `verify-rows`). On an RTX PRO 6000 that
holds every expert in VRAM, so decode is compute and VRAM-bandwidth bound,
a wider budget loses:

**Measured** (aggregate tok/s, MTP 3, greedy, 512 tokens a lane, 3 reps,
median; `.scratch/rows-card/results.jsonl`):

| cell | off | budget 8 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|---:|
| 8 lanes, code | 427 | 415 (no drafts) | 232 | 300 | 350 |
| 8 lanes, prose | 424 | 414 (no drafts) | 213 | 247 | 262 |
| 4 lanes, code | 332 | 404 | 209 | - | - |
| 4 lanes, prose | 329 | 369 | 156 | - | - |
| 4 simulated agents | 149.5 | 185.3 | 158.3 | - | 145.6 |
| 8 simulated agents | 213.8 | 224.2 | 147.2 | - | 174.0 |

The default's 8-row budget is the best cell everywhere it drafts. The wider
budgets accept well (at 8 lanes, code: 1.95, 2.80 and 3.55 tokens a lane a
round at 16, 24 and 32 rows; prose 1.80, 2.30, 2.64). They lose on round
time.

**The round time** (**estimate** from measured data: a batch round's time is
lanes x tokens a lane a round / aggregate tok/s, the tokens a lane a round
from the `ignis_speculative_*` deltas; the per-request `request done` events
of `server-full.jsonl` give the same within 3%):

| rows a round | shape (lanes x columns) | round |
|---:|---|---:|
| 1 | 1 x 1, off | 7.8 ms |
| 4 | 1 x 4 / 4 x 1 off | 13.1 / 12.1 ms |
| 8 | 4 x 2 / 8 x 1 off | 19.2 / 18.8 ms |
| 16 | 8 x 2 and 4 x 4 | 65.6-67.5 ms |
| 24 | 8 x 3 | 74.5-75.0 ms |
| 32 | 8 x 4 | 80.8-81.1 ms |

Up to 8 rows a row costs ~1.6 ms. From 16 rows on, ~0.9 ms. **Between 8 and
16 the round jumps by ~48 ms**: a cliff at row 9, not a slope.

**What switches at row 9** (the verify-rows finding, and the code):
- the routed experts: `fn_moe_experts_decode`
  (`kernel/src/flash_next/program.cu`) runs kern's decode route in launches
  of `IGNIS_MOE_DECODE_MAX_TOKENS` = 8 rows, each launch reading its own
  experts' weights;
- the FP8 linears: GEMV up to `IGNIS_FP8_GEMV_MAX_TOKENS` = 8, then the MMA
  route (`kernel/src/fp8_linear.cu`), 128 x 128 tiles, so a 2560-wide
  output runs on 20 CTAs of a 188-SM card, each looping over all of K;
- the BF16 linears (router, shared-expert gate): GEMV up to `kGemvTokens` = 8,
  then WMMA (`kernel/src/flash_next/linear.cu`);
- the hyper-connection mix (97 a round): its decode route up to `kDecodeRows`
  = 8, then the prefill route through `fn_linear` (`kernel/src/flash_next/hc.cu`);
- the GDN and QSA input projections lose their grouped-GEMV fusion
  (`rows <= IGNIS_FP8_GEMV_MAX_TOKENS`, `gdn.cu`, `qsa.cu`): one launch
  becomes three or four;
- the shared expert: GEMV to MMA, as the FP8 linears.

**The MoE is the smaller part.** **Estimate** (method):
- From 16 rows on, each further 8 rows add 6.2-7.5 ms. That includes one more
  MoE launch of 8 plus every other op's extra rows. So **one MoE launch of 8
  rows costs at most ~7.5 ms a round**; the register kernel's bench (25.4 us a
  layer at 1 token, ~1.4 us a layer per further distinct expert, RTX 5090)
  puts it nearer 4-5 ms.
- What a once-per-round weight read saves is the experts two launches share.
  Distinct experts per layer per round (**measured**, the expert-cache hit
  deltas / rounds / 48 layers / 2 projections, prefill included, so an upper
  value): 8 rows 56 (4 x 2) to 62 (8 x 1); 16 rows 94 (8 x 2, code); 32 rows
  137 (8 x 4, code). Two launches of 4 x 2 read ~112 at 16 rows: 19% more
  expert bytes, ~1.3 GB a round, **~0.7 ms at 1.79 TB/s**. At 32 rows four
  launches of 2 x 4 read ~190 (estimate): ~4 GB, ~2.2 ms.
- So of the 48 ms cliff, the MoE launches explain **at most ~7.5 ms**; **~40 ms
  is the dense routes switching** (the list above). That part is not
  attributed per op: no profile of a wide round exists (Phase 0 takes it).

**What the MoE route alone would buy.** **Estimate**: a native call costs the
fraction of the launches-of-8 that vLLM's fused MoE measures on the same
512-expert top-10 shape (literature [L9]: one 16-token call 0.80x two
8-token calls, one 32-token call 0.55x four), with one launch of 8 at
4.5-7.5 ms:

| 8 lanes | round now | round, MoE only | code | prose |
|---|---:|---:|---:|---:|
| budget 16 | 67.0 ms | 64.0-65.2 ms | 239-244 (0.56-0.57x off) | 221-225 (0.52-0.53x) |
| budget 24 | 74.8 ms | 66.6-69.9 ms | 321-336 (0.75-0.79x) | 263-276 (0.62-0.65x) |
| budget 32 | 81.0 ms | 67.4-72.9 ms | 390-421 (0.91-0.99x) | 290-313 (0.68-0.74x) |

Even if every extra launch were free (7.5 ms each), budget 32 reaches 485
tok/s on code (1.14x off) and 0.85x on prose. **A native wide MoE route alone does
not make a budget past 8 pay.**

**What a wide route for every op would buy.** **Estimate** (the ceiling): an
ideal wide round costs the 8-row round (19.0 ms) plus a per-row slope s for
each row past 8. Three bands:
- best, s = 0.875 ms/row: the measured marginal of today's routes past 16 rows
  (tensor-core linears plus launches of 8), the cliff removed;
- mid, s = 1.57 ms/row: the decode routes' own slope from 1 to 8 rows;
- worst, s = 2.04 ms/row: 1.3x that (a wide GEMV losing occupancy).

| 8 lanes | round best / mid / worst | code tok/s (x off 427) | prose tok/s (x off 424) |
|---|---|---|---|
| budget 16 | 26.0 / 31.6 / 35.3 ms | 600 / 494 / 442 (1.41 / 1.16 / 1.04x) | 554 / 456 / 408 (1.31 / 1.08 / 0.96x) |
| budget 24 | 33.0 / 44.1 / 51.6 ms | 679 / 508 / 434 (1.59 / 1.19 / 1.02x) | 558 / 417 / 356 (1.32 / 0.98 / 0.84x) |
| budget 32 | 40.0 / 56.7 / 68.0 ms | 710 / 501 / 418 (1.66 / 1.17 / 0.98x) | 528 / 373 / 311 (1.25 / 0.88 / 0.73x) |

At 4 lanes and budget 16 (4 x 4: 3.50 tokens a lane a round on code, 2.55 on
prose): code 538 / 444 / 396 against the default's 404 (1.33 / 1.10 /
0.98x), prose 392 / 323 / 289 against 369 (1.06 / 0.88 / 0.78x).

The bytes are not the bound. A round reads ~4.4 GB of trunk non-expert
weights, ~1 GB of the MTP head's and 4.5 GB (8 rows) to 9.9 GB (32 rows) of
distinct experts: 5.5-8.6 ms at 1.79 TB/s, against 19-81 ms measured. The
8-row round already runs at ~29% of the DRAM roofline. What the round pays
is per-launch latency and per-row compute, which is why the ceiling is
modelled from the measured slopes, not from bytes.

**So:** the gain is real on code at 8 lanes (mid band 1.16-1.19x off at
budgets 16-32) and marginal on prose (mid band 1.08x at 16, a loss past it).
It needs the dense routes and the MoE together. The literature agrees on the
direction: speculative gains shrink with concurrency, and MoE targets lose
part of them to the extra experts a verify reads (Literature).

## Solution

A **wide decode route**: every op a decode-phase call runs past 8 rows gets a
route built for 9-32 rows, so a 16-32-row verify round costs what its rows
cost, not a cliff. The MoE piece is the one this spec is named for: one
launch for the whole round, each selected expert's projection decoded once
and applied to every row that selected it. The dense pieces are the same
idea for the linears and the mix: the weights read once a round, enough CTAs
to fill the card at small M.

Phase 0 measures first. If the per-op profile does not leave room for the
mid band, the work stops there with a finding (the owner's rule: a
pre-registered no-go ends shipping, not the study).

The default (8 rows) and every round of up to 8 rows keep today's kernels and
today's bits. The new routes are taken only by decode-phase calls of 9-32
rows. A prefill chunk of 9-32 tokens keeps its tensor-core route.

## User Stories

1. As the owner serving Flash-Next on a 96 GB card to several agents at
   once, I want MTP to keep paying at 5-8 lanes, so that the card's decode
   rate grows with the agents instead of flattening at ~425 tok/s.
2. As the owner, I want the work stopped early, with a finding, if a profile
   shows the gain is not there.
3. As a maintainer, I want the 1-8-row kernels and their bits untouched, so
   that every existing test, graph and the 5090's served behaviour stay as
   they are.
4. As a maintainer, I want each wide op held to its existing reference bound
   and the end-to-end rule of ADR 0056, re-derived for the new tilings on
   both cards.

## Implementation Decisions

### Phase 0: the profile (no kernel code)

- Nsight Systems on the Linux card (the Windows hang is Windows-only): an
  8-lane round at budget 8 (8 rows, plain) and at budget 16 (8 x 2), resident
  experts, MTP 3, the rows-card prompts. Per op family: MoE routed, router
  and residency step, FP8 linears, BF16 linears and head, HC mixes, GDN, QSA
  and indexer, shared expert, MTP head, verify save/commit/accept. Each
  family's 16-row time against its 8-row time.
- `bench_moe` on the card: the decode route at 1-8 tokens, launches of 8 at
  16/24/32, and the prefill route (grouped, 64-row tiles) at 16/24/32, on the
  verify rounds' routing shapes (8 x 2, 4 x 4, 8 x 3, 8 x 4). This puts a
  measured number where this spec has the ~4.5-7 ms estimate.
- Optional, cheap: ADR 0056's option 1 as an experiment (a decode-phase flag
  that runs every op past 8 rows in launches of 8 on today's decode routes).
  If the 16-row round drops from ~67 ms towards ~38 ms (**estimate**: twice
  the 8-row round's ~14.5 ms of dense time, plus two MoE launches), the cliff
  is confirmed as the routes', in a few lines of code.
- The decision re-runs this spec's ceiling model with the measured
  per-family slopes (Acceptance 3).

### Phase 1: the MoE wide route

Today's register kernel (`kernel/src/moe_decode.cu`) already has the right
shape: a work unit is one distinct expert's tile, its trellis weights decoded
once into MMA B fragments, and every token of the call is an MMA row (tokens
that did not select the expert carry weight 0). At 8 tokens it fills only
rows 0-7 of each m16n8k16 tile (`row = g < tokens`). So:
- **16 rows**: the same units, the A operand's rows 8-15 used (`a[1]`,
  `a[3]`). Weight decode per unit unchanged; the MMA count unchanged.
- **32 rows**: two m-tiles per B fragment, each B fragment decoded once and
  used twice. The trellis decode (~5 integer instructions a weight) stays
  the dominant cost, so a unit costs ~1.1-1.3x (**estimate**).
- **The ≤ 8 constants** to lift, each with its test:
  - `IGNIS_MOE_DECODE_MAX_TOKENS` (`kernel/include/ignis_moe.h`) and
    `kDecodeMaxTokens` / `kDecodeMaxUnique` = 10 x tokens
    (`kernel/src/moe_workspace.cuh`): up to 320 distinct experts a call;
  - `Shared` (`moe_decode.cu`): `xh` fp32 [T][640] and `a` fp16 [T][648] are
    ~30 KB at 8 tokens, ~60 KB at 16 and ~120 KB at 32. 120 KB is past
    sm_120's ~99 KB per-block limit, and two CTAs per SM (which the 2026-10-05
    finding shows matter) need ≤ ~48 KB each. So the wide kernel drops the
    fp32 staging (the Hadamard written to fp16 directly) and stages 32 rows
    as two halves;
  - `unique_w` [unique][T] fp32 (41 KB at 32 x 320) becomes a per-assignment
    list; `unique_sel` stays a 32-bit token mask (exactly 32 tokens);
  - `build_unique` (`moe_decode_common.cuh`) assumes one thread per
    assignment (≤ 256 = blockDim); 320 assignments need a loop;
  - the workspace's decode regions `decode_gate_up` int64 [10D][D][1280] and
    `decode_h` f32 [10D][D][640] are quadratic in D: 131 MB at D = 32. They
    become per-assignment rows (10D rows: ~4 MB at 32).
- **A new route, not a widened one**: the wide kernel is a separate
  instantiation taken only past 8 rows; the 1-8-token launch keeps its code
  and bits. `fn_moe_experts_decode` (`program.cu`) calls it once instead of
  looping; `moe_decode_tokens()` and the plan lines follow; the MTP head's
  MoE (`kernel/src/flash_next/mtp.cu`) takes it too.
- The staged (tickets, ≤ 4 tokens) and cluster routes are untouched.

### Phase 2: the dense wide route

Taken only by decode-phase calls of 9-32 rows (a flag on the call, as ADR
0056 option 1 described; a prefill chunk of 9-32 tokens keeps its tensor-core
route):
- **FP8 linears** (`kernel/src/fp8_linear.cu`, `ignis_fp8_linear.h`): single,
  SwiGLU (shared expert) and grouped (the GDN / QSA fusions, at most 4
  segments). Either a GEMV with 16/32 accumulators per lane (register
  pressure; `IGNIS_FP8_GEMV_MAX_TOKENS` is also the grouped route's ceiling)
  or a skinny MMA: BM 16 or 32, a narrow N tile and split-K so that a
  2560-wide output fills the card. Phase 0's numbers pick one; the
  literature's precedent is the skinny MMA (llama.cpp moves from MMVQ to MMQ
  past ~5-8 columns as decode cost rises [L12]; TensorRT-LLM writes custom
  GEMMs for under 16 and under 30 tokens [L11]).
- **BF16 linears** (`kernel/src/flash_next/linear.cu`): the router, the
  shared-expert gate; `kGemvTokens` 8.
- **HC mix** (`kernel/src/flash_next/hc.cu`): `kDecodeRows` 8; the decode
  route's activation `act[kDecodeRows * kMaxDecodeRank]` fp32 in shared
  memory is 16 KB at 8 rows, 64 KB at 32; its fused inject paths
  (`fn_hc_mix_after`, `kMaxFusedRows`).
- **The fusions' gates** in `gdn.cu` and `qsa.cu`
  (`rows <= IGNIS_FP8_GEMV_MAX_TOKENS`) follow the grouped route's new ceiling.
- **The head** (`embed_head.cu`) over 16-32 rows: whatever Phase 0 shows.

### Numerics

ADR 0056 already holds a budget past 8 to numerical equivalence with
speculation off, not to its text: every logit within 64 BF16 ulps
(teacher-forced), a moved pick only at a near tie of 16 ulps, mean
log-probability within 0.02 nats, first-draft acceptance ≥ 0.5 and within 0.1
of the same prompts at 8 rows. The new routes change the summation order
again, so the four budget tests re-derive the tolerance on both cards, as the
ADR's consequences require. Up to 8 rows nothing changes: the near-tie rule
and every bit-exact test stay.

### Effort (estimate)

From the constants above and #306's per-step history (the staged MoE kernel
alone was ~870 lines and two benches; the decode-kernel restructurings of
2026-10-05/06 took several sessions and two were reverted):

| phase | work | agent sessions | card time |
|---|---|---:|---|
| 0 | profile, MoE microbench, optional launches-of-8 experiment, finding | 1 | ~2 h |
| 1 | wide MoE kernel (16, then 32), compact workspace, CTests, bench, plan lines | 2-4 | ~2 h |
| 2 | FP8 single / SwiGLU / grouped, BF16, HC route past 8, fusion gates, CTests | 3-5 | ~2 h |
| 3 | ADR 0056 tolerance on both cards, served cells, profile default, finding | 1-2 | ~0.5 day |

**Total: 7-12 agent sessions** (about 10-15 person-days), 2-3 calendar weeks
with reviews, about one day of RTX PRO 6000 time. **Risk: medium-high**: the
MoE decode kernel is structure-bound and has resisted restructuring before;
shared memory at 32 rows; the mid band itself straddles 1.0x on prose; and
the 64-ulp tolerance had a 1.1x margin on the 5090.

### Expected gain (estimate)

On the RTX PRO 6000, resident, 8 lanes, MTP 3, against speculation off
(427 / 424 tok/s):
- **MoE route alone:** 0.56-0.99x on code, 0.52-0.74x on prose at budgets
  16-32. Not worth shipping alone.
- **The whole wide route:** mid band **1.16-1.19x on code** (494-508 tok/s)
  and **0.88-1.08x on prose**, best case to 1.66x / 1.32x, worst case
  0.98-1.04x / 0.73-0.96x. Budget 16 is the safe one (prose ≥ 0.96x in every
  band); 24 is code's.
- **On the RTX 5090** (offloaded experts, PCIe-bound): no gain expected. A
  wider round copies more experts (1.16 -> 1.63 GB a round at 4 lanes,
  verify-rows finding); the routes do not change that.

## Testing Decisions

- Each new kernel route has a CTest at 9, 16, 24 and 32 rows on routing
  taken from verify rounds (lanes x columns, consecutive tokens sharing
  experts) and on uniform routing, against the existing reference: the MoE
  against ExLlamaV3's `reconstruct` and the fp64 bounds of `ignis_moe.h`,
  the linears against their fp64 reference and bounds. Determinism: run to
  run, from a graph replay, and whatever slot an expert occupies.
- The 1-8-row routes: their existing tests unchanged and green, and a pin
  that a call of 1-8 rows still runs the old route (the bits of today's
  `flash_next_moe_rows` 8-row launch).
- Plan lines: at budgets 0, 2, 6 and 8 byte for byte main's
  (`flash_next_bind`, as the verify-rows branch pins); at 16-32 recorded.
- ADR 0056's tests (`crates/core/tests/flash_next_speculative_gpu.rs`,
  `support/logit_drift.rs`) on the 5090 and the RTX PRO 6000.
- `cargo test` workspace-wide; the GPU tests on a free card (AGENTS.md).

## Acceptance

1. **Phase 0 profile.** A finding attributes the 16-row round's extra time
   (19 -> ~67 ms on the RTX PRO 6000) to op families, each family's 16-row
   time beside its 8-row time, and measures one MoE launch of 8 rows and the
   prefill route at 16/24/32 rows on the verify routing shapes.
2. **The MoE share stated.** The finding says whether the MoE launches are
   ≤ 10 ms of the 16-row round (this spec's estimate: ≤ 7.5) or more.
3. **GO / NO-GO (pre-registered).** The ceiling model is re-run with the
   measured per-family times, each family's wide cost taken as its 8-row
   cost x (rows / 8) at most. **GO** for Phases 1-2 if the model predicts, at
   8 lanes and budget 16, **≥ 1.10x speculation off on code and ≥ 1.0x on
   prose**. Otherwise **NO-GO**: the finding stays, nothing is built, the
   owner decides on the issue.
4. **Wide MoE kernel.** One launch serves 9-32 decode rows on the register
   route's units, each distinct expert's projection decoded once a call;
   within the fp64 bounds of `ignis_moe.h`; deterministic; the 1-8-token
   launch's bits unchanged. On the card, from the verify routing: one
   16-row call ≤ 0.85x the time of two 8-row launches, one 32-row call ≤ 0.65x
   of four (vLLM's fused MoE measures 0.80x and 0.55x on the same shape [L9]).
5. **Compact workspace.** The MoE decode regions grow linearly in the call's
   rows; plan lines at budget 8 are main's byte for byte.
6. **Dense wide routes.** The FP8 single / SwiGLU / grouped linears, the BF16
   linear and the HC mix each have a decode-phase route for 9-32 rows within
   their existing bounds; a prefill chunk of 9-32 tokens keeps its route; the
   GDN and QSA fusions run at up to 32 rows. Re-profiled on the card, each
   family's 16-row time is ≤ 1.5x its 8-row time.
7. **Numerics.** ADR 0056's four budget tests pass at budgets 16, 24 and 32
   on the 5090 and the RTX PRO 6000 with the tolerance re-derived from each
   card's drift column. A tolerance that has to exceed 64 ulps goes back to
   the owner.
8. **Speed on the RTX PRO 6000** (resident, MTP 3, the rows-card sweep: 8
   code and 8 prose prompts, 512 tokens, greedy, 3 reps, median):
   - the 16-row round ≤ 35 ms (now ~67);
   - 8 lanes at budget 16: **code ≥ 470 tok/s (1.10x the 427 off), prose ≥
     424 (1.0x)**;
   - 4 lanes at budget 16 ≥ the default's 404 (code) and 369 (prose);
   - 8 simulated agents ≥ 1.05x speculation off's 213.8 tok/s;
   - budgets 24 and 32 measured and reported beside them.
9. **Default.** The `rtxpro6000` profile takes the budget the cells of AC 8
   favour (the measured-better method is the default); the global default
   stays 8. The 5090's default cells (MTP at 1-3 lanes, budget 8) within
   ±3% of before.
10. **Docs.** A finding with an index row; spec 07 and the user README's
    `--spec-draft-rows` text say what a budget past 8 costs now.
11. `cargo test` passes workspace-wide; the Flash-Next GPU tests are green on
    a free card.

## Out of Scope

- **More than 8 decode lanes** (`IGNIS_DECODE_MAX_BATCH`): the same routes
  would serve plain rounds of 9-32 lanes, but the KV, the GDN replay record
  and the scheduler are a separate ticket.
- **The 5090's offloaded load**: PCIe-bound; a wider round copies more
  experts (verify-rows finding).
- **Bit-exactness past 8 rows**: ADR 0056 decided numerical equivalence.
- **Adaptive budget per round** (by acceptance or by lanes, like vLLM's
  per-batch-size draft count, TensorRT-LLM's `draft_len_schedule` or SGLang's
  adaptive steps [L5]): a follow-up once the wide budgets pay.
- **Tree drafts**, the staged (tickets) and cluster MoE routes, the 27B and
  anything vendored.

## Further Notes

### Open questions for the owner

1. If Phase 0 shows the dense routes are the cliff, should the ticket still
   be named for the MoE? (Proposed: keep the title, scope = every op past 8.)
2. One wide width (32) or two (16 and 32)? m16 MMA tiles make 16 the natural
   first width; 32 doubles the A operand and the shared memory.
3. Budget 16 or 24 for the profile? The estimate says 24 for code, 16 for
   prose; AC 8 measures both, and an adaptive budget is out of scope.
4. Is ≥ 1.0x on prose the right bar, given code at 1.10x?

### Literature (accessed 2026-10-18)

**(a) Speculative decoding vs batch size.** Every primary source shows the
gain falling with concurrency; MoE targets lose part of it to the experts the
extra rows activate.

| ref | engine, model, hardware | concurrency / batch | gain |
|---|---|---|---|
| L1 | TensorRT-LLM, DeepSeek-R1 MTP, 8x B200, 1K/2K | C1, MTP depth 1/2/3/4/5 | 1.78 / 2.25 / 2.28 / 2.21 / 2.15x (accept length 1.92-3.01) |
| L2 | SGLang (NextN 2 steps, 3 draft tokens), DeepSeek-V3, 8x MI300X, random 3200/800 | C 1/2/4/8/16/32/64 | 2.11 / 2.04 / 1.84 / 1.70 / 1.53 / 1.44 / 1.25x (end-to-end latency) |
| L3 | SGLang docs, DeepSeek-V3 MTP, H200 TP8 | bs 1 / 32 | up to 1.8x / 1.5x |
| L4 | SGLang, EAGLE-3, Llama-3.1-8B, H100 | bs 2/4/8/16/32/64 | 1.81 / 1.82 / 1.62 / 1.48 / 1.32 / 1.38x |
| L5 | SPEED-Bench, BS 32, 3 drafts | - | Qwen3-Next MTP 1.20x (SGLang); Qwen3-235B EAGLE3 1.06x; GPT-OSS-120B MTP 1.45x (one B200); Llama-3.3-70B EAGLE3 1.90x |
| L6 | Cascade (vLLM), five MoEs | bs 1 | drafts raise verification time 2-3x; slowdowns up to 1.5x |
| ours | ignis, Flash-Next MTP 3, RTX PRO 6000 (measured, above) | 1 / 4 / 8 lanes | 1.88x / 1.22x / 0.97x (code, budget 8) |
| ours | ignis, the 27B (dense) DFlash2 7, RTX PRO 6000 (profile finding) | 1 / 4 / 8 requests | 3.3x / 3.0x / 2.7x: a dense model's wide verify rounds still pay at 8 |

- [L1] https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog01_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html (first published 2025-05).
- [L2] https://rocm.blogs.amd.com/software-tools-optimization/mtp/README.html (2025-09-11).
- [L3] https://docs.sglang.io/basic_usage/deepseek_v3.html (undated; sglang 0.5.7).
- [L4] EAGLE-3, Table 3: https://arxiv.org/abs/2503.01840 (2025-03-03).
- [L5] SPEED-Bench, Table 1 and §8.4 ("at lower batch sizes ... longer drafts
  are preferred", larger batches prefer 1 draft): https://arxiv.org/abs/2604.09557
  (2026-02-10, v2 2026-05-28). Draft-count schedules by batch size: vLLM
  https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/ ;
  TensorRT-LLM `draft_len_schedule`, `max_concurrency`
  https://nvidia.github.io/TensorRT-LLM/latest/llm-api/reference/UserProvidedDecodingConfig.html
  (both undated docs, accessed 2026-10-18).
- [L6] https://arxiv.org/abs/2506.20675 (2025-06-17).
- Also: vLLM's DeepSeek MTP PR measured k = 1 at 1.34 / 1.64 / 1.28 / 1.11x
  for concurrency 1/2/4/8 (hardware not stated):
  https://github.com/vllm-project/vllm/pull/12755 (2025-02).

**(b) MoE models of this class on one 48-96 GB card.** Not comparable unless
the row says so: quantization, kernels, prompts and acceptance differ.

| ref | system | model, quant | card | C1 | C8 aggregate | comparable? |
|---|---|---|---|---:|---:|---|
| ours | ignis, spec off | Flash-Next, trellis 2.5-bit experts, FP8 dense, hq KV | RTX PRO 6000 Server | 129 | 427 | - |
| ours | ignis, MTP 3, budget 8 | same | same | 242 | 414 (no drafts) | - |
| L7 | vLLM, MTP 3 (community) | Flash-Next, NVFP4 mixed | RTX PRO 6000 Max-Q ("VRAM +6000" per the page) | 173 | 698 | closest: same model and card class; quant and prompts differ |
| L7 | vLLM, MTP 3 (community), TP2 | same | 2x RTX PRO 6000 Max-Q | 222 | 907 | two cards |
| L8 | vLLM / unstated (community) | Flash-Next, NVFP4 | RTX PRO 6000 | 81-88 off, 120-173 MTP | 341 (6 clients, off) | quant differs |
| L9 | vLLM (speculation not stated) | Qwen3-Next-80B-A3B, FP8 | H200, TP1 | 165 | 719 | same 512 / top-10 MoE shape; 2.7x the bandwidth |
| L10 | llama.cpp | gpt-oss-120b (117B / 5.1B active), MXFP4 | RTX PRO 6000 | 196 | - | same card, other model |

At 8 concurrent requests the one community measurement of the same model on
the same card class runs 1.6x ignis's best 8-lane cell, with MTP on. That is
the gap a wide route addresses; ignis's 1-lane MTP (242) is already above it.

- [L7] https://github.com/local-inference-lab/rtx6kpro/blob/master/models/qwen38-flash-next.md
  (undated; image tag 2026-09-22).
- [L8] https://forum.level1techs.com/t/qwen-3-8-flash-next-nvfp4-on-a-single-rtx-pro-6000-and-system-memory/254579
  (2026-08-28).
- [L9] https://github.com/vllm-project/vllm/pull/36559 (merged 2026-06-25):
  end-to-end C1 / C8 / C16 165 / 719 / 1,033 tok/s; fused-MoE kernel time a
  layer at B = 1 / 8 / 16 / 24 / 32: 30.0 / 62.3 / 99.9 / 118.6 / 136.6 us.
- [L10] https://github.com/ggml-org/llama.cpp/discussions/15396 (2025-08-18):
  tg128 196.31 t/s (Max-Q 170.62).

**(c) How engines run MoE decode at small batch.**
- [L11] TensorRT-LLM, DeepSeek-R1 at batch 1: "sparse experts as GEMMs":
  every token sent to each activated expert and masked, "because grouped
  GEMMs are memory-bound, the extra computations from redundant tokens have
  minimal impact"; custom dense GEMMs below 16 / 30 tokens. Same URL as L1.
  ignis's register kernel already works this way (every token an MMA row of
  every distinct expert's unit); it stops at 8 rows.
- [L12] llama.cpp: quantized decode uses MMVQ up to 8 columns (a build-time
  constant), MMQ past it, because MMVQ "re-decodes the weights for every
  output column; MMQ decodes them once"; per-quant switch points of 5-8 on
  the RTX 5090; the MoE path left on MMVQ because MMQ cost 38-62% there:
  https://github.com/ggml-org/llama.cpp/pull/26079 (2026-08-20). For MoE at
  batch > 1, tokens sorted by expert and batched MMQ: DeepSeek-V2-Lite (16B)
  Q4_0 on an RTX 4090 at microbatch 8 / 16 / 32: 255.5 -> 594.0, 257.6 ->
  857.6, 427.9 -> 1,493.7 t/s: https://github.com/ggml-org/llama.cpp/pull/13199
  (2025-04-30).
- [L13] vLLM fused MoE (Triton): BLOCK_SIZE_M 16 for M ≤ 32 by default,
  tuned configs keyed by batch size; a path that skips the align/sort step when
  tokens x top-k x 4 ≤ experts:
  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
  (main, accessed 2026-10-18); https://github.com/vllm-project/vllm/pull/29354 (2026-01-09).
- [L14] SGLang: DeepGEMM's masked grouped GEMM for decode (fixed shapes,
  graph-safe), contiguous grouped GEMM for prefill:
  https://lmsys.org/blog/2025-05-05-large-scale-ep/ (2025-05-05).

What the literature says for this spec: the engines that keep MTP paying at
4-8 requests run their verify rows through grouped or masked MoE kernels and
dense GEMMs sized for small M. A per-column re-decode (MMVQ-style) or a
tile sized for prefill is what they move away from. With 512 experts and top
10, a verify of 16-32 rows still touches 1-2 rows per expert, so the MoE
saving from a single launch is modest (vLLM: 0.80x at 16, 0.55x at 32); the
dense routes are where small-M design matters most here.
