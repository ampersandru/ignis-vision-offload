# A pinned hot set measured on the 5090: the host pool gives back the pinned bytes exactly and greedy text is unchanged; 1 GiB costs nothing measurable, 2 GiB about 1% of three-lane decode

- Kind: experiment
- Status: current
- Observed: 2026-10-18
- Last verified: 2026-10-18
- Scope: core / Flash-Next expert residency with a pinned hot set (`vram.pinned_expert_bytes`): the host plan's expert pool, the process's RAM, the expert cache's decode hit rate, decode tok/s and load time on the rtx5090 profile
- Related: [GitHub #329](https://github.com/gpillon/ignis/issues/329) (item 12),
  [ADR 0055](../adr/0055-a-pinned-hot-set-is-opt-in.md),
  [A partially exclusive expert pool on the 5090](2026-10-18-partial-exclusive-expert-pool.md),
  [spec flash-next/03](../specs/flash-next/03-expert-residency.md)
- Superseded by: none

## Question

The replay study predicted that pinning the calibration-hottest 1-2 GB of
experts in VRAM, outside the LRU and with no host copy, gives that RAM back
for under 1% of decode. Built behind `--vram-pinned-expert-bytes` (default 0),
does a served load on the RTX 5090 do that?

## Evidence

**Setup.** Branch `expert-pin` (ignis 0.7.0 + the pin), release build, the
served artifact `qwen3_8_flash_next_trellis_a25-v2.ninfer`, the rtx5090
profile (589,824 tokens, 3 lanes, yarn:4), MTP off,
`--model-idle-unload 0 --model-loop-guard off`. The machine had ~42-44 GB
available, short of the default host plan, so every launch ran the same lean
host lines: `--reuse-kv-host-pool-bytes 0 --reuse-retained-host 0
--kv-disk-bytes 0 --ngram-hot-bytes 0 --model-host-margin-bytes 1536M`. The
VRAM plan gave the expert cache 18,115,480,912 B (the study replayed 18.44 GB),
expected hit rate 0.596.

**Launches.** The baseline binary (main at `0a810c0b`) once; the pin binary at
`0`, `1G` and `2G` (GiB) three times each, in three rounds (0/1G/2G, 2G/1G/0,
0/1G/2G). Each launch: load, one unmeasured three-lane round of the prompts,
then one lane and three lanes, three reps each: greedy, `ignore_eos`,
thinking off, 320 tokens, three distinct prompts. Decode tok/s per request is
`(tokens − 1) / (last − first content chunk)`; the hit rate and the demand
stall are the `/metrics` decode counters around each phase. Reps 1-2 are
averaged (rep 0 warms the phase). The first 1 GiB launch was disturbed: its
reps ran 98-102 tok/s at one lane and 118-130 at three with unchanged hit
rates, and the later launches ran at the others' speed; it is left out of
the speed figures. Script and raw data: `.scratch/expert-pin/` in the
`expert-pin` worktree (`measure.py`, one JSON and one log per launch,
`summary.txt`).

**What was pinned.**

| field | experts | projections | pinned bytes | of which K = 4 | load staging |
|---|---:|---:|---:|---:|---:|
| `1G` | 443 | 886 | 1,075,085,312 | 812 projections | 282,034,176 |
| `2G` | 912 | 1,824 | 2,149,367,808 | 1,467 projections | 496,091,136 |

The hottest experts are mostly K = 4: at 2 GiB the gate/up K4 LRU keeps 303
of its 1,075 slots and the down K4 449 of 1,144.

**RAM.** The host plan's `expert_pool` line and the process's working set
after the load:

| field | `expert_pool` | working set | less than at 0 |
|---|---:|---:|---:|
| baseline, `0` | 37,795,446,784 | 38.162 GB | — |
| `1G` | 36,720,361,472 | 37.087 GB | 1.075 GB |
| `2G` | 35,646,078,976 | 36.013 GB | 2.149 GB |

The same three launches of each agree to 1 MB. The pinned records stream
through `load_staging` while the experts load (the plan's residency-phase
peak is the pool plus it).

**Decode.**

| | 1 lane tok/s | hit | stall ms/round | 3 lanes tok/s | hit | stall ms/round |
|---|---:|---:|---:|---:|---:|---:|
| baseline (main) | 105.65 | 96.285% | 2.152 | 140.07 | 95.113% | 8.165 |
| `0` | 105.90 | 96.285% | 2.166 | 139.76 | 95.106% | 8.197 |
| `1G` | 106.17 | 96.274% | 2.168 | 140.19 | 95.075% | 8.254 |
| `2G` | 105.76 | 96.218% | 2.213 | 138.19 | 94.967% | 8.474 |

Per launch at three lanes: `0` 139.90 / 139.70 / 139.68, `1G` 141.03 /
139.35, `2G` 138.15 / 138.15 / 138.28.

**Text.** Every launch's 12 completions (320 tokens each) equal the baseline
binary's, at `0`, `1G` and `2G`.

**Load.** The residency phase took 17.5-18.3 s at `0`, 19.9-20.9 s at `1G` and
19.9-20.4 s at `2G`, with the page cache warm.

## Finding

**Observed.** The pin gives back exactly its bytes: the plan's expert pool
and the process's working set drop by 1,075,085,312 B at `1G` and
2,149,367,808 B at `2G`. Greedy text is unchanged, so the pinned slots hold the
right records. With the field at 0 the binary decodes as main does (within
0.3%, hit rate equal to 0.007 points).

**Observed.** `1G` costs nothing measurable: tok/s within ±0.3% of `0` at one
and three lanes, the hit rate 0.01-0.03 points lower. `2G` costs 0.1% at one
lane and 1.1% at three (138.2 against 139.8 tok/s, every launch below every
`0` launch), the hit rate 0.07 and 0.14 points lower, and the three-lane
demand stall 0.28 ms per round higher (8.47 against 8.20).

**Compared with the study.** The study replayed 2.0 GB at +0.6-0.9% of a
three-lane round and 0.2-0.3% of a one-lane one, and 1.0 GB at 0.2% and 0.
At 2 GiB (2.15 GB) the card measures 1.1% and 0.1%: the same order, the
three-lane cost at the top of the study's bracket. Its hit-rate deltas at
2 GB (−0.04 and −0.08 points) are close to these (−0.07 and −0.14).

## Implications

- `--vram-pinned-expert-bytes 1G` is RAM for free on this host; `2G` costs about
  1% of three-lane decode. Past 2 GiB the K = 4 LRU pools are what the pin
  eats (72% of the gate/up K4 slots at 2 GiB), where the study saw the cost
  climb.
- Loading with a pin takes ~2 s longer, from reading the pinned records apart
  and the pool in runs around them.
- The default stays 0 (ADR 0055): the owner decides when the RAM is worth it.

## Limits and unknowns

- One prompt set, three prompts of 320 tokens, repeated: decode only, no
  prefill and no long session. Three launches per setting; the first `1G`
  launch was set aside as disturbed.
- Lean host lines (no n-gram hot rows, no KV-RAM arena, no retained host slots,
  a 1.5 GiB margin), the same on every launch, because the machine had too
  little RAM for the default plan that day. N-gram rows came from the page
  cache or the drive.
- The hit rates are this workload's, higher than the replay's teacher-forced
  chunks; the deltas are what compares.
- Only 1 and 2 GiB were measured, at one cache size.

## Follow-ups

- None needed for the feature. If a larger pin is ever wanted, the study's
  untried options (a split that re-plans the remaining pools, ranking by
  traffic per byte) are where to start.
