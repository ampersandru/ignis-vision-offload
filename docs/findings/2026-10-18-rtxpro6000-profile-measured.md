# On an RTX PRO 6000 the derived `rtxpro6000` profile holds: Flash-Next's MTP pays up to 1.85x, and the n-gram rows read off a cloud volume cost a fifth of decode

- Kind: experiment
- Status: current
- Observed: 2026-10-18
- Last verified: 2026-10-18
- Scope: serving / the built-in `rtxpro6000` profile (`crates/server/src/config/profile.rs`) on both families: Flash-Next MTP speculation, decode lanes, n-gram hot rows; the 27B's DFlash2 draft count; retained slots on the device; the host plan's available-RAM input in a container (`crates/core/src/residency/host_memory.rs`)
- Related: [MTP on the 5090](2026-10-07-flash-next-mtp-speculation-is-pcie-bound.md), [Flash-Next prefill chunk and the decode share](2026-10-07-flash-next-prefill-chunk-and-decode-share.md), https://github.com/gpillon/ignis/issues/327, https://github.com/gpillon/ignis/issues/329, `.scratch/rtxpro6000-profile/`
- Superseded by: none

## Question

The `rtxpro6000` profile shipped derived, not measured (2026-10-17): both
families at 1,048,576 tokens, Flash-Next at 8 lanes with MTP drafting 3
tokens, the 27B with DFlash2 at 7. Which of those numbers does a real card
keep, and what does a 96 GB card want that the profile does not set?

## Evidence

**Setup.** One NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, 97,887
MiB), driver 595.91.07, on an AWS g7e.4xlarge (16 vCPU, 128 GiB RAM) in a
ROSA HCP cluster (OpenShift 4.20.40, RHCOS 9.6, us-east-2a). The engine is
`ghcr.io/gpillon/ignis:develop` at 8adb2e9d
(`sha256:0a7597167706eadb36b720f25653483d7300228e73773cad63d011fbb956bd95`),
`--profile rtxpro6000 --model-loop-guard off --model-idle-unload 0`, in a
pod with a 112 GiB memory limit. The artifacts are the built-in catalog's,
on an EBS gp3 volume provisioned to 16,000 IOPS and 1,000 MiB/s. Knobs
changed through `PATCH /v1/config`, which reloads in place, and the models
through `POST /v1/models/switch`, so one pod ran the whole campaign. The
simulated agents alone ran on another build and host (below).

**Client.** `sweep.py` (in `.scratch/rtxpro6000-profile/`, stdlib Python) in
a CPU pod of the same namespace, so no ingress sits in the timed path:
- N streamed chat completions fired together, `ignore_eos`, 512 tokens,
  `temperature: 0` unless marked sampled (then the model card's defaults);
- each lane a distinct prompt from 8 code tasks or 8 prose tasks;
- the aggregate is the tokens inside the window where all N decode, over
  the window;
- 3 reps, the median reported. A rep's prompts depend only on its index, so
  configurations are compared prompt for prompt.

Every result line, the phase scripts, the field list and the server log are
in `.scratch/rtxpro6000-profile/`.

**The load plans.**
- **Flash-Next, 8 lanes, MTP 3.** Budget 100.4 GB:
  - weights 6.1 GB;
  - expert cache 37.8 GB (expected hit rate 0.994);
  - program 2.4 GB, residency tables 1.6 GB;
  - KV pool, resident, 51.0 GB = **11,149,376 tokens**.
  - The host pool of every expert (one `cudaHostAlloc`, ~38 GB) pinned
    without error on the 128 GiB host. It was not tried on a 64 GiB one,
    where a sglang deployment on the same driver had bisected a 32-34 GiB
    single-allocation ceiling.
- **The 27B, DFlash2 7, vision on.** Budget 100.2 GB: weights 18.6 GB, KV
  pool, resident, 77.0 GB = 8,351,168 tokens. 16 retained host slots take
  3.7 GB of RAM; as device slots they take 3.2 GB of VRAM, and the pool
  drops to 7,955,200 tokens.
- **Loads.**
  - Flash-Next cold, from the volume: device weights 4.5 s, residency 51 s;
    the n-gram 1 GiB hot rows 54 s on the first load (the cache file written),
    0.85 s from it after.
  - The whole n-gram table took 27.8 s.
  - A reload with the page cache warm takes ~22 s.

**Expert traffic.** Every Flash-Next decode cell (3 reps) missed the expert
cache 564-3,719 times and moved 2.2-13.5 GB to the device, a hit rate of
99.93-99.98%; the residency stall summed to at most 0.063 s a cell
(`ignis_expert_*` deltas in each result line).

**Flash-Next, MTP 3 against speculation off** (aggregate tok/s, greedy, the
1 GiB n-gram default):

| lanes | MTP code | off code | x | MTP prose | off prose | x |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 206 | 111 | 1.85 | 154 | 107 | 1.44 |
| 2 | 277 | 182 | 1.53 | 222 | 178 | 1.24 |
| 4 | 317 | 268 | 1.19 | 292 | 260 | 1.12 |
| 8 | 348 | 357 | 0.97 | 333 | 340 | 0.98 |

At two and four lanes on code one rep of three had a request end with the
decode error below; the client counted that request in chunks, not tokens,
so the rep reads low and the median shown is the lower of the two clean
reps. Their mean is 284 (1.56x) at two lanes and 320 (1.19x) at four.

**MTP sampled** (one lane, the whole n-gram table):
- code: 244 and 254 tok/s (two runs) against 130 off;
- prose: 167 and 178 against 131 off.

Acceptance falls from 0.80 (greedy code) to 0.75-0.77 sampled.

**MTP draft count** (one lane, the whole table, greedy):

| drafts | code | prose | code + prose | acceptance code / prose |
|---:|---:|---:|---:|---|
| 2 | 231 | 200 | 431 | 0.85 / 0.64 |
| 3 | 257 | 191 | 448 | 0.80 / 0.53 |
| 4 | 267 | 177 | 444 | 0.71 / 0.43 |
| 5 | 270 | 161 | 431 | 0.67 / 0.34 |

The 4-draft code reps span 208-274 tok/s, so 3 and 4 tie over the two kinds.
At two lanes 2 drafts ran 349 tok/s on code and 3 drafts 350; 3 drafts'
third rep had a request end with the decode error below, and its two clean
reps average 360, so 3 drafts may be ahead there.

**Simulated agents** (`sweep.py --type agents`, phase 9). N agents, started
0.3 s apart, each 8 turns of one growing conversation:
- a turn generates exactly a seeded 96-768 tokens (greedy, `ignore_eos`);
- then the agent waits a seeded 0.5-4 s ("a tool") and appends a seeded
  800-16,000-character tool result;
- seed 7, so both configurations get the same schedule.

This ran on the device-only expert build of branch
`resident-experts-vram-only` on a g7e.2xlarge (8 vCPU, 64 GiB RAM), with the
whole n-gram table, after a one-lane warm-up per configuration. Its decode
rates are within 4% of this build's (finding
2026-10-18-flash-next-experts-device-only).

| agents | MTP 3 makespan | off makespan | MTP tokens | off tokens | MTP tok/s | off tok/s | x |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 55.45 s | 68.47 s | 10,071 | 10,240 | 181.6 | 149.6 | 1.21 |
| 8 | 84.85 s | 88.92 s | 19,135 | 19,136 | 225.5 | 215.2 | 1.05 |

The tok/s is the tokens over the makespan. 3 of the 96 MTP turns ended with
the decode error below, none with speculation off. The client counted those
in chunks (94, 42, 330: an error carries no usage); the MTP tokens above
take theirs from the server's `ignis.request.done` instead (255, 88, 767).
So MTP's runs did the same work less 169 tokens at 4 agents (a lane cut
mid-turn as collateral) and 1 at 8. The median turn took 2.72 s against
3.83 s at 4 agents, and 5.19 s against 5.27 s at 8.

**A decode error under MTP.** Requests across the campaign's logs, on both
builds, ended with `finish_reason: error` after one
`ignis.runtime.leaf_error`, context `decode`:
`ignis_program_decode: speculative_window 2 is not the window a round of 1
lanes runs on this load (3)` (also window 2 at two lanes, and window 1 at
two lanes). Cause, found and fixed on branch `fn-mtp-window` (4c0ce315, a
test, reproduced on the RTX 5090 before and gone after):
- a request's reservation is its prompt plus `max_tokens`, rounded up to a
  64-token page; an MTP-3 round writes 6 positions;
- in the request's last few tokens, when that sum lands 0-5 tokens past a
  page edge, the host cut the window to the room left
  (`flash_next_window_within`), and the leaf runs only the load's window or
  a plain step (`program_verify`), so it refused the round;
- the request then died 1-4 tokens before its cap (255 of 256, 511 of 512,
  767 of 768), and in a round of several lanes the others died with it,
  mid-generation (88 of 256, 491 of 512);
- the warm-up's second rep (prompt 61 + 256 = 317, capacity 320) hit it on
  every load. The fix runs a round without room for its window as a plain
  step.

Cells with an errored request: the two MTP code cells above, 3 drafts at
two lanes, the agents' 3 turns, every warm-up, two first turns of the
Flash-Next 64K host-slots cell (46 of 48 tokens; their second turns still
restored), and two cells of the device-only comparison. No cell of
speculation off or of the 27B has one.

**A recorded trace** (`ignis-bench replay` of a 4-agent load trace, phase 8,
`replay-g4-*`) does not compare the two: the replay sends each recorded
request body as one user message, the outputs diverge between the two runs,
and the requests past 8 concurrent ones failed `engine_full`.

**n-gram rows: the whole table in RAM** (`--ngram-hot-bytes 27G`, every row,
no step reads the volume; `ignis_ngram_reads_total` stays 0) **against the
1 GiB default** (MTP 3):

| cell | 1 GiB hot rows | whole table | x |
|---|---:|---:|---:|
| 1 lane code | 206 tok/s | 255 | 1.24 |
| 8 lanes code | 348 | 427 | 1.23 |
| 1 lane prose | 154 | 191 | 1.24 |
| TTFT 32K | 16.9 s | 7.4 s | 2.3 |
| TTFT 128K | 68.6 s | 31.0 s | 2.2 |

With 1 GiB of hot rows a decode round still reads ~10 rows a token from the
volume (123K reads for 12K tokens at 8 lanes), and a 32K prefill ~320K.

**The 27B, DFlash2 7 against off and against 4 drafts** (aggregate tok/s,
greedy; 4 drafts ran in three cells only):

| requests | DFlash2 7 code | DFlash2 4 code | off code | DFlash2 7 prose | DFlash2 4 prose | off prose |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 216 | 78 | 179 | — | 78 |
| 4 | 813 | — | 268 | 500 | — | 269 |
| 8 | 1,362 | 1,218 | 506 | 813 | 862 | 506 |

**Retained slots on the device** (`--reuse-retained-device 16
--reuse-retained-host 0`): a conversation's second-turn TTFT, its first turn
16K or 64K tokens long. Every sample restored (`cached_tokens` within 2 of
the first turn's), 3-4 samples each.

| model | first turn | host slots | device slots |
|---|---:|---:|---:|
| Flash-Next | 16K | 0.170-0.182 s | 0.171-0.178 s |
| Flash-Next | 64K | 0.331-0.368 s | 0.317-0.323 s |
| 27B | 16K | 0.151-0.163 s | 0.124-0.153 s |
| 27B | 64K | 0.330-0.335 s | 0.318-0.345 s |

**The host plan in a container.** Every load's plan read 126.7-127.7 GB
available (`/proc/meminfo`'s `MemAvailable`, the whole node's) inside a
112 GiB (120.3 GB) container limit. The largest plan made, the whole
table's, took 69.9 GB and fit. The container's own cgroup, read in the pod
mid-campaign, was:
- `memory.max` 120,259,084,288 B, `memory.current` 112,050,900,992 B;
- `anon` 30.2 GB (the whole n-gram table among it);
- `shmem` 41.1 GB: the CUDA pinned pools, counted in `file` but on no file
  LRU;
- `active_file` + `inactive_file` 40.4 GB of page cache, mostly the artifact.

## Finding

- **Every derived number the campaign measured holds.** The context length
  holds only by the load plan: 1,048,576 tokens leave both resident KV
  pools at 8-11 million tokens, but no request went past 128K.
  - Flash-Next's MTP with 3 drafts ties 4 over code and prose together and
    beats 2 and 5.
  - Every lane decoding at once, it pays 1.85x / 1.44x at one lane and still
    1.12-1.19x at four. At eight it drafts nothing (row budget 8) and costs
    2-3%. 8 lanes give the most aggregate.
  - With simulated agents, whose lanes wait on tools and prefills between
    turns, MTP pays 1.21x at four agents and 1.05x at eight.
  - The 27B's DFlash2 at 7 drafts is 1.6-3.3x speculation off (1.6x on
    prose at eight requests, 3.3x on code at one). It beats 4 drafts in the
    three cells both ran except prose at eight requests (-6%).
- **MTP as the profile sets it can kill a request.** A round whose window
  the host cut near the end of a request's reservation was refused by the
  leaf, and the request ended with an error a few tokens before its cap
  (above). The profile's MTP needs that fix (branch `fn-mtp-window`) before
  it serves; speculation off and the 27B are not affected.
- **MTP's sign flips from the 5090 because almost no expert crosses PCIe.**
  On the 5090 a verify round copies more experts and loses at two lanes.
  Here the cache misses 0.02-0.07% of lookups and stalls at most 0.06 s a
  cell, so the round is compute (inference from the stall counter: the round
  itself is not timed).
- **The n-gram table is the 96 GB card's real lever, and it is a host
  setting.** With the 1 GiB default every decode round and every prefill
  chunk wait on reads of the artifact's table range. On a cloud block volume
  (16,000 IOPS, ms latency) that costs a fifth of decode (the whole table
  runs 1.24x) and more than half the prefill (2.3x).
  - The whole table in RAM takes them all off the disk. It costs ~28.8 GB of
    RAM and ~28 s of load.
  - The gap measured here is EBS's; a local NVMe was not measured.
- **Device-resident retained slots are not a profile number.** At 64K
  Flash-Next the device slots' range sits 8-51 ms below the host slots'; the
  27B's ranges overlap. The rest of the second turn's ~0.15-0.35 s is not in
  the restore (inference: rendering and prefilling the conversation's new
  tail; not timed apart).
- **Inside a container `MemAvailable` overstates what a load may pin.** It
  is the node's figure (126.7 GB here against a 120.3 GB limit), so a
  container whose limit is below it can plan past its limit. A plan from
  `auto` (or any large `--ngram-hot-bytes`) would then be OOM-killed instead
  of refused (inference: no plan here came near the limit).
  - Capping it by every cgroup v2 level's `memory.max` less the
    non-page-cache charge reads at most 120.3 GB at a load.
  - Mid-campaign, with the engine's own anon and shmem charged, the same
    formula reads ~48.6 GB.

## Implications

- `rtxpro6000` keeps every number and adds `ngram.hot_bytes: auto`
  (Flash-Next-only; the 27B passes it over). This is safe only with the
  cgroup-capped host plan, which ships in the same change, and only on
  cgroup v2: a cgroup v1 container's limit is not read.
- The host plan's available RAM on Linux is `MemAvailable` capped by every
  cgroup v2 limit above the process, each less its charge net of page cache
  (`active_file` + `inactive_file`; pinned and `shmem` pages count as taken).
  Without this, anvil-serving's recipe shape (52 GiB limit on a ~93 GB host)
  would plan the whole table and be killed.
- On a 64 GiB host (g7e.2xlarge) the ~38 GB pinned pool of every expert
  leaves `auto` less than the whole table. It plans a partial table, above
  the 1 GiB default, whose speed was not measured. The pinned pool itself
  is unverified there against the single-allocation ceiling. A 128 GiB host
  holds the whole table.
- Device-resident n-gram rows would give the whole-table speed with no host
  RAM: the table (28.8 GB) fits in the 51 GB the resident KV pool now takes,
  leaving ~5M tokens of KV.

## Limits and unknowns

- One card, the Server Edition, under Linux. The Workstation and Max-Q
  editions (and Windows) are not measured; the Max-Q's 300 W cap may lower
  every rate.
- The whole-table gain is measured against a cloud volume, not a local NVMe.
- The MTP lane table ran with the 1 GiB hot-rows default, before the profile
  set `auto`; with the whole table, MTP was compared with speculation off
  only at one lane sampled and with the simulated agents.
- `auto`, the profile's value, ran only as an explicit `27G` (the whole
  table on the 128 GiB host). On a host where `auto` plans a partial table
  nothing was measured.
- No request went past 128K tokens, although the profile sets 1,048,576.
- Greedy everywhere except one sampled MTP pair; synthetic prompts (8 code,
  8 prose), 512 tokens, loop guard off.
- The simulated agents ran on another build (device-only experts) and host
  (64 GiB), once per configuration (no reps). 3 of MTP's turns ended with
  the decode error (above); their tokens are the server's count.
- No recorded agent trace was replayed faithfully (above).
- In phase 5 the first sampled spec-off pair ran with MTP on: its `PATCH`
  was refused, because a live `draft_tokens: 3` has nothing to configure
  under `backend: off`. It is discarded and redone (`fn-off2-*`).
- An eight-lane MTP cell reports a nonzero acceptance: lanes finish
  staggered and the tail drafts. The window metric excludes the tail.
- The 38 GB pinned pool was not tried on a 64 GiB host.
- Prefill chunk width and the decode share were not swept here.

## Follow-ups

Every follow-up of this campaign is noted in
https://github.com/gpillon/ignis/issues/329, among them:
- device-resident n-gram rows on a card that holds every expert;
- `ignis-server model download` in the published image cannot run on a
  machine without an NVIDIA driver: the CUDA build links `libcuda.so.1`
  dynamically and dies before `main`, against the README's "needs no GPU";
- once an image carries the cgroup-capped host plan, check it on a live
  container: `ignis.runtime.flash_next_host_plan`'s `available_bytes` must
  read at most the container's `memory.max` (120,259,084,288 here) where it
  read the node's 126.7 GB. The walk is tested on file trees and the file
  layout was read in the pod, but the walk has not run on a live cgroup;
- the MTP decode error (items 4 and 6 there are this one bug), fixed on
  branch `fn-mtp-window` (4c0ce315); the cells it touched can be re-run once
  merged.
