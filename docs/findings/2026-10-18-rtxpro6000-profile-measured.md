# On an RTX PRO 6000 the derived `rtxpro6000` profile holds: Flash-Next's MTP pays up to 1.85x, and the n-gram rows read off a cloud volume cost a fifth of decode

- Kind: experiment
- Status: current
- Observed: 2026-10-18
- Last verified: 2026-10-18
- Scope: serving / the built-in `rtxpro6000` profile (`crates/server/src/config/profile.rs`) on both families: Flash-Next MTP speculation, decode lanes, n-gram hot rows; the 27B's DFlash2 draft count; retained slots on the device; the host plan's available-RAM input in a container (`crates/core/src/residency/host_memory.rs`)
- Related: [MTP on the 5090](2026-10-07-flash-next-mtp-speculation-is-pcie-bound.md), [Flash-Next prefill chunk and the decode share](2026-10-07-flash-next-prefill-chunk-and-decode-share.md), https://github.com/gpillon/ignis/issues/327, `.scratch/rtxpro6000-profile/`
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
through `POST /v1/models/switch`, so one pod ran the whole campaign.

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

**Flash-Next, MTP 3 against speculation off** (aggregate tok/s, greedy, the
1 GiB n-gram default):

| lanes | MTP code | off code | x | MTP prose | off prose | x |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 206 | 111 | 1.85 | 154 | 107 | 1.44 |
| 2 | 277 | 182 | 1.53 | 222 | 178 | 1.24 |
| 4 | 317 | 268 | 1.19 | 292 | 260 | 1.12 |
| 8 | 348 | 357 | 0.97 | 333 | 340 | 0.98 |

**MTP sampled** (one lane, the whole n-gram table):
- code: 244 and 254 tok/s (two runs) against 130 off;
- prose: 167 and 178 against 131 off.

Acceptance falls from 0.80 (greedy code) to 0.75-0.77 sampled.

**MTP draft count** (one lane, the whole table, greedy):

| drafts | code | prose | acceptance code / prose |
|---:|---:|---:|---|
| 2 | 231 | 200 | 0.85 / 0.64 |
| 3 | 257 | 191 | 0.80 / 0.53 |
| 4 | 267 | 177 | 0.71 / 0.43 |
| 5 | 270 | 161 | 0.67 / 0.34 |

At two lanes 2 and 3 drafts tie on code (349, 350 tok/s).

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

**The 27B, DFlash2 7 against off and against 4 drafts** (aggregate tok/s, greedy):

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

**The host plan in a container.** The plan read 126.7 GB available
(`/proc/meminfo`'s `MemAvailable`, the whole node's) inside a 112 GiB
(120.3 GB) container limit. The container's own cgroup, read in the pod
mid-campaign, was:
- `memory.max` 120,259,084,288 B, `memory.current` 112,050,900,992 B;
- `anon` 30.2 GB (the whole n-gram table among it);
- `shmem` 41.1 GB: the CUDA pinned pools, counted in `file` but on no file
  LRU;
- `active_file` + `inactive_file` 40.4 GB of page cache, mostly the artifact.

## Finding

- **The derived numbers hold, all of them.** Flash-Next's MTP with 3 drafts
  is the best draft count over code and prose together. It pays 1.85x / 1.44x
  at one lane and still 1.12-1.19x at four. At eight it drafts nothing (row
  budget 8) and costs 2-3%. 8 lanes give the most aggregate. The 27B's
  DFlash2 at 7 drafts is 2.3-3.3x speculation off and beats 4 drafts
  everywhere but prose at eight requests (-6%). 1,048,576 tokens of context
  leave both resident KV pools at 8-11 million tokens.
- **MTP's sign flips from the 5090 because no expert crosses PCIe.** On the
  5090 a verify round copies more experts and loses at two lanes. Here the
  cache holds them all (a few hundred misses per cell), and the round is
  compute.
- **The n-gram table is the 96 GB card's real lever, and it is a host
  setting.** With the 1 GiB default every decode round and every prefill
  chunk wait on reads of the artifact's table range. On a cloud block volume
  (16,000 IOPS, ms latency) that costs a fifth of decode (the whole table
  runs 1.24x) and more than half the prefill (2.3x).
  - The whole table in RAM takes them all off the disk.
  - That is `--ngram-hot-bytes auto` on a host with ~75 GB available, and
    every machine built around a 96 GB card has it.
  - The gap measured here is EBS's. It is smaller on a local NVMe, but the
    whole table is never slower: it costs RAM and ~28 s of load.
- **Device-resident retained slots are not a profile number.** They cut a
  64K Flash-Next restore by ~35 ms and leave the 27B within noise. The
  second turn's ~0.15-0.35 s is mostly elsewhere: rendering and prefilling
  the conversation's new tail.
- **Inside a container `MemAvailable` overstates what a load may pin.** It
  is the node's figure, so a container whose limit is below it can plan past
  its limit. A plan from `auto` (or any large `--ngram-hot-bytes`) is then
  OOM-killed instead of refused. Capping it by every cgroup v2 level's
  `memory.max` less the non-page-cache charge would have read ~48.6 GB here
  instead of 126.7 GB.

## Implications

- `rtxpro6000` keeps every number and adds `ngram.hot_bytes: auto`
  (Flash-Next-only; the 27B passes it over). This is safe only with the
  cgroup-capped host plan, which ships in the same change.
- The host plan's available RAM on Linux is `MemAvailable` capped by every
  cgroup v2 limit above the process, each less its charge net of page cache
  (`active_file` + `inactive_file`; pinned and `shmem` pages count as taken).
  Without this, anvil-serving's recipe shape (52 GiB limit on a ~93 GB host)
  would plan the whole table and be killed.
- A 64 GiB host (g7e.2xlarge) holds Flash-Next only with the 1 GiB default,
  so 19% slower decode (the whole table is 1.24x) and 2.3x the TTFT on block
  storage. It is also
  unverified against the pinned ceiling. A 128 GiB host holds the whole
  table.
- Device-resident n-gram rows would give the whole-table speed with no host
  RAM: the table (28.8 GB) fits in the 51 GB the resident KV pool now takes,
  leaving ~5M tokens of KV.

## Limits and unknowns

- One card, the Server Edition, under Linux. The Workstation and Max-Q
  editions (and Windows) are not measured; the Max-Q's 300 W cap may lower
  every rate.
- The whole-table gain is measured against a cloud volume, not a local NVMe.
- Greedy everywhere except one sampled MTP pair; synthetic prompts (8 code,
  8 prose), 512 tokens, loop guard off.
- In phase 5 the first sampled spec-off pair ran with MTP on: its `PATCH`
  was refused, because a live `draft_tokens: 3` has nothing to configure
  under `backend: off`. It is discarded and redone (`fn-off2-*`).
- An eight-lane MTP cell reports a nonzero acceptance: lanes finish
  staggered and the tail drafts. The window metric excludes the tail.
- The 38 GB pinned pool was not tried on a 64 GiB host.
- Prefill chunk width and the decode share were not swept here.

## Follow-ups

- Device-resident n-gram rows on a card that holds every expert.
- `ignis-server model download` in the published image cannot run on a
  machine without an NVIDIA driver: the CUDA build links `libcuda.so.1`
  dynamically and dies before `main`, against the README's "needs no GPU".
- Once an image carries the cgroup-capped host plan, check it on a live
  container: `ignis.runtime.flash_next_host_plan`'s `available_bytes` must
  read at most the container's `memory.max` (120,259,084,288 here) where it
  read the node's 126.7 GB. The parsers are tested, the file layout was read
  in the pod, but the walk itself has not run on Linux yet.
- The Kubernetes/OpenShift deployment used here (scale-to-zero ksvc,
  auth-gate, EBS volume, curl download Job pinned to the catalog) lives
  outside this repository (`gps-inference/infra/rosa-serverless-gpu-ignis`).
