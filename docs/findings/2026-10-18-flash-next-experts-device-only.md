# On a card that holds every expert, Flash-Next needs no host copy of them: 38 GB of pinned RAM back, the same decode and the same text

- Kind: experiment
- Status: current
- Observed: 2026-10-18
- Last verified: 2026-10-18
- Scope: expert residency on ADR 0045's resident load (`kernel/src/residency.cu` device-only mode, `crates/core/src/residency/{plan,load,device}.rs`, `crates/core/src/flash_next.rs` `build_residency`, the host plan in `crates/server/src/runtime.rs`)
- Related: [ADR 0045](../adr/0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md), [rtxpro6000 profile measured](2026-10-18-rtxpro6000-profile-measured.md), spec [flash-next/03](../specs/flash-next/03-expert-residency.md)
- Superseded by: none

## Question

Expert residency was built for the 5090: every expert projection in one
pinned, mapped host pool for the life of the load (~37.8 GB, one
`cudaHostAlloc`), and a VRAM cache holding a copy of some of them. On a ~96 GB
card ADR 0045's resident branch gives the cache the whole catalog's bytes.
Then:
- Does every projection actually live in VRAM?
- Does the host pool still earn its 38 GB?
- What does removing it cost?

## Evidence

**The split left projections out.** On the trellis-a25 artifact,
`plan_expert_cache` was run with a cache of exactly `ExpertCatalog::total_bytes`
(37,795,446,784 bytes), the resident branch's line (CPU probe, the real
catalog and calibration traffic):

| lanes | projections without a slot | cache bytes left unassigned | expected hit rate |
|---:|---:|---:|---:|
| 1 | 244 | 152 MB | 0.99984 |
| 8 | 1,944 (all K = 2) | 1.21 GB | 0.99436 |

Both hit rates are the ones the served plans logged: 0.99983 in
anvil-serving's report, 0.99436 in the rtxpro6000 campaign. The split models
one LRU of the calibration traffic: it holds no never-selected projection and
floors each class's share. The warm start admits selected experts only
(`warm_start_order`). So those projections came from the host pool on every
use: 455-4,245 misses per measured cell.

**The change.** A residency whose desc has `host_pool_bytes` 0 is
device-only:
- `create` refuses a class with fewer slots than projections and allocates no
  host pool;
- the loader admits every record from layer buffers it reads from the
  artifact (`ignis_residency_admit`, then `_admit_finish`, which refuses while
  any projection is missing). The reads are `stream_expert_layers`: four
  handles as the pool fill used, and a one-layer queue that holds the RAM to
  `stream_staging_bytes`, 4.7 GB on this artifact;
- a step that would still miss is refused with
  `IGNIS_RESIDENCY_STATUS_NO_HOST`, its projections left ABSENT so an expert
  op traps rather than reads, and a prefetch places nothing.

`plan_expert_cache` gives every class all its projections, and an expected
hit rate of 1.0, when the cache holds the catalog. `build_residency` takes
the device-only path exactly then. The host plan's `expert_pool` line is 0
there, and its staging line carries the layer buffers.

**Tests.**
- `ignis_kernel_residency_device_only_test` (new CTest) passed on the RTX 5090
  and on the RTX PRO 6000:
  - creation and admission refusals;
  - every slot-table entry on its own class's slot with its K and its
    record's bytes;
  - random decode and prefill steps with either lookahead, and a captured
    decode round, all hits, no copy, no stall.
- The existing residency CTests (trace in its four modes, routed) pass
  unchanged on both cards.
- `expert_residency_plan.rs` gains the cache-equals-catalog case.

**On the card** (RTX PRO 6000 Server Edition, g7e.4xlarge, Linux, driver
595.91.07). The develop image (main 851100c7, host pool) was compared with
the branch built from the same commit, in the same pod shape: same flags
(`--profile rtxpro6000`, `IGNIS_NGRAM_HOT_BYTES=27G`, loop guard off), same
prompts, greedy, 512 tokens, 3 reps after a warm-up cell:

| | develop (host pool) | branch (device only) |
|---|---:|---:|
| `expected_hit_rate` | 0.994 | 1.0 |
| host plan `planned_bytes` | 69.9 GB | 36.8 GB |
| pinned memory (cgroup `shmem`) | 41.1 GB | 3.3 GB |
| residency load phase (page cache warm) | 12.8 s | 5.7 s |
| 1 lane code / prose, tok/s | 262 / 197 | 263 / 198 |
| 8 lanes code, tok/s | 428 | 430 |
| expert misses in those cells | 455 / 634 / 4,245 | 0 / 0 / 0 |
| TTFT 32K | 7.37 s | 7.25 s |
| greedy text, 32 lanes | — | identical, every lane |

Every one of the 32 lanes (1-lane code and prose, 8-lane code, all reps)
produced the same text byte for byte.

**On a g7e.2xlarge** (8 vCPU / 64 GiB, the same RTX PRO 6000): the branch
loaded Flash-Next with the whole n-gram table. Host plan: 61.0 GB available,
36.8 GB planned, 24.2 GB left. The develop image's plan, 69.9 GB, does not fit
there. A cold load (fresh node, artifact off the EBS volume) took 2 min:
device weights 7.6 s, residency 67 s, n-gram 46 s. The same cells as above:
- 1 lane code / prose: 263.6 / 197.8 tok/s;
- 8 lanes code: 430.2 tok/s;
- TTFT 32K: 7.25 s;
- 0 misses, and the same 32 texts as the 4xlarge's.

Eight vCPUs instead of sixteen cost nothing measurable.

## Finding

- **On the resident load the host pool was 38 GB of RAM for ~0.6% of
  selections.** It served only the projections the cache split left slotless.
  Giving every projection its slot and dropping the pool costs nothing
  measurable:
  - decode equal or slightly faster, no miss left to wait on;
  - TTFT 1.6% lower;
  - the same text.
- **The host plan of such a load shrinks from ~74 GB to ~41 GB with the whole
  n-gram table** (36.8 GB of lines and the 4 GiB margin), and from ~46 GB to
  ~13 GB with the 1 GiB default. The pinned single allocation, the one a
  32-34 GiB driver ceiling would refuse, is gone.
- **Both builds share one thing that is not about residency:** the first one
  or two requests after a load decode at ~73 tok/s, against ~250 after them.
  Unexplained; the phase-7 cells run a warm-up cell first.

## Implications

- A machine around a 96 GB card no longer needs RAM for the experts. A 64 GB
  host holds Flash-Next with the whole n-gram table, which on a cloud volume
  is worth 1.24x decode and 2.3x prefill. Measured on a g7e.2xlarge above: the
  cheaper instance ($3.36/h against $4.00/h) serves at the 4xlarge's rates.
- The 5090's offloaded path is unchanged (cache < catalog): same pool, split,
  warm start and misses.
- The exclusive variant (a host pool of only what VRAM does not hold, with
  write-back on eviction) is a separate question for the offloaded path. A
  first link measurement on the 5090 (Gen 3) shows concurrent write-back
  costing the round 19-53%: `.scratch/residency-swap-5090/`.

## Limits and unknowns

- One card, one artifact (trellis-a25), Linux. Windows was compiled and the
  kernel tests run on the 5090; the full device-only load was not run there
  (a 32 GB card is never resident).
- The staging line (4.7 GB) is charged for the whole plan, though the load
  frees it before serving.
- The prefill staging ring (~1.5 GB of VRAM) is still reserved on a
  device-only load, where no step can stage. Giving it to the KV pool is a
  follow-up.

## Follow-ups

- The prefill staging ring on a device-only load.
- The slow first requests after a load.
