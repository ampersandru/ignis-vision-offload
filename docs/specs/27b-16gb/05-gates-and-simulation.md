# 05 - first light, gates and simulation: the 16 GB artifact served in ignis, its quality, 16 GB capacity on the 5090, the speed estimate

GitHub: #339 (Part A, 05a: first light) and #340 (Part B, 05b: gates and simulation), master #334.

Decided 2026-10-21: the owner approved this spec and vetoed none of the *(proposed)* items, so every one is decided.

Phase 1 has no 16 GB card (the owner, 2026-10-21). This spec is what phase 1 measures instead, and how:
- **quality**, end to end in ignis, against the owner's gates;
- **capacity**, simulated exactly on the 5090;
- **speed**, as an estimate stated as one, with a bench users can run to report real numbers.

ADRs: 0060 (the gates), 0063 (adaptation), 0003/0014 (gating practice), 0022 (BF16 as oracle).

## Part A (ticket 05a) - first light

The point where specs 01-04 meet: the 16 GB artifact served end to end by ignis on the 5090, under a 16 GiB budget.

**What it wires:**
- the 27B program binds a trellis body, fused planes or the split parts the repack carries (spec 01), by object
  presence;
- every layer's linears dispatch to spec 02's ops by tensor format;
- the token embedding is read from host memory (spec 03);
- the load runs the `rtx50-16gb` profile with its lanes and adaptation (spec 04).

**Acceptance (A):**

A1. The 16 GB artifact loads on the 5090 with `--vram-budget-bytes 17179869184` under the `rtx50-16gb` profile. The
    VRAM plan and any adaptation are logged, and the plan's lines match spec 04's formulas.
A2. It serves the coding canaries greedy, end to end: chunked prefill, then decode with DFlash2 at 2 lanes, the 128K
    leg included. Outputs are coherent and there are no errors. Tokens per round and round time are reported.
A3. A sanity KLD on two of the 2026-09-24 windows, engine path, reported beside M1's weights-only numbers for the same
    windows. This is not the gate (Part B); it catches a wiring error early.
A4. The 32 GB NVFP4 artifact still serves exactly as before: greedy outputs on the canaries are identical.
A5. `cargo test` workspace-wide; GPU tests green on a free 5090.

### As built: the wiring (2026-10-23)

- **The body is the head's.** `text/output_head` in a trellis format makes the body trellis, on both sides of the
  ABI: `ignis_artifact::body_27b` reads the directory, and `model_scope_27b_of` gives the binder and the load's
  descriptors spec 01's trellis text scope (`text_scope_27b_trellis`, K from the directory); the leaf's binder
  (`kernel/src/model.cu`) reads the head's qtype and binds the whole body under that schema. `ModelScope` stays the
  modules a load asks for. A tensor of the other body fails naming itself ("a load binds one body, never a mix").
  `fp8_exceptions` converts BF16 tensors only, so on the trellis body, whose `attention/output` of layers 3 and 7
  and `gdn/output` of layer 4 share the exception names, it is a no-op.
- **ABI.** Six leaf-only qtype codes, `IGNIS_QTYPE_TRELLIS_MUL1_K2` .. `_K6` = 64 .. 69, and
  `IGNIS_LAYOUT_TRELLIS_TILE16` = 64, past `ninfer::QType` / `QuantLayout` (vendored, untouched): a trellis tensor
  binds into an `ignis_trellis_weight`, never a `ninfer::Weight`. Its `qhigh` and `scales` are the record's `suh` and
  `svh` (the materializer's view names them); the binder checks both against the tiles' length and the record's
  bytes against its K.
- **Dispatch** (`kernel/src/trellis_body.h`): the three GDN bodies and the two GQA ones call the op in place of the
  vendored NVFP4 ops on a trellis load -- `in_proj_qkv` / `in_proj_z` into `qkv` / `zbuf`, the output and down
  projections as the op's residual add, the gate and up matrices as its SwiGLU, `k_proj` / `v_proj` into key and
  value, `q_proj` into a `[T][12288]` buffer whose heads' halves two `cudaMemcpy2DAsync` move into the query and the
  gate. The verify GDN body writes its projection into the conv record and runs the vendored
  `gdn_projected_conv_record_launch` over it. Every head site -- the four prefill ones, the decode and verify rounds,
  and the drafter's full head -- goes through `output_head`, the shortlist head staying the served Q4 one. The
  test-only `kA16Only` mode is the NVFP4 arm's alone.
- **The `trellis_workspace` plan line** (ADR 0030 amended, `ignis_model_reservations::trellis_workspace_bytes`,
  `VramLines::trellis_workspace`, the plan event, `/metrics`, the Monitor): one buffer at a stable address on the
  model's one stream (the drafter's calls run there too), zeroed once at load, `ignis_trellis_prepare` run by the
  plan and the load. Its size is the op's own answer for the widest call: the body's shapes at 1..64 tokens (a short
  prompt or a span's last chunk takes the decode route) and at the chunk, the head's at a round's columns across the
  lanes (`lanes x (k + 1)`) or one span-logits block (32). 36,225,280 bytes at chunk 1,024 with `rtx50-16gb`'s two
  lanes and seven drafts; 0 on the served body. The plan's scratch drops the vendored projections' workspaces and
  adds the q_proj buffer.
- **A fix in spec 02's op.** Its workspace's arrival counters sat in a room sized by the call's own output blocks,
  with the rotated activations after it, so a narrow call (a down projection) left a wider call's counters nonzero
  and that call reduced its splits early: every layer of the trellis body ran 0.17-0.86 from the oracle until the
  room got one size (`kArrivalBlocks`, out at most 524,288; +16 KB of workspace). `test_trellis_linear`'s
  `shared_workspace_arm` runs a load's calls of five shapes through one workspace, each against its own fresh one.
- **Tests and what they measured on the 5090:**
  - `kernel/tests/test_model_load_trellis.cpp` (the binder and the plan line) and `test_trellis_body.cu` (the q/gate
    split, `query_gate`, the head, the epilogues); both fail on the split's halves swapped.
  - `crates/core/tests/trellis_projections_gpu.rs`: every split part is the served plane's rows in spec 01's order,
    cosine 0.928-0.997 per block, the q/gate halves the other way round 0.005 or less.
  - `crates/core/tests/trellis_layers_gpu.rs`: GDN layer 0 and GQA layers 3 and 27, one and four tokens, against the
    f64 oracle over the same decoded weights: relative L2 0.0018-0.0047 (bound 0.01); the split's halves swapped:
    0.78-1.0.
  - `crates/core/tests/trellis_logits_gpu.rs`: 768 positions of code, top-1 agreement with the served program
    0.891, mean KL(served || trellis) 0.072 nats (floor 0.7, an error detector).
  - `crates/server/tests/trellis_16g_gpu.rs`: the server's load path, `--profile rtx50-16gb --vram-budget-bytes
    17179869184`: nothing adapted, 262,144 context, two lanes, a 509,184-token KV pool, weights 10,907,958,020
    bytes; greedy code coherent and deterministic, DFlash2 7.68 tokens per round.
  - `crates/artifact/tests/trellis_27b.rs`: the model scope binds the real artifact by its body at spec 01's device
    bytes; CPU tests for the codes, the body detection, the fp8 guard and the structs' layouts on both sides.
- **Left to the acceptance runs:** A1's plan log through `make start`, A2's canaries and 128K leg, A3's sanity KLD,
  A4's canary identity on the served artifact.

## Part B (ticket 05b) - gates and simulation

Blocked by 05a. Everything below this line is Part B.

## Problem Statement

- M1 measured the candidate body weights-only in a PyTorch harness (KLD 0.0340 vs the served NVFP4's 0.0458 on the
  2026-09-24 windows).
- The ignis artifact adds:
  - our kernels' numerics (spec 02);
  - hq-e8-2b KV (+~0.005 at 4K in the 2026-09-24 finding, more at 32-64K);
  - the host embedding;
  - DFlash2 on a lossy target.
- The 5090 emulates a 16 GB card's memory with `--vram-budget-bytes`, but not its bandwidth or SM count.

## Solution

1. **Quality gates on the ignis artifact.** The owner's, 2026-10-21:

   | # | gate | how it is measured |
   |---|---|---|
   | 1 | **KLD ≤ 1.2×** the served NVFP4's | same suite, same engine path: `span_logits.rs` dumps to `Y:`, scored with `ref_kld.py`, serving configuration (hq-e8-2b, chunk 1024) |
   | 2 | **MMLU-Pro** (n=281) not significantly below BF16 (McNemar p > 0.05) | through the engine's logits on the Flash-Next study's chunks, if M2 step 0 clears the harness; otherwise on the corrected harness M2 names |
   | 3 | **DFlash2 tokens per round ≥ 90%** of the NVFP4 target's | the in-house teacher-forced replay (61,692 recorded rounds, branch `dflash2-sampled-acceptance-267`), same traces on both artifacts |
   | 4 | **loop rate** no worse than today's | the loop set (`.scratch/loops/`), same prompts and seeds on both artifacts |

   Plus *(proposed)* the KLD per domain of the Flash-Next held-out set, reported (Python is the known weak spot: ppl
   +8.6% in M1).
2. **Capacity simulation on the 5090.** An explicit budget of 17,179,869,184 bytes, minus 0, 1 and 3 GB of simulated
   desktop. For each: the plan lines, the adaptations taken (spec 04), the context and lanes reached, and a short
   serving run (load, prefill 128K, decode both lanes) that must succeed. The plan lines are checked against
   `PIANO-VRAM.md`'s estimates.
3. **Speed estimate, stated as such.**
   - Measured on the 5090: decode round time at 1/2 lanes and 8K/64K/256K, its bytes, and the kernels' achieved
     bandwidth.
   - The same with spec 02's SM caps (36 / 70 / 84 SMs).
   - Scaled to each 16 GB card's bandwidth by the study's model (`.scratch/27b-16gb-study-2026-10-20/scripts/budget.py`).
   - Published as an estimate with its assumptions.
4. **A user bench.** *(proposed)* `ignis-bench` gains a `small-card` preset that records:
   - card name, total VRAM, PCIe link;
   - plan lines and adaptations;
   - decode tok/s on the coding prompts and on agent-like turns;
   - TTFT at 8K/64K.

   It writes one JSON a user can paste into an issue. This is how phase 2 finds real numbers if no card is rented.

## Testing Decisions

- Gates 1-4 are run once, at the end of phase 1, on a free 5090 (the owner's practice: benchmarks are error detectors,
  run once at phase end). Each result goes into a finding.
- The simulation scenarios run as GPU tests where cheap: load and plan. The 128K serving run is a bench leg.

## Acceptance

1. Gate 1: KLD of the ignis 16 GB artifact on the 2026-09-24 suite, serving configuration, ≤ 1.2× the served NVFP4's
   on the same run's reference.
2. Gate 2: MMLU-Pro on the cleared or corrected harness, McNemar p > 0.05 against BF16.
3. Gate 3: DFlash2 tokens per round on the replay ≥ 90% of the NVFP4 artifact's.
4. Gate 4: loop rate on the loop set no worse than the NVFP4 artifact's (same prompts, seeds, guard settings).
5. The capacity simulation's three scenarios load, adapt as spec 04 says, serve the 128K leg, and their plans are
   within 2% of the estimates (or the estimates are corrected).
6. The speed estimate is published with its assumptions and the SM-cap measurements, labelled as not measured on a
   16 GB card.
7. The `small-card` bench preset exists and writes the JSON above.
8. One finding per gate group: quality, capacity, speed estimate, with `docs/findings/README.md` rows.

## Out of Scope

- A speed gate on a real 16 GB card: phase 2 (the owner deferred it, Q8).
- 12 GB: phase 2.

## Further Notes

- The M1 harness (`.scratch/27b-16gb-study-2026-10-20/m1/m1_exl3.py`) is the weights-only reference point. Its numbers
  are not the gate.
- KLD dumps are 2.03 GB per 4K window, so they go to `Y:`. F: has ~18 GB free.
