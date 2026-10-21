# 05 - gates and simulation: the artifact's quality in ignis, 16 GB capacity on the 5090, the speed estimate

GitHub: to be opened (master to be opened).

Decided 2026-10-21: the owner approved this spec and vetoed none of the *(proposed)* items, so every one is decided.

Phase 1 has no 16 GB card (the owner, 2026-10-21). This spec is what phase 1 measures instead, and how:
- **quality**, end to end in ignis, against the owner's gates;
- **capacity**, simulated exactly on the 5090;
- **speed**, as an estimate stated as one, with a bench users can run to report real numbers.

ADRs: 0060 (the gates), 0063 (adaptation), 0003/0014 (gating practice), 0022 (BF16 as oracle).

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
