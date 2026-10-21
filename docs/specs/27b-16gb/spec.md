# The 27B on 16 GB cards — the feature, its tickets, and how to run them

GitHub: master and specs 01-05 to be opened after the owner's seam review (one ticket per spec, numbered ACs).

The feature-level document the tickets decompose (`docs/agents/issue-tracker.md`). Read this first, then the ticket's
spec.

## What and why

ignis serves Qwen3.8-27B on the 16 GB Blackwell **card class** (RTX 5060 Ti 16 GB, 5070 Ti, 5080), as well as on the
32 GB and 96 GB cards it serves today.

The 27B's served artifact cannot load on 16 GiB at any context (its text scope alone is 17.21 GB). The plan:
- a second artifact of the same family, whose **body format** is **trellis** at ~3 bits per weight;
- two changes that give VRAM back on every card:
  - no **prompt planes**;
  - the token embedding in host memory;
- a runtime that follows the card class.

On a 16 GiB card with little else on it, that serves 256K of context with DFlash2 and two lanes.

**The evidence** is the study of 2026-10-20/21 in `F:/ai/opencode/inference/.scratch/27b-16gb-study-2026-10-20/`
(untracked):
- `SINTESI.md`: options and literature;
- `PIANO-VRAM.md`: plan lines per card, §8-10 for 256K, no planes, MTP;
- `PIANO-LAVORO.md`: every owner decision;
- `RISULTATI-M1.md`: quality of the public EXL3 `SC_3.00bpw_H4` on our text;
- `m2/RISULTATI-M2.md`: DFlash2 acceptance on that target, and the MMLU harness check.

**Decisions:**
- [ADR 0060](../../adr/0060-the-27b-has-a-trellis-body-for-small-cards.md): a trellis body for the 16 GB card class.
- [ADR 0061](../../adr/0061-prefill-attention-reads-the-hq-codec.md): prefill attention reads the hq codec, so there
  are no prompt planes, on every card.
- [ADR 0062](../../adr/0062-the-token-embedding-lives-in-host-memory.md): the token embedding lives in pinned host
  memory, on every card.
- [ADR 0063](../../adr/0063-the-profile-follows-the-card-class.md): the profile follows the card class, and its
  defaults adapt to the budget.
- [ADR 0064](../../adr/0064-a-catalog-entry-states-its-card-class.md): a catalog entry states its card class, there is
  an alias within a family, and a switch refuses before teardown.
- The owner's decisions in the grill of 2026-10-21:
  - two phases;
  - quality gates;
  - one `decode_lanes` field;
  - the drafter head `auto`;
  - no display detection;
  - Flash-Next and images not in phase 1;
  - no 16 GB card in the project, so phase 1 is simulated on the 5090.

## Decided (2026-10-21)

The owner approved this spec set on 2026-10-21 ("accetto tutto") and vetoed none of the agent's proposals: every
*(proposed)* item in specs 01-05 is decided. ADRs 0060-0064 are accepted. The evidence behind the body is
[the 2026-10-21 finding](../../findings/2026-10-21-exl3-3bpw-27b-on-our-text-and-dflash2.md).

## Phase 1 (this spec set) and phase 2

**Phase 1** is the 27B on the 16 GB card class, excellent and releasable:
- capacity and quality are measured on the 5090;
- decode speed on 16 GB cards is stated as an estimate.

**Phase 2** is in [`phase2-notes.md`](phase2-notes.md), notes only:
- Flash-Next on 16 GB;
- the abliterated variant;
- the speed gate on a real card;
- per-card profiles;
- 12 GB.

The image model comes after phase 2.

## The tickets

| spec | what | blocked by | can start now |
|---|---|---|---|
| [03](03-vram-recovered-on-every-card.md) | no prompt planes; token embedding in host memory; the 5090's profile re-measured | — | **yes, everything**, on the 5090 |
| [04](04-the-runtime-follows-the-card-class.md) | one `decode_lanes` field; profile `auto` by card class + adaptation; catalog card class, alias, refusal before teardown; drafter head `auto` | — (the 16 GB entry's file pins come from 01) | yes: lanes, profile, adaptation and catalog work on the 5090 with an explicit budget |
| [02](02-the-dense-trellis-kernels.md) | dense trellis GEMV/GEMM for decode, verify and prefill, and the head | — for reduced shapes; 01 for full-shape checks | yes: reduced-shape fixtures straight from exllamav3 |
| [01](01-the-16gb-artifact.md) | the 16 GB artifact: body source, converter, DFlash2 graft, sidecar, catalog entry, HF upload | — (the body source is decided: repack, ADR 0060 §4) | yes: the repack, the copies and the sidecar need no kernel |
| [05](05-gates-and-simulation.md) | quality gates on the artifact in ignis; 16 GB capacity simulation on the 5090; speed estimate and a user bench | 01-04 for the final run | yes: the harnesses (M1's is reusable) |

The **order** is the owner's (Q20): 03, then 04, then 02 and 01, with 05 in parallel. 03 and 04 pay off on the 5090
before any 16 GB artifact exists.

## Environment

- **Study code and data:** `F:/ai/opencode/inference/.scratch/27b-16gb-study-2026-10-20/`. It includes the M1 harness
  `m1/m1_exl3.py`: a layer-streamed BF16 reference, with EXL3 reconstruction.
- **Python:** `F:/ai/ngram-venv`, with torch 2.13 cu130, transformers 5.17, flash-linear-attention 0.5.2 and
  exllamav3 1.5.3 (CUDA ext built).
  - Do not upgrade packages in it; another venv under `F:/ai/` if a newer exllamav3 is needed.
  - Never set `TORCH_CUDA_ARCH_LIST` when importing exllamav3: it rebuilds for ~6 minutes.
- **Checkpoint:** `Y:/models/Qwen3.8-27B` (BF16, 55.6 GB, on the NAS).
- **Public EXL3 quant:** `Y:/models/qwen38-27b-exl3-SC_3.00bpw_H4`. Other branches of `turboderp/Qwen3.8-27B-exl3`
  download to `Y:` with `HF_HUB_DISABLE_XET=1 hf download ... --max-workers 1`.
- **Disk:** F: has ~18 GB free. Model files and large dumps go to `Y:` (6 TB free).
- **Served artifact:** `F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer`, the source of the byte-for-byte
  copies in spec 01.
- **Worktrees:** `../.inference-qwen-worktrees/<branch>/`, one per ticket.

## The GPU: one card, one run at a time

- `make gpu-status` before any GPU work, in any worktree.
- Take the shared lock (`../.inference-qwen-worktrees/.swarm/gpu-lock.sh`).
- Stop `ninfer-serve` / the ignis server if they hold the card (standing permission); the owner restarts them.
- Long runs are launched detached (PowerShell `Start-Process` with an exit-code sentinel) and watched with an
  until-loop, never a foreground sleep.

## What every ticket must keep

- The 32 GB NVFP4 artifact, its kernels and its numbers are unchanged. Specs 03 and 04 must leave the 5090's behaviour
  equal or better, measured.
- `cargo test` workspace-wide, the kernel build clean, and the GPU tests green on a free 5090 (AGENTS.md).
- Every threshold or limit a ticket adds is a config field with a default (the owner's rule), never a `pub const`.
