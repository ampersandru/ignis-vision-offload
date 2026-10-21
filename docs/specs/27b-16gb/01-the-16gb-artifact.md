# 01 - the 16 GB artifact: a trellis body, the DFlash2 graft, the catalog entry

GitHub: to be opened (master to be opened).

Decided 2026-10-21: the owner approved this spec and vetoed none of the *(proposed)* items, so every one is decided.

The second artifact of the Qwen3.8-27B family, for the 16 GB card class (ADR 0060):
- the body and the output head in trellis at ~3 bits per weight;
- everything else copied byte for byte from the served artifact, the DFlash2 module included;
- published on Hugging Face and pinned in the built-in catalog.

ADRs: 0060, 0062 (embedding in host memory), 0064 (catalog card class and alias), 0043/0044 (checkpoint and exllamav3
as oracles), 0002 (the artifact format).

## Problem Statement

- No artifact of the 27B fits 16 GB. The served one is 19.41 GB; its text scope is 17.21 GB.
- The reference engine's DFlash2 graft tool (`F:/ai/q38/ninfer/tools/artifact/graft_dflash2_module.py`) accepts only
  its own NVFP4 recipe as a base (identity `nvfp4full`, exact object list).

## Solution

1. **The body: a repack of `turboderp/Qwen3.8-27B-exl3` branch `SC_3.00bpw_H4`** (Apache-2.0). The owner decided it
   on 2026-10-21, after M1/M2 (ADR 0060 §4).
   - Its records are bit-compatible with ours: `mul1`, multiplier `0x83DCD12D`, `out_scales: always`.
   - K per matrix ∈ {2, 3, 4, 5}; head at 4.
   - qkv/z and q/k/v stay split matrices, served as such (spec 02).
   - M1: KLD 0.0340 vs the served NVFP4's 0.0458 on our suite. M2: DFlash2 tokens per round ≥ 96% of an EXL3 5.0
     control.
   - The local copy is `Y:/models/qwen38-27b-exl3-SC_3.00bpw_H4` (revision of the branch pinned in the sidecar).
   - Our own conversion (fused planes, our calibration) stays the measured alternative, used only if spec 05's gates
     fail in ignis.
2. **The rest of the artifact:**
   - from the served `qwen3_8_27b_nvfp4full-v2.ninfer`, byte for byte:
     - the token embedding (W8G32; format open, W8 or BF16: ADR 0060);
     - the norms and the GDN small set (a/b, conv1d, A_log, dt_bias, GDN norm);
     - the 66 DFlash2 objects and the draft-head shortlist;
     - the vision tower;
     - the MTP layer (for phase 2's fallback; it costs file bytes, no VRAM);
     - the frontend resources;
   - identity `{model_id: qwen3.8-27b, weights_id: <new>}`;
   - *(proposed)* `weights_id`: `trellis-sc30`.
3. **A converter in this repo** (`tools/27b-16gb-converter/`, Python under `F:/ai/ngram-venv`). It writes the v2
   container with the trellis format codes, and a sidecar (`.conversion.json`) with:
   - the source repo and revision, every source shard pinned by sha256;
   - per-matrix K and reconstruction error against BF16;
   - the bytes copied from the served artifact, with that file's sha256;
   - the graft's provenance.
4. **Publish:**
   - a Hugging Face repo, *(proposed)* `gpillon/Qwen3.8-27B-trellis-3bpw-ignis`;
   - private until the privacy scan, the same procedure as Flash-Next's;
   - a card with measured values only (the owner's rule for public cards).

   And a catalog entry in `crates/server/src/download/catalog.yaml`:
   - served id *(proposed)* `qwen3.8-27b-16g`;
   - `card_class: 16`;
   - `aliases: [qwen3.8-27b]` (spec 04);
   - every file's bytes and sha256.

## Implementation Decisions

- The converter never reads the BF16 checkpoint for anything the served artifact already holds: the copies keep
  provenance simple and the endpoints identical to what the 5090 serves.
- *(proposed)* A split matrix keeps its exllamav3 name in the object table (`in_proj_qkv`, `in_proj_z`,
  `q_proj` / `k_proj` / `v_proj`). The layer code binds either a fused plane or its split parts by object presence.
- Vision is in the artifact but off in the 16 GB profile (spec 04).

## Testing Decisions

- Every trellis record round-trips: the tensor in the artifact equals its source EXL3 shard tensor bit for bit.
- Every copied object's bytes equal the served artifact's (sha256 per object).
- Per-matrix relative reconstruction error against BF16 is recorded in the sidecar. Any matrix above 0.6 is a layout
  error and fails the conversion, as in M1's harness.
- The artifact loads in ignis on the 5090 (with spec 02's kernels) under an explicit 16 GiB budget.
- Its quality is spec 05's gate.

## Acceptance

1. The source is pinned: repo, branch revision and every shard's sha256 in the sidecar, matching ADR 0060 §4.
2. The converter produces the artifact reproducibly: same inputs, same bytes. Its sidecar carries every provenance
   item above.
3. Every object passes the round-trip and copy checks. The artifact's object table is consumed whole by the binder (no
   unconsumed object).
4. Text-scope device bytes, with the embedding in host memory, at ≤ 10.5 GB for the 3.0 bpw body, reported with the
   plan.
5. The artifact is on Hugging Face (public after the scan, on the owner's go), with a card that cites turboderp's quant.
   It is in the built-in catalog with its pins, card class and alias.
6. `cargo test` workspace-wide.

## Out of Scope

- The kernels (spec 02) and the runtime (spec 04).
- The abliterated variant (phase 2: same converter).
- A trellis drafter (phase 2, if space is short): DFlash2 stays NVFP4.

## Further Notes

- The public quant's other branches (2.50, 3.00, 3.50, 4.00, SC_4.00_H5) are the fallbacks and comparison points if
  spec 05's gates send the choice elsewhere.
- Licences: the checkpoint and the public quant are both Apache-2.0. exllamav3 is MIT. The card cites turboderp's
  quant.
