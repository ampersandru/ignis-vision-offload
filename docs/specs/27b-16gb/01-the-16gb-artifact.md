# 01 - the 16 GB artifact: a trellis body, the DFlash2 graft, the catalog entry

GitHub: #338 (master #334).

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

## As built (2026-10-23)

- **Who writes the container.** `tools/27b-16gb-converter/convert.py run` writes the packer's work tree (units
  `frontend`, `text`, `mtp`, `vision`, `dflash2`: a `tensors.json`, one payload file per tensor, `DONE`, as
  `docs/specs/flash-next/layout.md` §2) and `converter.json`; `ignis-artifact-pack --family qwen3.8-27b-trellis`
  (`Family::Qwen27bTrellis`) assembles the container and merges `converter.json` into the sidecar. Python never writes
  the container, as for Flash-Next and the image model. The step that needs the GPU is the reconstruction errors only.
- **The artifact.** `qwen3_8_27b_trellis_sc30-v2.ninfer`, identity `qwen3.8-27b` / `trellis-sc30`, 13,379,877,124
  bytes, sha256 `2c4c9ee4d6b25c08da6c2d85840126e73d120564e230f926b6230b5b95ee8b3c`; sidecar
  `<artifact>.conversion.json`. 1,222 objects: 6 resources and 1,216 tensors, of which 401 trellis (the body's 400
  matrices and the head) and 821 copied. The 247 `*_input_scale_divisor` scalars of the NVFP4 planes are dropped with
  them; the quant's MTP layer (8 matrices) is not taken: the MTP layer is the served W8 one.
- **The object names** (`plan.py`; `ignis_artifact::text_scope_27b_trellis` on the Rust side). Under
  `text/layers/{i}/`:

  | served NVFP4 / BF16 plane | in the 16 GB artifact (trellis) | rows × columns |
  |---|---|---|
  | `gdn/query_key_value_z` | `gdn/in_proj_qkv`, `gdn/in_proj_z` | 10,240 / 6,144 × 5,120 |
  | `gdn/output` | `gdn/output` | 5,120 × 6,144 |
  | `attention/query_key_gate_value` | `attention/q_proj`, `attention/k_proj`, `attention/v_proj` | 12,288 / 1,024 / 1,024 × 5,120 |
  | `attention/output` | `attention/output` | 5,120 × 6,144 |
  | `mlp/gate_up` | `mlp/gate_proj`, `mlp/up_proj` | 17,408 / 17,408 × 5,120 |
  | `mlp/down` | `mlp/down` | 5,120 × 17,408 |

  and `text/output_head` (248,320 × 5,120) in trellis at K = 4. Every other object keeps its served name, format and
  bytes, in the served directory's order (the frontend resources sorted by name).
- **Row orders, for spec 05a's binding** (exllamav3's and Hugging Face's, not the served artifact's):
  - `attention/q_proj` is 24 heads of [q 256 | gate 256], interleaved per head (transformers' `q_proj(x).view(...,
    head_dim * 2)` then `chunk(2)`); the served `query_key_gate_value` is the blocks q | k | gate | v.
  - `gdn/in_proj_qkv` is [q 2,048 | k 2,048 | v 6,144], split by key and value width as transformers'
    `Qwen3_5GatedDeltaNet` does; `gdn/in_proj_z` is the 6,144-row output gate.
  - Each split matrix has its own `suh`/`svh` and its own K; a record's tile order and scales are exllamav3's
    (`docs/specs/flash-next/layout.md` §3), the records spec 02's `ignis_trellis_linear` decodes.
- **The token embedding stays W8G32** (ADR 0060 left W8 or BF16 open): copied byte for byte, it lives in mapped host
  memory (ADR 0062), so its format costs no VRAM, and the loader reads it already.
- **K per matrix** (the quant's): K2 34, K3 297, K4 66, K5 4; trellis bytes 9,772,122,112. **Reconstruction error**
  against BF16: max 0.3383 (`text/layers/0/mlp/down`), mean 0.1500, the head 0.0765; per matrix equal to M1's harness
  within 4.1e-5 (M1 cast to BF16 before the difference).
- **Acceptance:**
  1. Met: the sidecar pins `turboderp/Qwen3.8-27B-exl3`, branch `SC_3.00bpw_H4`, revision
     `86b95530f9d365d90dd24d01a997a8de0cdf6c55` (from the download's Hugging Face records), the two shards (each
     checked against its LFS oid) and the index, quantization config and config by sha256.
  2. Met: two packs of the same work tree are byte-identical (the sha256 above); the converter writes no timestamp or
     local path, so the same inputs give the same work tree (`test_the_same_inputs_give_the_same_work_tree`,
     `the_same_tree_packs_to_the_same_bytes`). The sidecar carries the provenance of the solution's point 3, the
     commit the converter ran from, and the served graft sidecar's NVFP4 records of the copied drafter, which
     `checksum::verify` reads clean (34 of 34).
  3. Met: `convert.py verify` reads every trellis object back equal to its source tensors and every copied object's
     sha256 equal to the served one's (VERIFY PASS, 401 + 821); the Rust test cross-checks copies of every kind through
     the crate's reader, and binds every object of the directory to `finish` (`tests/trellis_27b.rs`).
  4. Met: the text scope, the embedding in mapped host memory, is 9,825,087,488 bytes of device arena; with the DFlash2
     module and the draft head 11,264,998,400; with the vision tower too 11,560,717,824. The trellis op's workspace
     (`trellis::plan_bytes`, 72.4 MB at 16 decode and 2,048 prefill tokens: spec 02) comes on top in spec 05's plan.
  5. See the Hugging Face paragraph below.
  6. Met: `cargo test` workspace-wide.
- **Moved to spec 05a.** The testing decision "the artifact loads in ignis on the 5090 under a 16 GiB budget" is 05a's
  Acceptance A1: the 27B program binds the NVFP4 text scope only until 05a wires the trellis one.
