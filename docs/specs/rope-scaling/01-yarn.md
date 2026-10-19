# 01 — YaRN RoPE scaling as a load option

**GitHub:** #227 · **Branch:** `rope-yarn`

## Why

The checkpoint's trained rotary envelope is 262,144 positions
(`original_positions`, the reference's `TextConfig`). The GQA attention op
serves up to 1,048,576 visible keys under `hq-e8-2b` (524,288 under
bf16/int8 — `kGqaAttentionMaximumLinearVisibleKeys`), so the engine can
already *hold* a context past the trained range; it just rotates it at
frequencies the model was never trained on. YaRN is the frequency table that
makes those positions mean something, and the reference already serves it
behind `--rope-scaling none|yarn:F`.

## Seam

The table, not the kernel. `ninfer::ops::RopeFrequencies` (the vendored
`ops/rope.h`) already carries an arbitrary per-pair `inv_frequency` table and
the q-side `attention_factor`; the header states outright that "YaRN-shaped
tables are constructed by the owning target". The owning target here is the
Program (`kernel/src`, ours per ADR 0009), so the whole change is: build a
different table at load, hold it on the model, and let both GQA call sites
read it instead of rebuilding a linear one per call.

Nothing else moves:

- **DFlash2 drafter** keeps its own unscaled table at SWA-local positions
  (`drafter_rope()`), exactly as the reference does — target YaRN is
  Text-only.
- **Vision** keeps the 2-D vision table.
- **Linear default stays the linear builder.** `ops/rope.h` gives
  `attention_factor == 1` the exact legacy FP32 angle route and anything else
  an FP64-reduced one, so `none` must call `rope_linear_frequencies`, not the
  YaRN builder at factor 1. A load without the flag is bit-identical to
  today's.

## The table (HF `_compute_yarn_parameters`)

For pair `i` in `[0, rotary_dim/2)`, `linear = theta^(-2i/rotary_dim)`:

- `i < low` — extrapolate: `linear`
- `i > high` — interpolate: `linear / factor`
- otherwise — blend by `e = (i - low) / (high - low)`:
  `linear * ((1 - e) + e / factor)`

with the ramp bounds the floor/ceil of
`rotary_dim * ln(original / (beta * 2pi)) / (2 ln theta)`, clamped to leave a
non-empty extrapolation and interpolation segment, and
`attention_factor = temperature * ln(factor) + 1`.

At the checkpoint's constants (theta 1e7, rotary_dim 64, original 262144,
beta_fast 32, beta_slow 1) that is `low = 14`, `high = 22`.

## Surface

- `--rope-scaling none|yarn:F[,t=<c>][,beta_fast=<b>][,beta_slow=<b>]`,
  env `IGNIS_ROPE_SCALING`, default `none`. Grammar and defaults
  (t=0.1, beta_fast=32, beta_slow=1) are the reference's.
- `ROPE_SCALING=` Makefile knob, so `make config` prints it (ADR 0027).
- `ignis_model_load_options` grows four scalars (ADR 0016 `size` first);
  factor 0 or 1 means linear, a YaRN factor is in (1, 64].

## Acceptance

1. `rope_yarn_frequencies(1e7, 64, 262144, F, ...)` reproduces the reference's
   table: ramp bounds 14/22, the three segments, and
   `attention_factor = 0.1*ln(F)+1`. Kernel CTest with explicit vectors.
2. A load with no flag builds the linear table and crosses the ABI exactly as
   before (`attention_factor == 1`, the legacy FP32 route).
3. The four scalars are validated at the ABI boundary and a bad one fails the
   load with a named message — never a silently different table.
4. `--rope-scaling` parses the reference's grammar, rejects a factor outside
   (1, 64], a non-positive temperature, and a ramp that is not
   `beta_fast > beta_slow > 0`.
5. A load carrying only rope scaling (no speculation, no vision) still sends
   the options struct.
6. GPU: a load with `yarn:4` serves a short prompt as coherent text — the
   attention factor touches every position, so a short context is a real
   check.

## Out of scope (follow-ups)

- ignis does not validate `--max-context` against either attention envelope
  (524,288 linear / 1,048,576 hq). That check is the reference's
  `layouts_impl.h` and does not exist here; it is worth its own ticket.
- ~~No automatic factor derived from `--max-context`~~ — done, see the
  as-built note below.

## As built: `auto` (the default)

Owner's call (2026-10-18, "lo yarn:X sia calcolato automaticamente in base al
contesto"): `model.rope_scaling` (`--model-rope-scaling`) takes `auto`, and
`auto` is the default instead of `none`. The rule, `RopeSetting::resolve` in
`crates/core/src/rope_scaling.rs`, per model, from its effective
`model.max_context`: within the trained position envelope
(`TRAINED_POSITION_ENVELOPE`, 262,144) the table is `RopeScaling::NONE`,
byte for byte today's default (not YaRN at 1, which the rope op routes
differently); above it, YaRN at the exact ratio `max_context / 262,144` with
the default temperature and ramp (524,288 is 2, 786,432 is 3, 1,048,576 is
4); past 64x the envelope (16,777,216 tokens) `auto` has no table, and the
start is refused naming the context. Any explicit value (flag, env, file,
`PATCH /v1/config`) overrides it, and `auto` can be spelled, or the key
removed with a `PATCH` `null`, to go back.

**Amended 2026-10-19 (ADR 0059):** Flash-Next runs the same resolved table
as the 27B, under `auto` and under any explicit value, and so is its KV-disk
identity (spec [`02`](02-flash-next.md) is the kernel side). Written when
Flash-Next had no YaRN, this spec said its table was always `NONE` and named
an explicit YaRN value for it as ignored in a WARN; neither holds any more,
and a context `auto` cannot scale (past 16,777,216 tokens) is refused for it
as for the 27B.

The configured value (`GET /v1/config`, `config print`) stays as it was set;
`Config` and `EngineShape` carry the resolved table, and the KV-disk
identity, retained pages and the model switch's reload check key on it:
`needs_reload` compares the table for this field, not its spelling, so
`auto` and `yarn:4` at 1,048,576 are the same load and a patch between them
applies live. The load's `ignis.model.loaded` line logs `rope_scaling` (the
table run) beside `rope_scaling_setting` (`auto`, `none` or `yarn:F...`).
The built-in profiles no longer name `yarn:4`: the `rtx5090` 27B now runs
YaRN at 3 and Flash-Next at 2.25, the `rtxpro6000` runs both at 4
([ADR 0057](../../adr/0057-rope-scaling-is-auto-by-default.md)).
