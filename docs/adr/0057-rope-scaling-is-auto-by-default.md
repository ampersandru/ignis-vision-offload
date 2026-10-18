# ADR 0057 — rope scaling is `auto` by default: the YaRN factor follows the context

## Status

Accepted (2026-10-18, owner decision: "lo yarn:X sia calcolato
automaticamente in base al contesto; a 512k è 2, 768 è 3 e 1M è 4; da config
è overridabile"). Changes the default of `model.rope_scaling` that spec
rope-scaling/01 shipped as `none` (#227).

## Context

The checkpoints are trained over 262,144 positions, the trained position
envelope. A context past it needs the YaRN table, at the factor
`max_context / 262,144`: YaRN's own `s = L'/L`, and what Qwen3-Next's card
uses (4 over 262,144 original positions for ~1M, 2.0 for 524,288). A profile
that raised `max_context` had to remember to spell the matching `yarn:F`, per
family (the 27B and Flash-Next sit at different contexts on the same card),
and a mismatch is silent: a lower factor leaves positions extrapolated, a
higher one degrades short contexts for nothing.

Flash-Next takes no scaled table at all. Its kernel refuses one ("rotates at
its own linear table", `kernel/src/flash_next/program.cu`) and its load
(`flash_next_options`) passes none, so the `yarn:4` the profiles named
before never reached it either.

## Decision

**`model.rope_scaling` accepts `auto`, the new default.** Resolved at load,
per model, from that model's effective `model.max_context`: within the
trained position envelope it is no scaling (`RopeScaling::NONE`, the table
the engine has always used, not "YaRN at 1"), past it YaRN at the exact
ratio with the default temperature and ramp. Past 64x the envelope
(16,777,216 tokens, the widest factor a load takes) `auto` has no table, and
the start is refused naming the context. `none` and `yarn:F[,...]` still
override it from any source.

**Flash-Next's table is always `none`**, under `auto` and under an explicit
value alike, and so is its KV-disk identity: past 262,144 tokens Flash-Next
runs the linear table, with no YaRN. An explicit YaRN value that reaches it,
in its own section or as a group value meant for the 27B, is named as
ignored in a WARN at load (`ignis.model.rope_scaling_ignored`), never
refused: a value for the 27B must not break a Flash-Next start.

The built-in profiles drop their `yarn:4`: the `rtx5090` now runs the 27B
(786,432) at 3, and the `rtxpro6000` runs it (1,048,576) at 4 as before.
Each is the smallest factor that covers the context, which is what the owner
asked for. The 27B's 3 on the `rtx5090` profile is unmeasured against the
`yarn:4` it replaces.

## Considered options

- **1. Keep the profiles' explicit `yarn:4`**, `none` still the default.
  Rejected: a factor per profile that each model's `max_context` has to
  match by hand is the silent mismatch above, and the owner asked for the
  context to decide it.
- **2. `auto` rounds the ratio up to a whole factor** (655,360 tokens would
  be 3, not 2.5). Rejected: the owner's three points are exact ratios, and a
  rounded-up factor scales past what the context needs, which degrades short
  contexts for nothing. No measurement favours whole factors.
- **3. The exact ratio** (chosen): YaRN's `s = L'/L`, and the owner's 2, 3
  and 4 at 524,288, 786,432 and 1,048,576.

## Consequences

- A server started with no flags (context 16,384 or the engine default, at
  most 262,144) loads exactly as before. A start that raises the 27B's
  `max_context` past 262,144 now gets YaRN without a second flag, and a
  config that named only `max_context` changes its outputs.
- The configured value stays as set in `GET /v1/config` and `config print`;
  the table every identity keys on (KV-disk, retained pages, the switch's
  reload check) is the resolved one, so `auto` at 1,048,576 and `yarn:4` are
  the same load, and a `PATCH` between them applies with no reload. The
  `ignis.model.loaded` line names the table run (`rope_scaling`) beside the
  setting (`rope_scaling_setting`: `auto`, `none` or `yarn:F...`).
- Flash-Next past 262,144 tokens (589,824 on the `rtx5090`, 1,048,576 on the
  `rtxpro6000`) runs its linear table unscaled. This ADR records it and
  changes no kernel.
- The `make` knob `ROPE_SCALING` defaults to empty (the server's `auto`);
  `MAX_CONTEXT=524288` still gets `yarn:2`.
