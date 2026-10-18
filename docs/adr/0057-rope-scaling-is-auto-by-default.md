# ADR 0057 -- rope scaling is `auto` by default: the YaRN factor follows the context

## Status

Accepted (2026-10-18, owner decision: "lo yarn:X sia calcolato
automaticamente in base al contesto; a 512k e 2, 768 e 3 e 1M e 4; da config
e overridabile"). Changes the default of `model.rope_scaling` that spec
rope-scaling/01 shipped as `none` (#227).

## Context

The checkpoints are trained over 262,144 positions. A context past them needs
the YaRN table, at the factor `max_context / 262,144`; a profile that raised
`max_context` had to remember to spell the matching `yarn:F`, per family
(the 27B and Flash-Next sit at different contexts on the same card), and a
mismatch is silent: a lower factor leaves positions extrapolated, a higher
one degrades short contexts for nothing.

## Decision

**`model.rope_scaling` accepts `auto`, the new default.** Resolved at load,
per model, from that model's effective `model.max_context`: up to 262,144 it
is no scaling (`RopeScaling::NONE`, the table the engine has always used, not
"YaRN at 1"), above it YaRN at the exact ratio with the default temperature
and ramp. `none` and `yarn:F[,...]` still override it from any source.

The built-in profiles drop their `yarn:4`: the `rtx5090` now runs the 27B
(786,432) at 3 and Flash-Next (589,824) at 2.25; the `rtxpro6000` is
unchanged at 4. Both are the smallest factor that covers the context, which
is what the owner asked for; neither was measured against `yarn:4`.

## Consequences

- A server started with no flags (context 16,384 or the engine default, at
  most 262,144) loads exactly as before. A start that raises `max_context`
  past 262,144 now gets YaRN without a second flag, and a config that named
  only `max_context` changes its outputs.
- The configured value stays `auto` in `GET /v1/config` and `config print`;
  the table every identity keys on (KV-disk, retained pages, the switch's
  reload check) is the resolved one, so `auto` at 1,048,576 and `yarn:4` are
  the same load. The `kv pool` log line names the resolved table and its
  source (`auto` or `configured`).
- The `make` knob `ROPE_SCALING` defaults to empty (the server's `auto`);
  `MAX_CONTEXT=524288` still gets `yarn:2`.
