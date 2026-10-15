# 03 - a restore through a retained prefix is reported

GitHub: #325

A request whose prompt comes back from a **retained prefix** (ADR 0029: a
page-floored shared prefix, not a prompt checkpoint) reuses its tokens, and
the client sees it in `usage.prompt_tokens_details.cached_tokens`, but the
server's own telemetry does not say so. Found by the loop-guard GPU run of
2026-10-15 (`docs/findings/2026-10-14-reasoning-loop-guard.md`, "Telemetry";
raw log `.scratch/worktrees/loop-rewind-scratch/loop-gpu/ignis-server.loop-gpu.log`,
requests 13 and 66-70, e.g. 78,784 = 1,231 x 64 tokens restored).

## Problem Statement

- The request's `ignis.request.done` line carries no `reuse_source` and no
  `reused_prompt_tokens`: `reuse_source` is set only on the checkpoint claim
  (`crates/core/src/concrete.rs`, where `r.reuse_source = Some(m.source)`).
- `ignis_retained_state_hits_total` does not count it, although ADR 0017
  splits the retained-state families by `kind="checkpoint|prefix"`. Only
  `ignis_retained_reused_tokens_total{kind="prefix"}` moves.
- The done line's `prefilled_tokens` is the position reached, equal to
  `prompt_tokens` with or without reuse, so a reader of the log alone takes a
  prefix restore for a cold prefill. That misreading already put a wrong
  number in a finding ("the continuation re-prefills its whole prompt").

## Acceptance criteria

1. A request restored through a retained prefix has, on its done line,
   `reuse_source` (the tier the prefix came from) and `reused_prompt_tokens`
   (the tokens restored), and a field that says the reuse was a prefix, not a
   checkpoint (name it after the logging spec's conventions,
   `docs/specs/logging/spec.md`).
2. `ignis_retained_state_hits_total{kind="prefix",tier=...}` counts one hit per
   such restore; the checkpoint series is unchanged.
3. A request that reused nothing still reports no `reuse_source` (the absence
   is the "none", `crates/core/src/checkpoint.rs` `ReuseSource`).
4. Tests: a CPU test per AC 1-3 over the mock engine (a prefix restore, a
   checkpoint restore, a cold prefill), and the Playground Monitor reads the
   new hits series if it shows the family (ADR 0017: a metric change lands in
   the Monitor and the ADR's table in the same work).
5. ADR 0017's table and the logging spec name the new field; the loop-guard
   finding's "Telemetry" note points at this spec as fixed.
