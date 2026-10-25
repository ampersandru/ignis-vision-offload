# Testing against the canary oracle

## The canary oracle fixture (P1-05, GitHub #41)

`crates/bench/tests/fixtures/oracle_canary.json` is the committed fixture
G1 is measured against: the reference engine's (`ninfer-serve`, greedy,
the `qwen3_8_27b_nvfp4full-v2.ninfer` artifact) completions on the canary
suite (`crates/bench/src/canary.rs::CANARIES`), 32 greedy tokens per
prompt. `ignis-bench oracle compare` diffs a candidate engine's tokens
against it (§"Oracle (two levels)", spec `01-device-resident-forward`).

The canary suite itself (`ignis-bench canary`) judges sanity on everything a
turn generated. Its 64-token budget stays that way under the serving default,
so most canaries are read on the thinking channel and the report says so
(`answer=thinking-only`). Checking the *answer* is this fixture's job, token by
token with thinking off; the suite's own job is catching a degenerate engine,
which shows on either channel (GitHub #144).

Re-record it (needs the GPU and the reference stack, ADR 0006 — stop
`ignis-server` first):

```powershell
# 1. Start the reference engine on the same artifact, greedy, thinking off
#    (thinking on burns the whole token budget on the reasoning channel,
#    leaving no content tokens to compare). This is specific to the oracle,
#    which diffs answer tokens; `canary`, `ttft`, `g3` and `g4` read both
#    channels and need no such flag (GitHub #137).
F:\ai\q38\ninfer\build-ninja\apps\ninfer-serve.exe `
  F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer `
  --model-id qwen3.8-27b-nvfp4full-v2 --host 127.0.0.1 --port 8080 `
  --greedy --no-thinking --max-context 8192 --max-concurrency 1 --kv-capacity auto

# 2. Record the fixture.
cargo run -p ignis-bench -- oracle record `
  --endpoint http://127.0.0.1:8080 `
  --artifact F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer `
  --out crates/bench/tests/fixtures/oracle_canary.json --max-tokens 32

# 3. Self-check: compare the fixture against a fresh live recording (must be
#    100% — greedy + fixed seed on the same artifact is deterministic).
cargo run -p ignis-bench -- oracle compare `
  --fixture crates/bench/tests/fixtures/oracle_canary.json `
  --endpoint http://127.0.0.1:8080 `
  --artifact F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer

# 4. Stop ninfer-serve (frees the GPU) and commit the updated fixture.
```
