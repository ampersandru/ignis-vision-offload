# Benches and gate runs

## The G2 measurement instrument (P2-05, GitHub #87)

`ignis-bench ttft` measures time to first token at an **exact** prompt
length against any OpenAI-compatible endpoint, and `ignis-bench g2` turns
two such records into the G2 verdict (ADR 0015). Both engines are measured
by this one instrument; only the runs themselves need the GPU.

Every sample — the warmup included — gets its own deterministically
generated prompt, distinct from the first content token, and reads the
engine's own computed-prefill-token count back to prove the prefix was
cold. A sample whose computed prefill is not the prompt's own length is
**void** and fails its cell, and `g2` refuses a verdict over a record with
a void sample, a missing cell, or a session that does not match its
counterpart's.

The gate run (GitHub #88) needs the GPU exclusively and the reference
stopped and restarted around it (ADR 0006). One session, one shared
`--session` value:

```powershell
$Session = "g2-$(Get-Date -Format yyyyMMddTHHmmssZ)"
$Artifact = "F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer"

# 1. The reference, in the owner's production profile (hq-e8-2b KV, 1024
#    prefill chunk, CUDA graphs) -- the bar actually experienced, measured
#    as it is actually run (ADR 0015).
cargo run -p ignis-bench -- ttft `
  --endpoint http://127.0.0.1:8080 --artifact $Artifact `
  --cells 8192,32768 --label reference --profile "hq-e8-2b KV, 1024 chunk, graphs" `
  --session $Session --out .scratch/g2-reference.json

# 2. Stop the reference (it owns the GPU), start ignis-server on the same
#    artifact, then measure it on the same cells in the same session.
cargo run -p ignis-bench -- ttft `
  --endpoint http://127.0.0.1:8000 --artifact $Artifact `
  --cells 8192,32768 --label ignis --profile "BF16 KV, 1024 chunk, eager" `
  --session $Session --out .scratch/g2-ignis.json

# 3. The verdict. `--note` records what is *not* equal between the two
#    engines, rather than correcting for it.
cargo run -p ignis-bench -- g2 `
  --ours .scratch/g2-ignis.json --ref .scratch/g2-reference.json `
  --note "KV format differs: ignis BF16, reference hq-e8-2b" `
  --out .scratch/g2-verdict.json

# 4. Restart the reference.
```

Every ignis leg of a bench or a gate runs with the loop guard off:
`--model-loop-guard off`, or `LOOP_GUARD=off` on `make start` (`make config`
prints the flag). The server's default, `rewind`, takes a looping reasoning
back and nudges the model, so wherever it acts the output, the token counts
and the timing differ from a reference that has no guard (ADR 0052). The
in-process GPU tests that compare output (`openai_http_gpu.rs`,
`hq_canary_gpu.rs`, the `LiveServer` ones) build their server without a config
and run with it off; one built from flags or a config gets `rewind`, and a new
one that compares output passes `--model-loop-guard off`.

`ignis-server` takes the engine shape the run needs as flags (each over its
own env var, each validated before any loader work starts): `--model-prefill-chunk`
(default 1024, a nonzero multiple of 128), `--model-max-context` (default
40960 — a 32K prompt plus an 8K generation budget), `--model-kv-format` (`bf16`
or `hq-e8-2b`, default `hq-e8-2b` since GitHub #123 wired its attention
routes — ADR 0022's serving default) and
`--vram-kv-pool-bytes` (default: the KV pool policy's, ADR 0045 -- the rest of the
VRAM budget on the 27B, GitHub #210; 524,288 tokens shared by the lanes on
Flash-Next with its experts streaming; the plan, logged as
`ignis.runtime.vram_plan` or `ignis.runtime.flash_next_plan` with its
`kv_pool_policy`, `kv_pool_pages` and `kv_pool_tokens`, refuses a start that
cannot hold one `--model-max-context` sequence and a page per retained slot). The
pool is named in bytes or in tokens (`512Ktok`); the plan turns either into
pages at the model's own bytes per token (9,216 on the 27B under hq-e8-2b,
65,536 under BF16; 4,224 on Flash-Next under hq-e8-2b, KV and indexer keys).
On the 27B the resident-token capacity is logged at load as
`ignis.runtime.kv_pool`, so a run's KV profile is read off the events rather
than computed from a flag. A request that names no cap generates at most
`--model-default-max-tokens` (38,912; `0` = up to the context), and a test that
needs a request to run to the context sends `max_tokens` or that `0`.

Speculation is a load option too (GitHub #150): `--spec-backend dflash2
--spec-draft-tokens N` (N in 1..7, both flags or neither) binds the drafter's 66
`dflash2/*` objects and grows the prefill scratch by the drafter's context
append; its sequence pool carries the drafter window per slot (40 MiB each
slot, GitHub #152); `ignis.model.loaded` records it as
`speculation`. Absent, the load, its plan and its VRAM report are exactly what
they were before the option existed. `crates/core/tests/speculative_load_gpu.rs`
pins the VRAM delta; `crates/core/tests/dflash2_window_gpu.rs` checks a
prefill's window against a host f64 recomputation from the target's taps.
`crates/core/tests/speculative_round_gpu.rs` proves the verify round (GitHub
#153) with a fake drafter under the test-only `verify-only` backend, whose
sequence pool must be built for that backend too.
`crates/core/tests/dflash2_round_gpu.rs` proves the real drafter in the round
(GitHub #155): greedy spec-on against spec-off on the canary prompts at widths
1, 4 and 8, its graph replay against eager, a lane at extent 0 leaving its
window untouched -- and, continued by a round or by a prefill, writing that
round's anchor into its window before anything else (GitHub #157, checked
against a lane that never ran at extent 0) -- and a mid-generation
snapshot/restore. Every stream is cut
at the turn's first `<|im_end|>`/`<|endoftext|>`, where the server stops: past
it the model loops on chat markers and the verify and decode tilings part by
several logits on text no request emits. It prints acceptance
per round and the mean committed tokens per full-window round against the
reference's 3.4–5.75 band — read that line from the profile's output; a mean
below the band is a finding to file, not a test failure. The drafter's rewrite
checkpoint is retired (blob format 5): `dflash2_window_gpu.rs` asserts a
snapshot no longer lists it. `crates/runtime/tests/cuda_leaf_dflash2_gpu.rs` drives the same
drafter end to end through `RuntimeCompute`: the leaf passes no drafts, and
every round's speculative counters match the lane's extent and committed run.
The acceptance replay of recorded agent turns (GitHub #267,
`crates/core/tests/dflash2_replay_seams_gpu.rs`) walks a turn in
teacher-forced rounds. The recorded token after a lane's run is its pending
token, so it must be a vocabulary id; the pad past a recording's end is taken
only after a stop id, and leaves no pending token until a prefill gives one.
Only greedy lanes may be forced, and only on 27B loads.

Vision is a load option too (GitHub #177): `--vision-enabled` (with
`--vision-max-tokens N`, default 32,768 merged tokens) binds the 333 `vision/*`
objects in their stored Q4/Q5/Q6/W8/BF16 formats and reserves, inside
`ignis_model_load` and so before the sequence pool is built, the encoder
workspace for one item of `min(item bound, V)` and the embedding pool floored
at one `[5120, V]` item for `V = min(max_context, N)`. The item bound is the
processor's (`ProcessorOptions::max_item_tokens`, 16,384 for this artifact),
which the server hands to the load as `vision_item_max_tokens`;
`kernel/tests/test_vision_workspace.cpp` pins the encoder's workspace for a
32,768-token item. The encoder workspace is not an arena of its own
(GitHub #212): the prefill scratch is sized for the larger of the two, and
the VRAM plan shows it as one `workspace` line. `ignis.runtime.kv_pool`
reports what vision adds — the output transient plus the scratch's growth —
as `vision_reserved_bytes` beside the pool's `token_capacity`.
`crates/runtime/tests/vision_shared_workspace_gpu.rs` checks, on a DFlash2
load, that an image request, an image prefill interleaved with decoding text
lanes and a three-image prompt whose encodes land between prefill chunks
generate the tokens recorded before the arena was shared
(`IGNIS_RECORD_VISION_WORKSPACE_TOKENS=1` re-records them).
Absent, nothing vision-related is
bound or allocated. `crates/core/tests/vision_load_gpu.rs` pins the VRAM delta
(weights plus the reported reservation) and prints the reservation at the
default envelope; `kernel/tests/test_model_load_vision_options.cpp` pins the
envelope check and the leaf's vision schema host-side.

Since GitHub #195 `--vision-enabled` combines with `--spec-backend dflash2`: both scopes bind
in one load, the drafter's context append takes the span's KV positions (what
the reference's own prefill sink captures on a multimodal span), and the verify
round rotates its columns at `position + rope_delta` from its own staging, the
way a decode round already did. `crates/server/tests/vision_dflash2_gpu.rs` is
that load's check — the canary floor on it, greedy verify rounds against the
same load's own plain decode rounds, and the acceptance on the text after an
image against the 3.4–5.75 band. The verify round's rope staging exists only
when vision is loaded, so a text-only speculative load allocates nothing new.

Since GitHub #193 multimodal requests share prefixes under a media-aware
identity (the match key, ADR 0029). `crates/server/tests/vision_prefix_reuse_gpu.rs`
is its check on the real model: siblings over one system prompt sending
same-size flat swatches — identical token ids — where the other colour shares
only the system block and still names its own colour, and the same colour
reuses past the image, through a retained checkpoint with prompt reuse on and
through the sibling prefix cache alone with it off.

`crates/server/tests/vision_mixed_load_gpu.rs` (GitHub #181) is all of the
above at once, in the serving shape: one load behind the HTTP router with
`--vision-enabled`, `--spec-backend dflash2`, prompt reuse on, a 128-token prefill chunk, and
every image fetched by URL from a loopback server. Three text lanes decode
while an image request prefills between their rounds; same-size swatch
siblings each name their own colour; and with the KV pool sized to exactly one
4,096-token context, an Interactive request reserving all of it evicts the one
Agent image request to KV-RAM, which is restored and still describes its image.
The request log it prints names the evicted and restored request id and each
request's `spec.*` counters.

**The verify round's rope delta is a deliberate departure from the reference.**
The reference's DFlash2 round passes its proposal positions for both the cache
and the rotation and carries no rope delta in its decode ingress at all, while
its ordinary decode batch and its MTP round both add the sequence's — so on the
reference a multimodal sequence's verify columns and its decode rounds
disagree. The spec (`docs/specs/vision/01-image-input.md`, §Compute seam)
asks for `rope_position = position + rope_delta` on *every* decode and verify
round, and ignis follows the spec. Expect a live/live comparison against the
reference with `--vision --spec dflash2` to show the reference's own
divergence, not ours; the vision canary fixture was recorded without
speculation, so it stays the right oracle either way.

**BF16 is the oracle format (ADR 0022).** Every correctness check in the GPU
profile asks for it by name — `--model-kv-format bf16` at the server, and
`ignis_core::KvFormat::Bf16` on both `load_qwen38_27b` and `SeqPoolBudget`
in a test that drives the leaf directly. A test that inherits the default
runs hq, which is right for the serving-shape checks (the HTTP surface, the
TTFT instrument, the `CudaLeaf` smoke test) and wrong for anything carrying
a derived tolerance.

## The multimodal TTFT cell (GitHub #181)

`ignis-bench ttft --image <png>` adds one cell to a TTFT record: a fixed image
plus a short question (`--image-question`, default "Describe what this
screenshot shows in one sentence."), sent as content parts — the text first,
then the image as a `data:` URI. An error detector for the whole vision path,
not a gate. With `--image`, `--cells` may be omitted.

The committed image is `crates/bench/tests/fixtures/vision_ttft/screenshot.png`,
a 1280x800 editor screenshot drawn by `tools/vision-ttft/screenshot.py`
(deterministic; re-running it is a no-op). It sits on the 32-pixel grid, so it
expands to 1,000 vision tokens and the cell is 1,027 tokens with the default
question — `crates/bench/tests/ttft_image_real_frontend.rs` pins that against
the real frontend, and the reference counts it the same.

Cold in everything, like a text cell: each prompt (the warmup's too) starts
with its own nonce, and its image is the fixture with the top-left pixel
changed, so neither a prefix cache nor either engine's digest-keyed
preprocessing cache can serve it; the size, and so the length, is unchanged.
The record names the image (SHA-256, size, question), and `g2` refuses to pair
cells of one length over different images, or an image cell with a text cell.

A record carrying the image cell cannot be compared with one that has none, so
the image cell is measured as its own record. The session is
`scripts/vision-ttft-session.sh <out-dir>`, run from the repository root on a
free card after a release build: the reference with `--vision` (the owner's
hq-e8-2b-262k preset), ignis with `--vision-enabled`, and ignis without it, **each
launched twice** (ADR 0021). Every launch writes a text record (1024, 8192 and
32768 tokens, cut from the ninfer corpus) and, with vision, an image record;
`scripts/ttft-pool.py` pools each engine's two launches into one record, and
`g2` compares the pooled records: image against image, text against text, and
ignis's text with `--vision-enabled` against without. The 1.5 threshold `g2` prints is
G2's own; for this cell the ratios are error detectors.

**Measure a launch once.** The prompts are fixed — the same nonces, the same
changed pixels — so a second `ttft` run against a live engine is served from
its prefix and media caches, and ignis reports no cached tokens for the void
rule to catch it. Restart the engine instead.

**Watch the card's headroom.** Both ignis loads at 262,144 context with
DFlash2 took ~29.8 GiB of their own when measured, before #215 (retained
state was then sized from what was free after load), so the desktop's own VRAM decides whether the card
oversubscribes; when it does, every cell comes back slower and erratic rather
than failing. The script samples `nvidia-smi` beside every launch — distrust
one whose peak reaches the total
([finding](../../findings/2026-09-16-vision-ttft-live-live.md)).

## The KV-format A/B: 2x2 over engine and format (GitHub #139)

What it answers: how much the hq-e8-2b KV format costs against BF16, whether
that cost is ours or the format's, and what the card is actually doing while it
is paid. The result of the 2026-09-13 run is
[`docs/findings/2026-09-13-hq-vs-bf16-decode-cost.md`](../../findings/2026-09-13-hq-vs-bf16-decode-cost.md);
this is how to redo it.

**The one rule that makes it a comparison.** Give every leg the *same KV token
capacity*. A BF16 token costs 65,536 bytes against hq-e8-2b's 9,216 (7.11x, see
[hq-e8-2b KV capacity](../../findings/2026-09-11-hq-e8-2b-kv-capacity.md)), so the
same byte budget buys 7.11x more pages under hq and a "just change the format"
run confounds the format with its page count. 65,536 tokens is the value to
pick: it is 1,024 pages, the geometry spec 03's ITL table and #114's cap
arithmetic assume, and the ITL fixture's peak of 65,344 tokens fits it with 192
to spare. Under hq that means asking for the smaller pool explicitly
(`--vram-kv-pool-bytes 576M`), because its default 4 GiB buys 465,984 tokens.

For the record: shrinking the hq pool from 7,281 pages to 1,024 moved ITL p50 by
0.04% and p95 by 0.05%, so the page count is not itself a performance term —
but that is a measured result, not an assumption to skip the matching on.

Needs the GPU exclusively (ADR 0006) and `ninfer-serve` stopped for the ignis
legs. Four runs, ~4 minutes each. Launch the legs with the explicit commands
below: `make start` brings up the G5 configuration (262144 context, DFlash2),
a different engine from these legs.

```powershell
$Artifact = "F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer"
$Corpus   = "F:\ai\q38\ninfer\bench\fixtures\bench_corpus.ids"
cargo build --release -p ignis-server -p ignis-bench --features cuda

# Per leg: start the telemetry sampler, start the engine, run the cell, stop both.
# Stop the sampler as soon as ignis-bench exits -- see the pitfall below.
nvidia-smi --format=csv,noheader -l 1 `
  --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,temperature.gpu `
  > .scratch/gpu-<leg>.csv

# ignis, hq-e8-2b / BF16 (--model-kv-format bf16 --vram-kv-pool-bytes 4G: 65,536 BF16
# tokens; with no --vram-kv-pool-bytes the pool is the rest of the VRAM budget)
.\target\x86_64-pc-windows-msvc\release\ignis-server.exe --model-artifact $Artifact `
  --server-bind 127.0.0.1:8000 --model-kv-format hq-e8-2b --model-max-context 40960 `
  --vram-kv-pool-bytes 576M --model-prefill-chunk 1024 --server-request-timeout 900 `
  --model-idle-unload 0 --model-loop-guard off

# the reference, same two formats
F:\ai\q38\ninfer\build-ninja\apps\ninfer-serve.exe $Artifact `
  --host 127.0.0.1 --port 8080 --kv-dtype hq-e8-2b --max-context 40960 `
  --kv-capacity 65536 --max-concurrency 8 --prefill-chunk 1024

# the cell, once per leg (port 8000 for ignis, 8080 for the reference)
.\target\x86_64-pc-windows-msvc\release\ignis-bench.exe g3 `
  --endpoint http://127.0.0.1:8000 --artifact $Artifact --corpus $Corpus `
  --label ignis --profile "hq-e8-2b KV, 1024 chunk, max-context 40960, 65536-token pool" `
  --session "kvab-<leg>-$(Get-Date -Format yyyyMMddTHHmmssZ)" `
  --out .scratch/g3-<leg>.json
```

Each engine logs the capacity it resolved — `ignis.runtime.kv_pool`'s
`page_count`, and ninfer's `KV capacity explicit resolved=... tokens pages=...`.
Check both say 1,024 pages / 65,536 tokens before believing the leg.

Then reduce each capture over the ITL cell, which is the last cell a `g3` run
measures and whose length the record itself states
(`window_end_ms - window_start_ms`):

```powershell
python scripts/gpu-telemetry-summary.py .scratch/gpu-<leg>.csv --label "<leg> ITL" --last 90
```

**Pitfall: a sampler left running past the cell dilutes the window.** Idle
samples pull the means down hard — the same hq leg reads 96% util / 306 W over
its ITL cell and 83% / 274 W if eleven seconds of idle tail are included. Stop
the sampler when the bench exits, or pass `--from`/`--to` instead of `--last`.

**Pitfall: a cold machine is not a measurement.** The first `g3` run of a
session came in 14% under the records committed the day before and converged on
them over four runs (C=1 46.6 -> 51.3 -> 51.0 -> 54.2 against 55.6). Discard the
first leg, or run one throwaway before the matrix.

**This is not live/live and cannot decide a gate.** Each leg is its own session,
so `g3-gate` will refuse to build a verdict from these records, by design (ADR
0015, ADR 0021). The A/B answers a question about *where time goes*; a verdict
needs both engines measured back to back under one `--session`.

## The harness's own request deadline (GitHub #138)

Every `ignis-bench` subcommand drives its endpoint through one
`HttpEndpoint`, which sets an **explicit** per-request deadline: 1800 s by
default, and `IGNIS_BENCH_REQUEST_TIMEOUT=<seconds>` overrides it (`0`
removes it entirely). Each run prints the deadline it is using on stderr
before its first request.

This is not a knob a normal run needs — it exists because the *implicit*
one cost a gate session. `reqwest::blocking::Client::new()` carries an
undeclared 30-second total timeout that covers reading the response body,
so a streamed measurement request lives entirely inside it. The G4 needle
cell at 131,072 tokens (~30 s of prefill on this hardware) died on it
against **both** engines, every attempt, reporting `read SSE: error
decoding response body` — which reads like an engine fault and was chased
as one across four launches. The 65,536-token cell, at ~15 s, passed every
time.

Two consequences for anyone measuring here: the deadline is now the
harness's own declared number rather than a library default, and a request
that does hit it says so by name instead of surfacing as a decode error.
Raise it (or set `0`) for a cell whose legitimate wall time could approach
half an hour; never lower it to "make a run finish".
