# Speech recognition in ignis — the feature, its tickets, and how to run them

GitHub: master #350; one ticket per spec, 01-06 = #351-#356.

The feature-level document the speech tickets decompose
(`docs/agents/issue-tracker.md`). Read this first, then the ticket's spec.

## What and why

ignis gets a **speech model**: **Qwen3-ASR**, so a user can dictate a prompt
instead of typing it, in the Playground or in their own coding agent (Q7).
The audio never leaves the machine. The model is a third **model kind**:
it takes audio and gives back its transcript, and never converses (Q8, Q9).
It is loaded beside the text or image model, never resident in VRAM. It runs
on the CPU, and in phase 3 also on free KV pages lent to it for one
**transcription** (ADR 0066). It is loaded at first use and unloaded when
idle (Q53).

Surfaces:
- `POST /v1/audio/transcriptions`, OpenAI's shape, the de-facto standard
  (Q2, Q23);
- `wss://…/v1/realtime?intent=transcription`, the OpenAI Realtime GA
  transcription subset (Q20);
- the Playground's **dictation**, always over Realtime (Q3, Q33).

**Decisions:**
- [ADR 0066](../../adr/0066-the-speech-model-waits-on-the-host-and-borrows-free-kv-pages.md):
  the speech model waits on the host and borrows free KV pages.
- [ADR 0067](../../adr/0067-the-speech-models-cpu-executor-is-ours-in-rust.md):
  the speech model's CPU executor is ours, in Rust.
- The grill of 2026-10-23/24, Q1-Q68 and the validated mockup of the
  Playground (Q17). Q59-Q63 are the owner's review of the vendoring the
  grill first chose (Q27, Q41): the CPU executor is ours, in Rust, after a
  spike that met every criterion fixed before it ran, and the GPU loan
  builds on our leaf alone. The log is kept untracked in
  `F:/ai/opencode/inference/.scratch/speech-grill-2026-10-23/`, with the
  mockup (`dictation-canvas/`). Every acceptance criterion in specs 01-06
  cites the decision it comes from, as `(Qnn)`. Q48 sets this packaging:
  one spec set, one ticket per spec, two ADRs, four glossary entries.
- The owner's decisions (2026-10-24) on every detail the grill left open,
  cited as `(owner, 2026-10-24)`:
  - the catalog ids;
  - the Hugging Face repos;
  - the provisional-tail extension's names;
  - the executor's crate, `crates/speech`;
  - the hold-to-talk shortcut;
  - every detail the specs first carried as a proposal, the config field
    names among them.
- The evidence: [Qwen3-ASR on the CPU](../../findings/2026-10-24-qwen3-asr-on-cpu.md)
  and [Rust quantized matmul matches ggml on the CPU](../../findings/2026-10-24-rust-quantized-matmul-matches-ggml.md).

## The model

| | 0.6B | 1.7B |
|---|---|---|
| audio encoder (AuT) | 18 layers, d 896, 14 heads of 64, 186 M params | 24 layers, d 1024, 16 heads of 64, 317 M params |
| decoder | Qwen3-0.6B: 28 layers, hidden 1024, GQA 16/8 of 128 | Qwen3-1.7B: 28 layers, hidden 2048, GQA 16/8 of 128 |
| front end | 128-bin log-mel, n_fft 400, hop 160, 16 kHz | same |
| audio tokens | 13 per second (1 s chunks of 100 frames); encoder attention bidirectional inside 8 s windows (104 tokens) | same |
| KV | 112 KiB per token | 112 KiB per token |
| longest clip | 1,200 s (15,600 audio tokens) | same |
| prompt | `<\|im_start\|>system\n{context}<\|im_end\|>\n<\|im_start\|>user\n<\|audio_start\|>…<\|audio_end\|><\|im_end\|>\n<\|im_start\|>assistant\n` | same |
| output | `language <Name><asr_text>…`, or `language None<asr_text>` | same |
| licence | Apache-2.0 | Apache-2.0 |

## Variants and the catalog (Q47, Q49)

| served id | weights | on the owner's dictations (Italian WER, no context / context) | i9, 10 threads, x real time | role |
|---|---|---|---|---|
| `qwen3-asr-1.7b` | 1.7B: decoder Q4_K_M + encoder Q8_0, ~1.64 GB | 3.7% / 2.4% | 4.8x | `speech.model: auto` (Q49) |
| `qwen3-asr-1.7b-q8` | 1.7B: decoder and encoder Q8_0, ~2.52 GB | 3.3% / 2.6% | 3.1x | in the catalog (Q49) |
| `qwen3-asr-0.6b` | 0.6B: decoder and encoder Q8_0, ~1.02 GB | 9.1% / 9.2% | 6.7x | "light", the fallback for weak CPUs (Q34, Q49) |

0.6B Q4_K_M (12.7%, fails Q34's 10% gate) and the bf16 files are rehosted
but not in the catalog (Q49). Every file is converted by us from the official
Qwen weights with llama.cpp b11840's converter. Our Q8_0 is compared with
ggml-org's as a check. Every file is rehosted in the owner's Hugging Face
repo, with Qwen's LICENSE, attribution and a modification notice. The upload
needs the owner's OK and a privacy scan at upload time (Q47).

**Ids and repos** (owner, 2026-10-24):
- **The served ids follow the 27B's:** the bare id is the recommended artifact
  of its size, as `qwen3.8-27b` is, and a suffix names a variant, as
  `qwen3.8-27b-16g` does. Only one 0.6B is in the catalog, so it takes the
  bare id.
- **The repos** are `gpillon/Qwen3-ASR-1.7B-GGUF-ignis` and
  `gpillon/Qwen3-ASR-0.6B-GGUF-ignis`. They follow `<upstream model>-<format>-ignis`
  (`gpillon/Qwen3.8-27B-trellis-3bpw-ignis`), one per size, as Qwen and
  ggml-org publish them.
- Each repo holds its size's decoder and encoder files in bf16, Q8_0 and
  Q4_K_M: the catalog's entries and the rehosted rest.

## Phases (Q33, Q37, Q56)

1. **Phase 1:**
   - the CPU executor, with incremental encode, audio-prefix prefill and
     partial decode with rollback (01);
   - `/v1/audio/transcriptions` (02);
   - Realtime with manual commit (03);
   - the Playground (04).
2. **Phase 2:** `server_vad`, with Silero on the CPU (05).
3. **Phase 3:** the GPU loan (06).

Q56 swapped phases 2 and 3: `server_vad` first, the loan last.

Q34 makes the CPU go **only** through incremental Realtime transcription.
Q33 puts the Playground on Realtime for push-to-talk too. So Realtime is on
phase 1's critical path, and Q16's "batch first, live after" no longer
holds.

## The tickets

| spec | what | blocked by | can start now |
|---|---|---|---|
| [01](01-cpu-executor.md) | the executor in Rust (GGUF reader, quantized kernels, the Q4_K layout, the pool), its two graphs, the pipeline and its incremental API, runtime AVX2 detection, threads, the oracle, the GGUF files, the HF repo, catalog entries | — | yes, everything |
| [02](02-transcriptions-api.md) | the speech model in the server (`speech.model`, availability, `/v1/models`, compat names), `/v1/audio/transcriptions`, formats, honoured-or-refused, deadlines, logging, metrics, the amendments below | 01 for the real executor | yes, against a fake executor |
| [03](03-realtime.md) | `/v1/realtime?intent=transcription`: sessions, PCM, manual commit, partials, the provisional-tail extension, context at session open | 02's speech slot; 01's incremental API for the real path | yes, against a fake executor |
| [04](04-playground.md) | dictation in chat, Decide and the image studio, from the Q17 mockup | 03 | its components, against a fake socket |
| [05](05-server-vad.md) | phase 2: `server_vad` | 03 | the VAD port and its oracle |
| [06](06-gpu-loan.md) | phase 3: the KV-page loan, the CUDA adapter, its kernels | 01-03 | the plan and the policy, on CPU tests |

**Order.** 01 and 02 start together; 02 is built against a fake executor,
as image serving was against a fake leaf. 03 follows 02's speech slot. 04
follows 03. Phase 1 closes with 03's latency criterion and 01's quality gate
on the real executor. Then 05 (phase 2), then 06 (phase 3) (Q56).

## Configuration: the `speech` group

Every field is declared once (ADR 0046). Every threshold is a field with a
default, never a constant (the owner's rule).

| field | default | from | spec |
|---|---|---|---|
| `speech.model` | `auto` (the 1.7B Q4_K_M entry); an id; or `off` | Q39, Q49 | 02 |
| `speech.compat_names` | `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-transcribe`, `gpt-live-transcribe` | Q11, Q24 | 02 |
| `speech.threads` | `auto` = physical cores − 1 | Q38 | 01 |
| `speech.max_clip_seconds` | 1200 | Q19, Q36 | 02 |
| `speech.context_max_tokens` | 512 | Q50 | 01, 02 |
| `speech.pinned_weights` | `false` (memory-mapped file) | Q31 | 01 |
| `speech.idle_unload` | 300 s without a transcription, `model.idle_unload`'s default | Q53 | 01 |
| `speech.partial_interval_ms` | 2000 | Q37 | 01, 03 |
| `speech.partial_cost_factor` | 1: the next refresh waits at least this many times the last one's compute time; 2 and 3 measured no faster (spec 03, As built) | Q76 | 03 |
| `speech.rollback_tokens` | 10 (Q75; the official package's `unfixed_token_num` is 5) | Q37 | 01 |
| `speech.unfixed_partials` | 2 (the official package's `unfixed_chunk_num`): the first partials fix nothing. A field since the integration fixes, read with `rollback_tokens` when a session opens | #351 | 01, 03 |
| `speech.max_sessions` | 4 open Realtime sessions | Q44 | 02, 03 |
| `speech.max_queue` | 4 waiting batch requests | Q44 | 02 |
| `speech.vad_threshold`, `speech.vad_prefix_padding_ms`, `speech.vad_silence_duration_ms` | 0.5, 300, 500 | Q22 | 05 |
| `speech.vad_leading_silence_ms` | 500: the digital silence the model reads before each `server_vad` item, never shown to the client (AC 7 met with it: 4.68% against the manual commit's 4.03%; 7.58% without) | Q77 | 05 |
| `speech.loan_grace_seconds` | 30 | Q32 | 06 |
| `speech.loan_wait_seconds` | 5 | Q35 ("a short wait") | 06 |

Every field name above, and every default no Q gave, is the owner's
(2026-10-24), but `speech.partial_cost_factor` and
`speech.vad_leading_silence_ms`: the owner made them fields with defaults
chosen from measurement (Q76, Q77, 2026-10-25), and the names are the build's,
in the group's style.

## Requirements (Q28, Q34)

- **A CPU with AVX2** (Haswell, 2013, or later) for the CPU executor. The
  executor detects AVX2, FMA and F16C at run time and picks its kernels by
  what it finds; one binary serves every x86-64 CPU. Without AVX2 the speech
  model runs only on a loan, or is unavailable (ADR 0066, 0067).
- **The 4-core figure goes into the requirements** (Q34). On a 4-core
  container sharing the i9's L3 and memory bandwidth, our CPU executor at
  `speech.threads: auto` (3 threads there) runs `qwen3-asr-1.7b` at 2.9-3.0x
  real time and `qwen3-asr-0.6b` at 3.6-3.7x; `qwen3-asr-1.7b-q8` runs at
  1.6-1.7x, under the 2x go band (1.9-2.1x at 4 threads, where llama.cpp
  b11840 does 1.9-2.2x). On the i9 at `auto` (9 threads) the three run at
  4.1-4.7x, 2.7-3.1x and 6.0-6.6x (spec 01, As built). That container is
  optimistic, and a real budget 4-core with slower RAM can fall under 2x. The
  documented fallback for a weak CPU is `speech.model` set to the 0.6B Q8_0
  entry; `qwen3-asr-1.7b-q8` reached 2x only at 4 threads on the 60 s clip
  and stayed under it on the 10 s clip. Live partials every 2 s
  cost more than real time on four cores with every variant (spec 01, As
  built; spec 03's concern).
- **Live dictation's commit is slower than the 2 s target** (spec 03, AC 13,
  measured through the real Realtime route on a 67 s Italian dictation at
  `speech.threads: auto` and the default refresh): 2.4-4.4 s from the commit
  to `completed` on the i9 (two runs at each `speech.partial_cost_factor`, 1 to
  3; the eleven dictations' median 1.7-2.2 s, 3.2 s in the one run on a
  slower machine) and 5.8-6.1 s on the 4-core container at factor 1 (median
  3.8 s over the eleven). The executor's own commit takes 1.9-2.0 s on the i9
  and 4.0 s on the container; the rest is the wait for a refresh in flight.
  No factor measured better than 1 (Q76), so 1 is the default. A slower or
  busier machine adds more: the first measurement, on a machine whose window
  pushes took about 60% longer, gave 6.7 s and 11.8 s. Not met; the levers
  left are in spec 03, As built.
- **Host RAM:** 1.4 GB (0.6B) to 3.1 GB (1.7B Q8_0) resident while the
  model is in use, KV at a 4,096-token context included. Nothing once
  `speech.idle_unload` has unloaded it (Q53).

## Existing artifacts the implementation must amend

Read at `711b1d85`; every line checked in this worktree.

| where | what it says today | change | spec |
|---|---|---|---|
| `crates/server/src/openapi.rs:79-87` (84-85, 86-87) | not served: "audio in or out (the model has no audio tower or head)", "more than one loaded model" | audio *in* is served. A speech model is loaded beside the other. Audio *out* stays not served. `/v1/audio/translations` joins the list | 02 |
| `crates/server/tests/openapi_http.rs:284-288` (286) | pins `"audio"` and `"more than one loaded model"` in that description | pins the new wording | 02 |
| `README.md:541` and `README.md:544-546` | "Audio, in or out"; "More than one model loaded at once" | as the OpenAPI description, plus a speech section | 02 |
| `docs/specs/server/09-honoured-or-refused.md:181-182` | the not-served list | as above; the rows at `:115` (`modalities`) and `:117` (`audio`) stay, since they refuse audio out | 02 |
| `docs/specs/responses-api/01-responses-streaming-and-websocket-mode.md:421` | "The Realtime API (`/v1/realtime`)" out of scope | the transcription subset is served by spec speech/03. Realtime conversation sessions stay out | 03 |
| `crates/server/src/openai_fields.rs:161-166`, `:174-179` | `modalities` other than `["text"]` and a non-null `audio` refused on chat | unchanged (no audio out); reviewed, the wording still holds | 02 |
| `web/src/api/model.test.ts:197` | `kind: "audio"` is a kind the client does not know | `kind: "speech"` is known; `"audio"` stays the unknown example | 04 |
| `crates/server/src/api.rs:1818` *(found)* | `GET /v1/models`: `kind` is "`text` or `image`"; "A server serves one model at a time" | `speech`; the speech model listed beside the loaded one | 02 |
| `crates/server/src/api.rs:2054` *(found)* | a switch: "every other `/v1` route answers `503 model_switching`" | the speech routes keep serving on the CPU during a switch (Q40) | 02 |
| `crates/core/src/compute.rs:58-63` *(found)* | `ModelKind { Text, Image }` | a third kind | 02 |
| `CONTEXT.md` **Servable model** *(found)* | "What `GET /v1/models` lists" | the speech model is listed without being switched to | 02 |
| `CONTEXT.md` **Idle unload** *(found)* | use is "a request in flight on a model route" | a transcription is not use of the text model (Q40) | 02 |

## Environment

- **Study data:** `H:\speech-study\` (bulk data on H:, never Y:):
  - `results\`: every CSV and report input;
  - `dictations\`: the owner's eleven recordings with their references.
    Private: never committed, never quoted beyond short error examples;
  - `models\`: the GGUF files measured;
  - `src\llama.cpp\`: a read-only clone at tag b11840 (`36e36e63a`), the
    comparison and benchmark baseline, never in the tree or the build
    (ADR 0067);
  - `rust-spike\`: the spike's source and raw results;
  - `scripts\`, `logs\`.
- **Python:** `H:\speech-study\venv` (torch 2.14.1+cpu, transformers
  5.20.2). The oracle's reference checkpoints are `Qwen/Qwen3-ASR-1.7B-hf`
  @`bcd2b5b7` and `Qwen/Qwen3-ASR-0.6B-hf` @`7f1569a4`.
- **Worktrees:** `../.inference-qwen-worktrees/<branch>/`, one per ticket.

## CPU and GPU runs

- **CPU measurements:** never two at once, and never beside another heavy
  CPU job. Each run notes the load before it starts, as the finding's runs
  did.
- **GPU** (Q38's validation in spec 02, and spec 06): `make gpu-status`
  first, then the shared lock (`../.inference-qwen-worktrees/.swarm/gpu-lock.sh`). One run on
  the card at a time. ninfer may be stopped; the owner restarts it.

## Rules

- **Tests:** every change ships with a test, and `cargo test` must pass
  workspace-wide. GPU tests are `--ignored` and run under the GPU profile.
- **Formatting:** never `cargo fmt`; diffs are semantic only.
- **Copied code** (ADR 0051): only MIT, BSD, Apache-2.0 or MIT-CMU.
  Provenance goes in the file header and in `NOTICE` or `kernel/NOTICE`. The
  CPU executor copies none (ADR 0067).
- **Native code outside `cuda`** (Q63, Q42): the executor is pure Rust and
  adds no native code. The C a build without `cuda` already compiles
  (`ring`, and the oniguruma and esaxx parts of `tokenizers`) goes through
  the `cc` crate, with no CMake. The one candidate for more is libopus,
  through `opus`/`opusic-sys` and the `cmake` crate, for 02's Opus input. The
  owner accepts it if it does not complicate the build or cost too much: it
  comes in phase 1 only if a verification build passes on both targets,
  otherwise later. That verification weighs that it would be the only CMake
  in a build without `cuda`.
- **Names** (Q57, owner): OpenAI's names and naming style only where a
  field, event or value is there for compatibility with OpenAI clients.
  What ignis adds carries ignis's names: the provisional tail's opt-in
  `session.ignis.provisional_tail` and its event
  `ignis.input_audio_transcription.provisional` (03), the `ignis_` metrics
  and the `ignis.` log events.
- **Privacy** (Q13): audio is never written to disk or to a log. A
  transcript follows the rule for prompt text (`docs/specs/logging/spec.md`
  §29: not logged by default).
- **Merging:** a ticket's branch merges into local `main` only when every
  criterion holds. Never push, never file or close an issue without the
  owner's word.

## Nothing left undecided

Every detail the specs first carried as a proposal was decided by the owner
on 2026-10-24, and each is cited `(owner, 2026-10-24)` where it appears.

The four details the rewrite after Q59-Q63 left open were decided the same
day:
- Q64: no setting serves the scalar kernels, not even for debugging (01);
- Q65: the whole executor's gate is Q34's go band on the i9 and on the
  4-core container, and pooled WER within 1 pt of llama.cpp b11840's on the
  same clips; the ratio to llama.cpp's time is reported, not gated
  (01, acceptance 24);
- Q66: NEON kernels for aarch64 in a later ticket, after phase 1 (01);
- Q67: no AVX-VNNI or AVX-512 kernels in phase 1 (01).
