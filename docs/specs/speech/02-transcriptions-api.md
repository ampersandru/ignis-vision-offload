# 02 - the transcriptions route, and the speech model in the server

GitHub: #352 (master #350). Phase 1. Feature: `spec.md`. Built against a fake
executor first; 01 for the real one.

ADRs: 0066 (the speech model), 0058 (deadlines), 0049 (idle unload), 0047
and 0064 (catalog, aliases), 0046 (config v2), 0017 (metrics); spec
server/09 (honoured or refused).

Decided 2026-10-23/24 in the speech grill. Details the grill left open were
decided by the owner on 2026-10-24, cited as `(owner, 2026-10-24)`.

## Problem Statement

- **One model in the server.** `Server.active` holds one `ActiveModel`, a
  `Runner::Text` or `Runner::Image` (`crates/server/src/active.rs:86-92`).
  `ModelKind` is `Text | Image` (`crates/core/src/compute.rs:58-63`).
  Nothing can hold a second model.
- **No audio route.** The only audio code refuses: `modalities` and `audio`
  on chat (`openai_fields.rs:161-166, 174-179`). The OpenAPI description and
  the README list "audio in or out" and "more than one loaded model" as not
  served (`spec.md`'s amendment table).
- **The serving gate** answers every `/v1` route `503 model_switching` during
  a switch (`api.rs:2054`). Q40 wants transcription served then.
- **Clients** call `POST /v1/audio/transcriptions`: Aider with `whisper-1`
  hard-coded, Cline through a provider base URL.

## Solution

### The speech model beside the active one

- **Its own slot on `Server`**, beside `active`, not a third `Runner`. Its
  state: ready, unavailable with a reason, or off (ADR 0066). `ModelKind`
  gains `Speech`.
- **`speech.model`** (Q39, Q49):
  - `auto` (the default) is the 1.7B Q4_K_M catalog entry;
  - an id is that catalog entry;
  - `off` is no speech model.
  The model is enabled when its files are on disk. Nothing is ever
  downloaded by itself.
- **Unavailable, and why** (Q35, Q39), in the words the Playground shows
  (spec 04):
  - turned off (`speech.model = off`);
  - no files, naming `ignis model download <id>`;
  - the CPU lacks AVX2 and no loan fits.
- **`GET /v1/models`** lists the speech model in `data` only while it is
  available, with `kind: "speech"` (Q11, Q39, Q55). Available means it can
  serve, whether or not it is loaded right now (Q53).
  - **Its place** (owner, 2026-10-24): after `data[0]`, which stays the text
    or image model, with `active: true`.
  - **The reason** (Q55): a top-level `speech` object,
    `{"available": …, "reason": "…"}`, always present, beside `status` and
    `last_switch`. The Playground can then disable the mic before the press.
- **The rest of the server's lifecycle** (Q40):
  - the speech routes are outside the switch's serving gate, and serve on the
    CPU during a switch, beside the image model and with the text model
    idle-unloaded;
  - a transcription does not reset the text model's idle clock.

### Which model a request names (Q11, Q24)

| `model` | answer |
|---|---|
| absent | the speech model |
| the speech model's served id | the speech model |
| one of `speech.compat_names` | the speech model; `model` is rewritten to its served id before anything reads it, as ADR 0064's alias is |
| a text or image model's id | `409 model_kind_mismatch`, naming both kinds (Q11 says 400; the repo's `model_kind_mismatch` is 409 everywhere) |
| anything else | `404 model_not_found`, the existing rule (`api.rs:577`) (owner, 2026-10-24) |

`speech.compat_names` defaults to `whisper-1`, `gpt-4o-transcribe`,
`gpt-4o-mini-transcribe`, `gpt-transcribe` and `gpt-live-transcribe` (Q24).
These are not catalog aliases: the catalog's `aliases` take only a model
family's own id, and only below a card class (ADR 0064). A compat name
equal to a served id is refused at start, and compat names are not listed on
`/v1/models` (owner, 2026-10-24). `/v1/audio/translations` is not served (Q24):
`404`, like the other routes not served by choice.

### `POST /v1/audio/transcriptions` (multipart)

Every field is honoured, inert, or refused with a 400 naming it in `param`.
None is dropped in silence (spec server/09).

| field | honoured | refused | from |
|---|---|---|---|
| `file` | WAV, FLAC, MP3, AAC (symphonia); the Ogg and WebM containers demuxed; Opus decoded only if the verification build of libopus passes (`spec.md`, Q63) | any other format: `400` naming the accepted set | Q12, Q42, Q63 |
| `model` | as above | as above | Q11, Q24 |
| `prompt` | becomes the context (the system turn) | — | Q15, Q23 |
| `keywords[]` | joined into the context, after the prompt, as one line | a keyword containing `<`, `>`, CR or LF refuses the request, OpenAI's rule | Q23; owner, 2026-10-24 |
| `language` | forces the language. The docs warn that forcing Italian Italianises English jargon | a code outside the model's 30 languages | Q6, Q23, Q51; owner, 2026-10-24 |
| `languages[]` | one entry forces; several mean auto-detect, with the detected language reported | both `language` and `languages[]` ("don't send both") | Q23; owner, 2026-10-24 |
| `response_format` | `json` (default), `text`, `verbose_json` | `srt`, `vtt`, `diarized_json` | Q23 |
| `timestamp_granularities[]` | `segment`, with `verbose_json` | `word` | Q23 |
| `stream` | `true`: Server-Sent Events, below | — | Q23 |
| `temperature` | absent or `0` | anything else: "this server decodes greedily" | owner, 2026-10-24 |
| `chunking_strategy` | absent or `null`: one block | `"auto"` or a `server_vad` object: no automatic split in v1 (Q19), and spec 05 keeps it refused | Q19; owner, 2026-10-24 |
| `include[]` | — | `logprobs` | owner, 2026-10-24 |
| `known_speaker_names[]`, `known_speaker_references[]` | — | refused: no diarization | Q23 |

**Answers** (Q23, Q51):
- `json`: `{text, usage: {type: "duration", seconds}}`. `languages: [{code}]`
  carries the detected language, and is omitted when the label is `None`
  (owner, 2026-10-24).
- `text`: the transcript as the body.
- `verbose_json`: `{text, language, duration, segments, usage}`.
  - `language` is the model's own label, omitted when `None` (Q51).
  - **Segments without an aligner** (owner, 2026-10-24): one segment
    spanning the clip (start 0, end the duration), with the whole text and
    its tokens. The fields the engine does not compute are 0.
- `stream=true`: `transcript.text.delta` events, then `transcript.text.done`
  with `text` and `usage` in audio seconds. Then `data: [DONE]`, which the
  OpenAI SDK stops on and LocalAI sends (owner, 2026-10-24).

**Context** (Q15, Q23, Q50): `prompt` and `keywords[]` are capped at
`speech.context_max_tokens` (512). Over the cap the server keeps the last
512 tokens (owner, 2026-10-24). The Playground sends the last reply's tail
before the composer text (spec 04), so what is dropped is the oldest.

**Length** (Q19, Q36): a clip longer than `speech.max_clip_seconds` (1200,
the model's limit) is `400` naming the limit. Nothing is split
automatically. The body limit is the media one (`MEDIA_REQUEST_BODY_LIMIT`,
384 MiB), set per route as `/v1/images/edits` does: 20 minutes of 48 kHz
stereo 16-bit WAV is ~230 MB. That adds no new threshold (owner,
2026-10-24).

### Admission and deadlines

- **One compute worker** on the CPU (Q44). Realtime sessions take precedence
  and interleave window by window (spec 03). Batch requests fill the gaps,
  first in, first out, a window at a time. A batch wait is proportional to
  the audio, and the docs say how long (Q36).
- **Two limits** (Q44): at most `speech.max_sessions` (4) open Realtime
  sessions, and at most `speech.max_queue` (4) batch requests waiting. Past
  either, the request gets `503` with `Retry-After`, as Q35 answers. The
  `Retry-After` estimates when room frees, capped at 15 s like the engine's
  own (`RETRY_AFTER_OVERDUE_MAX_SECS`, `engine.rs:203`). The Playground shows
  it as its busy state (Q17).
- **Unavailable** (owner, 2026-10-24). With the speech model off or its
  files missing, the route answers `503 speech_unavailable` naming the
  reason, with no `Retry-After`, since waiting changes nothing.
- **Not loaded yet** is not unavailable. The first request loads the model
  (Q53) and is served after it.
- Deadlines follow ADR 0058 (Q19): `server.request_timeout` answered whole,
  `server.stream_timeout` streamed, from admission. **Past it** (owner,
  2026-10-24): `504 request_timeout`, or an `error` event when streamed, as
  for an image job. A transcription's answer has no field that says
  "incomplete".

### Privacy, logs, metrics

- Audio is decoded in memory. It is never written to disk or to a log (Q13).
- A transcript follows the rule for prompt text: not logged by default
  (`docs/specs/logging/spec.md` §29) (Q13).
- **One log line per transcription** (Q45), for the batch route and every
  Realtime item alike. Event `ignis.speech.transcription` (owner,
  2026-10-24), carrying:
  - the audio seconds;
  - the executor (`cpu` or `gpu_loan`);
  - the variant (the served id);
  - x real time;
  - the wait from commit (for a batch request, from its last byte received);
  - the detected language;
  - the outcome.
  No audio, and no text beyond Q13's rule.
- **`/metrics`** (Q45), from the server telemetry under ADR 0017's contract,
  never parsed from the log. Q45's names carry the repo's `ignis_` prefix:
  - `ignis_speech_transcriptions_total{executor, outcome}`;
  - `ignis_speech_audio_seconds_total`;
  - a histogram of the wait from commit to completed,
    `ignis_speech_commit_wait_seconds`;
  - gauges of open sessions and queue length, `ignis_speech_sessions_open`
    and `ignis_speech_queue_length`.
  The histogram's and gauges' names are the owner's (2026-10-24).

### Documentation

- The OpenAPI document:
  - documents the route, every field's rule, the answers and the events;
  - its not-served list says audio *out* and `/v1/audio/translations`, and
    no longer says "more than one loaded model": a speech model is loaded
    beside the other (ADR 0066);
  - `openapi_http.rs:286` pins the new wording.
- `README.md`:
  - `:541` and `:544-546` change to match the OpenAPI document;
  - a speech section covers the route, the compat names (Aider works with
    `OPENAI_API_BASE` and its `whisper-1`), the formats, the requirements
    and the waits;
  - it carries the `language` warning (Q51).
- `docs/user/README.md` gets the same section. `server/09` gets this route's
  rows and the new not-served list (`:181-182`).
- `CONTEXT.md`'s **Servable model** says the speech model is listed without
  being switched to.

## Acceptance

1. `speech.model` resolves `auto` to the 1.7B Q4_K_M entry, an id to its
   entry, and `off` to none. The speech model is enabled when its files are
   on disk, and nothing downloads by itself (Q39, Q49).
2. Each unavailable state reports its reason: off; files missing, naming
   `ignis model download <id>`; no AVX2 and no loan. `GET /v1/models` always
   carries a top-level `speech {available, reason}` object (Q35, Q39, Q55).
   The route answers an off or missing model with `503 speech_unavailable`
   and no `Retry-After` (owner, 2026-10-24).
3. `GET /v1/models` lists the speech model in `data` only while it is
   available, with `kind: "speech"`, after `data[0]` (still the text or image
   model) and with `active: true` (Q11, Q39, Q55; owner, 2026-10-24).
4. `model` absent, the speech model's id, or any of `speech.compat_names`
   (default list as above, changeable) is served by the speech model. A text
   or image id is `409 model_kind_mismatch` naming both kinds (Q11, Q24).
5. WAV, FLAC, MP3 and AAC are transcribed; Ogg and WebM containers are
   demuxed; Opus is decoded only if libopus built. Any other format is `400`
   naming the accepted set. No audio byte reaches the disk (Q12, Q42, Q13).
6. Every field in the table is honoured, inert or refused as written, and
   none is dropped in silence (Q20, Q23; owner, 2026-10-24).
7. `json`, `text` and `verbose_json` answer as above. `verbose_json`'s
   `language` is the model's label, omitted when `None`. `usage` is
   `{type: "duration", seconds}` (Q23, Q51).
8. `stream=true` sends `transcript.text.delta` events and one
   `transcript.text.done`, whose `text` equals the deltas joined and whose
   usage is in audio seconds (Q23).
9. `prompt` and `keywords[]` reach the executor as the context, capped at
   `speech.context_max_tokens` (Q15, Q23, Q50).
10. A clip over `speech.max_clip_seconds` (default 1200) is `400` naming the
    limit, and nothing is split (Q19, Q36).
11. One compute worker serves Realtime sessions first, window by window, and
    batch requests first in, first out, in the gaps. Past `speech.max_queue`
    waiting batch requests (default 4), or `speech.max_sessions` open
    sessions (default 4), the answer is `503` with a `Retry-After` of at most
    15 s (Q44, Q35, Q17).
12. The route has ADR 0058's deadlines (Q19). A passed one answers
    `504 request_timeout`, or an `error` event when streamed (owner,
    2026-10-24).
13. During a switch, with the image model loaded, and with the text model
    idle-unloaded, transcriptions are served on the CPU. None of them resets
    the text model's idle clock (Q40).
14. A run's log at the default level contains no audio bytes and no
    transcript text. The test checks a run's log for the transcript's words
    (Q13).
15. Each transcription writes one log line with the fields above, and no
    audio or transcript text. `/metrics` exports the two counters, the wait
    histogram and the two gauges, from the server telemetry (Q45, Q13).
16. On the 5090, the 27B's greedy decode, one lane, is less than 10% slower
    while a 60 s CPU transcription runs at `speech.threads: auto`. If not,
    the default is lowered until it holds, and the number is written in
    "As built" (Q38).
17. The OpenAPI document, its pinning test, `README.md`, the user guide and
    server/09 are amended as listed. `/v1/audio/translations` answers `404`
    and is listed as not served (Q24, Q51, Q36).
18. `cargo test` passes workspace-wide (AGENTS.md, not a grill decision).

## Out of Scope

- The Realtime route: spec 03.
- `chunking_strategy` with VAD and any automatic split (Q19; spec 05 keeps
  it refused).
- Word timestamps, SRT and VTT, and diarization (Q23).
- Audio output, and `/v1/audio/translations` (Q24).

## As built (GitHub #352)

Built on `speech-352` against a scripted executor
(`crates/server/src/speech/executor.rs`, `FakeExecutor`). Where the build
decided something the spec left open, or differs from it:

- **The seam.** `SpeechExecutor` has exactly spec 01 §4's operations plus
  `name`, `availability` (a reason a person reads), `load`, `unload` and
  `is_loaded`. `Transcript` carries the model's label (`None` is `None`) and
  the decoded token ids, for `verbose_json`'s segment. **The adapter to
  `ignis-speech` is the one open item**: `main` wires `NoExecutor`, so a real
  server reports the speech model unavailable ("this build has no speech
  executor") until #351's crate is adapted.
- **The context cap lives in the executor** (it owns the tokenizer): the
  route passes the context and `speech.context_max_tokens` to `open`. AC 9 is
  tested up to that call.
- **`GET /v1/models`.** The speech entry is `kind: "speech"`, `active: true`,
  `available: true`, right after `data[0]`. The `speech` object always exists;
  `reason` is absent while it can serve (as `reason` is elsewhere in the
  list). The unavailable reasons are checked in the order a person fixes them:
  off, an id the catalog lacks, a host that cannot run it, files missing. The
  download command is spelled `ignis-server model download <id>`, the binary's
  real name.
- **A speech entry is no model to switch to.** The catalog has no `kind` key,
  so `speech::is_speech_id` (`qwen3-asr-` prefix) keeps those entries out of
  the switch listing, the implicit switch and `POST /v1/models/switch` (`409
  model_kind_mismatch`). #351's ids follow that prefix; if they change, so does
  the constant.
- **The route.** Order of answers: `404`/`409` on `model`, `503
  speech_unavailable`, the field rules (`400`), the audio (`400`), admission
  (`503 engine_full`), the deadline (`504`). `speech_unavailable` is also the
  answer to a model that will not load. The gate lets the route through in
  every status, a warm-up included, and `MODEL_ROUTES` does not list it.
- **Deadline.** Counted from admission, the queue's wait included, as this
  spec says (an image job's runs from its start). The job stops at its next
  window.
- **The stream** sends `transcript.text.delta` as the executor's stable text
  grows after each 8 s window, the rest at the commit; a clip under one window
  streams its whole text as one delta. Events are bare `data:` lines, as
  OpenAI's.
- **`verbose_json`** also carries `task: "transcribe"`, as OpenAI's does.
  Seconds are rounded to the millisecond. `keywords[]` follow the prompt on a
  line of their own, joined by `, `.
- **Languages.** The 30 are the model card's table
  (`speech/languages.rs`), by ISO code, English name, or `tl` for Filipino.
- **The worker.** One thread: interactive tasks first, batch jobs a window at
  a time, and the idle clock (`speech.idle_unload`) in the same loop. The
  Realtime lane exists (`Speech::submit_interactive`, `open_session`) and is
  tested, but no route uses it before spec 03. `Retry-After` is the running
  clip's length over the speed measured so far (3 x real time before any),
  between 1 and 15 s.
- **Config.** All 16 fields of the `speech` group are declared, live and
  patchable; the ones spec 03, 05 and 06 read are documented as not served yet.
  A compat name equal to a served id (a catalog entry, a family, the start
  model, the speech model) is refused at start.
- **Metrics** are recorded by the request, not the log; families are absent
  until the first transcription. The Monitor reads them (snapshot, derive,
  view, tests) and ADR 0017 lists them.
- **Opus.** The libopus verification build failed on the msvc target: the
  `cmake` crate (`opusic-sys`' build dependency) does not compile in the host
  graph here (`dlltool.exe` not found, the windows-gnu host toolchain). Opus
  stays refused with the accepted set named. Symphonia (MPL-2.0, pure Rust)
  decodes WAV, FLAC, MP3, AAC and demuxes Ogg and Matroska; there is no Vorbis
  decoder.
- **Open:** AC 16 (the 5090's decode while a 60 s transcription runs) needs the
  real executor and the GPU; the executor adapter above; Opus.
