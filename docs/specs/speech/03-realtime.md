# 03 - Realtime transcription: sessions, manual commit, incremental partials

GitHub: #353 (master #350). Phase 1. Feature: `spec.md`. Needs 02's speech slot;
01's incremental API for the real path, a fake executor before it.

ADRs: 0066 (the speech model), 0058 (deadlines), 0049 (idle unload); spec
responses-api/01 (the Responses socket, its auth); spec server/09
(honoured or refused).

Decided 2026-10-23/24 in the speech grill. Details the grill left open were
decided by the owner on 2026-10-24, cited as `(owner, 2026-10-24)`.

## Problem Statement

- **The CPU goes only through incremental transcription** (Q34). Every
  catalog variant ran at 2-10x real time. A whole 60 s clip waits 7-18 s
  after the user stops, but encoding and prefilling while they speak leaves
  only the last window to do.
- **The standard is OpenAI's Realtime GA transcription session**
  (`wss://…/v1/realtime?intent=transcription`, `session.type:
  "transcription"`). llama.cpp follows it but serves no Qwen3-ASR on it.
  vLLM has its own event names. Codex speaks it, at 24 kHz with
  `turn_detection: null`.
- **ignis has a WebSocket**, the Responses socket, with browser auth by
  subprotocol (`api.rs:275`, `responses/socket.rs:65`). It has no audio
  route. Spec responses-api/01 puts Realtime out of scope (`:421`).

## Solution

### The socket

- **`GET /v1/realtime?intent=transcription`** upgrades to a WebSocket (Q20).
  Auth is the Responses socket's: `Authorization: Bearer <key>`, or from a
  browser the subprotocol `openai-insecure-api-key.<key>`.
- A conversation session (no `intent=transcription`, or a `model` in the URL,
  which is how OpenAI opens one) is refused explicitly (Q20): the upgrade
  answers `400`, naming `intent` (owner, 2026-10-24).
- On open:
  - the server sends `session.created` with the effective configuration;
  - the speech model is loaded, if it is not already (Q53);
  - the context is prefilled (Q50);
  - an idle-unloaded text model starts loading in the background (Q40);
  - in phase 3, the loan starts (Q32, Q56, spec 06).
- **Sessions share one compute worker** (Q44). They take precedence over
  batch requests and interleave window by window: each session's next unit
  (a window, a partial, a commit) takes its turn. Batch requests run first in,
  first out, in the gaps (spec 02).
- **At most `speech.max_sessions` (4) sessions are open** (Q44). Past that,
  the upgrade is refused `503` with a `Retry-After` of at most 15 s, as spec
  02 answers (Q35). The Playground shows it as its busy state (Q17) and opens
  one session per dictation (spec 04).

### Client events (Q20, Q21, Q22)

| event or field | honoured | refused | from |
|---|---|---|---|
| `session.update`, `session.type: "transcription"` | answered with `session.updated` and the effective configuration | — | Q20 |
| `audio.input.format` | `{type: "audio/pcm", rate: 24000}`, resampled to 16 kHz; `rate: 16000`, a declared extension | `audio/pcmu`, `audio/pcma` (G.711) | Q21 |
| `audio.input.transcription.model` | absent, the speech model's id, a compat name | a text or image id: `error`, `model_kind_mismatch` | Q11, Q24 |
| `audio.input.transcription.prompt`, `.keywords` | the context, prefilled at once | as spec 02 | Q15, Q23, Q50 |
| `audio.input.transcription.language`, `.languages` | as spec 02 | as spec 02 | Q6, Q23, Q51 |
| `audio.input.turn_detection` | `null`: manual commit | in phase 1, `server_vad` (spec 05 serves it from phase 2); `semantic_vad` always | Q20, Q22, Q33, Q54, Q56 |
| `audio.input.transcription.delay` | inert: the cadence is `speech.partial_interval_ms` | — | owner, 2026-10-24 |
| `audio.input.noise_reduction` | `null` | `near_field`, `far_field` | owner, 2026-10-24 |
| `include` | — | `item.input_audio_transcription.logprobs` | owner, 2026-10-24 |
| `session.ignis.provisional_tail` | `true`: the provisional-tail opt-in, below | — | Q43; owner, 2026-10-24 |
| `input_audio_buffer.append` | base64 PCM16 mono at the session's rate, up to 15 MiB an event, OpenAI's limit | — | Q20; owner, 2026-10-24 |
| `input_audio_buffer.commit` | ends the item; its transcription completes | an empty buffer: `error`, as OpenAI | Q20 |
| `input_audio_buffer.clear` | discards the buffer and the item's partial state, answered with `input_audio_buffer.cleared` | — | Q20 |
| `transcription_session.update` (the retired beta) | — | `error`, naming `session.update` | Q20 |
| conversation events (`response.create`, `conversation.item.create`, …) | — | `error` | Q20 |

**`turn_detection` in phase 1** (Q54, Q56). Q22 makes `server_vad` the
default, and Q56 builds it in phase 2. In phase 1 only, a session that sends
no `turn_detection` runs with `null`, declared in `session.created` and
`session.updated`, and an explicit `server_vad` is refused (honoured or
refused). A client expecting server VAD then sees that it must commit. From
phase 2, a missing `turn_detection` means `server_vad` (spec 05).

### Server events

- `session.created`, `session.updated`.
- `input_audio_buffer.committed` (`item_id`, `previous_item_id`) and
  `input_audio_buffer.cleared`.
- `conversation.item.input_audio_transcription.delta` (`item_id`,
  `content_index`, `delta`): **stable text only**. Deltas only ever append
  (Q43).
- `conversation.item.input_audio_transcription.completed` (`item_id`,
  `content_index`, `transcript`, `usage: {type: "duration", seconds}`). Its
  `transcript` equals the item's deltas joined (Q43, Q23).
- `conversation.item.input_audio_transcription.failed` (`item_id`,
  `content_index`, `error`).
- `input_audio_buffer.speech_started` / `speech_stopped`: only under
  `server_vad`, spec 05 (Q20, Q22).
- `error`, for every refused event or field.
- **The provisional tail** (Q43 b): the event
  `ignis.input_audio_transcription.provisional` (`item_id`, `content_index`,
  `text`) (owner, 2026-10-24).
  - It carries the whole current tail, the text still inside the rollback
    zone, which replaces the previous one, and is empty once the item
    completes.
  - A session opts in through `session.update` with
    `session.ignis.provisional_tail: true` (owner, 2026-10-24). A session
    that does not never receives it, so a strict client never sees it.

### Partials and commit (Q37, Q43)

- Appended audio is resampled when it is 24 kHz and fed to 01's session:
  - each complete 8 s window is encoded and prefilled at once;
  - every `speech.partial_interval_ms` a partial decode runs. Its new stable
    text goes out as `delta`, its tail as the provisional event.
- `commit`: `committed`, then 01's `commit()`. Whatever stable text is left
  goes out as `delta`, then `completed`.
- The next item reuses the session's prefilled context (01's prefix cache)
  and starts a new audio prefix.
- **Resampling:** the `rubato` crate (MIT/Apache-2.0), 24 kHz to 16 kHz
  (owner, 2026-10-24).

### Limits

- Each committed item respects `speech.max_clip_seconds`. A longer one fails
  with the limit named (Q44, Q19).
- An idle session closes after `server.socket_idle_close`, by the Responses
  sockets' rule (Q44).
- **Deadlines** (owner, 2026-10-24): `server.stream_timeout` applies per
  item, from its first append (ADR 0058).

### Documentation

- Spec responses-api/01 `:421` says the transcription subset is served here
  and Realtime conversation sessions stay out.
- The OpenAPI document describes the route the way it describes the
  Responses socket (`responses/socket.rs:109`): the upgrade, the auth, the
  events and the refusals.
- `README.md` and the user guide: the route, 24 kHz and the 16 kHz
  extension, manual commit, the opt-in tail.

## Acceptance

1. `GET /v1/realtime?intent=transcription` upgrades with a bearer token or
   the `openai-insecure-api-key.<key>` subprotocol. A conversation session's
   upgrade is refused `400`, naming `intent` (Q20; owner, 2026-10-24).
2. `session.update` with `session.type: "transcription"` is answered with
   `session.updated`, which carries the effective configuration (Q20).
3. 24 kHz PCM16 is resampled to 16 kHz. `rate: 16000` is accepted as a
   declared extension, and `session.created` reports the effective rate.
   G.711 is refused (Q21).
4. `append`, `commit` and `clear` behave as in the table and send
   `committed` and `cleared`. An empty commit is an `error` (Q20).
5. On each of the 11 dictations fed at real-time pace, the deltas are stable
   text only, and the deltas joined equal `completed.transcript` (Q43, Q20).
6. `ignis.input_audio_transcription.provisional` reaches only a session that
   opted in with `session.ignis.provisional_tail: true`. It carries the
   current tail and is empty at completion (Q43; owner, 2026-10-24).
7. `completed` carries `usage` in audio seconds. An item over
   `speech.max_clip_seconds`, or one whose transcription fails, gets
   `failed` with the error (Q20, Q19, Q44, Q23).
8. In phase 1 `server_vad` is refused, and `semantic_vad` always.
   `transcription_session.update` is refused naming `session.update`, and
   conversation events are refused. Every other field is honoured, inert or
   refused as the table says (Q20, Q22, Q33, Q54; owner, 2026-10-24).
9. The context is prefilled at session open or at `session.update`: a
   commit's work prefills no context token (Q50, Q15).
10. `transcription.model` follows spec 02's resolution. A text or image id
    is an `error` with `model_kind_mismatch` (Q11, Q24).
11. `language` and `languages` follow spec 02, and nothing is forced unless
    asked (Q6, Q51).
12. Opening a session loads the speech model if it is not loaded (Q53). It
    also starts loading an idle-unloaded text model in the background, and
    the session is served on the CPU meanwhile (Q40).
13. On the i9, with 1.7B Q4_K_M at `speech.threads: auto`, an Italian
    dictation of at least 60 s, fed at real-time pace, goes from `commit` to
    `completed` in at most 2 s. The 4-core container's figure is written in
    "As built" and in `spec.md`'s requirements (Q37, Q34).
14. Up to `speech.max_sessions` sessions (default 4) run at once,
    interleaved window by window, ahead of batch requests. One more is
    refused at the upgrade with `503` and a `Retry-After` of at most 15 s. A
    session idle past `server.socket_idle_close` is closed (Q44, Q35, Q17).
15. In phase 1, a session that sends no `turn_detection` runs with `null`,
    and `session.created` and `session.updated` declare it (Q54, Q56).
16. Spec responses-api/01 `:421`, the OpenAPI document and the README are
    amended as above (Q20).
17. `cargo test` passes workspace-wide (AGENTS.md, not a grill decision).

## Out of Scope

- `server_vad`: spec 05, phase 2 (Q56). `semantic_vad` is never served
  (Q22).
- G.711 (Q21), WebRTC, and Realtime conversation sessions (Q20).
- The GPU loan: spec 06, phase 3 (Q56).

## As built (GitHub #353)

Built on `speech-353` against the scripted executor (`FakeExecutor`), in
`crates/server/src/speech/realtime/` (the socket, `config.rs` for
`session.update`, `pcm.rs` for the audio). Where the build decided something
the spec left open, or differs from it:

- **The seam is unchanged.** Each item is one executor session (spec 01 §4):
  opened when the socket opens and again right after each commit, so the
  next item's context is prefilled before its first append (AC 9), and the
  commit's own unit pushes the rest and decodes, opening nothing. A
  `session.update` that changes the context or the forced language reopens
  the current item's session when it holds no audio; with audio, the change
  applies from the next item. `speech.rollback_tokens` is a setting the
  adapter to `ignis-speech` passes, not a §4 operation: not in the trait.
- **The worker has a lane per session.** Every executor call is one unit
  queued in the session's lane, in order (an open, a window pushed as soon as
  one is in, a refresh, the commit). The worker takes the lanes in turn, one
  unit each, and every queued unit runs before the next batch window, so a
  session with a backlog is not interleaved with a batch job (AC 14). The
  executor session lives in a slot the units share; they never run at once.
- **Resampling** reuses the transcription route's streaming windowed-sinc
  resampler (`speech/audio.rs`, #352) instead of the `rubato` crate the spec
  names: it is already tested for pitch, length and aliasing, and a second
  resampler would be a second dependency doing the same job. Swapping in
  `rubato` touches `pcm.rs` alone; the owner's call.
- **Items.** An item gets its `item_id` at its first append, so deltas carry
  it before the commit. An item over `speech.max_clip_seconds` or past its
  deadline stops being transcribed at once (its audio goes nowhere after
  that), an `error` says why at once (no `item` field carries it: the item is
  not committed yet), and its `failed` comes after its `committed`, at the
  commit; an item already committed gets `failed` at once. A model that will not load fails
  the item it was opened for, at that item's commit (`speech_unavailable`);
  the next item tries the load again. An empty commit is one with no sample
  since the last (OpenAI asks 100 ms).
- **`session.update`** merges field by field: a field left out keeps its
  value, `null` is the field's default (`transcription: null` resets the
  whole transcription object), and one refused field refuses the update.
  `transcription.delay` is checked against OpenAI's five values and echoed.
  Refusals are `error` events with code `unsupported_value` for what OpenAI
  serves and ignis does not, `invalid_request_error` for a malformed value,
  `model_kind_mismatch`/`model_not_found` for the model; `param` is the
  field's full path (`session.audio.input.format.rate`), and `error.event_id`
  the client event's. Every server event carries an `event_id`.
- **The provisional tail** is sent after a refresh that changed it, and `""`
  when the item completes, fails or is cleared, if a tail was shown. A refresh
  is queued every `speech.partial_interval_ms` only when audio came since the
  last one and the last one has answered.
- **The upgrade**, in order: not a WebSocket upgrade `400`; no
  `intent=transcription`, or a `model` in the URL, `400` naming `intent`; the
  speech model unavailable `503 speech_unavailable`; `speech.max_sessions`
  open `503 engine_full` with `Retry-After: 1` (room frees when a person ends
  a session). The browser's subprotocol `realtime` is selected back, the
  credential never. Frames are capped at 16 MiB (the text body limit); an
  append whose `audio` holds more than 15 MiB of base64 is an `error`.
- **Loading.** An open session holds the speech model: the worker's idle
  clock does not unload it while any session is open, and runs again from
  the end of the last (`speech.idle_unload` counts "no transcription and no
  session"). Opening a session loads whatever model the text model's idle
  clock unloaded, the image model too: dictation feeds the image studio as
  well (spec 04).
- **Logs and metrics.** Each committed item writes one
  `ignis.speech.transcription` line and counts in the existing families, its
  wait from the commit; an item still being transcribed when the socket closes
  counts as cancelled. No new family, so the Monitor and ADR 0017 are
  unchanged. The idle close logs `ignis.ws.idle_closed`, the Responses
  sockets' event, now with a `route` field (`responses`, `realtime`). A
  transcript that does not begin with the stable text sent (an executor
  breaking spec 01 §4) is still sent, with a WARN
  `ignis.speech.stable_text_revised`.
- **Open:** AC 5 on the owner's eleven dictations and AC 13 (commit to
  `completed` in 2 s for 60 s of Italian on the i9, and the 4-core figure)
  need the real executor and its adapter, the integration step after #351.
