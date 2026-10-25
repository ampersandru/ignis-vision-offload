# 08 - the diarizer: speakers beside the speech model

GitHub: (ticket under master #350). Track: Transcribe (spec 07 is the Playground view). Feature: `spec.md`. Needs 05
(`server_vad`).

ADRs: 0051 (adapted code, licences), 0067 (the speech model), 0068 (the CPU executor), 0069 (this diarizer).

Decided 2026-10-25 in the Transcribe grill (Q81-Q118, the log in `.scratch/speech-grill-2026-10-23/`). Where this
spec settles a protocol detail the grill left to it (Q118), it says `(spec 08)`. Evidence: the finding
[Nemotron on the CPU](../../findings/2026-10-25-nemotron-on-the-cpu.md).

## Problem Statement

- The Transcribe view (spec 07) labels each utterance with its speaker. Qwen3-ASR gives no speakers: no diarization,
  no speaker tokens, no word timestamps (ADR 0069).
- Nemotron 3 Diarization tells Italian speakers apart on the CPU (DER 3.6-4.0% on twenty synthetic conversations),
  at a cost set by its geometry: the card's low geometry takes half of a 4-core machine, a 2.32 s chunk costs 2.6x
  less at no DER cost, and the offline geometry costs 25-47x less than real time (the finding).
- Under `server_vad`, an item ends only at a silence. Two voices that change with no pause land in one item, and a
  room that never falls silent holds an item open until `speech.max_clip_seconds` (1200 s) fails all of it.
- Spec 07 feeds a file through Realtime faster than real time, and nothing bounds the audio a session queues.

## Solution

### The diarizer (ADR 0069)

- **Nemotron 3 Diarization** (`nvidia/Nemotron-3-Diarization`, 100M parameters, OpenMDW-1.1): a 31-layer RoPE
  Transformer with a Sortformer head, 8 speakers in arrival order, an arrival-order speaker cache (AOSC) and a FIFO.
  The fourth **model kind**, the **diarizer**: it takes audio and gives back who spoke when, and never transcribes
  (Q118).
- **Ours, in Rust, on the CPU** (Q104, ADR 0068's rule), in `crates/speech`. Its pieces:
  - the front end;
  - the 31 layers;
  - the head;
  - the streaming state: the speaker cache, its compression at the update period, and the FIFO.
  These are restated from NeMo's Apache-2.0 source as an adapted op under ADR 0051: provenance in the file header
  and in `NOTICE`, as `vad.rs` is.
- It reuses the executor's GGUF reader, kernels and pool. **The stored format** (F32, BF16 or Q8_0) is chosen by
  measurement in this ticket: in the spike PyTorch FP32 beat ggml Q8_0 on it, since each step is one large GEMM over
  about 540 frames (the finding).
- **Weights:** a GGUF converted by our own tool, with the OpenMDW-1.1 licence beside it. The Hugging Face repo waits
  for the owner's OK and a check that OpenMDW-1.1 allows the redistribution (Q115).
- **Never resident in VRAM** (ADR 0067). Loaded at the first session that asks for speakers, unloaded with the speech
  model at `speech.idle_unload` (Q116). About 0.2-0.4 GB of host RAM, by the stored format.

### The catalog and availability (Q118)

- Its own catalog entry, `kind: diarizer`, downloaded with `ignis-server model download <id>`.
- `speech.diarization_model`: `auto` (the catalog's diarizer entry), an id, or `off`. A text, image or speech id is
  refused at start and on a PATCH, as `speech.model` refuses a text id.
- **Not in `/v1/models`**: no request names it. The `speech` object that `GET /v1/models` carries in every status
  gains a `diarizer` member: available, or unavailable with its reason (files missing with the download command, a
  CPU without AVX2, `off`).
- **The Server view** shows the diarizer beside the speech model: its id, whether it is loaded, and its reason when
  unavailable.

### On the worker (Q113)

- The diarizer runs on the speech model's worker, as one more kind of interactive unit in its session's lane: one
  step of its geometry.
- **Its steps rank below the speech model's windows and commits and above its partials.** A partial is already
  skipped, never queued, while a window is pending (spec 03). It is skipped, too, while a diarizer step of its
  session is pending. On a slow CPU the partials thin out first, and the labels and commits keep their pace.
- It reads the session's 16 kHz stream after resampling, the same samples the VAD reads, from the session's first.
  It never reads the leading silence that only the speech model is fed.

### Geometry (Q106)

- **Live:** the default is a long chunk, SPKCACHE 264 / FIFO 264 / CHUNK 25 / RIGHT CONTEXT 4 / UPDATE PERIOD 222
  frames of 80 ms (2.32 s of latency). It measured 3.60% DER against the card's low geometry's 3.97%, at 2.6x less CPU.
- **File:** the card's offline geometry, 264 / 40 / 340 / 40 / 300 (30.4 s), for a session fed faster than real time.
- **The session picks** `live` or `file` in its opt-in (below). Every geometry parameter of both is a `speech.*`
  field with the measured default. The names follow the group's style, chosen at the build, as Q76's and Q77's were.

### One speaker per utterance (Q107, Q108)

- **Speaker of an item.** Each 80 ms frame's speaker is the established channel with the highest activity. An item's
  speaker is the established channel that holds most of its speech frames.
- **The ghost rule** (Q108, Q116): a channel is established once it holds `speech.diarization_min_speaker_seconds` of
  speech, counted on the frames where it is the most active. Until then its frames count for the established channel
  with the next highest activity, and its items carry that one.
  - The session's first channel is established at once.
  - `0` turns the rule off: every channel is established at its first frame.
  - The default is chosen on the twenty conversations. The spike's ghosts held 0.9-5.1 s and its one real split
    14.5 s.
- **The cut at a change.** When the speaker of the frames changes to another established channel inside an item, the
  server ends the item at the change, as the VAD does on a silence:
  - the change is the first frame of the new speaker's run. A channel established only now counts from the start of
    the run that established it, so the cut can reach back by up to `speech.diarization_min_speaker_seconds` plus
    the geometry's latency (spec 08);
  - the events (spec 08): `input_audio_buffer.speech_stopped` with `audio_end_ms` at the change, then
    `input_audio_buffer.committed`; then `input_audio_buffer.speech_started` for a new item with `audio_start_ms` at
    the change. So every item keeps its pair of times;
  - the new item takes no prefix padding, which would take the previous speaker's speech. It is fed to the speech
    model after `speech.vad_leading_silence_ms` of digital silence, as every `server_vad` item is (spec 05, Q77)
    (spec 08);
  - **the audio past the change** (spec 08): the ended item's last encoder window is computed again on its audio up
    to the change, and the audio from the change on is the new item's, encoded afresh. At most one window's worth of
    encoding is done twice, plus the cut's lag.
- **Text in a diarized session** (spec 08): an item can end after its audio was heard, and with no word timestamps
  the text past a change cannot be told from the text before it. So in a session with speakers the server fixes no
  text before the item ends:
  - partials still run and still feed the provisional tail, which carries all of the item's text;
  - the internal stable text speeds them up as before, but a cut item is decoded once more from its audio up to the
    change;
  - at the item's end the server sends one `conversation.item.input_audio_transcription.delta` with the whole
    transcript, so the deltas joined still equal `completed.transcript` (spec 03, AC 5).
  - A session without speakers is unchanged. A session with speakers that did not opt in to the provisional tail
    sees an item's text only at its end.
- **At most 8 speakers** (Q110), the model's channels. A ninth voice lands on one of them.

### The length guard (Q117)

- `speech.max_utterance_seconds`: `0` off, otherwise N. The starting default is 120 s, and AC 12's measurement sets
  it. It applies to every `server_vad` session, with or without speakers, so that a room that never falls silent
  loses no audio to `speech.max_clip_seconds`.
- **Where it cuts** (spec 08): once an item holds N seconds, the server ends it at the first frame under the VAD's
  `threshold`, a pause too short to end it, or at N plus `speech.utterance_cut_search_ms`, whichever comes first.
  The search looks forward, never back, so no text already fixed is past the cut. The events and the new item are as
  at a change. The same speaker continues in the new item.
- **A session can set its own limit** (spec 08): `session.ignis.max_utterance_seconds` (null = the server's, 0 = off).
  The Playground's dictation sends 0. Its first server commit ends it (Q80), and a long dictation with short pauses
  must not end at the guard. The dictation is bounded by `speech.max_clip_seconds`, as before.

### The protocol (Q111; names rule Q57)

- **Opt-in:** `session.ignis.diarization`: `null` or absent (off), `"live"` or `"file"` (the geometry). It is
  accepted only on `server_vad` sessions. On a manual-commit session it is refused naming `turn_detection`.
  `session.updated` declares it. When the diarizer is unavailable the field is refused with its reason, and the
  session stays open without speakers.
- **The speaker:** `ignis.input_audio_transcription.speaker` (`item_id`, `content_index`, `speaker`: the channel
  index 0-7, numbered in arrival order) goes after an item's last delta and before its `completed`.
- OpenAI's events do not change. A client that does not opt in sees nothing new except the length guard's cuts.
- `session.ignis` takes `provisional_tail`, `diarization` and `max_utterance_seconds`. An unknown key stays refused,
  naming the three.

### A session's queued audio (Q99)

- `speech.max_queued_audio_seconds`: the audio a session holds that no `completed` has covered yet. Past it, the
  append is refused with an `error`, code `ignis_audio_queue_full`, and the session stays open. The default is above
  spec 07's client pace (60 s) and is chosen with the file AC.
- A session fed at real time never reaches it; a file fed flat out does when its client does not pace it.

### Configuration

| field | default | from |
|---|---|---|
| `speech.diarization_model` | `auto`; an id; `off` | Q116, Q118 |
| the live geometry (5 fields) | 264 / 264 / 25 / 4 / 222 frames | Q106 |
| the file geometry (5 fields) | 264 / 40 / 340 / 40 / 300 frames | Q106 |
| `speech.diarization_min_speaker_seconds` | measured on the twenty conversations; `0` off | Q108, Q116 |
| `speech.max_utterance_seconds` | 120, then AC 12's measurement; `0` off | Q117 |
| `speech.utterance_cut_search_ms` | measured with AC 12 | spec 08 |
| `speech.max_queued_audio_seconds` | above 60, measured with the file AC | Q99 |

Every field is declared once (ADR 0046), patchable live where the group's fields are, and in the generated reference.

## Acceptance

1. The diarizer runs in Rust on the CPU. Teacher-forced on the same audio, its per-frame speaker probabilities match
   the checkpoint's Transformers code in FP32 within a bound fixed before the run, recorded as AC 1 of spec 05 was:
   a multiple of the reference's own distance from fp64, with a negative control past it. The streaming state
   (speaker cache, compression, FIFO) is in the comparison. Its provenance is in its header and in `NOTICE`
   (Q114, Q104).
2. The stored format is the one measured best of F32, BF16 and Q8_0 on speed at equal DER; the measurement is in
   "As built" (Q114, ADR 0069).
3. On the twenty synthetic conversations of the finding (`H:\speech-study\nemotron-spike\data\conv\`, the
   `--ignored` leg), the port's DER at collar 0.25 is within +0.5 pt of NeMo-Speech.cpp's at the same geometry, at
   the live and the file geometries (Q114).
4. A small public real-meeting set (AMI, English) is scored and reported in "As built"; it does not gate (Q114).
5. `speech.diarization_min_speaker_seconds`'s default is the value that minimizes DER on the twenty conversations
   with no conversation's count of speakers worse than with `0`. `0` turns the rule off (Q108, Q116).
6. A speaker change inside an item ends it at the change. The events are `speech_stopped`, `committed` and
   `speech_started`, with the times at the change. The new item has no prefix padding and has the leading silence
   (Q107, spec 08).
7. In a session with speakers no delta is sent before an item ends, and the deltas joined equal
   `completed.transcript` on every item of the twenty conversations, cut items included (spec 03, AC 5; spec 08).
8. `ignis.input_audio_transcription.speaker` goes after each item's last delta and before its `completed`. A session
   that does not opt in gets none (Q111).
9. `session.ignis.diarization` is honoured on `server_vad` sessions and refused on manual commit. With the diarizer
   unavailable it is refused with the reason, and the session stays open. `session.updated` declares it (Q111).
10. The diarizer's steps rank below windows and commits and above partials. A test with a slow fake executor shows
    the partials thinning while every diarizer step and commit is served (Q113).
11. On the i9, at the live geometry, with Qwen3-ASR `qwen3-asr-1.7b` live in the same session, a 5-minute
    conversation fed at real time keeps the diarizer within its geometry's latency plus 2 s at every item end. The
    4-core container's figures are reported, with the share of the machine (Q113, Q34's style).
12. `speech.max_utterance_seconds` cuts as specified. On items of 60, 120 and 240 s with no pause of 500 ms (built
    from FLEURS and VoxPopuli), WER, commit time and RAM are measured, and the default is the value that measures
    best, 120 s unless the data say otherwise. `speech.utterance_cut_search_ms`'s default comes from the same
    measurement. `session.ignis.max_utterance_seconds` overrides it, and the Playground's dictation sends `0`
    (Q117, Q80).
13. `speech.max_queued_audio_seconds` refuses the append past it with `ignis_audio_queue_full`, and the session stays
    open (Q99).
14. `speech.diarization_model` resolves `auto`, an id and `off`, and refuses a text, image or speech id. The
    diarizer has its catalog entry (`kind: diarizer`), is downloadable with `ignis-server model download`, is not
    in `/v1/models`, and its availability is in the `speech` object there. The Server view shows it (Q118).
15. The diarizer loads at the first session that asks for speakers and unloads with the speech model (Q116).
16. Every new field is in the generated configuration reference; the OpenAPI document and
    `docs/user/api/realtime-transcription.md` describe the opt-in, the event, the cuts, the guard, the queue limit
    and `session.ignis.max_utterance_seconds`; the speech guide states the diarizer's CPU and RAM figures from AC 11
    (AGENTS.md).
17. A short committable fixture of public audio (a VoxPopuli two-speaker excerpt, CC0) runs the speaker leg in every
    `cargo test`. The twenty conversations stay on the study disk (the privacy rule).
18. `cargo test` passes workspace-wide (AGENTS.md).

## Out of Scope

- Nemotron 3.5 ASR, or any second speech model (ADR 0069).
- A per-utterance embedding model (CAM++): the fallback if AC 1 or AC 3 fails, a ticket of its own then.
- `diarized_json` on `/v1/audio/transcriptions`: a parked idea (Q94).
- Speaker enrolment, and corrections that teach the model (Q109).
- The GPU loan for the diarizer: phase 3's spec 06 does not cover it.
- New metric families: none in this ticket. The session's log line names its speaker count and the diarizer's step
  time.

## Where it goes

- `crates/speech/src/`: the diarizer beside `vad.rs` and the executor (its graph, its streaming state, its loader).
- `crates/server/src/speech/realtime/config.rs`: `session.ignis.diarization`, `max_utterance_seconds`, the unknown-key
  sentence.
- `crates/server/src/speech/realtime/mod.rs`: the cuts, the events, the per-item speaker, the text rule of a diarized
  session, the queue limit.
- `crates/server/src/speech/worker.rs`: the diarizer's unit and its rank.
- `crates/server/src/config/schema.rs`: the fields. `crates/server/src/download/catalog.rs` and the catalog file: the
  entry and `kind: diarizer`.
- `web/src/speech/realtime.ts`: the dictation sends `session.ignis.max_utterance_seconds: 0`.
- `web/src/server/`: the diarizer in the Server view.
- The tool that converts the weights, beside `tools/speech-vad/`.
