# 08 - the diarizer: speakers beside the speech model

GitHub: #361 (master #350). Track: Transcribe (spec 07 is the Playground view). Feature: `spec.md`. Needs 05
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
- `diarizer.model`: `auto` (the catalog's diarizer entry), an id, or `off`. A text, image or speech id is refused at
  start and on a PATCH, as `speech.model` refuses a text id.
- **Not in `/v1/models`**: no request names it. The `speech` object that `GET /v1/models` carries in every status
  gains a `diarizer` member: available, or unavailable with its reason (files missing with the download command, a
  CPU without AVX2, `off`).
- **The Server view** shows the diarizer beside the speech model: its id, whether it is loaded, and its reason when
  unavailable.

### On the worker (Q113; amended by the owner, 2026-10-26)

- **The diarizer runs on a thread of its own, beside the speech model's worker**, with a pool of its own:
  `diarizer.threads` threads (`auto`: a third of the physical cores, at least 2; 3 on the i9), created when the
  diarizer loads and dropped when it unloads, computing with the speech executor's ISA. The one thread serves every
  session's speakers' units (a stream's open, its steps, its tails), the sessions taking turns a unit each, as the
  speech worker serves its lanes.
- **Its steps never wait for the speech model's units and never stop them.** A partial is skipped, never queued, while
  a unit of its session's speech is pending on the speech worker (spec 03), never for a diarizer step, which on its
  own thread is pending almost always while audio flows.
- **Not the speech model's pool.** Its dispatch takes one lock per op: a Qwen3-ASR decode token is about 200 small
  ops, a diarizer op on a live step's 557 rows takes 2-5 ms, so a shared pool would put a diarizer op in front of
  every one of the decoder's.
- The owner's choice, on one condition, confirmed by the measurements (As built): the diarizer's thread must not
  hurt the other critical paths, the commit's latency above all, the live text and the other sessions. As first
  decided it ran on the speech model's worker, its steps ranked below the windows and commits and above the partials,
  and a step that fell due stopped the refresh under way; on a long monologue the live text then stood still for the
  whole item (As built).
- It reads the session's 16 kHz stream after resampling, the same samples the VAD reads, from the session's first.
  It never reads the leading silence that only the speech model is fed.

### Geometry (Q106)

- **Live:** the default is a long chunk, SPKCACHE 264 / FIFO 264 / CHUNK 25 / RIGHT CONTEXT 4 / UPDATE PERIOD 222
  frames of 80 ms (2.32 s of latency). It measured 3.60% DER against the card's low geometry's 3.97%, at 2.6x less CPU.
- **File:** the card's offline geometry, 264 / 40 / 340 / 40 / 300 (30.4 s), for a session fed faster than real time.
- **The session picks** `live` or `file` in its opt-in (below). Every geometry parameter of both is a field of the
  `diarizer` group, applicable only to the `Nemotron3Diarization` family, with the measured default (Configuration,
  below). The names follow the group's style, chosen at the build, as Q76's and Q77's were.

### One speaker per utterance (Q107, Q108)

- **Speaker of an item.** Each 80 ms frame's speaker is the established channel with the highest activity. An item's
  speaker is the established channel that holds most of its speech frames.
- **The ghost rule** (Q108, Q116): a channel is established once it holds `diarizer.min_speaker_seconds` of
  speech, counted on the frames where it is the most active. Until then its frames count for the established channel
  with the next highest activity, and its items carry that one.
  - The session's first channel is established at once.
  - `0` turns the rule off: every channel is established at its first frame.
  - The default is chosen on the twenty conversations. The spike's ghosts held 0.9-5.1 s and its one real split
    14.5 s.
- **The cut at a change.** When the speaker of the frames changes to another established channel inside an item, the
  server ends the item at the change, as the VAD does on a silence:
  - the change is the first frame of the new speaker's run. A channel established only now counts from the start of
    the run that established it, so the cut can reach back by up to `diarizer.min_speaker_seconds` plus
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

### Configuration: a `diarizer` group and a family (owner, 2026-10-25)

- **A group of its own,** `diarizer`, as the image model has `image` and the speech model `speech`: the diarizer is the
  fourth model kind (ADR 0046's groups; ADR 0069).
- **A family of its own:** `ModelFamily` gains `Nemotron3Diarization`. Every field that is a concept of Sortformer, the
  geometries and the speaker cache's constants, is `applicable_to: Only([Nemotron3Diarization])` under its plain key
  `diarizer.<field>`. A diarizer of another family refuses them rather than applying them in silence (ADR 0046's
  rule for a field one family has). A field two diarizer families would want sized differently takes ADR 0046's
  family scope, `diarizer.<family>.<field>`, when that family comes.
- **The speaker cache's constants default to `auto`:** the value the checkpoint's `streaming_config` gives, which the
  conversion tool writes into the GGUF's metadata. An override is validated by a `Range`. The effective value is
  logged at load and shown in the Server view, and a session reads them when it opens, as it reads
  `speech.rollback_tokens`.
- **What stays in `speech`:** the length guard and the queue limit. Both hold for every `server_vad` session, with
  speakers or without.

| field | applicable to | default | from |
|---|---|---|---|
| `diarizer.model` | every diarizer | `auto` (the catalog's diarizer entry); an id; `off` | Q116, Q118 |
| `diarizer.threads` | every diarizer | `auto` (a third of the physical cores, at least 2); a count | owner, 2026-10-26 |
| `diarizer.min_speaker_seconds` | every diarizer | measured on the twenty conversations; `0` off | Q108, Q116 |
| the live geometry (5 fields: speaker cache, FIFO, chunk, right context, update period) | `Nemotron3Diarization` | 264 / 264 / 25 / 4 / 222 frames | Q106 |
| the file geometry (the same 5 fields) | `Nemotron3Diarization` | 264 / 40 / 340 / 40 / 300 frames | Q106 |
| `diarizer.prediction_score_threshold` | `Nemotron3Diarization` | `auto` (the checkpoint's 0.25) | owner, 2026-10-25 |
| `diarizer.latest_frames_score_boost` | `Nemotron3Diarization` | `auto` (0.05) | owner, 2026-10-25 |
| `diarizer.strong_boost_rate` | `Nemotron3Diarization` | `auto` (0.75) | owner, 2026-10-25 |
| `diarizer.weak_boost_rate` | `Nemotron3Diarization` | `auto` (1.5) | owner, 2026-10-25 |
| `diarizer.min_positive_scores_rate` | `Nemotron3Diarization` | `auto` (0.5) | owner, 2026-10-25 |
| `diarizer.speaker_cache_silence_frames_per_speaker` | `Nemotron3Diarization` | `auto` (1) | owner, 2026-10-25 |
| `speech.max_utterance_seconds` | the speech model | 120, then AC 12's measurement; `0` off | Q117 |
| `speech.utterance_cut_search_ms` | the speech model | measured with AC 12 | spec 08 |
| `speech.max_queued_audio_seconds` | the speech model | above 60, measured with the file AC | Q99 |

Every field is declared once (ADR 0046), patchable live where the group's fields are, and in the generated reference.
The constants' names are the checkpoint's own keys.

## Acceptance

1. The diarizer runs in Rust on the CPU. Fed the same audio chunk by chunk, with the same streaming state, its
   per-frame speaker probabilities match the checkpoint's Transformers code in FP32 within a bound fixed before the
   run, recorded as AC 1 of spec 05 was:
   a multiple of the reference's own distance from fp64, with a negative control past it. The streaming state
   (speaker cache, compression, FIFO) is in the comparison. Its provenance is in its header and in `NOTICE`
   (Q114, Q104).
2. The stored format is the one measured best of F32, F16 and Q8_0 on speed at equal DER; the measurement is in
   "As built" (Q114, ADR 0069). (Amended by the owner, 2026-10-26: F16 in place of BF16, which the executor's
   kernels do not have; the same size, a finer mantissa.)
3. On the twenty synthetic conversations of the finding (`H:\speech-study\nemotron-spike\data\conv\`, the
   `--ignored` leg), the port's DER at collar 0.25 is within +0.5 pt of NeMo-Speech.cpp's at the same geometry, at
   the live and the file geometries (Q114).
4. A small public real-meeting set (AMI, English) is scored and reported in "As built"; it does not gate (Q114).
5. `diarizer.min_speaker_seconds`'s default removes the synthetic conversations' ghost labels it can without losing a
   real meeting's participant (AMI, AC 4) or more than +0.5 pt of DER against `0`. `0` turns the rule off (Q108, Q116).
   (Amended by the owner, 2026-10-26, twice: the first wording, the value that minimizes DER, picked `0`, since the
   rule only adds confusion; the second, the most speaker counts right on the twenty conversations, picked 5 s, which
   lost quiet participants of real meetings.)
6. A speaker change inside an item ends it at the change. The events are `speech_stopped`, `committed` and
   `speech_started`, with the times at the change. The new item has no prefix padding and has the leading silence
   (Q107, spec 08).
7. In a session with speakers no delta is sent before an item ends, and the deltas joined equal
   `completed.transcript` on every item of the twenty conversations, cut items included (spec 03, AC 5; spec 08).
8. `ignis.input_audio_transcription.speaker` goes after each item's last delta and before its `completed`. A session
   that does not opt in gets none (Q111).
9. `session.ignis.diarization` is honoured on `server_vad` sessions and refused on manual commit. With the diarizer
   unavailable it is refused with the reason, and the session stays open. `session.updated` declares it (Q111).
10. The diarizer runs on a thread and a pool of its own: its steps never wait for the speech model's units and never
    stop them, and a partial is never skipped for a step. A test with slow fakes shows both progressing: a step runs
    whole inside a refresh, which no step stops, refreshes begin while a step runs, and every diarizer step and commit
    is served (Q113; amended by the owner, 2026-10-26: the first wording ranked the steps above the partials on the
    speech model's worker, and the partials thinned out).
11. On the i9, at the live geometry, with Qwen3-ASR `qwen3-asr-1.7b` live in the same session, a 5-minute
    conversation fed at real time keeps the diarizer within its geometry's latency plus 2 s: every frame's speaker
    probabilities come within that of the frame's first sample (the session log line's `lag_max_ms`). The 4-core
    container's figures are reported, with the share of the machine (Q113, Q34's style). (Amended by the owner,
    2026-10-26: the first wording bounded every item end, but an item cut at a change gets its speaker later by
    design, the cut reaching back to the new speaker's run, up to `diarizer.min_speaker_seconds` before.)
12. `speech.max_utterance_seconds` cuts as specified. On items of 60, 120 and 240 s with no pause of 500 ms (built
    from FLEURS and VoxPopuli), WER, commit time and RAM are measured, and the default is the value that measures
    best, 120 s unless the data say otherwise. `speech.utterance_cut_search_ms`'s default comes from the same
    measurement. `session.ignis.max_utterance_seconds` overrides it, and the Playground's dictation sends `0`
    (Q117, Q80).
13. `speech.max_queued_audio_seconds` refuses the append past it with `ignis_audio_queue_full`, and the session stays
    open (Q99).
14. `diarizer.model` resolves `auto`, an id and `off`, and refuses a text, image or speech id. The diarizer has its
    catalog entry (`kind: diarizer`), is downloadable with `ignis-server model download`, is not in `/v1/models`,
    and its availability is in the `speech` object there. The Server view shows it (Q118). The `diarizer` group's
    Sortformer fields are `Only([Nemotron3Diarization])`, judged against the diarizer's own slot, so a text, image
    or speech family active beside it does not refuse them, and a test shows it (amended by the owner, 2026-10-26:
    a family judges only its own slot's fields, ADR 0046; no second diarizer family exists to refuse them). The speaker
    cache's constants resolve `auto` to the GGUF's values from the checkpoint, take an override within their
    ranges, and log and show the effective value (owner, 2026-10-25).
15. The diarizer loads at the first session that asks for speakers and unloads with the speech model (Q116).
16. Every new field is in the generated configuration reference; the OpenAPI document and
    `docs/user/api/realtime-transcription.md` describe the opt-in, the event, the cuts, the guard, the queue limit
    and `session.ignis.max_utterance_seconds`; the speech guide states the diarizer's CPU and RAM figures from AC 11
    (AGENTS.md).
17. A short committable fixture of public audio (a VoxPopuli two-speaker excerpt, CC0) runs the speaker leg in every
    `cargo test`: its log-mel and its speaker cache replayed on the reference's recorded probabilities. The forward
    pass on it needs the weights, which are not in the tree, so it is an `--ignored` leg on the study disk. The twenty
    conversations stay on the study disk (the privacy rule). (Amended by the owner, 2026-10-26.)
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
- `crates/server/src/speech/worker.rs`: the diarizer's own thread (`DiarizerWorker`); `crates/server/src/speech/cpu.rs`:
  its pool.
- `crates/server/src/config/schema.rs`: the `diarizer` group and the `speech` fields. `crates/core/src/compute.rs`:
  `ModelFamily::Nemotron3Diarization`. `crates/server/src/download/catalog.rs` and the catalog file: the entry and
  `kind: diarizer`.
- `web/src/speech/realtime.ts`: the dictation sends `session.ignis.max_utterance_seconds: 0`.
- `web/src/server/`: the diarizer in the Server view.
- The tool that converts the weights, beside `tools/speech-vad/`; it writes the checkpoint's `streaming_config` into
  the GGUF's metadata.

## As built

Built 2026-10-26 on branch `speech-diarizer` (#361). The port is `crates/speech/src/diarizer/`; its oracle and tools are
`tools/speech-diarizer/` (`record.py`, `convert.py`, `der.py`, `long_items.py`); its measurements' data are on the study
disk (`H:\speech-study\nemotron-spike\oracle-361\`, `port-361\`, `H:\speech-study\length-guard-361\`, `H:\speech-study\ami\`).

### The port (AC 1, 17)

- **Source.** Ported from the checkpoint's Transformers code, `nemotron3_diarization` in transformers 5.20.2, which
  restates NeMo's streaming Sortformer (both Apache-2.0); the header of `diarizer/mod.rs` and `NOTICE` name both.
  ADR 0069 said "restated from NeMo"; the Transformers code is the one read and run as the oracle.
- **The graph, as read from that code:** the embedder stacks 8 log-mel frames and projects them (no convolutional
  subsampling); the speaker cache and the FIFO hold the embedder's output, so each step encodes cache, FIFO, chunk
  and look-ahead again through the 31 layers; a kernel-3 convolution upsamples each 80 ms row to 8 rows of 10 ms;
  the cache scores 80 ms frames by the mean of their 8 rows' sigmoids. A frame is speech when a channel reaches 0.5,
  the line the checkpoint's own cache draws (`probs > 0.5`).
- **A step waits for the samples the checkpoint's processor would give it** (the first chunk centred, the later ones
  from half a transform before their first frame), 104 samples (6.5 ms) more than the log-mel needs, so that a
  stream's last steps are the processor's.
- **AC 1, the oracle.** The reference runs in fp64 with its speaker cache and FIFO forced, at every step, to the fp32
  run's frames: unforced, a tie at the boundary of the cache's top-k (whose order torch leaves undefined) sent the fp64
  twin down another path (conv15, live: 0.19). Bound = 10 x the reference's distance from fp64, at least 1e-5:
  live 5.5e-5, file 1.2e-4, small 2.8e-5; the log-mel 1.6e-3 (natural log).
  - The forward pass, tensor by tensor on the live geometry's first step: 0.5-1.5e-6 relative L2 per row.
  - Every 10 ms probability, with the reference's state: fixture 3.6-5.1e-6; conv03, conv08, conv15 at live and file
    2.5-5.9e-6. A stream that forgets its state between steps: 0.96-0.99.
  - The streaming state replayed on the recorded probabilities keeps the reference's frames at every step (33 steps of
    the small geometry with many compressions, and 359 of the conversations), except the two steps where torch's top-k
    met a tie at its boundary (conv03 file step 1, conv15 live step 54): the port takes equal scores by the lower index.
  - Free-running, the port's own state follows the reference's on the fixture and on 3 of the 6 conversation runs;
    on the others a near-tie made it keep other frames from there on (0.01-0.19), which is AC 3's to judge.
- **AC 17.** The forward pass needs the weights (107-397 MB), which are not in the tree, so every `cargo test` checks
  the log-mel and replays the speaker cache on the committed VoxPopuli fixture (16 s of conv04, CC0), and the forward
  legs are `--ignored` on the study disk.

### Quality (AC 3, 4, 5)

- **AC 3, passed.** The port's 10 ms probabilities segmented as NeMo-Speech.cpp segments its own (`der.py nsc`: its
  defaults, onset 0.641 / offset 0.561 / pads 0.229 / 0.079 s / gaps under 0.296 s filled / segments under 0.511 s
  dropped), scored by the finding's scorer at collar 0.25 on the twenty conversations: live 3.57 % against
  NeMo-Speech.cpp's chunk25 3.60 % (-0.03 pt), file 3.77 % against its offline 3.56 % (+0.21 pt). Without the same
  segmentation the comparison would measure the post-processing: the spike's Transformers rows (sigmoid > 0.5, no
  padding) differ from NeMo-Speech.cpp's by about 1.4 pt on the same 5 conversations.
- **AC 4, reported (does not gate).** AMI test meetings ES2004a, IS1009a and TS3003a (Mix-Headset, 46 min, English;
  pyannote's AMI-diarization-setup `only_words` references), live geometry, the same segmentation, collar 0.25:
  pooled 30.2 % (missed 27.7, false alarm 0.5, confusion 2.0); per meeting 14.1 %, 16.5 % and 51.0 %. TS3003a's speech
  is mostly missed (50.8 %), and not for its level: brought from -45.7 to -25 dBFS it scores the same. Speaker
  confusion stays low (0.1-7.5 %); what real meetings cost this model is detection, as the finding expected.
  - **No fallback for speech under 0.5 (scenario A, owner, 2026-10-26).** Of the 457 items `server_vad` makes of the
    three meetings with reference speech, 9 (about 1 s) have no frame a channel holds at 0.5 and get no speaker: the
    speech the diarizer misses, the VAD misses too, so no item is made of it. A fallback to the most active channel
    would label those 9 (8 right); it was not built.
  - **The ghost rule on real meetings** (F16, the speakers' timeline, four speakers in each): no ghost voice appeared
    with the rule off (speakers found 4, 4, 3 for ES2004a, IS1009a, TS3003a); the rule only took away. Items labelled
    right (ES2004a / IS1009a): 0 s 95.1 / 91.9 %, 1 s 94.4 / 91.9 %, 2 s 93.8 / 90.7 %, 3 s 92.6 / 90.7 %, 5 s 90.7 /
    90.7 %; at 4-5 s TS3003a's count fell to 2 and at 5 s IS1009a's to 3, a quiet participant (4 s of speech) merged
    into another. DER 40.7 % (0 s) to 40.9 % (5 s).
- **AC 5.** The ghost rule scored on the speakers' timeline (spec 08's rules over the 80 ms frames, no padding), live:
  0 s 4.70 % (speaker count right in 75 % of the conversations), 1-2 s 4.71 % (80 %), 3 s 4.91 % (85 %), 5 s 5.06 %
  (95 %), 6 s 5.44 % (95 %), 8 s 6.10 % (90 %, conv17's count worse than with 0); file 0 s 5.29 % (80 %) to 6 s 6.05 %
  (90 %). The rule only adds confusion (a channel's frames before it is established count for another), so the
  letter of AC 5 picks 0; **the default is 2 s** (the owner, 2026-10-26, with AC 5 amended after the AMI check below: 5 s was chosen first, for 95 % of the synthetic counts right, then dropped).

### Cost (AC 2, 11)

- **AC 2, the stored format: F16.** The DER, live, the same segmentation: F32 3.57 %, F16 3.57 % (the same counts and
  labels), Q8_0 4.15 % (+0.58 pt, two more counting errors: its matmuls quantize the activations to 8 bits too). The
  speed, conv01 live, 9 threads, the i9 idle: F32 6.2x real time (333 ms a step), F16 6.4x (322 ms), Q8_0 5.2x
  (392 ms). F16 is as good and as fast as F32 at half the RAM (199 MB against 397 MB). BF16 was not measured: the
  executor's kernels have no BF16 path; F16 takes its place (the same size, a finer mantissa).
- **A step, by part** (`examples/diarizer_profile`, the live geometry's 557 rows, F16, 9 threads): 447 ms, of which
  the attention 124 ms (a task per head and block of 16 queries, tiled in registers: the decoder's attention op takes
  2.1x as long on these rows), the MLP's matmuls 112 ms and the q, k, v projections 46 ms. Torch FP32 took about
  245 ms on the same rows in the spike.
- **AC 11, passed on the i9.** `crates/server/tests/speech_diarizer_study.rs` (`--ignored`): conv07 (297 s, three
  speakers) through `/v1/realtime` at real-time pace, `server_vad`, speakers live, Qwen3-ASR `qwen3-asr-1.7b` with
  refreshes every 2 s, on the one worker: a frame's probabilities came at most 3.6 s after its first sample (bound
  4.32 s: the live geometry's 2.32 s plus 2 s); 149 steps, 332 ms mean, 474 ms max; 39 items, 3 speakers.
  - As first built it failed (5.6 s): a step waited behind a refresh already running, seconds of decode on a long
    item; with refreshes nearly off it was 3.7 s. So the speakers' units go in a lane of their own, taking turns with
    the session's speech units a unit each, and a step that falls due stops the refresh under way at its next unit of
    compute, as a commit does (Q79). The refreshes thin out on long items. (Decision D6.) **Superseded, see below:**
    measured on the idle i9 with a long monologue (`speech_diarizer_refreshes`), the refreshes did not thin out but
    stopped: 0 in items of 137 and 147 s with speakers, against 26-32 without (the live text still for the whole
    item); the owner chose a diarizer thread of its own (2026-10-26), [below](#the-diarizers-own-thread-ac-10-11).
  - An item's speaker can come later than the bound by design: a cut at a change reaches back to where the new
    speaker's run began, up to `diarizer.min_speaker_seconds` before. The session's log line's `lag_max_ms` is the
    diarizer's pace, not that.
- **AC 11, the 4-core container** (podman, `--cpuset-cpus 0-3`, a Linux build of `examples/diarize`, F16, conv07 live,
  the diarizer alone): 3 threads 3.2x real time, 636 ms a step, 275 s of CPU for 297 s of audio, 23 % of the four
  cores; 4 threads 4.1x, 488 ms, 24 %. The card's low geometry took 54 % there in the spike. The run with Qwen3-ASR
  beside it was not made there: Qwen3-ASR's live refreshes alone do not keep up on that machine (spec 01).
- **RAM:** the F16 weights, 199 MB, mapped; a stream holds its speaker cache and FIFO (528 rows of 512, 1 MB) and the
  samples its next step reads; a diarized item keeps its audio for a cut (64 KB/s, 7.7 MB at the 120 s guard).

### The diarizer's own thread (AC 10, 11)

The owner's choice of 2026-10-26, on the condition that it hurt neither the commit's latency, nor the live text, nor
the other sessions; built on branch `speech-diarizer-thread` and confirmed by the measurements below.

- **Built:** one thread, `ignis-diarizer` (`DiarizerWorker`), serves every session's speakers' units in lanes that take
  turns a unit each; `CpuDiarizer` loads the weights with a pool of `diarizer.threads` (`ignis-diarizer-<i>`), dropped
  with them, and the speech executor's ISA. A session's stream sits behind a lock of its own: a speech unit holds the
  session's slot for its whole call, so a shared lock would have put the steps back behind the refreshes. The
  preemption (D6), the speakers' lane on the speech worker and the refresh gate's count of the steps are gone; the
  queue limit, the tail on a stall, the cuts and the log line are unchanged.
- **`diarizer.threads`.** conv01 (198 s) through `examples/diarize`, live, F16, the i9 idle: 2 threads 884 ms a step
  (median; 1,169 ms max), 2.4x real time; 3 threads 629 ms (1,102), 3.3x; 4 threads 564 ms (1,473), 3.5x. The third
  thread adds 38 %, the fourth 8 %: `auto` is a third of the physical cores, at least 2, so 3 on the i9's 10.
- **Before and after**, the same i9 (`9585b5c5`, the first design with the leg's commit timing, against this
  build), each leg run with no `cargo`, `rustc` or `link` in `tasklist` and the desktop's own 7-22 % of the 20 logical
  CPUs (`typeperf`), nothing else running:

  | leg | before | after |
  |---|---|---|
  | `speech_diarizer_refreshes`, 125192-240: refreshes in the two long items (130-147 s), speakers off | 28, 24 (still at most 19.0, 14.9 s) | 29, 27 (14.1, 12.9 s) |
  | the same, speakers `live` | 0, 0 (136.2, 145.3 s) | 32, 24 (13.9, 15.0 s) |
  | AC 11, conv07: `lag_max_ms` (bound 4,320) | 3,858 | 3,946 |
  | AC 11: `committed` to `completed`, 38 items: median, p90, max | 2,240, 4,077, 5,920 ms | 2,201, 2,977, 3,513 ms |
  | AC 11: a step, mean and max | 353, 528 ms (the speech model's 9 threads) | 732, 1,634 ms (3 of its own) |
  | spec 03's dictation leg, d10 (67 s), no speakers: commit to `completed`, two runs | 2,678, 2,726 ms | 2,506, 2,678 ms |

  With speakers the live text keeps the pace it has without them; the commits are no slower, and their tail is
  shorter, since a step no longer takes a turn in front of them; a step takes twice as long on 3 threads beside the
  decode, and the speakers still keep within AC 11's bound; a session without speakers sees no change (no pool exists
  until the diarizer loads). The dictation's 2,000 ms of spec 03's AC 13 failed on this box before as after: the leg's
  own figure, unchanged by this.
- The 4-core container was not measured with the thread: there `auto` is 2 threads beside the speech model's 3.

### The text in a session with speakers (AC 7)

- **AC 7, passed.** `speech_diarizer_study.rs`'s second leg (`--ignored`): the twenty conversations (77 min), each in a
  session of its own, fed at twice real time with speakers `live`, Qwen3-ASR `qwen3-asr-1.7b` and the diarizer (F16)
  on the one worker, the test profile, the i9 shared with another session's run: 347 items, 9 of them cut at a change
  of speaker (conv03, conv05, conv12, conv15 five times, conv20; none by the length guard). No item had a delta before
  its `committed`; each had at most one delta, its deltas joined equal to its `completed.transcript`, and its speaker
  after them and before `completed`. No `error`, no `failed`. Speakers heard: 2 to 6 a conversation.
- The worker ran behind the 2x feed (each conversation's last item completed 77-417 s after its audio ended), so most of
  the 119 changes the references hold with under 0.5 s of silence were found after their item had ended and split
  nothing, as designed. A worker that keeps up should cut more of them (not measured here).

### The length guard (AC 12)

- **Items:** continuous Italian from two VoxPopuli speakers (`long_items.py`): their segments concatenated, every run of
  frames under `server_vad`'s threshold longer than 0.3 s shortened to it, then the first segments reaching 60, 120
  and 240 s (67-78, 121-131 and 246-251 s); the longest quiet run left is 0.32 s, so the VAD never ends them.
- **Qwen3-ASR 1.7B (Q4_K_M + Q8_0 encoder)**, partials every 2 s of audio pushed as fast as it goes, the machine
  shared (times are upper bounds):

  | item | WER 125192 | WER 28429 | commit 125192 | commit 28429 | audio tokens |
  |---|---|---|---|---|---|
  | 60 s | 23.0 % | 25.4 % | 5.8 s | 6.6 s | 873-1017 |
  | 120 s | 21.3 % | 25.2 % | 6.5 s | 9.2 s | 1576-1699 |
  | 240 s | 15.4 % | 26.1 % | 30.2 s | 26.1 s | 3195-3258 |

  The WER does not grow with the length (its level is VoxPopuli's references: spoken-form numbers, words at the
  segments' edges missing from them, and one sentence the model dropped in the 60 s item already). The commit grows
  four-fold from 120 to 240 s, and each partial re-reads the stable text, so a 240 s item's refreshes cost far more
  than its audio. The decoder's cache holds 112 KB per token (28 layers, 8 KV heads of 128, f16): about 0.24 GB at
  120 s and 0.49 GB at 240 s per session. **`speech.max_utterance_seconds` stays 120.**
- **The cut search:** from any point of these speakers' speech the wait to the first frame under the VAD's threshold
  is p50 1.9-2.8 s, p90 5.0-8.2 s, p95 6.6-10.0 s, max 13-21 s: Silero stays over 0.5 across the gaps between the
  words of fluent speech. **`speech.utterance_cut_search_ms` defaults to 10,000 ms** (the owner, 2026-10-26): the p95 of both speakers, so 95 % of the cuts fall in a dip and an item runs at most N + 10 s.

### The server (built in slices E-G, decisions the spec left open)

- `session.ignis.diarization` changes apply at once while the current item holds no audio, else from the next item;
  a new stream numbers its speakers afresh. A diarizer that fails mid-session is an `error`, `ignis_diarizer_failed`,
  and the session goes on without speakers; an unavailable one refuses the opt-in with `ignis_diarizer_unavailable`.
- **The tail.** An item's speaker waits for the frames over its end, which in a stream that flows the next step brings
  with its look-ahead. A stream that stalls, no audio for as long as its next step's missing samples take at real time
  plus the client's last piece again (a client at real-time pace sends its next piece that long after the last), gets
  them from a copy of the stream finished: what it heard past its last step, scored with no look-ahead (a file's end,
  a pause in what the client sends, a commit with no audio after it). The tail's frames cut nothing and serve the
  items whose end they cover, those ended after it too, until the steps reach as far. The wait is at most the
  geometry's latency (2.32 s live, 30.4 s file) and the last piece. The log line counts the tails (`tails`) apart from the
  steps; their compute is not in `step_mean_ms`.
- A change found only after its item ended does not split it (the item takes its majority speaker). An item with no
  speech gets no speaker event. The speaker is the model's channel, so numbers can skip one the ghost rule held back
  (spec 07 numbers the speakers by first appearance).
- The length guard's pause cut lands at the end of the quiet VAD frame, its timed cut at the end of the frame where the
  item reaches N plus the search; idle-timeout items and items a commit makes are not guarded.
- **The queue limit counts the audio not yet computed**, each sample once: the larger of the speech model's (samples
  not yet queued and in its units still in the worker) and the speakers' (samples not yet sent and in their units in
  the worker), not the audio since the last `completed`: read literally, a dictation longer than the
  limit (its guard off) would be refused at real time. (Decision D3.)
- **ADR 0046's applicability is judged per model slot:** a family judges only the fields of its own slot (the active
  model, or a model loaded beside it, as the speech model and the diarizer are), so a `diarizer.*` value is not
  refused while a text model is active. (Decision D4.)
- The override ranges: threshold 0.001-1, boost 0-10, rates 0-4, positive rate 0-1, silence frames 0-125,
  `min_speaker_seconds` 0-60, `max_queued_audio_seconds` 1-3600 (no 0-off).
- `diarizer.model`: `auto` takes the first diarizer entry in catalog order (built-in before the operator's). The
  built-in entry is `nemotron-3-diarization`, the F16 GGUF on `gpillon/Nemotron-3-Diarization-GGUF-ignis` (public,
  OpenMDW-1.1's licence beside it), uploaded with the owner's OK after the licence check (Q115, decision D5).
- A live change of `diarizer.model` takes the next session's load; a session open with speakers keeps its stream,
  and with it the old weights and their pool, until it ends. Meanwhile `GET /v1/models` reads the new diarizer, not
  loaded, and the old one's file is not refused by a removal (on Windows the removal fails while the file is mapped).
  The owner chose (2026-10-26) to document this rare case rather than track the weights an open stream still holds.
- **The Server view** reads the diarizer from `GET /v1/models`' `speech.diarizer`: beside the loaded model's figures,
  **Speech** (the speech model's id, the `kind: speech` entry of `data`) and **Diarizer** (its id, `Loaded` or `Not
  loaded`), either `Unavailable` with the server's reason under the figures. The Server view had shown no speech
  model before, so both are new; the diarizer's catalog row keeps its Download and Remove, and no row loads either.
