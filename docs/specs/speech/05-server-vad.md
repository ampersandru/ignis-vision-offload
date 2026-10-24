# 05 - phase 2: `server_vad`

GitHub: #355 (master #350). Phase 2 (Q56). Feature: `spec.md`. Needs 03.

ADRs: 0051 (adapted code, licences), 0066 (the speech model).

Decided 2026-10-23/24 in the speech grill (Q22, Q54, Q56). Details the grill
left open were decided by the owner on 2026-10-24, cited as
`(owner, 2026-10-24)`.

## Problem Statement

- OpenAI's Realtime transcription sessions default to `server_vad`, where
  the server finds the speech and commits on its own (threshold 0.5, prefix
  padding 300 ms, silence 500 ms). A client written for it never sends
  `commit`.
- Qwen3-ASR has no VAD. Its `language None` output marks no speech after
  the fact, not while the audio streams.
- Until this spec, spec 03 refuses `server_vad`. In phase 1 a session that
  sends no `turn_detection` runs with manual commit (Q54).

## Solution

- **A small model-based VAD** of the Silero class: ~2 MB, MIT, run on the
  CPU, ported to Rust (Q22). It uses Silero VAD's published weights and graph
  (owner, 2026-10-24). The port is adapted code under ADR 0051 (provenance in
  its header and in `NOTICE`), and the weights ship with their licence.
- **`turn_detection: {type: "server_vad", …}`** honours `threshold`,
  `prefix_padding_ms` and `silence_duration_ms` (Q22). On speech it sends
  `input_audio_buffer.speech_started` (`audio_start_ms`, `item_id`). After
  `silence_duration_ms` of silence it sends `speech_stopped` (`audio_end_ms`)
  and commits the item itself, keeping `prefix_padding_ms` of audio before
  the speech start (Q20, Q22).
- **The default.** From this phase on, a session that sends no
  `turn_detection` gets `server_vad`, as OpenAI's do (Q22, Q54). The
  Playground keeps sending `null` and committing at Done (Q33).
- **`semantic_vad`** stays refused (Q22).
- **The other `server_vad` fields** (owner, 2026-10-24): `idle_timeout_ms`
  is honoured. `create_response` and `interrupt_response` are inert, since
  they belong to conversation sessions.
- **Config fields** for the server's defaults, by the owner's rule for
  thresholds (owner, 2026-10-24): `speech.vad_threshold` 0.5,
  `speech.vad_prefix_padding_ms` 300, `speech.vad_silence_duration_ms` 500.
- **The batch route's `chunking_strategy` stays refused** (owner,
  2026-10-24). Q19 rules out an automatic split in v1, and a VAD split is one.

## Acceptance

1. The VAD runs in Rust on the CPU. Its per-frame speech probabilities match
   the reference implementation within a recorded threshold on the FLEURS
   clips and the dictations. Its licence is in `NOTICE` (Q22; owner,
   2026-10-24).
2. `server_vad` honours `threshold`, `prefix_padding_ms` and
   `silence_duration_ms`, with OpenAI's defaults 0.5 / 300 / 500 (Q22).
3. `speech_started` and `speech_stopped` are sent with their times, and the
   server commits after the silence, keeping the prefix padding (Q20, Q22).
4. A session that sends no `turn_detection` runs `server_vad`, and
   `session.created` says so (Q22, Q54, Q56).
5. `semantic_vad` is refused (Q22).
6. The Playground's dictation is unchanged: `null`, committed at Done (Q33).
7. On each dictation fed at real-time pace with `server_vad`, the items it
   commits, joined, have a pooled WER at most 1 pt worse than 03's manual
   commit of the whole dictation, after Q52's 1-pt rule
   (Q52; owner, 2026-10-24).
8. `cargo test` passes workspace-wide (AGENTS.md, not a grill decision).

## Out of Scope

- `semantic_vad` (Q22).
- An automatic split of long batch clips (Q19).
- The GPU loan: spec 06, phase 3 (Q56).

## As built (GitHub #355)

Built on `speech-355` from `speech-asr` at `994d36d0`. Where the build decided
something the spec left open, or differs from it:

- **The VAD** is `ignis_speech::vad` (`crates/speech/src/vad.rs`), an adapted
  op under ADR 0051: Silero VAD 6.2.3's 16 kHz graph (snakers4/silero-vad tag
  v6.2.3, commit 5cd79456) restated in Rust, single-threaded scalar f32, with
  no ONNX runtime, no C and no build script. Each 512-sample frame (32 ms) is
  read with the 64 samples before it, and the LSTM state carries from frame
  to frame. The oracle leg reads 858 s of audio in 8.5 s on the i9, WAV reads
  included: under a hundredth of one core.
- **The weights** are the wheel's `silero_vad.jit`, the model
  `load_silero_vad()` returns (sha256 `e1122837…`). `tools/speech-vad/record.py
  convert` wrote them once into `crates/speech/vad/silero-vad-16k.safetensors`
  (1,239,884 bytes, sha256 `8597e191…`, the same bytes on every run). They are
  built into the binary, so a bare `cargo build` fetches nothing and the tests
  run offline. The wheel also ships a `silero_vad_16k.safetensors`, whose
  weights are not the `.jit`'s (conv1's differ by up to 4.0), so it is not
  used. The licence is beside the file and in `NOTICE`.
- **AC 1's oracle** records the `.jit`'s per-frame probabilities as the
  wheel's `utils_vad.py` reads a clip: `reset_states()`, 512-sample chunks,
  the last zero-padded. The graph written out in torch from the converted
  file reproduces the `.jit` to 4.2e-7, which checks the file and the graph.
  The bound is ten times the `.jit`'s own distance from fp64: 3.9e-5. The
  port is 4.0e-6 from the `.jit` on the fixture clip (in every `cargo test`),
  at most 7.7e-6 on the 40 FLEURS clips and 4.6e-6 on the dictations (the
  `--ignored` leg). The negative control, the port with its state reset at
  every frame, is more than 100 times the bound away. The fixture clip's and
  FLEURS' records are in `crates/speech/tests/fixtures/vad/`; the dictations'
  are on the study disk (`results/vad/`).
- **Where it runs.** On the socket's task, on the 16 kHz stream after
  resampling, frame by frame as audio arrives; never as a worker unit, so a
  32 ms frame never waits behind an 8 s window. The `audio_*_ms` count the
  16 kHz samples the session has heard, from its first, and fall on 32 ms
  steps. At 24 kHz they count after the resampler, whose delay shifts them by
  a few milliseconds against the client's own count; the tests send 16 kHz.
- **The rules.**
  - A frame is speech at or over `threshold`, in both directions. Silero's own
    iterator ends speech under `threshold − 0.15`; that 0.15 would be a
    threshold with no field, so it is not used.
  - An item starts at the first speech frame. Its padding is up to
    `prefix_padding_ms` of the audio before that frame, but never earlier than
    the end of an earlier item's last speech frame, a commit or a clear: it may
    take the silence that ended the item before, never its speech, so two
    items' times overlap by that silence when speech resumes at once (speech
    under the threshold in it can reach both). With padding that stopped at
    the item before, AC 7's first run had an item start right at its speech
    and come out as one word. Letting the padding take that silence fixed it,
    and moved the pooled WER from 8.06% to 7.58%; the rule was found on AC 7's
    own dictations.
  - The item ends at the end of the first frame where the silence since its
    last speech frame reaches `silence_duration_ms`. It holds that silence,
    and `audio_end_ms` is there.
  - `idle_timeout_ms` follows OpenAI's documented shape: that much audio with
    no speech, since the session opened, the last item ended or the field
    changed, is committed after `input_audio_buffer.timeout_triggered` and
    transcribed (usually to an empty transcript). It counts audio time, and
    takes 1 ms to `speech.max_clip_seconds`; OpenAI's SDK types state no
    range, so none narrower is refused.
- **Commit and clear under `server_vad`.** A client commit ends the item under
  way, with no `speech_stopped` (OpenAI's `speech_started` says as much). With
  no speech heard since the last item it is `input_audio_buffer_commit_empty`,
  where OpenAI would commit what its buffer holds: audio the VAD heard no
  speech in goes nowhere here. Neither a commit nor a clear resets the
  resampler, since the stream goes on and its clock with it. A clear drops
  the utterance and the padding the VAD holds.
- **`session.update`** merges `server_vad`'s fields as it does the
  transcription's: a field left out keeps its value, and `null` is the
  server's default (`speech.vad_*`, read when the session opens; no idle
  timeout). The three lengths take at most `speech.max_clip_seconds`, since an
  item holds no more. An object needs its `type`. `create_response` and
  `interrupt_response` are echoed when set. The fields hold at once; a switch
  between `null` and `server_vad` while the item holds audio holds from the
  next item, as a changed context does.
- **AC 6.** The Playground is unchanged: `web/src/speech/realtime.ts` sends
  `turn_detection: null` (`realtime.test.ts` pins it), and
  `speech_realtime_vad.rs` shows a `null` session getting no VAD event and
  every sample in its item. The realtime tests' 16 kHz helper (`at_16_khz`)
  now asks for manual commit too.
- **AC 7 is not met.** On the 11 dictations, `qwen3-asr-1.7b` at the defaults
  (0.5, 300, 500), fed 100 ms at a time with a second of silence after the
  last word, the items joined score 7.58% pooled WER (47 edits in 620
  words); a manual commit of each whole dictation through the same path
  scores 4.03% (25 edits). Six dictations of eleven are within a word;
  d04, d06, d07, d10 and d11 are three to seven words worse. In their
  transcripts no speech is lost to a cut: the errors are the kind a different
  start brings, numbers in spoken form ("ventisette b" for 27B), an
  identifier read out, a word heard differently ("web file" for WAV file). One run of two more arms places the
  cost:
  - a control, the whole dictation as **one** item from where
    `server_vad`'s first item starts (its leading silence trimmed to the
    300 ms padding, nothing cut), scores the same 7.58% pooled: d06 goes from
    4 edits to 13, d03 from 3 to 9. Dictation by dictation the control and
    `server_vad` differ (d03 is worse trimmed, d04 worse cut), so the run
    shows that trimming the leading silence alone costs as much as the whole
    gap, not that the cuts cost nothing. The model's output depends on how
    much audio comes before the speech, and the 1-pt gate against the
    untrimmed commit is below that;
  - `server_vad`'s items, each with its predecessors' transcript as its
    context, which is not built (Q50's context is the client's), score
    6.94%.

  The owner's call: restate AC 7 against the trimmed commit, build a
  mitigation (more leading audio, a longer padding default or synthetic
  silence; the predecessors' text as context), or accept the gap. The
  `--ignored` leg (`crates/speech/tests/vad.rs`) asserts AC 7 as written.
- **`chunking_strategy`** on the batch route stays refused, unchanged.
