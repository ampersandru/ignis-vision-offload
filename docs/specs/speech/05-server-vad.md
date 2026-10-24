# 05 - phase 2: `server_vad`

GitHub: to be filed. Phase 2 (Q56). Feature: `spec.md`. Needs 03.

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
