---
title: "Speech"
description: "The speech model: which one, its compat names, its limits and its queue."
sidebar:
  order: 12
---

The `speech` group configures the [speech model](../api/transcriptions.md)
(Qwen3-ASR, on the CPU, beside the model on the card). Every field is read
live: `PATCH /v1/config` changes it with no reload, and `threads` and
`pinned_weights` hold from the model's next load. The generated list, with
every default, is [the configuration reference](../reference/configuration.md).

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--speech-model` | `IGNIS_SPEECH_MODEL` | `auto` | `auto` is `qwen3-asr-1.7b`, the 1.7B model with a Q4_K_M decoder; a catalog id picks another entry; `off` turns the speech model off. It is on when its files are on disk (`ignis-server model download <id>`); nothing downloads by itself. |
| `--speech-compat-names` | `IGNIS_SPEECH_COMPAT_NAMES` | `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-transcribe`, `gpt-live-transcribe` | The `model` names OpenAI clients send that the speech model answers to (Aider hard-codes `whisper-1`). `;`- or `,`-separated. Not catalog aliases and not listed by `GET /v1/models`. A name equal to a served model id is refused at start. |
| `--speech-threads` | `IGNIS_SPEECH_THREADS` | `auto` | The CPU threads the model computes with; `auto` is the physical cores less one, counted within the CPUs the server may run on (a container's cpuset or CPU quota). Each load logs the count and whether `auto` chose it (`ignis.speech.threads`). |
| `--speech-max-clip-seconds` | `IGNIS_SPEECH_MAX_CLIP_SECONDS` | `1200` | The longest clip one transcription takes (1..1200, the model's limit). A longer one is a `400`; nothing is split. |
| `--speech-context-max-tokens` | `IGNIS_SPEECH_CONTEXT_MAX_TOKENS` | `512` | The most tokens of `prompt` and `keywords[]` a transcription reads; past it the last are kept. 0 reads none. |
| `--speech-pinned-weights` | `IGNIS_SPEECH_PINNED_WEIGHTS` | off | Pin the weights in host memory instead of mapping the file: the model's tensors are copied into the server's own memory at load (and locked in RAM where the OS allows it, on Linux), so the files are not held open. |
| `--speech-idle-unload` | `IGNIS_SPEECH_IDLE_UNLOAD` | `300` | Seconds with no transcription after which the model is unloaded; the next one loads it again. 0 never unloads. |
| `--speech-max-queue` | `IGNIS_SPEECH_MAX_QUEUE` | `4` | The transcriptions that may wait behind the one running; one more is `503 engine_full` with `Retry-After`. 0 runs one at a time. |
| `--speech-max-sessions` | `IGNIS_SPEECH_MAX_SESSIONS` | `4` | The most Realtime sessions open at once. Read by the Realtime route, which is not served yet. |
| `--speech-partial-interval-ms` | `IGNIS_SPEECH_PARTIAL_INTERVAL_MS` | `2000` | How often a Realtime session's live transcript refreshes. Read by the Realtime route, not served yet. |
| `--speech-rollback-tokens` | `IGNIS_SPEECH_ROLLBACK_TOKENS` | `10` | The tokens of a live transcript held back as provisional. Read by the Realtime route, not served yet. |
| `--speech-vad-threshold`, `--speech-vad-prefix-padding-ms`, `--speech-vad-silence-duration-ms` | `IGNIS_SPEECH_VAD_*` | `0.5`, `300`, `500` | `server_vad`'s threshold, the audio kept from before speech, and the silence that ends a turn. Not served yet. |
| `--speech-loan-grace-seconds`, `--speech-loan-wait-seconds` | `IGNIS_SPEECH_LOAN_*` | `30`, `5` | The loan of free KV pages to the speech model. Not served yet. |

The fields marked "not served yet" are declared so that a configuration file
written today keeps its meaning: the routes and the loan that read them come
with their own tickets ([ADR 0066](../../adr/0066-the-speech-model-waits-on-the-host-and-borrows-free-kv-pages.md)).
