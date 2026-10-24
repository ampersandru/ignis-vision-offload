---
title: "Live transcription"
description: "Dictation over a WebSocket: OpenAI's Realtime transcription session on /v1/realtime, with the text as it settles."
sidebar:
  order: 6
---

`GET /v1/realtime?intent=transcription` opens **OpenAI's Realtime
transcription session** on the [speech model](transcriptions.md): the client
streams audio while the person speaks, and the server sends the text as it
settles. On the CPU this is the fast path: the model encodes and prefills the
audio while it arrives, so a commit leaves only the last few seconds to
decode, where [`POST /v1/audio/transcriptions`](transcriptions.md) starts on
the whole clip once it is in
([ADR 0066](../../adr/0066-the-speech-model-waits-on-the-host-and-borrows-free-kv-pages.md)).

It is the GA protocol's transcription subset: a client written for OpenAI's
`wss://api.openai.com/v1/realtime?intent=transcription` connects with the
base URL changed. Conversation sessions (a `model` in the URL, voices,
responses) are not served.

## Opening a session

```text
GET /v1/realtime?intent=transcription
Authorization: Bearer <key>
```

From a browser, which cannot set that header, offer the key as a subprotocol:
`new WebSocket(url, ["realtime", "openai-insecure-api-key.<key>"])`. The server
selects `realtime` and never echoes the key. Without a key configured, no
header is needed.

The upgrade is refused before the socket opens:

| Answer | When |
|---|---|
| `400`, `param: "intent"` | the URL has no `intent=transcription`, or names a `model`: that is a conversation session |
| `401 invalid_api_key` | a key is set and the upgrade carried neither the bearer token nor the subprotocol |
| `503 speech_unavailable`, no `Retry-After` | the speech model cannot serve; the message says why, as `GET /v1/models`' `speech` object does |
| `503 engine_full`, with `Retry-After` | `speech.max_sessions` (4) sessions are open; one more waits for one to close |

A plain `GET` of the route, with no upgrade, is answered what an upgrade would
be, and claims no session: the same `400` for a wrong `intent`, the same `503`s,
and, when the socket could open, `400 upgrade_required`. A page that cannot read
a refused upgrade's status asks this way why it was refused.

On open the server loads the speech model if it is not loaded, prefills the
context, and, if the text model was [unloaded for being idle](idle-unload.md),
starts loading it in the background: a person dictating a prompt sends it
next. The session is served on the CPU meanwhile, and through a
[model switch](model-switching.md). It is no use of the text model and never
keeps it loaded.

The first event is `session.created`, with the configuration in effect:

```json
{ "type": "session.created", "event_id": "event_…",
  "session": { "type": "transcription", "id": "sess_…",
    "audio": { "input": {
      "format": { "type": "audio/pcm", "rate": 24000 },
      "transcription": { "model": "qwen3-asr-1.7b", "prompt": "", "language": null },
      "noise_reduction": null,
      "turn_detection": null } },
    "include": [],
    "ignis": { "provisional_tail": false } } }
```

`turn_detection: null` means the client commits: the server does not detect
the end of speech yet.

## Configuring it

`session.update` changes what it names and is answered `session.updated` with
the whole configuration in effect. A field left out keeps its value, `null`
puts it back to its default, and one refused field refuses the whole update
with an `error`, the session keeping the configuration it had.

```json
{ "type": "session.update", "session": { "type": "transcription",
  "audio": { "input": {
    "format": { "type": "audio/pcm", "rate": 16000 },
    "transcription": { "prompt": "A talk about GPUs.", "keywords": ["ignis", "Qwen3"] } } },
  "ignis": { "provisional_tail": true } } }
```

| Field | |
|---|---|
| `audio.input.format` | `{type: "audio/pcm", rate: 24000}`, OpenAI's, resampled to 16 kHz; or `rate: 16000`, the model's own rate, an ignis extension that skips the resampling. `audio/pcmu` and `audio/pcma` (G.711) are refused. |
| `audio.input.transcription.model` | Absent, the speech model's id, or one of `speech.compat_names` (`gpt-4o-transcribe`, `whisper-1`, ...): the session reports the speech model's id. A text or image model's id is an `error` with `model_kind_mismatch`; any other name `model_not_found`. |
| `audio.input.transcription.prompt`, `.keywords` | The context, as for [a transcription](transcriptions.md#the-request): names and terms to spell right. It is **prefilled when the update arrives**, so it costs nothing at the commit. A change while an item holds audio applies from the next item. |
| `audio.input.transcription.language`, `.languages` | As for a transcription: one language forces it, several detect among them, both together are refused. Leave it off unless the model guesses wrong. |
| `audio.input.transcription.delay` | Accepted and inert: the live text refreshes at most every `speech.partial_interval_ms` (2000). |
| `audio.input.turn_detection` | `null`. `server_vad` is refused (not served yet) and `semantic_vad` always: commit with `input_audio_buffer.commit`. |
| `audio.input.noise_reduction` | `null`. `near_field` and `far_field` are refused. |
| `include` | Empty. `item.input_audio_transcription.logprobs` is refused. |
| `ignis.provisional_tail` | `true` to receive the provisional tail, below. |

`transcription_session.update`, the retired beta's event, is an `error` naming
`session.update`, and every conversation event (`response.create`,
`conversation.item.create`, ...) is an `error`. No `error` closes the socket.

## Sending audio

```json
{ "type": "input_audio_buffer.append", "audio": "<base64>" }
```

`audio` is base64 of 16-bit little-endian mono PCM at the session's rate, up
to 15 MiB of base64 an event. Nothing is sent back. The audio between two
commits is one **item**, which gets its `item_id` with its first append.

- `input_audio_buffer.commit` ends the item: `input_audio_buffer.committed`
  `{item_id, previous_item_id}` at once, then its transcription. A commit with
  no audio since the last one is an `error`, `input_audio_buffer_commit_empty`.
- `input_audio_buffer.clear` discards the item, its audio and its partial text,
  answered `input_audio_buffer.cleared`.

## The text

While audio arrives the server refreshes the item's transcript at most every
`speech.partial_interval_ms`, and once the item is committed it decodes the
rest. Refreshes are best effort: each one decodes the item so far again, so
while the session's audio is still being computed a refresh is skipped rather
than queued behind it. A CPU that cannot keep up sends fewer deltas, never a
later transcript: the audio and the commit go first.

- `conversation.item.input_audio_transcription.delta` `{item_id,
  content_index: 0, delta}` carries **stable text only**, text the model will not
  revise, so deltas only ever append.
- `conversation.item.input_audio_transcription.completed` `{item_id,
  content_index, transcript, usage: {type: "duration", seconds}}`. Its
  `transcript` is the item's deltas joined.
- `conversation.item.input_audio_transcription.failed` `{item_id,
  content_index, error}`: the item held more than `speech.max_clip_seconds`
  (1200) of audio, it ran past `server.stream_timeout` from its first append
  (`request_timeout`, [deadlines](deadlines.md)), or the speech model failed on
  it. It comes at the commit, or at once if the item is already committed; an
  item that fails before its commit also gets an `error` at once, with the same
  reason, so the client can stop sending its audio. The session goes on with the
  next item.

**The provisional tail** is ignis's. The last few tokens of each refresh
are not stable yet: the next refresh may revise them (how many is
`speech.rollback_tokens`, 10, which the speech executor reads when a session
opens, so a change holds from the next session; the first `speech.unfixed_partials`
(2) refreshes fix no text at all, so the language and the opening words are
decided on more than one interval of audio). A session that set `ignis.provisional_tail: true` receives them as

```json
{ "type": "ignis.input_audio_transcription.provisional", "item_id": "item_…", "content_index": 0, "text": " per il" }
```

after each refresh that changed them: the whole current tail, which replaces
the previous one, and `""` when the item ends. Show the deltas, then the tail
greyed after them. A session that does not opt in never receives the event, so
a client that rejects unknown events is safe.

## Several sessions, and the queue

One compute worker serves the speech model. Up to `speech.max_sessions` (4)
sessions are open at once; they take turns a unit of work at a time (a window
of audio, a refresh, a commit) and every unit goes ahead of the
[transcription requests](transcriptions.md#waiting-and-refusing), which run in
the gaps.

A session with no committed item still being transcribed and no frame from its
client for `server.socket_idle_close` seconds (300; 0 never) is closed with a
normal close frame (1000) whose reason says so, as the
[Responses socket](responses.md) is, and the server logs `ignis.ws.idle_closed`
with `route: "realtime"`. Open a new session to go on.

While a session is open the speech model stays loaded, however quiet the
session; `speech.idle_unload` counts from the end of the last one.

## What is logged and counted

As for a transcription: neither the audio nor the text is logged. Each
committed item writes one `ignis.speech.transcription` line, its wait counted
from the commit, and counts in `ignis_speech_transcriptions_total` and
`ignis_speech_commit_wait_seconds`; `ignis_speech_sessions_open` is the open
sessions ([metrics](../reference/metrics.md)).
