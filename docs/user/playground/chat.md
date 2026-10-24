---
title: "Chat"
description: "The Playground's chat: a reply with its reasoning and figures, the prompt box and its dictation, the model picker, the context meter and the session log."
sidebar:
  order: 2
---

The **Playground** tab is a streaming chat with the loaded model. Sessions are
on the left ([Sessions](sessions.md)), the settings on the right
([Settings](settings.md)), the conversation in the middle, and under all three
the session's log. Every reply shows what the model reasoned, what it called,
what it answered, and the figures the browser measured while it streamed.

![A reply: reasoning folded, the answer, the figures line and the context meter's details](../images/playground/chat-reply.png)

## Sending a prompt

Type in the box at the foot: **Enter** sends, **Shift+Enter** starts a new
line. While the reply streams the button reads **Stop**, which ends the turn,
agents included. The **+** button adds to the prompt:

- **Image**: pictures sent with the prompt as `image_url` parts. The server
  must run with image input on ([vision](../configuration/vision.md)); the
  built-in `rtx5090` profile turns it on. A pasted image lands here too. Each
  thumbnail can be **marked**: circles or strokes drawn on it, so the model
  sees what you point at.
- **File**: a text or PDF file the model reads in pieces with its `read_file`
  tool; it is offered while the Attachments tool is on
  ([Tools and agents](tools-and-agents.md#attachments)).

An image alone is a prompt. Nothing is sent until you press Send.

## Dictating a prompt

The **mic** beside Send dictates into the box, with the
[speech model](../api/transcriptions.md) on the server. Click it, or press
**Ctrl+Shift+Space** with the focus in the box: held for longer than 400 ms it
is push-to-talk, and letting go ends the dictation; tapped, it starts one, and
the next tap ends it.

While you speak:

- the mic becomes **Done**, and **Send** waits: a dictation is never sent by
  itself;
- a strip above the box shows a level meter, the words as they settle (light),
  the last few words the model may still change (grey), and a timer;
  **Listening…** before the first word;
- the box's underline runs.

**Done** ends it. The strip reads **Finishing** while the server decodes the
last words; then the text lands at the cursor, selected, with a space added
where it would touch a word. Edit it as you like, and press Send when it is
ready. **Esc**, or **Discard** on the strip, throws the dictation away and
leaves the box as it was.

The line under the box follows along:

| When | It says |
|---|---|
| idle | Enter sends, Shift+Enter adds a line. Dictate with the mic, or hold Ctrl+Shift+Space while you speak. |
| listening | Listening. Done puts the text at the cursor; Esc discards it. |
| finishing | Finishing the last words. |
| for 6 s after | Inserted N words at the cursor. Nothing is sent until you press Send. |
| for 4 s after Esc | Dictation discarded. |
| every session taken | Speech is busy with another dictation. Trying again in N s. |
| refused again | The dictation still could not open: speech is busy. Press the mic to try again. |
| the server is up but the socket never opened | Dictation could not start: the server is up, but this address does not carry WebSockets. |
| the connection closed | Dictation stopped: the connection to the server closed. Insert what was heard, or discard it. |
| the server failed it | Dictation stopped: *the server's reason*. Insert what was heard, or discard it. |
| Done with no audio, or nothing said | Nothing was heard. |
| the server unreachable, the mic broken | Dictation could not start: *why*. |

**Busy.** The server holds `speech.max_sessions` (4) dictations at once
([speech](../configuration/speech.md)). One more is refused; the page counts
down 3 seconds and tries once more. A browser cannot read why a socket was
refused, nor its `Retry-After`, so the page asks the same route with a plain
`GET`, which the server answers as it would the upgrade: `503 engine_full` is
busy, `503 speech_unavailable` is the unavailable reason, and a server that
would have opened the socket says so, which means something between the page
and the server (a proxy that does not carry WebSockets) lost the upgrade.

**A stopped dictation.** If the connection closes, or the server fails the
dictation, the strip keeps what was heard (the settled words and the last grey
ones): **Insert what was heard** puts them at the cursor, **Discard** drops
them.

**One at a time.** The page dictates into one field at once. Pressing another
mic, in the [Image studio](image-studio.md) or [Decide](decide.md), ends the
dictation under way, and its text still lands in its own field.

### What a dictation sends

Each dictation is one session on the server's Realtime transcription route,
`GET /v1/realtime?intent=transcription`, opened at the press and closed when
the text lands. Pressing the mic is also what loads the speech model, if it is
not loaded. The session asks for:

- **manual commit** (`turn_detection: null`): **Done** commits;
- **16-bit PCM at 16 kHz**, the model's own rate: the browser downsamples the
  mic and sends 100 ms of audio at a time;
- **the provisional tail** (`session.ignis.provisional_tail: true`): the grey
  words;
- **a context**, as the transcription's `prompt`: the last reply, then what is
  already in the box, so names and terms are spelled the way the conversation
  spells them. The server reads its last `speech.context_max_tokens` (512)
  tokens.

No language is sent: the model detects it. The audio is never written to disk
or to a log.

### When the mic is struck through

A struck-through mic cannot start, and the line under the box says why:

| It says | Why | What to do |
|---|---|---|
| The microphone needs https or localhost. … | The browser gives a page a microphone only over https or on localhost. | Open the Playground through an https proxy, or start the server with `--server-expose cloudflare-quick` ([access](../configuration/access.md)). |
| No speech model on this server. Download one with `ignis-server model download <id>`. | The speech model's files are not on disk. | Run that command ([fetching a model](../guides/models.md#fetching-one-ignis-server-model-download)). |
| Speech is turned off on this server (`speech.model = off`). | The server was started with the speech model off. | Set `--speech-model auto` ([speech](../configuration/speech.md)). |
| Microphone access is blocked for this site. … | You, or the browser, refused the microphone. | Allow it in the browser's site settings, then press the mic again: this one stays pressable. |
| Speech is unavailable on this machine: the CPU lacks AVX2 … | The speech model runs on a CPU with AVX2. | — |

The server's reasons come from `GET /v1/models`' `speech` object, read when
the page opens and with each message; a reason the table has no row for is
shown in the server's own words.

## What a request carries

On a real server the chat talks over **one WebSocket** to `/v1/responses`
([the Responses API](../api/responses.md)): every session and every agent
shares it, and a turn sends only the items that are new since the last one.
Choose **HTTP** under Settings, or let the page fall back to it when the socket
does not open, and each turn is a `POST /v1/chat/completions`; then the browser
carries at most five streams at once, agents included. When the page falls
back, a bar under the header says so once.

A request never names a `model`: the chat follows the model the server has
loaded, so it cannot start a switch by accident (the
[two names pitfall](../concepts/switching-and-idle-unload.md)). It carries
the settings (`reasoning_effort`, `thinking_budget`, `temperature`, `top_p`,
`max_tokens` or `max_output_tokens`, and `class`, the lane tag), the tools
that are on, and the system prompt.

## A reply

From top to bottom:

1. **Reasoning**, folded under its summary once the reply ends (it reads
   **Thinking…** and stays open while it streams). An orange **Budget reached**
   mark says the [thinking budget](settings.md#thinking-and-its-budget) closed
   it. Reasoning the [loop guard](../concepts/reasoning.md) took back is shown
   struck through, where the loop was.
2. **Queued**, over the WebSocket only, when every lane was busy and the
   server held the request in its queue instead of refusing it.
3. **The answer**, as Markdown: headings, lists, tables, code blocks with a
   **Copy** button, formulas (KaTeX: `$…$`, `$$…$$`, `\(…\)`, `\[…\]`), and a
   **Preview** of an `html` block. **Markdown in replies** off shows the raw
   text.
4. **Tool sections**: Agents, Web, Tool calls, Pictures, Questions
   ([Tools and agents](tools-and-agents.md)).
5. An **error line** for a request the server refused, with its status and
   message, such as `503: … engine full` over HTTP.
6. **The figures line**.
7. **Regenerate**, **Edit** and **Fork**: regenerate drops this reply and what
   follows and asks again; edit changes the text in place; fork starts a new
   session from here. A prompt has **Edit** too, with **Save and resend**.

![A reply's reasoning, opened](../images/playground/chat-reasoning.png)

## The figures line

| Figure | What it measures |
|---|---|
| **Queued** | How long the request waited for a lane (only when it did). |
| **TTFT** | Send to the first token of any kind: reasoning, answer or tool call, less the queue. |
| **Decode** | Tokens per second from the first token to the last: `(completion tokens − 1) / (last − first)`. |
| **Total** | Send to the end of the stream. |
| **Tokens** | Prompt and completion tokens, from the stream's final `usage`. |
| **Rewound** | Tokens the loop guard took back (`rewound_tokens` in `usage`), when it did. |
| **Finish** | The `finish_reason`, or `stopped`. |

They are **browser-observed**: measured around the stream, over the network,
not inside the engine. A decode rate here includes the network and the page;
the server's own figures are in the [Monitor](monitor.md) and its
[metrics](../reference/metrics.md).

## The model picker

Under the box, the picker names the model the **next message** goes to, and a
chip beside it says what sending will do first: **Loaded**, **Loads when you
send** (after an idle unload), or **Switches from … when you send**. The menu
lists the text models the server knows, with their size and whether a message
can go to them; one that is not on disk points at the Server view, where the
download command is.

![The model picker](../images/playground/model-picker.png)

Choosing another model makes the next message switch the server
(`POST /v1/models/switch`), wait for the load and then send; the prompt says
`Switching to … from …` while it does. The choice holds for that message only:
the chat then follows the server again. With the image model loaded, the next
message goes back to a text model on its own. How switching works and what it
costs: [Models that come and go](../why-ignis/models-that-come-and-go.md).

## The context meter

**Context** shows how much of the model's context the session uses: the solid
part is the last reply's prompt plus completion, the pale part the room the
next reply reserves (its Max tokens). Hover it for the numbers: the context,
the last prompt, the last completion, what is in use, what the next reply
reserves and what is left after that. The limit is the loaded model's
`max_model_len` from `GET /v1/models`. Reasoning is not sent back on the next
turn, so the next prompt can be shorter than the meter suggests. Over the
limit the bar turns red: the server would refuse the request, so lower Max
tokens or start a new session.

## The session log

**This session**, at the foot, collects a row per request: the time, the lane
tag (agents show as `agent <name>`), the thinking effort and budget (with
`reached` when the budget closed the reasoning), TTFT, decode rate, total,
prompt and completion tokens, and the finish reason. A failed request shows its
error across the row. It is the quickest way to compare two settings on the
same prompt.

![The session log, open](../images/playground/session-log.png)

## When the model is not serving

The empty chat says what the server is doing: **Light it up** when a model
serves, **Warming up** while one loads, **Switching models** during a switch,
and that a model unloaded while idle loads again with the next message. While
no model can take a message the prompt box reads `Waiting for the model…` and
Send is off.

## Read more

- [Settings](settings.md): thinking, the budget, sampling, the lane tag, the transport.
- [Reasoning that finishes](../why-ignis/reasoning-that-finishes.md): the loop guard and the budget behind the marks above.
- [Chat completions](../api/chat-completions.md) and [Responses](../api/responses.md): the same requests from your own code.
- [Speech to text](../api/transcriptions.md) and [the speech settings](../configuration/speech.md): the model behind the mic.
