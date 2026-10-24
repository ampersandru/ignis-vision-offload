---
title: "Chat"
description: "The Playground's chat: a reply with its reasoning and figures, the prompt box, the model picker, the context meter and the session log."
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
