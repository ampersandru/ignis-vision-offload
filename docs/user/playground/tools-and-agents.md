---
title: "Tools and agents"
description: "The chat's tools: subagents on agent lanes, web search, questions to you, JavaScript, a plan, memory, the workspace's files and pictures."
sidebar:
  order: 4
---

The chat gives the model nine tools, all on by default
([Settings](settings.md#tools)). The tools run **in the browser**: the server
only sees the model decide to call one and, on the next request, the result.
That makes the Playground an agent loop you can watch: a turn streams a reply,
runs the calls it made, sends the results back, and goes on until a reply
calls nothing, up to **Tool rounds** replies. A call to a tool the model was
not given is shown under **Unknown tools** with its arguments, and not run.

## Agents

With **Agents** on, the model can hand sub-tasks to subagents. Each one is its
own request, on an **agent lane** (`class: "agent"`), and up to eight run in
parallel on the server's decode lanes, while the conversation keeps its own.
The answers go back to the model, which writes the final reply. Agents get the
other tools that are on, except asking you, the plan and pictures.

![Three subagents, done, each with its time and decode rate](../images/playground/agents.png)

The **Agents** section shows a card per agent: its name, its task, a heat bar
while it runs, then `Done in <time>` and its decode rate. Click a card to open
the **agent reader**: its system prompt, its task, its reasoning, the tools it
called, its answer and its own figures. The session log gets a row per agent
request, tagged `agent <name>`.

![The agent reader](../images/playground/agent-reader.png)

This is the load ignis is built for: one main conversation and a burst of
subagents on one card. Why the burst does not push the conversation out:
[Lanes that know who is asking](../why-ignis/lanes.md) and
[One round for every lane](../why-ignis/decode-round.md).

## Web

**Web** lets the model search (five results) and read pages (the first 20,000
characters). The calls run from your browser, not from the server: with a
Tavily API key (typed under the Web switch, kept in this browser and sent only
to Tavily) searches go to Tavily; without one they go to `search.tiago.zip`,
then to DuckDuckGo Lite through `r.jina.ai`, and pages are read through
`r.jina.ai`. ignis takes no part. Each call is a row under **Web** you can
open: the results with their links, or a preview of the page.

![A search and a page read](../images/playground/web.png)

:::caution[Your queries leave the machine]
A web search sends the model's query to the search service from this browser.
Turn **Web** off for a conversation that must stay on the machine.
:::

## Ask me

**Ask me** lets the model stop and ask you something, with answers to pick or
your own. The turn waits under **Questions** until you answer, then goes on.
Only the conversation asks; agents do not.

![A question from the model](../images/playground/ask.png)

## Date and time

**Date and time** writes the day, date, time and time zone into the ignis
system prompt once per session. It is frozen on purpose: a prompt that changes
every turn could not be reused from the server's cache. **Update every
prompt** sends the moment of each turn as a developer message after the
conversation instead.

## Run JavaScript

**Run JavaScript** gives the model a sandbox for exact results: a Web Worker
with no network, no page and no storage, a 10 s limit and 10,000 characters of
output. With **Safety check** on (the default), a request on an agent lane
reviews each piece of code first, greedy and without thinking, and code it
does not call safe is blocked. A row under **Tool calls** shows the code, the
check's verdict, the result and the console.

## Plan

**Plan** lets the model keep a checklist of its steps in the reply, updated as
it works: done, in progress, pending.

![Code the model wrote, checked by the safety review and run](../images/playground/plan-and-code.png)

## Memory

**Memory** lets the model save short notes about you in this browser and see
their titles in every session (up to 4,000 characters of titles); it reads a
note's body when it needs it. **Memory** in the header opens the notes: search
them, edit them, delete them, or write one yourself. They live in this
browser's `localStorage`, never on the server.

![What ignis remembers](../images/playground/memory.png)

## Workspace

**Workspace** lets the model work on the files of the
[chat's workspace](chat.md#the-chats-workspace), with three tools, declared
whenever the switch is on, whether the chat has a file yet or not:

- `list_files`: each file's name, kind (text, PDF, picture, audio or other),
  size, last change and who made it: you, the chat itself, one of its agents,
  or another chat.
- `read_file`: a text file as numbered lines, a range at a time (`offset` is
  the first line, `limit` the most lines), up to 20,000 characters a call; a
  PDF reads as its text, the same way.
- `write_file`: creates a text file or replaces it whole. A name may hold `/`
  for folders, as in `site/index.html`, and is unique in its workspace.

Every write is a new version of the file, with its author and time; the
earlier versions are kept. The model's first file creates the workspace.
Agents get the same tools, and their writes are their own. A file you attach
with **File** in the prompt box becomes a file of the workspace when the
prompt is sent, and the prompt says so after your words:
`[Attached to the workspace: brief.pdf]`.

A file the model writes shows in its reply as a card: its name, the version
that reply wrote, its size, **Preview** and **Download**. An HTML file opens on
a live preview in a sandboxed frame (scripts run, no access to the page), with
its source, **Save as PDF** and a full-screen view at desktop, tablet or phone
width.

![An HTML file the model made, previewed](../images/playground/html-file.png)

The ignis system prompt's workspace part and the three tools are the same in
every chat, whatever the workspace holds and whatever you attach, so a file
never changes the head of the prompt and the server's cache of the
conversation still holds
([ADR 0072](../../adr/0072-the-playgrounds-system-prompt-never-changes.md)).
With the switch off the model has no workspace tool and **File** leaves the
prompt box's **+** menu; pictures can still be attached, and the files stay in
this browser.

## Pictures

**Pictures** lets the model make and edit pictures with the image model. The
page switches the server to the image model, makes up to four variants per
call, and switches back to the text model before the next request: each reply
that makes a picture costs two model loads, about half a minute on an RTX
5090. The model is told each picture's name (`made-1`, or `user-1` for one you
sent), size and seed, not what it looks like. A picture can be marked, opened
full screen, downloaded, or opened in the [Image studio](image-studio.md).
Agents do not get this tool. The weights are for non-commercial use only
([images](../api/images.md)).

## Read more

- [Chat](chat.md): the reply these sections live in.
- [The Responses API over a WebSocket](../why-ignis/responses-websocket.md): why a burst of agents is not held to the browser's connection limit.
