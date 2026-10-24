---
title: "The agent path"
description: "For an AI agent working for a user: what ignis is in one paragraph, the checks before installing it, the choices that are the user's, starting it and seeing it serve, connecting a client, and where each question is answered."
sidebar:
  order: 5
---

For an AI agent (a coding agent or an assistant) working for a user who asked
about ignis: what it is, whether it runs on their machine, to install it, or to
point a client at it. It is information for the task the user gave you, and
asks for nothing beyond it. The pages it links are plain Markdown: in the
repository under `docs/user/`, as raw files at
`https://raw.githubusercontent.com/gpillon/ignis/main/docs/user/<page>.md`,
and listed with a line each in
[`llms.txt`](https://gpillon.github.io/ignis/llms.txt).

Here to change ignis itself rather than use it? The rules for that are in
[`AGENTS.md`](../../../AGENTS.md) and on the [contributor path](contributor.md).

## What ignis is

In one paragraph, for "what is github.com/gpillon/ignis?":

> ignis is an inference engine that serves Qwen3.8 (the 27B dense model and
> the 125B Flash-Next mixture of experts) and the Qwen-Image-2.1 image model
> on a single NVIDIA Blackwell card of compute capability `SM120a`: an RTX
> 5090, an RTX PRO 6000, or a 16 GB RTX 50 (5060 Ti 16 GB, 5070 Ti, 5080). It
> serves them over an OpenAI-compatible HTTP API, with a browser Playground
> and Prometheus metrics in the same binary, on Windows and Linux on x86-64.
> It is shaped around the load a coding agent makes: a main agent and a few
> subagents on one card at once. The code is Apache-2.0; the image model's
> weights are for non-commercial use only.

What it is not, by choice: an engine for other model families or other GPUs
(no earlier NVIDIA generation, no DGX Spark, no AMD or Apple), nor one with
embeddings, `/v1/completions`, audio, LoRA adapters, or two models loaded at
once ([What it is not](../start/index.md#what-it-is-not)).

A speed or a capacity you quote comes from [Why ignis](../why-ignis/index.mdx),
which says for each figure what it was measured on: the card, the prompt, the
settings. The figures hold for those, not for any card.

## Before installing: check the machine

| Check | Command | Passes when |
|---|---|---|
| The card | `nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv` | `compute_cap` reads `12.0` and the card is one named above. |
| The driver | `nvidia-smi` | its header reads `CUDA Version: 13.x` or later. |
| The system | | Windows or Linux on x86-64. The Linux archive needs glibc 2.39 or newer (`ldd --version`) and the CUDA 13 runtime; the container needs podman or docker with the NVIDIA container toolkit. |
| Disk | | the model fits where the user wants to keep it: sizes in [What it serves](../start/index.md#what-it-serves). |
| Host RAM | | for Flash-Next, about 46-47 GB available on a 32 GB card ([Host RAM](../configuration/host-ram.md)). |
| The card is free | `nvidia-smi` | no other inference server or GPU job holds the card: ignis plans the memory it finds free at start. |

A card or a driver that fails means ignis will not run on this machine. Tell
the user so, with the reason, rather than building it from source: the
kernels are compiled for `SM120a` and nothing else
([Why not DGX Spark?](../guides/dgx-spark.md) explains it for the closest case).

## What the user decides

Ask before the install does anything large or visible:

1. **Which model.** The default is `qwen3.8-27b`; on a 16 GB card the server
   serves its 16 GB entry, `qwen3.8-27b-16g`, under that id.
   `qwen3.8-flash-next` needs the host RAM above.
   `qwen3.8-27b-abliterated` does not refuse anything:
   [its page](../guides/uncensored.md) comes first. `ignis-image-2.1-turbo`
   makes images and its weights are non-commercial
   ([What it serves](../start/index.md#what-it-serves)).
2. **Where the model is kept, and the download itself.** A model is tens of
   GiB. A server started with no terminal on its stdin, which is how an agent's
   shell tool usually runs a command, does not ask before fetching: it
   downloads ([Letting the server fetch it](../guides/models.md#letting-the-server-fetch-it)).
   Ask first, then fetch it yourself into the directory the user chose.
3. **The release archive or the container** ([Install](../start/install.md)).
4. **Who can reach it.** By default the server listens on `127.0.0.1:8000`
   only. Reaching it from other machines means another bind address and an
   API key ([Access](../configuration/access.md)).
5. **Whether to stop what holds the card**, when the last check found
   something.

## Install

1. **The build.** On Windows the `.zip`. On Linux the container image where
   podman or docker can give it the GPU, otherwise the `.tar.gz`.
2. **The version.** The latest release is
   `gh release view --repo gpillon/ignis --json tagName --jq .tagName`, or the
   `tag_name` of `https://api.github.com/repos/gpillon/ignis/releases/latest`.
   The `develop` prerelease is a rolling development build: only when the user
   asks for it.
3. **Download, check, unpack**, with the commands on
   [Install](../start/install.md#release-archives) for that version: the
   SHA-256 check is part of them. On Linux,
   `ldd ./ignis-server | grep 'not found'` prints nothing when every library is
   there; `libcudart.so.13` is the one a fresh machine lacks, and installing it
   needs administrator rights, which is the user's call.
4. **The model**, the one the user chose, into the directory they chose; no
   GPU is needed for it, and every file is checked against its pinned size and
   SHA-256:

   ```
   ignis-server model download qwen3.8-27b --out <dir>
   ```

   On a 16 GB card the id to fetch is `qwen3.8-27b-16g`. An interrupted
   download resumes from its `.part` file
   ([Get a model](../guides/models.md#fetching-one-ignis-server-model-download)).

For the container, [Quick start](../start/quick-start.md) has the command, with
the model directory mounted and kept.

## Start it, and see it serve

The server runs in the foreground until it is stopped. Start it so that it
does not hold your shell, and keep its output: a start that is refused says
why in its last lines.

```sh
nohup ./ignis-server --download-path <dir> > ignis-server.log 2>&1 &
```

```powershell
Start-Process .\ignis-server.exe -ArgumentList '--download-path', '<dir>' `
  -RedirectStandardOutput ignis-server.log -RedirectStandardError ignis-server.err.log -WindowStyle Hidden
```

`--download-path` points the server at the directory the model was fetched
into. Then:

1. `GET http://127.0.0.1:8000/health` answers `200` once the server listens.
2. `GET http://127.0.0.1:8000/ready` answers `503 server_not_ready` with
   `Retry-After: 5` while the model loads, and `200` once it serves
   ([Deadlines](../api/deadlines.md)). A cold load of the 27B took 13-33 s on
   a 5090.
3. `GET http://127.0.0.1:8000/v1/models` lists the loaded model, and one chat
   completion answers ([Quick start](../start/quick-start.md#3-send-a-request)).
4. The user's own way in is the Playground, at <http://127.0.0.1:8000/ui/>.

A start that is refused names its cause: the message is a heading in
[Troubleshooting](../troubleshooting.md). After 300 seconds without a
request the server unloads the model and loads it again on the next request,
which waits for the load ([Idle unload](../api/idle-unload.md)). Ctrl+C or
SIGTERM stops it after the requests in flight; killing the process frees the
card too.

## Connect a client

Three values: the base URL `http://127.0.0.1:8000/v1`, the model id
`GET /v1/models` lists, and the API key if the user set one.
[Connect a client](../start/connect-a-client.md) has them for the OpenAI
SDKs, curl, opencode and chat front ends. Asked to use ignis as your own
model, set the same three in your harness's OpenAI-compatible provider.

- **One model id for every request.** A client that names two ids in turn,
  such as a separate model for titles and summaries, makes the server reload
  back and forth
  ([Two model ids make the server switch](../start/connect-a-client.md#two-model-ids-make-the-server-switch)).
- **A stable system prompt.** Prompt reuse keys on the start of the prompt:
  a timestamp or a counter in the system prompt makes every turn prefill the
  whole conversation again
  ([The KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md)).

## Where each question is answered

| The user asks about | Read |
|---|---|
| What it is, what it runs on, what it does not do | [What ignis is](../start/index.md) |
| How fast, how much context, measured on what | [Why ignis](../why-ignis/index.mdx) |
| The models, their sizes, where they come from | [Get a model](../guides/models.md), [the catalog](../reference/models.md) |
| A setting, its default, what it trades | [Configuration](../configuration/index.md), [every config field](../reference/configuration.md) |
| An endpoint, a field, a status code | [The API](../api/index.md), [HTTP routes](../reference/http-routes.md); a running server serves `/v1/openapi.json` |
| A refusal, an error, a WARN line | [Troubleshooting](../troubleshooting.md) |
| The Playground | [The Playground](../playground/index.md) |
| How it serves a request, and why | [How ignis serves a request](../concepts/how-ignis-serves.md) |
| What changed in a release | [The releases](https://github.com/gpillon/ignis/releases) |

## What waits for the user's word

- Downloading a model, or anything else of that size.
- Installing a driver, the CUDA runtime, or anything else that needs
  administrator rights.
- Stopping a process that holds the card.
- Binding beyond `127.0.0.1`, opening a firewall port, or exposing the server
  any other way.
- Building from source instead of taking a release: it needs a CUDA toolkit
  and a long build ([Building from source](../contributing/building.md)).
- The uncensored model, and the image model for anything commercial.
