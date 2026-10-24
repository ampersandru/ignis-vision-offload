---
title: "Get a model"
description: "The catalog, `model download`, letting the server fetch its model, the endpoint and token, and pointing at an artifact you already have."
sidebar:
  order: 1
---

The engine loads a `.ninfer` artifact: weights, tokenizer and chat template in
one container, verified against its provenance sidecar (`.graft.json` or
`.conversion.json`) at load. A mismatch refuses the start.

## The catalog

The models the server knows how to fetch are its **catalog**
([ADR 0047](../../adr/0047-a-catalog-the-operator-can-extend.md)). Four are built
into the binary, each pinned file by file — byte count and SHA-256 — at a fixed
revision of its Hugging Face repository:

| Id | Repository | Size |
|---|---|---|
| `qwen3.8-27b` (the default) | [`gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer) | 18.1 GiB |
| `qwen3.8-27b-abliterated` | [`gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer) | 18.1 GiB |
| `qwen3.8-flash-next` | [`gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis`](https://huggingface.co/gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis) | 67.8 GiB (with its MTP companion) |
| `ignis-image-2.1-turbo` | [`gpillon/ignis-image-2.1-turbo`](https://huggingface.co/gpillon/ignis-image-2.1-turbo) | 15.6 GiB (the [image model](../api/images.md); non-commercial licence) |

The id is the **served id**: what `--model-id` names, what `/v1/models`
reports and what a request puts in `model`. `ignis-server model list` shows
every entry and how much of it is on disk:

```
ignis-server model list                 # id, built-in or operator, repo@revision, size, complete/partial/absent, family
ignis-server model list --format json
```

## Fetching one: `ignis-server model download`

```
ignis-server model download                          # the configured --model-id (qwen3.8-27b by default)
ignis-server model download qwen3.8-flash-next       # one or more ids
ignis-server model download --all                    # every entry
ignis-server model download qwen3.8-27b --out /mnt/usb   # somewhere other than --download-path
```

Every file of the entry — the artifact, its sidecar, Flash-Next's companion
container — is streamed to a `.part` file, checked against the size and
SHA-256 pinned in the catalog, and renamed into place only then, small files
first. An interrupted download resumes from its `.part`; a tampered one is
discarded. A file already at its name with its pinned size is kept; one with
another size is refused and left alone. The command never asks, ignores
`--download-enabled`, and needs no GPU and no NVIDIA driver: a build without
`cuda` fetches as well, and so does the CUDA build — the container image
included — on a machine with no driver, such as a Kubernetes CPU node. The
CUDA build links none of the driver's libraries; it reaches them only when a
model loads, and a GPU load on a machine without the driver is refused as
`no NVIDIA driver (libcuda.so.1) on this machine` (`nvcuda.dll` on Windows).
A driver too old for the build's CUDA runtime (a lower CUDA major version than
the runtime's; a lower minor is fine) is refused naming both versions and asking
to update the NVIDIA driver.
Progress goes to stderr, the artifact paths to stdout, and the exit code is 0
only when every entry verified. It reads the same configuration a start does
(config file, environment, flags).

A model fetched into a running server's `--download-path` is switchable at
once: a request naming its id switches the server to it, as a
`--switch-known-models` entry would (`--switch-allow-implicit`, on by default;
an explicit `--switch-known-models` entry wins for its id). A request naming a
catalog id that is not on disk is refused as an unknown model — a switch never
starts a download.

A client configured with two known models reloads the server back and forth: opencode sends its `small_model` requests (titles, summaries) to the id named by `small_model`, so a `model` and a `small_model` naming different ignis ids switch the server on alternate requests, 13-33 s each. The same holds for a front end that picks its ids from `GET /v1/models`, which lists every servable model: Open WebUI's task model (titles, tags, follow-ups) or Hermes set to another listed id than the chat's does it too. Point both at the same id, or start with `--switch-allow-implicit false`. The Playground's chat and Decide never cause a switch: their requests name no model, so the server serves the one loaded, and the header and the context limit follow it, read again from `GET /v1/models` when a turn starts. The Playground switches only when asked to: **Load** on its Server view, **Generate** and **Edit** on its Image view with a text model loaded ([in the Playground](../api/images.md)), the chat's `make_image` tool (to the image model and back), and the chat's picker for the next message's model ([the Playground](../playground/index.md)); each of these switches with `POST /v1/models/switch` or by naming the image model, never by a chat request naming a model.

## Letting the server fetch it

A GPU build started **without** `--model-artifact` / `IGNIS_MODEL_ARTIFACT`
looks for the `--model-id` entry's artifact under `--download-path` (default
`./models`, flat, under the names the repository publishes) and fetches the
whole entry when it is not there
([ADR 0033](../../adr/0033-the-server-fetches-its-own-model.md)):

```
ignis-server                                      # asks first: the size, the source, the destination
ignis-server --model-id qwen3.8-flash-next        # the same, for another catalog entry
ignis-server --download-enabled false             # never fetches: the placeholder template and the CPU mock
ignis-server --download-path /weights
```

On a terminal you are asked (`[y/N]`, on stderr, naming the endpoint the bytes
come from); without one — a container, a daemon, CI — nobody can answer, so it
downloads, through the same verified transfer as `model download`. A failed
fetch refuses the start. A build without `--features cuda` never downloads
weights it could not run.

A `model download` and a download the server did land on the same file names —
as does a `hf download <repo> --revision <rev> --local-dir models` done by
hand — so any of them satisfies the others.

## Where the bytes come from: endpoint and token

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--download-endpoint <url>` | `IGNIS_DOWNLOAD_ENDPOINT` | `https://huggingface.co` | The server a file is fetched from, as `{endpoint}/{repo}/resolve/{revision}/{file}` — Hugging Face, or any mirror answering the same route. |
| `--download-token <token>` | `IGNIS_DOWNLOAD_TOKEN` | unset | Sent as `Authorization: Bearer …` to the endpoint. Never shown by `GET /v1/config` or `config print`, never logged. |
| `--download-catalog <path>` | `IGNIS_DOWNLOAD_CATALOG` | unset | An operator catalog ([Your own models: an operator catalog](operator-catalog.md)). |

The token is `--download-token` when it is set; otherwise `HF_TOKEN`, but only
when the endpoint is `huggingface.co` itself — a personal Hugging Face
credential is never handed to a mirror. No token is sent across a redirect to
another host (Hugging Face sends a large file from its CDN). None of the three
can be changed by `PATCH /v1/config`.

## Pointing at one you already have

```
set IGNIS_MODEL_ARTIFACT=./models/qwen3_8_27b_nvfp4full-v2.ninfer   # Windows
export IGNIS_MODEL_ARTIFACT=./models/qwen3_8_27b_nvfp4full-v2.ninfer
```

A named path is the operator's word: missing, it refuses the start, and it is
never matched against the catalog. Served with no `--model-id`, it takes its
family's default id.

The artifact carries a grafted DFlash2 drafter module, which `--spec-backend dflash2`
loads for speculative decoding and which costs no VRAM when speculation is off.
