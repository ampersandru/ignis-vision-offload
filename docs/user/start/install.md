---
title: "Install"
description: "Get ignis: the container image, the release archives, and what each needs on the host."
sidebar:
  order: 2
---

## The container image

A `v*` tag publishes a `linux/amd64` image to `ghcr.io/gpillon/ignis`, tagged
with the version and `latest`. The image carries the CUDA runtime but no
driver: the host's NVIDIA driver is injected by the container runtime, and the
model is mounted, never baked in.

Every push to `main` also publishes the image as `ghcr.io/gpillon/ignis:develop`
(and a rolling `develop` prerelease on the Releases page holding the matching
archives). That is a development build, **not a release**: it is rebuilt on each
push, replaced by the next one, and never tagged `latest` or with a version.

```
podman run --rm --device nvidia.com/gpu=all -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e IGNIS_MODEL_ARTIFACT=/models/qwen3_8_27b_nvfp4full-v2.ninfer \
  ghcr.io/gpillon/ignis:latest
```

(`docker`: `--gpus all` in place of `--device`.) Every flag has an `IGNIS_*`
environment variable; anything after the image name is passed to the server.
The image sets no default command, so the Playground is on — the server's own
default. So are metrics: the Playground's Monitor reads them at `/ui/metrics`
on port 8000, while the keyless Prometheus listener stays on the container's
own loopback. To scrape it from outside, add
`-e IGNIS_SERVER_METRICS_BIND=0.0.0.0:9464 -p 9464:9464`.

The server is the image's entrypoint, so it is the container's PID 1, and it
needs no init in front of it: `ignis-server` (not its `model download`
subcommand) stops on SIGTERM (or Ctrl+C) from the moment it starts. Before it
serves — while the model downloads or loads — the process exits at once,
which frees the GPU, its last line `ignis.process.stopped` naming the
`phase` it was in; once it serves it drains in-flight requests and unloads
the model first. A load while serving (a switch, a reload, the load after an
idle unload) is not waited for either: a request waiting for it is answered
`503 server_stopping`, and so is one that would begin a load (`POST
/v1/models/switch`, a `PATCH /v1/config` reload, a request naming another
model or needing the unloaded one); once the drain is over the process exits
under the load, its last line naming the `phase` `GET /v1/models` reports it
in (`switching`, or `loading` after an idle unload). A second signal while the
shutdown runs ends the process at once (`phase: shutting_down`). Its probes
are [`/health` and `/ready`](../api/index.md).

With no `IGNIS_MODEL_ARTIFACT` the image fetches the model into its own working
directory (`/home/ignis/models`) and loses it with the container. Mount a
directory and point the download at it to keep what it fetches:

```
podman run --rm --device nvidia.com/gpu=all -p 8000:8000 \
  -v /path/to/models:/models \
  ghcr.io/gpillon/ignis:latest --download-path /models
```

**The image's user.** The server runs as `ignis` (uid 10001), never as root,
so whatever it reads under a mount must be readable by uid 10001. A model
directory or Hugging Face cache that root wrote with `0700` directories is
not readable, and the load fails on it. Two Flash-Next features also *write*
next to the artifact by default (`model`): the n-gram cache
(`--ngram-persist-path`) and the [KV-disk](../configuration/kv-disk.md) tier (`--kv-disk-path`).
On a read-only mount, like the `:ro` one above, or one uid 10001 cannot write,
the server still starts, with two costs:
- every start rebuilds the n-gram hot rows from the artifact, with the WARN
  `n-gram cache not writable; reading the artifact`;
- the load runs without KV-disk, with the WARN `ignis.kv_disk.off`.

To keep both, give them a directory uid 10001 can write (rootless podman maps
uids: there, `podman unshare chown 10001 /path/to/ignis-cache`):

```
mkdir -p /path/to/ignis-cache && sudo chown 10001 /path/to/ignis-cache
podman run --rm --device nvidia.com/gpu=all -p 8000:8000 \
  -v /path/to/models:/models:ro -v /path/to/ignis-cache:/cache \
  -e IGNIS_MODEL_ARTIFACT=/models/qwen3_8_flash_next_trellis_a25-v2.ninfer \
  -e IGNIS_NGRAM_PERSIST_PATH=/cache -e IGNIS_KV_DISK_PATH=/cache \
  ghcr.io/gpillon/ignis:latest
```

To smoke-test the image with no GPU and no model — the deterministic CPU mock
([ADR 0006](../../adr/0006-exclusive-gpu-testing.md)) — say so:

```
podman run --rm -p 8000:8000 ghcr.io/gpillon/ignis:latest --download-enabled false
```

`Containerfile` builds the same thing locally (`podman build -t ignis:dev .`).
Its `artifacts` stage is what CI exports the Linux tarball from, so the release
binaries and the image binaries are the same build
([ADR 0032](../../adr/0032-release-through-the-container-build.md)).

## Release archives

A `v*` tag also publishes a Windows `.zip` and a Linux `.tar.gz`, each with its
SHA-256. Both are compiled for `SM120a` only. The Linux tarball needs the CUDA
13 runtime on the host; the Windows zip carries `cudart64_*.dll`.
