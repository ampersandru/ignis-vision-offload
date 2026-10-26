---
title: "Air-gapped machines"
description: "Carry the files in, or point the server at a mirror."
sidebar:
  order: 3
---

- **Files carried in.** On any machine with a network — no GPU needed —
  `ignis-server model download <id> --out <dir>`, then copy `<dir>` into the
  air-gapped machine's `--download-path`. The server finds the files there and
  fetches nothing.
- **A mirror.** An Artifactory or Nexus Hugging Face proxy, or a static server
  laid out as `{repo}/resolve/{revision}/{file}`:
  `--download-endpoint https://mirror.corp/hf` (and `--download-token` if it
  needs one). No catalog is needed: the built-in pins still say whether the
  bytes are the owner's.
- **Your own models.** An operator catalog (above), with the endpoint that
  serves them.
- **Nothing on the network at all.** `--download-enabled false` keeps the
  server from fetching at start and refuses the Server view's **Download**
  (`POST /v1/models/download`, `403 download_disabled`) as well, so the server
  never opens a connection to an endpoint; the page then offers the
  `model download` command to copy instead. With a mirror,
  **Download** fetches from it like a start does
  ([Downloading from the Server view](models.md#downloading-from-the-server-view-post-v1modelsdownload)).
  **Remove** still frees the disk: a removal uses no network
  ([Removing a model](models.md#removing-a-model-post-v1modelsremove)).
