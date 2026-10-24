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
