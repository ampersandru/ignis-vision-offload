---
title: "Your own models: an operator catalog"
description: "Add models to the catalog with a YAML or JSON file in the built-in catalog's format."
sidebar:
  order: 2
---

`--download-catalog` names a YAML or JSON file in the built-in catalog's own
format. Its entries are added after the built-in ones, fetched and verified the
same way, and listed as `operator`:

```yaml
models:
  - id: acme-qwen3.8-27b-ft          # the served id
    repo: acme/qwen-ft-ninfer          # <owner>/<name>
    revision: v1                       # commit, tag or branch; required
    artifact: acme_ft.ninfer           # the file the server loads
    files:                             # every file, all pinned
      - { name: acme_ft.ninfer.graft.json, bytes: 48211,       sha256: 9c1e… }
      - { name: acme_ft.ninfer,            bytes: 19406942468, sha256: abb1… }
```

The whole file is refused — the start, and every `model` command — when it
cannot be read, or when an entry has an unknown key, an empty or duplicate id
(case-insensitive), **an id the built-in catalog already has**, a `repo` not of
the form `<owner>/<name>`, an empty `revision`, an `artifact` not among its
`files`, no sidecar (the artifact's name plus `.graft.json` or
`.conversion.json`), a `sha256` that is not 64 lowercase hex digits, a `bytes`
of 0, or a file name that is not a plain name (a path separator, `..`, an
absolute path). A relative `--download-catalog` set in a config file is read
from that file's directory; from a flag or an env var, from the working
directory.

A built-in id cannot be re-pinned by a file: a mirror of the owner's models is
an endpoint, a variant of them a new id.
