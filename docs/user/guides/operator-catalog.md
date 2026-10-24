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
    card_class: 32                     # optional: the smallest card class it loads on, in GB: 16, 32 or 96
    # aliases: [qwen3.8-27b]           # optional: the family's default id it answers to on a smaller card
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
of 0, a file name that is not a plain name (a path separator, `..`, an
absolute path), a `card_class` that is not 16, 32 or 96, or an
alias that is not a model family's default id (`qwen3.8-27b`,
`qwen3.8-flash-next`, `ignis-image-2.1-turbo`), is the entry's own id, names
another family than the entry's own id or its other aliases, or is already
another entry's — a built-in one's included. A relative `--download-catalog` set in a config file is read
from that file's directory; from a flag or an env var, from the working
directory. An entry with no `card_class` states no class: it is never refused
by class, so a catalog written before card classes still starts, and only the
VRAM plan says it does not fit.

A built-in id cannot be re-pinned by a file: a mirror of the owner's models is
an endpoint, a variant of them a new id. Your entry may answer to a built-in
default id as an alias — your own 16 GB build answering to `qwen3.8-27b` —
since the built-in entry keeps its id and pins and still serves it wherever it
fits: the alias is consulted only on a card below its class, and the answer
names your entry.
