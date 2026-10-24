---
title: "Flash-Next n-gram startup cache"
description: "The n-gram hot rows: where they are kept and what a start costs without them."
sidebar:
  order: 8
---

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--ngram-hot-bytes <bytes\|auto>` | `IGNIS_NGRAM_HOT_BYTES` | `1G` | The Flash-Next n-gram rows held in RAM, index included (accepts `K`/`M`/`G`; `0` holds none). Every other row is read from the artifact on NVMe when a step needs it. `auto`: what the host plan leaves after its other lines and its margin (below). The 27B has no n-gram table and refuses the flag. (`make` knob `NGRAM_HOT_BYTES`.) |
| `--ngram-persist <true\|false>` | `IGNIS_NGRAM_PERSIST` | `true` | Keep a compact copy of the selected Flash-Next hot rows on disk. `false` reads and writes no persistent n-gram cache. |
| `--ngram-persist-path <model\|auto\|dir>` | `IGNIS_NGRAM_PERSIST_PATH` | `model` | Cache directory. `model` is the artifact's own directory, on the disk that already holds the model. An explicit relative directory is relative to the server's working directory. |

The table is 28.8 GB (320,001,536 rows of 90 bytes); the artifact ranks its
15,642,665 most frequent rows (1.37 GiB with their index), and the hot rows
are the head of that ranking that fits the budget. What a budget holds:

- **Up to 1.37 GiB:** that many ranked rows. The default 1 GiB holds 11.4M of
  them.
- **Past the ranking and short of the whole table:** the whole ranking and no
  more. Nothing says which other rows are worth the RAM, so `4G` holds what
  `2G` holds; the start warns that it does (`ignis.runtime.ngram_hot_budget`)
  and the plan charges the ranking, not the size named.
- **The whole table (`27G` or more: 28.8 GB is 26.8 GiB):** every row, loaded
  at start by one sequential read of the table, and no step reads the NVMe again
  (`ignis_ngram_reads_total` stays at 0). It takes as long as reading 28.8 GB
  from the artifact's disk, and that load is never written to a cache file:
  the artifact already is its copy on disk.

`auto` picks among those, from what the host plan leaves after its other
lines held while serving and its margin (a device-only load's layer buffers
are freed before the rows are read, and take none of it), less 256 MiB for
the load's own reads: the whole
table when that holds it, otherwise the whole ranking when it fits, otherwise
the rest rounded down to whole GiB and never below the 1 GiB default, which
the plan then refuses exactly where it would refuse the default. So `auto` never refuses a start the
default would make, and the rows it picks, which key the cache file, change
only when free RAM crosses one of those steps rather than at every start. The
host plan (`ignis.runtime.flash_next_host_plan`) logs the budget as named
(`ngram_hot_budget`), as resolved (`ngram_hot_budget_bytes`) and its plan
line (`ngram_hot_rows_bytes`: the budget, capped at what the table can hold at
it, so `4G` is charged as the 1.37 GiB ranking);
`ignis.ngram.hot_rows` logs the rows loaded and how long the load took
(`duration_ms`).

`model` writes `<artifact stem>.ngram-<key>.bin` beside the artifact. `auto`
uses `%LOCALAPPDATA%\ignis\cache\ngram` on Windows, and
`$XDG_CACHE_HOME/ignis/ngram` on Linux (when XDG_CACHE_HOME is absolute),
otherwise `$HOME/.cache/ignis/ngram`. The directory is created on a cache miss.
A read-only model directory falls back to the normal load each time: name
`auto` or a directory there.

The first load gathers the normal hot rows and saves them; subsequent loads
read the compact file and validate its SHA-256 checksum. Loading still closes
the artifact mapping before any hot-row reads, even with persistence disabled.
A changed artifact path, file stamp, table layout, selected row order/count,
or cache format produces a different key. New packs also record a content
identity assembled from the n-gram source files' verified SHA-256 digests.
Legacy artifacts work without repacking: they use the filesystem stamp and
descriptors. Deliberately changing source bytes while preserving those stamps
and the source digest requires separate artifact verification.

Corruption, an unavailable cache directory, a concurrent writer or a failed
write falls back to the normal artifact load. Publication is atomic and uses
an OS lock; interrupted writes are never accepted as complete cache files.
The cache is disposable and does not change the model artifact. With the
default 1 GiB hot-row RAM budget a compact file is about 980 MiB. Saving a new
key removes the same artifact's older cache files in that directory, so one
file per artifact remains. This cache is separate from prompt/KV reuse.
