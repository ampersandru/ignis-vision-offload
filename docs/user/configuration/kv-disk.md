---
title: "KV-disk"
description: "The disk tier under the host KV pool for Flash-Next's long contexts."
sidebar:
  order: 7
---

The tier below KV-RAM ([ADR 0045](../../adr/0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md)):
when a sequence has to leave the device and KV-RAM has no room for it, or
KV-RAM needs room for a newer one, its state is written to a file instead of
being dropped, and read back when its turn comes -- exactly where it stopped,
with no token re-prefilled. Retained prompt checkpoints KV-RAM gives up go
there too, and a later request on the same conversation restores from the
file when that saves at least the restore floor (8,192 tokens on Flash-Next,
16,384 on the 27B) over re-prefilling.

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--kv-disk-bytes <bytes>` | `IGNIS_KV_DISK_BYTES` | `4G` on Flash-Next (about three whole 262K-token lanes, ~1.25 GB each), `0` (off) on the 27B | The tier's ceiling (accepts `K`/`M`/`G`). At start it is cut to the volume's free space less a 10 GiB margin, and printed with the directory on `ignis.kv_disk.ready`; a WARN says when it was cut, and a volume with no room above the margin starts without the tier. A write that would cross the margin is refused and the state stays where it was. `0` turns the tier off. (`make` knob `KV_DISK_BYTES`.) |
| `--kv-disk-path <model\|auto\|dir>` | `IGNIS_KV_DISK_PATH` | `model` | Where the files go, by the n-gram cache's rule: `model` is the artifact's own directory, `auto` is `LOCALAPPDATA/ignis/cache/kv-disk` on Windows and `XDG_CACHE_HOME/ignis/kv-disk` or `HOME/.cache/ignis/kv-disk` on Linux, anything else is that directory. (`make` knob `KV_DISK_PATH`.) |

- Each process writes in its own `ignis-kv-disk/<pid>-<nonce>/` directory
  there, locked for its life, and removes it at shutdown; a dead process's
  directory is removed at the next start. Nothing is reused across a restart,
  and two servers on one location never touch each other's files.
- One file per state, written with unbuffered IO in 32 MiB windows through
  64 MiB of pinned staging reserved at load, on two IO threads of its own. A
  file's header is written last, with a CRC32 per window: a file torn by a
  crash, corrupted, or written by another load is never restored. A live
  request whose file is refused is prefilled again from its tokens.
- The model thread never waits on the disk: a move is a few steps during
  which the other lanes keep decoding, and a Flash-Next prefill's n-gram
  reads go before the tier's.
- When the device, KV-RAM and the disk are all full of live work, a new
  request waits in the queue until room returns: nothing is refused or loses
  work for it (the in-flight cap's 503 is unchanged).
- With metrics on (the default) the tier is `tier="disk"` on the retained-state series and
  `ignis_kv_disk_*` (bytes, spills, failures); the Monitor shows a Disk row.
