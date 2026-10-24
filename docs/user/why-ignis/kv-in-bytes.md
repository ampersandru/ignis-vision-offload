---
title: "KV at 9 KB a token"
description: "The hq-e8-2b KV format holds 7.11 times the tokens of BF16, and the pool is budgeted in bytes, not in sequences."
sidebar:
  order: 5
---

Every token a conversation has read lives in the KV cache, and the KV cache is
what runs out first when a main agent and its subagents share one card. ignis
stores it in **hq-e8-2b**, a compressed format: a token of the 27B's context
costs **9,216 bytes instead of BF16's 65,536, 7.11 times the capacity** in the
same VRAM. The pool is a number of bytes, not a number of sequences: one long
conversation and eight short ones draw from the same pages.

## How it works

The 27B has 64 layers, but only 16 of them are full-attention layers that keep
a KV cache, each with 4 KV heads of 256 dimensions. The other 48 are
linear-attention (GDN) layers, whose recurrent state has a fixed size per
sequence however long it gets. So a token costs:

| Format | Per head, per token | × 16 layers × 4 heads × K and V | A token |
|---|---|---|---|
| BF16 | 256 dims × 2 bytes = 512 B | × 128 | **65,536 B** |
| hq-e8-2b | 64 B of 2-bit code + an 8-byte metadata row = 72 B | × 128 | **9,216 B** |

The ratio is 7.11 rather than 8 because the metadata adds 12.5% on top of the
code. Both formats cost a fixed number of bytes per token, so the pool is a
plain function of a byte budget: `vram.kv_pool_bytes` takes bytes or tokens,
and the server derives the other and prints both. Pages hold 64 tokens, and a
request reserves the pages for its whole bound (prompt plus generation cap)
when it is admitted, so the pool can never run dry halfway through a reply.

BF16 stays in the engine on purpose: it is the format every accuracy test
compares against. And since 0.7.3 the prefill attention reads the compressed
cache directly, so a long prompt no longer needs a decompressed copy of
itself in VRAM.

One cleanup came free with the format. Before every prefill attention call the
engine used to zero a 288 MB workspace per layer per chunk. Under hq-e8-2b
every byte the next launch reads it writes itself first (checked by filling
the workspace with garbage before the call), so the zeroing was dead work and
is gone.

The whole picture, from what a sequence is made of to the reuse tiers:
[KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md).

## The numbers

| Figure | Measured on | Source |
|---|---|---|
| 9,216 against 65,536 bytes a token, **7.11x** | the 27B's geometry | [hq-e8-2b KV capacity](../../findings/2026-09-11-hq-e8-2b-kv-capacity.md) |
| Eight lanes at a 40,960-token context: **3.02 GB** in hq-e8-2b against 20.0 GiB in BF16 | arithmetic, not a measured out-of-memory | same |
| The workspace zeroing removed: −2.12% of the GQA layers' device time on a 70K prefill, −191 ms median | RTX 5090, three A/B pairs | [GQA workspace memset](../../findings/2026-09-18-gqa-workspace-memset.md) |
| The quality price: KLD from BF16 +0.018 at 32K and +0.022 at 64K, about one point of top-1 | 27B, six windows, chunk 1,024 | [KLD against BF16](../../findings/2026-09-24-kld-against-bf16.md) |
| The decode price: both ignis and the reference engine pay about 14.5% of inter-token p95 for hq against BF16 | four concurrent requests | [hq vs BF16, live](../../findings/2026-09-13-hq-vs-bf16-live-live.md) |

:::caution[Compression is not free]
hq-e8-2b costs nothing inside the first prefill chunk, and its divergence
grows with the distance attended: at 32K-64K it adds 50-57% on top of the
weights' own KLD. For agent contexts of 20K-100K tokens the KV format is a
quality lever of the same order as the weight format. BF16 is one flag away
when that matters more than capacity.
:::

## See it, control it

- **The format**: [`model.kv_format`](../reference/configuration.md#modelkv_format),
  `hq-e8-2b` by default or `bf16`; changing it reloads the model.
- **The pool's size**: [`vram.kv_pool_bytes`](../reference/configuration.md#vramkv_pool_bytes),
  in bytes or tokens (`900Ktok`); unset, the 27B's pool is whatever the
  [VRAM plan](every-byte-planned.md) leaves.
- **At load**: the `ignis.runtime.vram_plan` log line prints `kv_pool_pages`
  and `kv_pool_tokens`.
- **Live**: `ignis_kv_pool_pages`, `ignis_kv_pool_used_pages` and
  `ignis_kv_page_bytes` ([metrics](../reference/metrics.md)); the
  Playground's [Monitor](../playground/monitor.md) draws the pool's occupancy.

## Limits

- 9,216 bytes is the 27B's geometry. Flash-Next has its own, and its pool
  follows how many experts stay resident.
- The 7.11x is capacity. The decode cost and the KLD cost above are real, and
  measured.

## Read more

- [KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md): what a sequence is made of.
- [VRAM and the KV pool](../configuration/vram-and-kv-pool.md): sizing the pool.
- [State that outlives its request](state-that-outlives.md): what happens to the pages when a request ends.
