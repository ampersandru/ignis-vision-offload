---
title: "The advanced path"
description: "Tune ignis: how a request is served, the configuration model, VRAM and the KV pool, reuse, speculation, host RAM, models, and running it in production."
sidebar:
  order: 2
---

For an operator who wants to know what each knob trades, and set it. Every
setting is one field with a flag, an env var and a config-file key; the
explanation pages say what a mechanism does and why, the configuration pages
which field moves it.

## Understand the engine

Start with **[Why ignis](../why-ignis/index.mdx)**: each topic there says what
was measured, on what, and where it stops, and links the page below that
explains its mechanism.

1. **[How ignis serves a request](../concepts/how-ignis-serves.md)** — lanes,
   prefill and decode, the KV pool, speculation, one model at a time.
2. **[The VRAM plan](../concepts/vram-plan.md)** — how a load lays out every
   byte before it starts, and why a start is refused rather than run out of
   memory.
3. **[The KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md)**
   — pages, the pool, prefix reuse and retained state, the host tier and
   KV-disk.
4. **[Speculative decoding](../concepts/speculative-decoding.md)** — DFlash2 for
   the 27B, MTP for Flash-Next, and when each pays.
5. **[Models and artifacts](../concepts/models-and-artifacts.md)** — the 27B,
   Flash-Next and the image model, and what each needs in VRAM and host RAM.

## Configure it

6. **[Configuration](../configuration/index.md)** — fields and groups,
   precedence, per-model values, the config file, profiles, live changes.
7. **[Profiles and defaults](../concepts/profiles.md)** and the
   [built-in profiles](../reference/profiles.md) — what each card's profile
   chooses.
8. The fields by topic: [context and speculation](../configuration/context-and-speculation.md),
   [Flash-Next MTP](../configuration/flash-next-mtp.md),
   [VRAM and the KV pool](../configuration/vram-and-kv-pool.md),
   [KV-disk](../configuration/kv-disk.md),
   [the n-gram cache](../configuration/ngram-cache.md),
   [host RAM](../configuration/host-ram.md), [vision](../configuration/vision.md),
   [prompting](../configuration/prompting.md).
9. **[Every config field](../reference/configuration.md)** — the generated list,
   with each field's default, rule and whether it changes live.

## Run it for real

10. **[Access](../configuration/access.md)** — keys, CORS origins, the metrics
    listener, public exposure.
11. **[Telemetry and metrics](../guides/telemetry.md)** and the
    [metric list](../reference/metrics.md).
12. **[Air-gapped machines](../guides/air-gapped.md)**,
    [an operator catalog](../guides/operator-catalog.md),
    [Knative](../guides/knative.md).
13. **[Model switching and idle unload](../concepts/switching-and-idle-unload.md)**
    — what a switch costs, and when the server frees the card.

:::tip
`ignis-server config print` shows what a start would run — file, env vars and
flags resolved — and the Server view of the Playground shows where each
running value came from.
:::

:::caution
A raised context or a lowered headroom is checked by the VRAM plan at load:
a value that does not fit refuses the start and names what it needed. Change
one thing at a time and read the refusal ([Troubleshooting](../troubleshooting.md#starting)).
:::
