---
title: "The user path"
description: "From nothing to a chat: install ignis, fetch a model, open the Playground, connect your tools. No configuration needed."
sidebar:
  order: 1
---

For someone who wants ignis running and answering, on a card it supports,
with its defaults. Nothing on this path asks you to choose a setting: the
defaults are the ones measured on the owner's own cards.

1. **[What ignis is](../start/index.md)** — the models it serves, the cards and
   systems it runs on, and what it deliberately does not do; then
   **[Why ignis](../why-ignis/index.mdx)**, what sets it apart, with its numbers.
2. **[Install](../start/install.md)** — the container image or a release
   archive, and what each needs from the host: the NVIDIA driver, and for the
   archives the CUDA runtime.
3. **[Quick start](../start/quick-start.md)** — one command to a running
   server, the Playground in the browser, one request with curl.
4. **[Get a model](../guides/models.md)** — the built-in catalog, `ignis-server
   model download`, and pointing at an artifact you already have.
5. **[The Playground](../playground/index.md)** — every view, with screenshots:
   the [chat](../playground/chat.md) and its [settings](../playground/settings.md),
   [subagents and tools](../playground/tools-and-agents.md), [Decide](../playground/decide.md),
   the [image studio](../playground/image-studio.md), the [Monitor](../playground/monitor.md)
   and the [Server view](../playground/server.md).
6. **[Connect a client](../start/connect-a-client.md)** — the base URL and the
   key, the OpenAI SDKs, coding agents and chat front ends.
7. **[Troubleshooting](../troubleshooting.md)** — what a refusal or a WARN line
   means, and what to do about it.

:::tip
The server holds one model at a time and gives it the whole card. A client
that names two different models in turn makes the server reload back and
forth; point every request at the same id
([Connect a client](../start/connect-a-client.md#two-model-ids-make-the-server-switch)).
:::

:::tip
The first start after a download, or the first request after the server
unloaded an idle model, waits for the load. Later requests do not
([Idle unload](../api/idle-unload.md)).
:::

When the defaults stop being enough — a longer context, a smaller card, a
machine without internet — continue on the [advanced path](advanced.md).
