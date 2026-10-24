---
title: "Server"
description: "The Server view: the loaded model, the models a switch can go to, and the running configuration as a form you can apply."
sidebar:
  order: 9
---

The **Server** view changes the running server from the page. At the top, the
**model band**: what is loaded, its state and context, and every model the
server knows. Below it, the **configuration**: every field the server runs
with, as a form. Applying a change sends one `PATCH /v1/config`; the server
judges every value, applies what it can live, and reloads the model only for
a field that needs it.

![The model band and the start of the configuration](../images/playground/server.png)

## The model band

The headline is the loaded model's id with its **State** (`Serving`,
`Switching`, `Loading`, `Unloaded`, `Not serving`), its **Context** in tokens
and its **Family**. Under it, one row per model the server's catalog knows
(`GET /v1/models/catalog`): where the entry comes from (the built-in catalog,
an [operator catalog](../guides/operator-catalog.md), or a known model), how
much of it is on disk, and one action:

- **Loaded** for the model that serves;
- **Load** for one on disk: it switches to it by id (`POST /v1/models/switch`),
  whatever `switch.allow_implicit` says;
- **Copy download command** for one not on disk, or only partly: it copies
  `ignis-server model download <id>`. The page never downloads anything.

During a switch the band says what is happening and for how long; afterwards
it keeps the outcome: `Switched from … to …`, or why the switch was refused
or failed and which model serves now. A model the server
[unloaded while idle](../api/idle-unload.md) shows as **Unloaded**, not as an
error; the next message loads it again, and **Load** does it now. While a chat
reply streams, the band says how long a switch would let it finish
(`switch.drain_timeout`) before cancelling it.

![A switch from the 27B to Flash-Next, under way](../images/playground/server-switching.png)

## The configuration

**Scope** picks what the form edits: **All models** (the shared values) or one
model family (its own values, which win over the shared ones). The groups
(Server, Model, VRAM, Prompt reuse, Model switch, Speculative decoding,
Vision, Media, image, Download, N-gram table, KV-disk) are listed on the left;
an orange count marks the groups holding a change.

![The Model group: values, their source, and the fields that reload](../images/playground/server-config.png)

Each row is one field ([every field](../reference/configuration.md)): its
name, the first sentence of its description (**More** for the rest), and
where its value comes from: `From profile rtx5090`, `Set by
IGNIS_MODEL_THINKING_BUDGET` (a flag or an environment variable), `Shared`,
or `Own value`. A **reload** mark says a change to it reloads the model. A
field set only at start (the bind address, the API key, the artifact) is shown
read-only, `Set at start`.

Editing fills a draft and sends nothing. The dock at the foot counts the
changes and offers **Discard** and **Apply**, which reads **Apply and reload
&lt;model&gt;** when the draft holds a field that reloads it. The label is the
confirmation; there is no dialog.

- A live change answers at once: `Applied and written to <path>`, the config
  file in use, or why it was not written. A value a flag or an environment
  variable would override at the next start is named.
- A reload is followed in the model band, then `Reloaded <model> and written
  to <path>`.
- A refused value is named by the server; its row is marked `Refused by the
  server`, and nothing was applied.

**View as config file** shows the configuration in force, live changes
included, as `ignis.yaml`, to copy or download.

## Read more

- [Configuration](../configuration/index.md): fields, precedence and profiles.
- [Models that come and go](../why-ignis/models-that-come-and-go.md): what a switch does and costs.
- [Model switching](../api/model-switching.md): the same switch from your own code.
