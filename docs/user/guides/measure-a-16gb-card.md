---
title: "Measure a 16 GB card"
description: "Run `ignis-bench small-card` against your server and paste the record into a GitHub issue: the card, the VRAM plan, what the load adapted, decode speed and time to first token."
sidebar:
  order: 5
---

The `rtx50-16gb` profile is derived, not measured: the project has no 16 GB
card ([the profile](../concepts/profiles.md#rtx50-16gb)). If you run the 27B
on an RTX 5060 Ti 16 GB, a 5070 Ti or a 5080, one command against your running
server writes a record of what it does there, and a GitHub issue with that
record is how the profile gets real numbers.

## Run it

Start the server as you normally do, let it load the model, then:

```bash
ignis-bench small-card --endpoint http://127.0.0.1:8000 --out small-card.json
```

`ignis-bench` ships beside `ignis-server` in the release archives
([Install](../start/install.md#release-archives)) and in the container image.
With the container, run it inside, where both of the server's listeners are
(`<container>` is its name or id, `podman ps`), and keep the record from its
output:

```bash
podman exec <container> ignis-bench small-card --endpoint http://127.0.0.1:8000 > small-card.json
```

The record is printed on stdout and, with `--out`, written to that file too;
the progress goes to stderr. On the project's 5090 inside a 16 GB budget it
took two minutes, most of it the 64K time-to-first-token cell, which a 16 GB
card's slower prefill makes longer; `--cells 8192` leaves that cell out.

Measure the server with nothing else using it: the decode figures are one
request at a time, and another client's requests would land in them.

## What it records

- **The card**: its name and memory, and the PCIe link it served at beside the
  most the card and the slot allow, from `nvidia-smi` (read right after the
  decode requests, since an idle link slows down).
- **The load**: the server's version, the model and the profile, the VRAM plan
  line by line with its budget and the KV pool's pages (from the metrics
  listener), the configuration values that shape the load and where each came
  from (default, profile, file, env or flag).
- **What the load adapted**: the context and the decode lanes it runs at beside
  the ones configured, and every step it took to fit the card
  (`adaptation.adapted`, empty when it took none) — on a card whose desktop
  takes 3 GiB, `max_context 262144 -> 65536 (adapt_context_step 65536,
  adapt_spare_tokens 65536)` ([when a profile does not fit](../concepts/profiles.md#when-a-profile-does-not-fit)).
- **Decode speed**: tokens per second on the coding prompts and on three
  synthetic agent turns (a system prompt, tool calls and their results, a
  coding request), 256 tokens each, thinking off, and how many draft tokens
  DFlash2 got accepted meanwhile.
- **Time to first token** at 8,192 and 65,536 prompt tokens, three cold samples
  each after a warm-up. A cell longer than the context the load runs is not
  measured: the record says so and names the largest cell that fits, which
  `--cells` asks for.
- **Notes**: every piece it could not read, and why.

`--metrics` points at the metrics listener when it is not where the server's
`server.metrics_bind` says (the endpoint's host on that port); `--metrics off`
skips it, and the plan lines are then empty. `--samples` and
`--decode-tokens` change the sample counts. A piece it cannot read is `null`,
never a failed run.

## What it never records

No API key, no path, no address but the endpoint's, written without any
`user:password@`. The configuration is a fixed list of fields, never the whole
document. The preset sends no key either: a server started with
`--server-api-key` answers it `401`, and the record then says nothing was
measured. Run it against a server without a key.

## Send it

Open an issue at
[github.com/gpillon/ignis/issues](https://github.com/gpillon/ignis/issues/new)
titled with your card, and paste the whole file in a fenced block:

````markdown
```json
{
  "schema": "ignis-small-card/1",
  ...
}
```
````

Say anything the record cannot know: what else holds the card (a desktop, a
second application), the operating system, and any flag you started the server
with that you would like looked at.
