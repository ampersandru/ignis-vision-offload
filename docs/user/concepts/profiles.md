---
title: "Profiles and defaults"
description: "What a hardware profile is, where it sits between the config file and the built-in defaults, what the rtx5090 and rtxpro6000 profiles choose and why, and how to define your own."
sidebar:
  order: 8
---

Most settings have a sensible default that works anywhere. A few do not: the
longest context a card can hold, how many lanes it can feed, whether a draft
head pays for its memory. Those depend on the card. A **profile** is a named
bundle of such defaults for one kind of card. It is a smarter default, never an
override: anything you set yourself wins over it.

## Where a profile sits

Every field is resolved from five places, highest first
([configuration](../configuration/index.md)):

```text
flag  >  environment variable  >  config file  >  profile  >  built-in default
```

So a profile fills in only what you left unsaid. `--model-max-context 131072`
beats the profile's context; a profile's value beats the hardcoded default; and a
field the profile does not mention keeps its hardcoded default. An empty value
counts as unset.

Choose a profile with `--profile <name>`, `IGNIS_PROFILE`, or `profile:` in the
config file. With none of them the profile is `rtx5090`; to skip the layer
entirely and run on the hardcoded defaults, name `none`. A name that is not
defined is refused with the list of the ones that are.

:::caution
The profile is chosen by name. The server does not look at which card it runs on,
so a start on another card uses `rtx5090` unless a profile is named, and a plan
the card cannot hold is then refused at load
([the VRAM plan](vram-plan.md)). Name the profile that matches the card, or
`none`.
:::

Profile values can be scoped to a model family in the same shape as the file:
`qwen38` for the 27B, `qwen38flashnext` for Flash-Next. A field only one family
has, such as the number of decode lanes or the pinned expert bytes, is passed over
by the other family, not refused. `GET /v1/config/origins` says where each value
in force came from, `profile` included.

## The built-in profiles

Both carry numbers measured on the card they name, with the finding for each
choice linked below.

### `rtx5090`

The default: a 32 GB RTX 5090.

| Field | 27B | Flash-Next |
|---|---|---|
| Context (`model.max_context`) | 917,504 | 589,824 |
| Vision | on | not applicable |
| Speculation | DFlash2, 7 draft tokens | off |
| Decode lanes | 8, fixed | 3 |
| Prefill chunk | 2,048 | 1,024 (the default) |
| VRAM headroom | 1,200 MiB | 1,536 MiB (the default) |
| Pinned hot experts | not applicable | 2 GiB |

Why each:

- **The 27B's context and headroom.** Raising `max_context` under vision until
  the plan refused it, then trading headroom down from 1,536 to 1,200 MiB, reached
  917,504 tokens with DFlash2 still on; 950,272 is refused by 59 MB. The
  alternative trade, turning speculation off, reached about 864K tokens, and the
  profile chose to keep DFlash2. Past the checkpoint's trained 262,144 positions
  the `auto` rope scaling applies YaRN at `context / 262,144`: a factor of 3.5 for
  the 27B and 2.25 for Flash-Next
  ([ADR 0057](../../adr/0057-rope-scaling-is-auto-by-default.md),
  [ADR 0059](../../adr/0059-flash-next-applies-yarn-on-q-and-k.md)).
- **The 27B's prefill chunk.** With vision on, the image encoder sizes the
  workspace, so a 2,048-token chunk costs no VRAM over 1,024. It prefilled
  1.9 and 2.2 percent faster at 46K and 105K, with a lower KLD against BF16 and
  decode unchanged; the cost is that a decoding lane's longest pause while a
  prompt prefills grows by about 65 percent
  ([finding](../../findings/2026-10-21-the-27b-prefill-chunk-at-2048.md)).
- **Flash-Next's context and lanes.** 589,824 is 512K plus two 32K
  allowances, the floor asked for on both the expert cache (at least 15 GiB; it
  measured about 16.6) and the context. Three lanes is Flash-Next's own default.
- **Speculation off for Flash-Next.** Decode on this card is bound by the experts
  copied over PCIe, which MTP does not reduce
  ([speculative decoding](speculative-decoding.md)).
- **Pinned hot experts.** Keeping the calibration-hottest 2 GiB of experts in the
  cache gives back 2.149 GB of host RAM for 0.1 to 0.2 percent of one-lane and
  0.2 to 1.1 percent of three-lane decode, with greedy text unchanged. At 4 GiB
  the cost reached 4.3 and 5.9 percent
  ([finding](../../findings/2026-10-18-pinned-hot-set-measured-on-the-5090.md),
  [ADR 0055](../../adr/0055-a-pinned-hot-set-is-opt-in.md)).

### `rtxpro6000`

The RTX PRO 6000 Blackwell series (Workstation, Max-Q and Server Edition, 96 GB
each), the same architecture as the 5090. Measured on a Server Edition under
Linux ([finding](../../findings/2026-10-18-rtxpro6000-profile-measured.md)).

| Field | 27B | Flash-Next |
|---|---|---|
| Context | 1,048,576 | 1,048,576 |
| Vision | on | not applicable |
| Speculation | DFlash2, 7 draft tokens | MTP, 3 draft tokens |
| Decode lanes | 8, fixed | 8 |
| Prefill chunk | 2,048 | 1,024 (the default) |
| VRAM headroom | default | default |
| N-gram hot rows | not applicable | `auto` |

Why each:

- **A million tokens of context for both.** It is the 27B's whole compressed-KV
  attention envelope, and the plan holds it. `auto` rope scaling makes both
  models YaRN at 4. No request that long was run.
- **Headroom left at the default.** Nothing has to be traded for vision on 96 GB.
- **Eight lanes.** The expert cache holds nearly every expert, so a lane costs
  its state and its share of the KV pool, and the aggregate peaks at eight.
- **MTP for Flash-Next, at the default row budget.** Few experts cross PCIe here,
  so speculation pays at one to four active lanes and with simulated agents. A
  wider row budget lost on this card, so the profile keeps 8 rows
  ([speculative decoding](speculative-decoding.md)).
- **`auto` n-gram hot rows.** The whole n-gram table in RAM keeps the disk out of
  every decode round, and `auto` takes it wherever the host plan leaves room,
  never refusing a start the 1 GiB default would make
  ([n-gram cache](../configuration/ngram-cache.md)).
- **No pinned hot set.** The profile leaves `vram.pinned_expert_bytes` at its
  default, 0.

## Defining your own

A config file's `profiles:` section holds profiles in the file's own shape, one
per name, and `profile:` selects one:

```yaml
profile: workstation
profiles:
  workstation:
    vram:
      headroom_bytes: 2G
    model:
      qwen38:
        max_context: 262144
    spec:
      qwen38:
        backend: dflash2
        draft_tokens: 7
```

A built-in profile is only the default contents of the same mechanism, so one you
define resolves exactly as they do. A profile in the file with a built-in's name
replaces it for that file. A good profile is found the way the built-in ones
were: start from a candidate context, raise it until the load is refused, back
off, and write down what you measured
([the VRAM plan](vram-plan.md)).

`ignis-server config print` shows what a start would run, `ignis-server help
--fields` lists every field a profile can set, and the Server view of the
Playground edits the running configuration
([the Playground](../playground/index.md)).
