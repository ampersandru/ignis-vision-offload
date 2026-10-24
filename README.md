<p align="center">
  <img src="web/assets/images/logo-trasparent.png" alt="Ignis" width="140">
</p>

<h1 align="center">Ignis</h1>

<p align="center">
  <b>A single-GPU inference engine for Qwen3.8 (the 27B and the 125B Flash-Next MoE) and Qwen-Image-2.1, built for the load an agent makes.</b>
</p>

<p align="center">
  OpenAI-compatible HTTP API &nbsp;·&nbsp; Rust core &nbsp;·&nbsp; C++/CUDA kernel leaf &nbsp;·&nbsp; one Blackwell card
</p>

<p align="center">
  <img alt="license Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-d94b1f?style=flat-square">
  <img alt="target SM120a" src="https://img.shields.io/badge/target-SM120a%20%2F%20RTX%205090%20%7C%20RTX%20PRO%206000-d94b1f?style=flat-square">
  <img alt="models Qwen3.8-27B NVFP4, Qwen3.8-Flash-Next and Qwen-Image-2.1-Turbo" src="https://img.shields.io/badge/models-Qwen3.8--27B%20NVFP4%20%7C%20Flash--Next%20125B%20MoE%20%7C%20Qwen--Image--2.1-d94b1f?style=flat-square">
  <img alt="API OpenAI-compatible" src="https://img.shields.io/badge/API-OpenAI--compatible-d94b1f?style=flat-square">
  <br>
  <a href="https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer"><img alt="weights on Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20weights-Qwen3.8--27B%20NVFP4%20%2B%20DFlash2-ffb000?style=flat-square"></a>
  <a href="https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer"><img alt="uncensored weights on Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20weights-uncensored%20(abliterated)-ffb000?style=flat-square"></a>
  <a href="https://huggingface.co/gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis"><img alt="Flash-Next weights on Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20weights-Qwen3.8--Flash--Next%20trellis%202.5%20bpw-ffb000?style=flat-square"></a>
  <a href="https://huggingface.co/gpillon/ignis-image-2.1-turbo"><img alt="image model weights on Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20weights-Qwen--Image--2.1--Turbo%20(non--commercial)-ffb000?style=flat-square"></a>
</p>

<table align="center">
  <tr>
    <td align="center"><b>~170 tok/s</b><br><sub>one lane, average; peaks ~370</sub></td>
    <td align="center"><b>1,065 tok/s</b><br><sub>eight lanes, aggregate</sub></td>
    <td align="center"><b>~9,700 tok/s</b><br><sub>cold prefill, 46K prompt</sub></td>
    <td align="center"><b>~100 tok/s</b><br><sub>Flash-Next 125B MoE, one lane</sub></td>
    <td align="center"><b>7.11x</b><br><sub>KV capacity of BF16</sub></td>
    <td align="center"><b>~36 ms</b><br><sub>a decision, zero tokens decoded</sub></td>
    <td align="center"><b>896K</b><br><sub>context, with vision and DFlash2</sub></td>
    <td align="center"><b>~7.9 s</b><br><sub>a 2048² image, end to end</sub></td>
  </tr>
</table>

<p align="center"><sub>Measured on one RTX 5090, 64 GB DDR4-3200 MT/s, PCIe 3.0. How, and against what: <a href="#how-fast">How fast</a>.</sub></p>

---

Ignis is a deliberately specialized engine: **one model family, one class of
card**. It gives up generality and takes back speed. It loads an NVFP4
Qwen3.8-27B onto a single Blackwell card — an **RTX 5090** or an **RTX PRO
6000**, both `SM120a` ([why not DGX Spark?](docs/user/guides/dgx-spark.md)) — serves streaming and non-streaming
completions over the OpenAI v1 API, and is shaped around the workload a
developer actually produces — *one main agent plus a handful of subagents
hitting the same card at once*. Eight resident decode lanes run as one
batch-wide round; the paged KV cache is budgeted in bytes rather than in
sequences; conversation state survives the request that built it; a loop in
the model's reasoning is caught while it streams and taken back; images are
evidence, not an afterthought; and a decision can be answered without
generating a single token. A Playground, a live Monitor and Prometheus
metrics ship in the binary.

It also serves **Qwen3.8-Flash-Next**, the 125B MoE with 6B active, on the
same 32 GB card. Its 48 × 512 experts are re-quantized to a trellis code at 2.5
bits per weight on average (2 to 4 bits per expert, from calibration error); a
VRAM cache holds the hot ones and the rest stream from pinned host RAM, about
**100 tok/s at one lane on a PCIe 3.0 board**
([weights and numbers](https://huggingface.co/gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis)).
On a 96 GB RTX PRO 6000 every expert stays in VRAM with no copy in host RAM,
and the built-in `rtxpro6000` profile drafts with the model's own MTP head:
about 258 tok/s at one lane on code and 427 tok/s across eight
([measured](docs/findings/2026-10-18-rtxpro6000-profile-measured.md)).
And it makes pictures: **Qwen-Image-2.1-Turbo**, on FP8 GEMM and 8-bit
attention kernels for the 5090, serves OpenAI's image generation and edit routes — a
2048x2048 image in about 7.9 s end to end, an edit from up to ten input images,
the prompt optionally rewritten first by the 27B
([Images made](#images-made-on-the-same-card)). Its weights are for
non-commercial use only.

The server switches between these models without a restart — a request that
names another model is held and answered by it — and unloads an idle one after
five minutes so the card is free for other work; the next request loads it
again.

Rust owns everything above the *step* — scheduling, admission, KV accounting,
serving. The forward pass and all GPU compute live in a C++/CUDA static
library behind a flat, device-resident step-level C ABI. The engine is its own
dogfood target, and partly there already: it serves some of the coding agents
that build it, and the rest of that is what the remaining work is for.

## TL;DR

Take a build from **[Releases](https://github.com/gpillon/ignis/releases)** — a
Windows `.zip`, a Linux `.tar.gz`, or the `linux/amd64` container image on
`ghcr.io/gpillon/ignis`. Run it on a Blackwell card. (Every push to `main` that
changes code also publishes a rolling development build, **not a release**: the image
`ghcr.io/gpillon/ignis:develop` and the `develop` prerelease.)

```
ignis-server --model-max-context 524288 \
  --spec-backend dflash2 --spec-draft-tokens 7 --vision-enabled
```

No model yet? It offers to fetch one. Those flags set the 512K context `make`
serves (YaRN x2, which the default `auto` rope scaling picks for it), DFlash2
speculative decoding and image input; the Prometheus
metrics the Playground's Monitor reads are on by default. A bare `ignis-server`
already gets DFlash2 and image input from the built-in `rtx5090` profile, with
an 896K context (YaRN x3.5). Flash-Next is `--model-id
qwen3.8-flash-next`; the flags `make` serves it with are on its
[weights card](https://huggingface.co/gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis).
The image model is `ignis-image-2.1-turbo`: once `ignis-server model download
ignis-image-2.1-turbo` has fetched it, a request to `/v1/images/generations`
naming it switches the running server to it.

Then the OpenAI API is on <http://127.0.0.1:8000/v1>, the Playground and its
Monitor on <http://127.0.0.1:8000/ui/>, and Prometheus on
<http://127.0.0.1:9464/metrics>. The weights are on Hugging Face: the 27B in two
flavours, the standard one and an uncensored one, and Flash-Next:
[The weights](#the-weights). Everything
else — the container, building from a checkout, every flag — is
[Quick start](#quick-start) and [docs/user](docs/user/README.md).

## How fast

**Ignis on one RTX 5090** — Qwen3.8-27B NVFP4, hq-e8-2b KV, DFlash2 speculative
decoding with 7 draft tokens, greedy, short prompts. The served context is
**512K tokens**: the checkpoint's native 262,144 stretched by YaRN x2, as the
TL;DR flags and `make` set it. These runs used a 262,144-token maximum:

| load | throughput | source |
|---|---:|---|
| one lane, eight coding prompts (write, edit, explain), geometric mean | **367 tok/s** | [upstream quick wins](docs/findings/2026-09-24-upstream-quick-wins-ab.md) |
| one lane, predictable code / free prose | **491** / **167 tok/s** | [retained slots on the host](docs/findings/2026-09-28-retained-slots-on-the-host.md) |
| eight lanes at once, aggregate | **1,065 tok/s** | [upstream quick wins](docs/findings/2026-09-24-upstream-quick-wins-ab.md) |

**Other engines, same card, same model** — single-stream decode as their
authors published it:

| engine | weights and speculation | single-stream decode | source |
|---|---|---:|---|
| vLLM (main `g41f179b57`) | NVFP4, FP8 KV, DFlash2 drafter with 6 tokens | ~304 tok/s code at 8K, 221 at 60K | [HF discussion #132](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/132) |
| llama.cpp | NVFP4-MTP GGUF, q8_0 KV, MTP with 4 tokens | 122–142 tok/s | [DataCamp, 2026-08-25](https://www.datacamp.com/tutorial/how-to-run-qwen3-8-27b-locally) |
| llama.cpp | MTP | 70.5 tok/s | [HF discussion #132](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/132) |

Those rows are other people's runs: their prompts, their drafters, their
context lengths. They are not a same-session benchmark, so read them as an order
of magnitude, not a ratio. **Prefill was where ignis trailed; 0.7.3 made it
30-53% faster on long prompts.**
- **ignis**, measured cold: **9,717 tok/s on a 46K prompt, 7,865 on a 105K one**.
  The prefill attention now reads the compressed KV directly, instead of
  expanding it into BF16 planes first: +30% at 46K and +53% at 105K against
  0.7.2's kernels in one A/B session
  ([no prompt planes](docs/findings/2026-10-21-prefill-attention-without-prompt-planes.md)).
  A 2,048-token prefill chunk adds 2% more
  ([the 2,048 chunk](docs/findings/2026-10-21-the-27b-prefill-chunk-at-2048.md)).
  Decode is unchanged, and the KL divergence from BF16 is lower.
- **One same-session bench, someone else's**
  ([mkyt/qwen3-27b-5090](https://github.com/mkyt/qwen3-27b-5090), a 5090 at a
  400 W limit, 2026-10-11), cold 73K prompt:
  - vLLM 4,800 tok/s;
  - NInfer 7,000;
  - that engine 8,800 (its attention runs on FP4/FP8 tensor cores).
- The ~12,500 tok/s at 60K once published for vLLM in
  [HF discussion #132](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/132)
  is not reproduced there.

**The one same-session comparison** is against
[NInfer](#lineage), the engine whose kernels ignis started from: same artifact,
same card, one session, two process launches per side, before ignis had
speculative decoding. ignis is **1.06x the reference at one lane and 3.07x at
four**, and 3% behind it on inter-token latency p95
([hq vs BF16 live/live](docs/findings/2026-09-13-hq-vs-bf16-live-live.md)).

## What sets it apart

Every number below is a measurement on a 5090.

### The decode round

- **One batch-wide round for all eight lanes**, replayed from **per-width CUDA
  graphs** (widths 1..8), not one launch sequence per sequence.
- The round is **1,166 graph nodes and 15.81 ms of device time that is weight
  streaming at the card's bandwidth** — the backbone GEMM at 100% of roofline,
  the output head at 86%. Graph submission costs 2.2% of it, and the whole
  remaining headroom is the 4.8% the device spends idle.
- Prefill and decode **interleave at chunk level**, so a long prompt never
  stalls the lanes already generating.

### Speculative decoding (DFlash2)

- The served artifact carries a **grafted DFlash2 drafter**; `--spec-backend dflash2
  --spec-draft-tokens 7` runs draft-and-verify rounds against it.
- The drafter's vendored top-k gave one warp to each of seven columns over a
  248k vocabulary and spent 3,145 µs — 16.4% of decode kernel time — to move
  2.0 µs of memory. **Our row-split replacement does it in 44 µs at a
  bit-identical answer: 17.6% of a decode round, +21% decode throughput at one
  lane**. A vendored kernel *measured* as the bottleneck may be replaced — an
  exemption to verbatim vendoring.

### KV that costs bytes, not sequences

- **hq-e8-2b** is the serving KV format: a sequence-token costs **9,216 bytes
  against BF16's 65,536 — 7.11x the capacity**, so eight lanes at a 40,960
  context need ~3.02 GB instead of 20 GiB.
- BF16 is retained, not deprecated: it is the format every correctness oracle
  runs against.
- The GQA workspace zeroing is **dead work under hq** — every byte the next
  launch reads it writes itself — and removing it is worth 2.12% of GQA layer
  device time on a 70K prefill.

### State that outlives its request

- **Cross-request reuse**: a prompt checkpoint captured at the generation
  opener, a retained prefix shared between siblings, and a lazy spill into a
  pinned **KV-RAM** host tier with unified eviction.
- A device-to-device prefix clone is **0.25 ms for 148 MiB — 96x cheaper than a
  PCIe round trip and ~8,000x cheaper than re-prefilling the head**; a
  full-context snapshot/restore round trip is ~90 ms, about 100x cheaper than
  the re-prefill it replaces.
- The retained checkpoints and prefixes live in **pinned host memory**, not in
  VRAM, so the KV pool gets that room: **475K -> ~680K tokens at the default
  load, at level throughput**
  ([retained slots on the host](docs/findings/2026-09-28-retained-slots-on-the-host.md)).
- Below KV-RAM, a **KV-disk** tier takes the state of an evicted sequence and
  gives it back where it stopped: a ~1 GB spill takes ~0.63 s beside two
  decoding Flash-Next lanes, costing their inter-token p50 3-8%
  ([KV-disk contention](docs/findings/2026-10-08-kv-disk-contention-on-the-volume.md);
  on by default for Flash-Next).
- What a request resumed is reported, not guessed: `cached_tokens` on the
  usage, and the done line names the tier and the kind of reuse it came from.

### A VRAM plan, decided at load

- Every byte the process will hold is **reserved and laid out before the first
  request**, and the plan is printed: weights, workspaces, lane state, decode
  and drafter graphs, retained slots, media embeddings, and whatever is left
  becomes the KV pool. The Monitor shows that plan beside what is in use —
  [see it](#the-playground).
- The load that used to grow the process **2,734 MiB in 14 minutes now grows it
  6 MiB in 24.5**, and the printed plan matches what the OS reports to within
  16 MiB. No paging, no surprise at minute forty.
- In 0.7.3 the 27B handed **up to 2.4 GB of VRAM back to the KV pool**. The
  prefill attention no longer holds prompt planes (up to 1.02 GB, ~0.6 GB with
  image input on), and the 1.35 GB token embedding lives in pinned host
  memory, at ~1 µs per prompt token. On the 5090 the built-in profile's context went from 786,432 to
  **917,504 tokens, with image input and DFlash2 still on**
  ([embedding in host memory](docs/findings/2026-10-21-token-embedding-in-host-memory.md)).

### Lanes that know who is asking

- A request states its own **lane tag** — `interactive` or `agent` — through the
  `class` extension field or an `@<lane>` suffix on the model name. The tag
  drives protection, backfill priority and eviction order, so a burst of
  subagents fills the lanes a foreground conversation is not using instead of
  evicting it.
- Admission is a full state machine: protection, backfill class, temporal
  credit, frontier distance.

### The Responses API, over a WebSocket

- `/v1/responses` is OpenAI's Responses API — the wire Codex CLI, the OpenAI
  SDKs and the Agents SDK speak: streaming events, function tools,
  `tool_choice` (`required` and a named function force the call), reasoning
  items passed back and forth.
- **WebSocket mode** on `GET /v1/responses`: one connection per agent,
  `previous_response_id` plus only the new items on each turn instead of the
  whole history, a `generate: false` prewarm that really prefills and keeps
  the prompt state, and several streams multiplexed on one socket.
- **A full engine queues the request** on the socket, in one first-in
  first-out order across every connection, where HTTP would answer `503` and
  leave the retry to the client. Closing the socket cancels what it had
  running.
- The Playground talks to the server over that socket, so parallel agents
  are not held to the browser's six connections per host.

### Reasoning that finishes

- **A loop in the reasoning is taken back while it streams.** About a third of
  first turns on a long enumeration loop, at the model card's sampling — on
  NInfer with the same weights too, so the loop is the model's
  ([reasoning loops](docs/findings/2026-10-12-reasoning-loops-are-the-models.md)).
  The guard scores the reasoning as it is written; once a loop persists, it
  cuts back to the last clean window, puts a nudge in the model's own words,
  and the generation goes on in the same response. The client sees a tag where
  the loop was and `rewound_tokens` in the usage; the reasoning sent back on
  the next turn is repaired to what the model saw.
- Scored on recorded runs, it catches **25 of 25 harmful loops** (6 of 7 on
  40 fresh seeds), with **no fire on the reasoning of real opencode and qwen-code
  sessions**; on 79K- and 138K-token agent contexts a rewind restores the
  prompt in 0.08-0.10 s
  ([a loop persists](docs/findings/2026-10-16-a-loop-persists.md),
  [the guard](docs/findings/2026-10-14-reasoning-loop-guard.md)).
- **An end of turn drawn inside the reasoning closes it** — it becomes
  `</think>`, and the model goes on to the answer or the tool call it was
  writing instead of returning reasoning and nothing else.
- **A thinking budget whose close answers**: past the budget the reasoning is
  closed with a sentence that tells the model to answer from what it has,
  without one more tool call — the deferral that kept two agent sessions
  looping for 15 and 32 turns ([a close that answers](docs/findings/2026-10-01-a-close-that-answers.md)).

### Jev-like decisions, answered without generating (`/v1/decide`)

- A **classification is not a completion**. `POST /v1/decide` prefills the
  evidence once and reads the answer out of the logits at a single position.
- Seven primitives: **`noul`** (yes/no), **`choice`** and **`score`**; one
  constrained digit per step, **`number`**, **`point`** and **`box`**; and
  **`locate`**, which finds a line in a log or a passage in a text from the
  attention heads and a labelled choice, generating nothing.
- On 144 authored decisions the declared options hold a median **99.8%** of the
  model's distribution, and the readout costs **~36.5 ms and zero decoded
  tokens** at 0.934 balanced accuracy on the layout first measured; moving the
  evidence into the system block takes it to **0.963**.
- `point` reads a click target off a 4096px screenshot to within 84 px worst
  case, inside the button on every scene tested.
- `locate` finds the line on fresh cluster logs of **100K to 1M tokens 21 times
  out of 23, median 2.6 s**, with no long prefill and no token decoded
  ([zero-decode locate](docs/findings/2026-09-28-zero-decode-locate-very-long-logs-and-prose.md)).
- The same handler is served as **`/v1/systemone`**, so a Jev client reaches
  this engine by changing the URL and nothing else.

### Images as evidence

- `--vision-enabled` loads the tower and reserves its workspace; images arrive inline
  as data URIs or as URLs the server fetches, prepares and caches.
- The encoder output is kept past its request, keyed by content digest and
  grid: **four questions over one 4096x4096 screenshot go from 27.92 s to
  13.24 s** on one encode instead of four, and 9.49 s when the picture was
  already seen.
- The tower's quadratic curve is the checkpoint's own scheme, not a bug: 88.2%
  of a 16,384-column encode is the attention kernel, holding 164.7 TFLOP/s with
  0.1% device idle, so 3.67 s is the worst case per image — and the only lever
  is sending a smaller one.

### Images made, on the same card

- **Qwen-Image-2.1-Turbo** (`ignis-image-2.1-turbo`) is served on OpenAI's
  routes, `POST /v1/images/generations` and `POST /v1/images/edits`: the
  checkpoint's seven presets (2048x2048 to 2752x1536) and 1024x1024, 1 to 10
  input images per edit, transparent backgrounds, progress streamed as
  server-sent events.
- The 8-step denoising loop is **1.20 s at 1024x1024 and 6.11 s at
  2048x2048**, its FP8 GEMM at 67-73% of the published bar for the card and
  its 8-bit attention at 88-93% ([the image leaf](docs/findings/2026-10-14-image-model-on-the-5090.md)).
  Served end to end, a 2048x2048 generation is **7.87 s** and a two-image
  edit 9.48 s ([spec 05, live run](docs/specs/image-model/05-image-serving.md)).
- `prompt_enhancement` has the 27B rewrite the prompt with Qwen's own system
  prompts before the image model reads it; with the 27B loaded, the rewrite
  runs first and the server switches once.
- In the Playground, the Image view keeps **image sessions** — every step,
  marks drawn on a picture or sent as a mask, "change only the marked area" —
  and the chat has a `make_image` tool: the page switches to the image model,
  makes the pictures, and switches back.
- **The weights are under the Qwen Research License, non-commercial use
  only**: do not serve the image model in a commercial product.

### Models that come and go

- **One card, one model at a time, no restart.** `POST /v1/models/switch`
  moves the server between the 27B, Flash-Next and the image model; a request
  whose `model` names another known model triggers the same switch and is
  held until that model serves — no retry loop in the client. On the 5090 the
  27B to the image model is 10.0 s, and back 11.3 s.
- **Idle unload**: after five minutes with no request the model is unloaded
  — VRAM, pinned host memory and KV-RAM all freed for other programs — and the
  server stays up; the next request loads it again and is answered.
  Polling `/v1/models` or `/metrics` neither keeps it loaded nor loads it.
- **`GET /v1/models` lists what a request can be served by.** The loaded
  model comes first, then every model whose weights are on disk, so a client's
  model picker shows them all. A model whose weights are gone is a 404 and
  never tears the loaded one down.
- **A catalog it fetches from**: `ignis-server model download <id>`, every
  file pinned at a revision and its size and SHA-256 checked before use; an
  operator catalog file and a mirror endpoint for private or air-gapped
  machines.

### A context you can rescale

- **`auto` by default**: both models are trained to **262,144 positions**, and
  rope scaling follows each model's context — no scaling up to 262,144, then
  YaRN at `max_context / 262,144` (3.5 for the 27B's 917,504 on the 5090, 4 at
  **1,048,576**; factor up to 64).
  `--model-rope-scaling yarn:F` or `none` overrides it. What a long-context
  probe must ask is a question about *relative* position — a literal needle
  at 320K is recalled with YaRN and without it.

### Batteries in the binary

- **The Playground** at `/ui/` — chat with tools and subagents, parallel
  sessions, image input, an Image studio, a Decide tab, a live Monitor and a
  Server view that loads models and edits the running configuration. Built
  into the server, not a separate service.
- **One declaration per setting**: every option is a flag, an `IGNIS_*`
  variable and a key in a YAML or JSON config file, with per-model values and
  a measured `rtx5090` profile applied by default (and `rtxpro6000` for the
  96 GB card); `GET`/`PATCH /v1/config`
  reads and changes the running server, reloading the model only for a field
  that needs it.
- **`POST /v1/tokenize`** counts a chat body's prompt exactly as it would be
  served — template, tools, thinking controls — without serving it.
- **Prometheus** on its own loopback listener, on by default (`--server-metrics false` turns it off), with the VRAM plan, retained
  state, KV pool occupancy, decision counters and request lifecycle.
- **It fetches its own model.** Started with no artifact, a GPU build asks, then
  downloads and verifies it.
- **`--server-api-key` and `--server-expose`** — a keyed public URL through a quick Cloudflare
  tunnel, key always required.
- **Linux and Windows**, a `linux/amd64` container image on `ghcr.io`, and
  release archives cut from the same build.

## The Playground

Agents running in parallel, each on its own lane, with the engine's own timings
beside every reply:

![The Playground running four subagents](docs/user/images/chat_agents_salute.png)

A decision over prose — the winner, the whole distribution, and a score read as
levels — answered out of one prefill with nothing generated:

![A text decision and its distribution](docs/user/images/decide_1.png)

The same endpoint over an image: a point, a box, and a yes/no, each a digit at a
time, with the model's self-reported uncertainty on every axis:

![A decision over an image, with point and box](docs/user/images/decide_2.png)

The Monitor scrapes the engine's own metrics: throughput, lane occupancy,
latency quantiles:

![The Monitor, live](docs/user/images/metrics_1.png)

...and what the load actually reserved, against the constants that bound it:

![The VRAM plan and retained state](docs/user/images/metrcs_2.png)

The Image view, a studio of image sessions: the prompt as the 27B rewrote it,
a transparent 2048x2048 picture, and the next step's edit already pointed at
it. Every step is kept in the browser, and a picture can be marked before it is
edited again:

![An image session in the Image view](docs/user/images/image_generation.png)

The Server view: the model serving and the switch that brought it, every model
the catalog knows with what is on disk and a **Load** button, and the running
configuration, for all models or one family, field by field with where each
value comes from ([docs/user](docs/user/playground/index.md)):

![The Server view, after a switch to the image model](docs/user/images/server_config.png)

A model the idle clock unloaded is not an error: the header says so, and the
next message loads it again before it is answered. The picker under the prompt
box says what sending will do first:

![The chat after an idle unload](docs/user/images/playground.png)

## Quick start

### The weights

Four artifacts on Hugging Face: two of the 27B, both 18.07 GiB, both carrying
the DFlash2 drafter, Flash-Next, and the image model:

| | Hugging Face | file |
|---|---|---|
| **Standard** | [`gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer) | `qwen3_8_27b_nvfp4full-v2.ninfer` |
| **Uncensored** | [`gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer) | `qwen3_8_27b_nvfp4full-v2-huihui-abliterated.ninfer` |
| **Flash-Next** (125B MoE, 67.8 GiB with its MTP companion) | [`gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis`](https://huggingface.co/gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis) | `qwen3_8_flash_next_trellis_a25-v2.ninfer` |
| **Image** (Qwen-Image-2.1-Turbo, 15.57 GiB, **non-commercial use only**) | [`gpillon/ignis-image-2.1-turbo`](https://huggingface.co/gpillon/ignis-image-2.1-turbo) | `ignis_image_2_1_turbo-v2.ninfer` |

The server fetches the standard one by itself the first time it starts without
a model, and checks every file's size and SHA-256 before using it. To download
any of them ahead of time — no GPU needed, `--out <dir>` for a machine with no
network:

```
ignis-server model download qwen3.8-27b
ignis-server model download qwen3.8-27b-abliterated
ignis-server model download qwen3.8-flash-next
ignis-server model download ignis-image-2.1-turbo
```

The uncensored image is the standard one with the
[huihui-ai abliteration](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)
applied: 1,255 of its 1,325 objects are byte-identical, and only the 70 matrices
the abliteration changed are re-encoded. Start it with `--model-id
qwen3.8-27b-abliterated` (fetched like the standard one when it is not on disk),
or `make dev UNCENSORED=1` from a checkout; it is served under that id, so a
client can tell the two apart. **It does not refuse**, so put the
guardrails in the application or tool layer and do not expose it to untrusted
users.

### Container

```
podman run --rm --device nvidia.com/gpu=all -p 8000:8000 \
  -v /path/to/models:/models \
  ghcr.io/gpillon/ignis:latest --download-path /models
```

(`docker`: `--gpus all` in place of `--device`.) With no model in that
directory the server fetches one and verifies it. The Playground is then at
<http://127.0.0.1:8000/ui/>.

### From a checkout

```
make doctor          # toolchain, rust target, artifact, web deps
make dev             # build web + kernel + GPU server, then run it
make dev VISION=1    # the same, with the vision tower loaded for image input
make dev UNCENSORED=1  # the same, on the uncensored (abliterated) weights
make mock            # the same with no GPU, no kernel, no artifact
```

`make` alone lists every target; `make config` prints the exact server command a
run will use.

### A request

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'
```

Full flag table, API surface, model handling, container and release detail:
**[docs/user](docs/user/README.md)**.

Every field of the OpenAI request bodies is honoured, accepted as inert (a
value asking for what the server already does, like `n: 1` or `store: false`),
or refused with a `400` naming it in `param` — never dropped in silence. The
table is in the API reference at `/v1/docs/`.

## What it does not serve

Choices, not gaps — so nobody goes looking for them:

- **Embeddings, or a generic reranker.** Either is a second model on the card;
  the VRAM plan gives the card to one. A classification is `/v1/decide`, read
  off the loaded model's own readout.
- **`/v1/completions`.** A raw prompt skips the chat template the model was
  tuned on, and the turn boundaries reuse is keyed on; chat completions and
  responses cover it.
- **`/v1/batches`.** An offline queue is a storage service, and this server
  holds no request past its connection: send the requests at once, and the
  scheduler batches them.
- **Audio, in or out.** The model has no audio tower and no audio head.
- **LoRA adapters.** They are trained against BF16 weights, and the checkpoint
  is NVFP4: merge the adapter and re-export the checkpoint instead.
- **More than one model loaded at once.** The VRAM plan is decided at load,
  for one: the server [switches](#models-that-come-and-go) between models by
  a full reload instead.

## Building it

Two parts, in this order: the C++/CUDA kernel leaf, then the Rust workspace that
links it.

- **Toolchain** — Rust; MSVC C++ build tools (Windows) or GCC (Linux); the CUDA
  Toolkit (`nvcc`, target `SM120a`); CMake + Ninja; Node.js for the Playground.
- **Kernel leaf** — `kernel/build.ps1` on Windows, `kernel/build.sh` on Linux.
  Both configure and build into `kernel/build/`, which `crates/*/build.rs` link.
- **Workspace** — `cargo build --release -p ignis-server --features cuda` for
  the real engine; without `--features cuda` (or without an artifact) the server
  runs a deterministic CPU-only mock, which is how protocol and scheduler work
  gets done with no card.

### The repo

| Path | What lives there |
|---|---|
| `crates/core/` | Scheduler, admission, paged KV accounting, request state, host tier. |
| `crates/artifact/` | The `.ninfer` reader: reader, binder, materializer. |
| `crates/runtime/` | Safe wrapper over the step ABI (`CudaLeaf`, decode graphs). |
| `crates/server/` | HTTP, OpenAI schemas, decisions, metrics, telemetry. |
| `crates/logging/` | Structured logging: tracing layers, hotpath lint, trace context. |
| `crates/bench/` | Trace-replay harness, gate and canary runner. |
| `crates/vendor/` | The vendoring tool: manifest, hashes, patch records. |
| `kernel/` | The C++/CUDA leaf: the program and the vendored ops (CMake + nvcc). |
| `web/` | The Playground: React + Vite, embedded into the server at build time. |

`make` wraps all of it. Tests: `cargo test` is workspace-wide, CPU-only and
never touches the GPU; GPU work is checked by a separate explicit profile that
**fails** rather than skips when the card is busy. The 5090 fits one run at a
time — check `make gpu-status` before starting anything on it.

## Documentation

| | |
|---|---|
| [`docs/user/`](docs/user/README.md) | Running it: flags, API, models, container, releases, telemetry. |
| [`docs/adr/`](docs/adr/) | Every architectural decision, and why it was taken. |
| [`docs/findings/`](docs/findings/README.md) | Durable, evidence-backed measurements. |
| [`docs/agents/`](docs/agents/) | Conventions for agents working in this repo. |
| [`CONTEXT.md`](CONTEXT.md) | The glossary. One vocabulary, one meaning per term. |

The OpenAPI description of the HTTP surface is served at `/v1/openapi.json`,
with a browsable reference at `/v1/docs/`.

## Lineage

Ignis started as a branch of [**NInfer**](https://github.com/Neroued/ninfer),
in the fork [`gpillon/ninfer`](https://github.com/gpillon/ninfer), where its
first features were built on top of NInfer's engine. That branch soon showed
its limits. Parts of NInfer did not fit the tool it was meant to become, and
much of what that tool needed was not native to NInfer but lived in other
forks, to be cherry-picked one by one. The branch was becoming more
cherry-picks than work of its own. So rather than keep carrying it, Ignis was
rewritten from scratch, around everything NInfer was missing: its own forward
pass, sequence state, step ABI, scheduler and serving layer, with those
features native instead of picked.

To have a working engine from the start, the rewrite brought NInfer's CUDA
kernels in as they were: the kernel leaf **vendors them verbatim** under a
manifest that pins the reference commit and every file's content hash — a
claim you can diff. The pinned reference is
[`gpillon/ninfer`](https://github.com/gpillon/ninfer), itself downstream of
[`cometkim/ninfer`](https://github.com/cometkim/ninfer) (kernel-perf,
hyperquant KV, NVFP4-full, the original DFlash2 port) →
[`natpate/ninfer-windows`](https://github.com/natpate/ninfer-windows) (the
Windows port) → [`Neroued/ninfer`](https://github.com/Neroued/ninfer) (the
original engine). The `.ninfer` model artifact is still NInfer's format.

The vendored kernels are a starting point, not a destination: one at a time,
where measurement shows one is the bottleneck, it is rewritten. The latest is
the 27B's prefill attention, which no longer expands the compressed KV history
into BF16 planes before attending: up to 1.02 GB of device memory freed and
cold prefill up to 53% faster. The kernels added for Flash-Next and the image
model, which have no NInfer reference, are our own, or adapted or vendored from
a named, permissively licensed source recorded in `kernel/NOTICE`.

Ignis is **not** a fork or a port of NInfer. It is a rewrite that started from
NInfer's kernels, and is rewriting them in turn.

## Credits

With thanks to the NInfer project and its contributors:

- **cometkim** — kernel-perf integration (PDL decode chain, split-K prefill,
  per-request error boundary), the DFlash2 base port, the hyperquant KV cache,
  the 1M-context envelope, the NVFP4-full target.
- **Mirko Covizzi** — RTX 5090 Laptop compatibility, MTP adaptive
  verification-width tuning.
- **mr-september** — warmup/readiness decoupling, frontend streaming fixes.
- **dylan (dylanbrodiefafard)** — the RAM KV cache concept, LRU lane eviction,
  the decode CPU-spin fix.
- **Neroued** — the original NInfer engine.

## License

Apache License 2.0.
