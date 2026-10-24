---
title: "System One decisions, without decoding"
description: "A System One-compatible API (POST /v1/systemone, POST /v1/decide) that reads a yes/no, a choice or a score from one position of the model's distribution: one prefill, no tokens generated."
sidebar:
  order: 11
---

An agent asks small questions all day: is this the right queue, did the test
pass, which of these three files matters. Answering each with a generated
completion is slow and hard to parse. ignis answers them through the **System
One API**, the typed-decision API the Jev model is served under: the same
`POST /v1/systemone` route and body, so a System One client points its base
URL at ignis and keeps working, here on a local model. It reads the answer
from the model's distribution instead of generating one. The evidence is prefilled once, each option
gets a one-token label, and the answer is the restricted softmax over those
labels at the single position after the prompt. A decision costs **one prefill
and decodes zero tokens**: 36.5 ms at the median in the measured run.

## How it works

`noul` (yes/no), `choice` and `score` are pure logit readouts. The server gives
each option a single-token label, runs the prompt, and normalizes the
probability of the labels at the last position. The top probability comes back
as `confidence`, and the share of the full next-token distribution that the
declared options hold is the **answer mass**, exported as a histogram. A low
mass means the model wanted to say something else, the one failure nothing in
the answer itself can show.

`number`, `point` and `box` generate one constrained digit per step with the
prefix forced (`{"x":` then three digits), so they cost a prefill plus a round
per digit. `locate` finds a line in a log, an item in a list or a sentence in a
text without generating: calibrated attention heads at the end of the prompt
narrow the text to a few candidates and a labelled `choice` decides. A log is
first folded into templates plus values, so nothing near its length is
prefilled.

`POST /v1/decide` and `POST /v1/systemone` are the same handler, and the
errors follow System One's own table; only `model` must be left out or name
the model ignis serves. A decision defaults to the `agent` lane class: it holds no residency and takes no
decode lane, so a fan-out of questions queues behind conversations instead of
cutting ahead of them. The request and response are in [Decisions](../api/decide.md).

![Three questions over one message in the Decide tab: a yes/no, a choice with its distribution, a score](../images/playground/decide-text.png)

## The numbers

| Figure | Measured on | Source |
|---|---|---|
| The declared options hold a median 99.8% of the distribution (p5 0.986, min 0.929); the unrestricted winner is a declared letter on 144 of 144 | 144 authored decisions, 27B, BF16 KV, thinking off | [Typed-option logit readout](../../findings/2026-09-19-typed-option-logit-readout.md) |
| Prefill p50 36.5 ms, p95 40.7 ms, 28.0 decisions/s serial, no prefix reuse | same, prompts of about 132 tokens | same |
| Balanced accuracy 0.934, and 0.963 with the evidence moved into the system block (a difference of 4 rows) | same 144 rows | [The evidence belongs in the system block](../../findings/2026-09-20-the-evidence-belongs-in-the-system-block.md) |
| `point` inside the target on 236/240, 240/240, 120/120 and 60/60 of the four pre-registered sets | served endpoint, attention-head method | [The sub-cell reading holds through decide](../../findings/2026-09-23-the-sub-cell-reading-holds-through-decide.md) |
| `point` errors of 84, 18 and 32 px on three synthetic 4096 px screenshots, every point inside its button | first constrained-digit experiment | [Constrained digit readout](../../findings/2026-09-19-constrained-digit-readout-points.md) |
| `locate` on logs of 100K, 200K and about 1M tokens: 21 of 23 (91.3%) against a floor of 85% registered in advance; median 2.6 s, p90 6.2 s | one cluster capture | [Zero-decode locate, very long logs and prose](../../findings/2026-09-28-zero-decode-locate-very-long-logs-and-prose.md) |
| `locate` on prose: 88.0% overall, but 7 of 12 (58.3%) at 1M tokens, which failed its per-group floor | same | same |

:::caution[What these numbers are not]
The 36.5 ms is one short prompt, serial, with no prefix reuse. The 144 rows
come from one authored fixture whose own provenance says it is model-reviewed,
not human-adjudicated, so 0.934 and 0.963 are not a quality claim for your
task, and the finding says answer mass and calibration are sturdier than
accuracy. The 0.9 abstention threshold needs re-measuring on any workload
before it is trusted. `point` is bounded by the model's roughly 32 px cell; the
84 px is the worst of three synthetic scenes, and the pre-registered sets are
the stronger evidence.
:::

## See it, control it

- **The route**: `POST /v1/decide`, or `POST /v1/systemone`; types `noul`,
  `choice`, `score`, `number`, `point`, `box` and `locate`, with `within` as a
  JSON Pointer for `locate`.
- **In the response**: `confidence` and `ranking`; for `locate` also `found`,
  `segment`, `value` and `pointers`. `usage.output_tokens` is 0 for the three
  logit readouts.
- **Metrics**: `ignis_decisions_total` (by primitive),
  `ignis_decision_answer_mass` and `ignis_locates_total`, in the
  [metrics](../reference/metrics.md).
- **Logs**: `ignis.decide.locate`, `ignis.decide.located` and
  `ignis.decide.pointing_head`.
- **Playground**: the [Decide tab](../playground/decide.md) builds requests and
  shows the full distribution behind each answer; decisions are stored in the
  browser.

## Limits

- Only `noul`, `choice` and `score` are pure logit readouts. `number`, `point`
  and `box` decode a digit per step, and `locate` runs several prefills and head
  passes.
- `locate`'s 91.3% is for logs. On prose the 1M-token group failed its
  registered floor.
- A decision has no reasoning: the loop guard does not apply, and the answer is
  the model's first reading of the evidence.
- The wire shape is a third party's, and the [Decisions](../api/decide.md) page
  says so.

## Read more

- [Decisions](../api/decide.md): the primitives, `locate` and the response.
- [Images as evidence](images-as-evidence.md): decisions over a screenshot.
- [Lanes that know who is asking](lanes.md): why a decision defaults to `agent`.
