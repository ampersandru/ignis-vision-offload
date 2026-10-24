---
title: "Lanes that know who is asking"
description: "A request says whether it is the conversation or a subagent, so a burst of subagents fills idle lanes and yields first when memory is short."
sidebar:
  order: 8
---

One card, eight lanes, and two kinds of request that look the same on the
wire: the conversation you are reading, and the subagents it sent off to search
the repository. When memory runs short, the server should not evict the
conversation to make room for a subagent's tenth grep. In ignis a request
**states which it is**, with a lane tag of `interactive` or `agent`, and the
tag drives who is protected, who backfills first and who loses residency.

## How it works

The tag is an ignis extension: the `class` field, or an `@<lane>` suffix on
`model`. Absent or unrecognised is `interactive`. The exception is
`/v1/decide`, where absent means `agent`: a decision holds no residency and
never takes a decode lane, so `interactive` would only let twenty queued
questions cut ahead of conversations.

Admission is a state machine pinned by tests
([ADR 0004](../../adr/0004-full-admission-in-v1.md)). When a queued request is
blocked, because no lane is free or its KV reservation does not fit, the
active requests are frozen and the earliest-finishing set whose release makes
the blocked one feasible is chosen as *donors*. A donor is not evicted while
that protection is open. Meanwhile other requests may backfill: a
*persistent* one fits the blocked request's future capacity and never borrows
the donors' pages, a *temporal* one does not fit it but finishes, by its own
work, before the donors' capacity is needed. One unit of work is one decode
token.

When memory does run short, eviction uses one priority across every tier
([ADR 0023](../../adr/0023-unified-eviction-priority.md)). On the GPU,
eligibility and protection come first, then class (`agent` before
`interactive`), then least recently used. A need moves only state that ranks
below the requester, and sequences not in the decode round go before lane
holders. Below the GPU, class is the only criterion left.
`reuse.retained_interactive_ttl` keeps a main conversation's checkpoint ranked
as interactive until it has been idle for 300 s.

When every lane is busy, an HTTP request gets `503 engine_full` with a
`Retry-After` estimate, and a request on the Responses WebSocket is queued
instead ([the Responses API over a WebSocket](responses-websocket.md)).

Where admission sits in the request's path:
[How ignis serves a request](../concepts/how-ignis-serves.md#2-admission-decides-whether-and-where-it-runs).

![A subagent's request in the agent reader: its own prompt, answer and figures, on an agent lane](../images/playground/agent-reader.png)

## The numbers

| Figure | Measured on | Source |
|---|---|---|
| None for the tag's effect. No finding isolates it: there is no interactive-against-agent A/B | not measured | [ADR 0004](../../adr/0004-full-admission-in-v1.md), the design |
| A browser holds 6 connections per origin over HTTP/1.1; with six agents streaming, every same-origin request waited while the engine answered each in about a millisecond | the Playground, six parallel streams | [Browser connection limit](../../findings/2026-09-18-browser-connection-limit-caps-parallel-streams.md) |

:::caution[A design, not a benchmark]
The lane tag is a design statement. The guide attaches no figure to it,
because none was measured. The admission rules are pinned by tests and the
eviction order is stated in an ADR, but no run compares a subagent burst with
and without the tag.
:::

## See it, control it

- **Per request**: `class: "interactive"` or `"agent"` on chat completions,
  responses and decide, or `model: "<id>@agent"`
  ([chat completions](../api/chat-completions.md)).
- **Config**: [`reuse.retained_interactive_ttl`](../reference/configuration.md#reuseretained_interactive_ttl)
  sets how long a main conversation's checkpoint ranks as interactive.
- **When full**: `503 engine_full` with `Retry-After` over HTTP (an overdue
  estimate waits at most 15 s); `response.queued` on the WebSocket.
- **Metrics**: `ignis_scheduler_requests` (by state) and
  `ignis_requests_rejected_total` ([metrics](../reference/metrics.md)).
- **Logs**: `ignis.request.admitted` carries the backfill class;
  `ignis.request.evicted` names what was moved.
- **Playground**: subagents run on `agent` lanes, and the engine timings sit
  beside each reply ([tools and agents](../playground/tools-and-agents.md),
  [Monitor](../playground/monitor.md)).

## Limits

- Two classes only. The engine the admission policy came from had three.
- One prefill lane: class orders the queue, it does not let a second prompt
  prefill at the same time.
- Class sets order, not a latency guarantee. When every tier is full of live
  work, the new request waits.
- `<id>@agent` is also a model name to the server. See
  [explicit and implicit switches](../concepts/switching-and-idle-unload.md#explicit-and-implicit-switches)
  for what a client that mixes names can trigger.

## Read more

- [How ignis serves a request](../concepts/how-ignis-serves.md): admission in the path of a request.
- [KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md#where-it-lives-three-tiers): where evicted state goes.
- [State that outlives its request](state-that-outlives.md): the tiers a lane's state moves through.
- [Decisions without decoding](decisions.md): why `/v1/decide` defaults to `agent`.
