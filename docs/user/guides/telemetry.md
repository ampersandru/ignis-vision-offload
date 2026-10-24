---
title: "Telemetry and metrics"
description: "Logs, OpenTelemetry traces and the Prometheus metrics listener."
sidebar:
  order: 10
---

Everything goes through one structured log on stdout, in the format
`IGNIS_LOG_FORMAT` picks (pretty on a terminal, JSON otherwise);
`IGNIS_LOG_LEVEL` sets the level.

- The request lifecycle is the `ignis.request.*` events (`admitted` / `ttft` /
  `done`), with the request id doubling as the OTel trace id
  ([ADR 0012](../../adr/0012-request-id-as-trace-id.md)). A cancelled request
  ends with `cancelled` instead of `done`: its `cause` is `timeout` (past its
  deadline, `--server-request-timeout` or `--server-stream-timeout`, whatever
  it was answered), `consumer` (its client hung up, its queued
  response was cancelled, or a loop-guard rewind replaced it) or `shutdown`
  (a model switch stopped it), beside how long it ran, whether it had a lane
  and the tokens it generated.
- The scheduler counters are the DEBUG event `ignis.scheduler.interval`, emitted
  whenever `waiting`, `running` or `kv_evictions` change
  ([ADR 0025](../../adr/0025-scheduler-interval-as-a-log-event.md)):

```text
2026-09-13T10:00:00.000Z DEBUG ignis.scheduler.interval - scheduler counters changed {tick=123, waiting=2, running=3, kv_evictions=1}
```

- The VRAM layout decided at load is the `ignis.runtime.vram_plan` event.

With metrics on (the default; `--server-metrics false` turns them off), the Prometheus exposition covers the request lifecycle and
rejections, KV pool pages and evictions, the KV-RAM arena, retained slots and
what reuse held or gave up, the VRAM plan, decoded tokens, forced thinking
closes, and decision counters with the answer-mass histogram ([ADR 0017](../../adr/0017-prometheus-metrics.md)).
The Playground's **Monitor** tab scrapes the same exposition. It scrapes every 2-5 s while the tab is visible and stops while the tab is hidden (so a tab left open does not keep a scale-to-zero deployment awake), scraping again when the tab shows.
