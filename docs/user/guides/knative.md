---
title: "Run behind Knative"
description: "Scale-to-zero and connection draining: the socket idle close and the Monitor's polling."
sidebar:
  order: 20
---

Behind Knative, or any proxy that drains connections before it stops a
container, an open Responses WebSocket is a connection in flight. Knative's
queue-proxy waits for it to drain before the server even gets its SIGTERM, up
to the revision's `timeoutSeconds`, and counts it as concurrency, so the
service does not scale to zero while one is open. The server closes a socket
left idle for `--server-socket-idle-close` seconds (default `300`,
[WebSocket mode](../api/responses.md)), and the Playground closes its own a
minute after its tab is hidden, so a tab left open holds the pod, and its GPU,
for minutes rather than for the revision's timeout. Lower it
(`-e IGNIS_SERVER_SOCKET_IDLE_CLOSE=120`) to release the pod sooner; a
client's next request opens a new socket either way.

The Monitor is the other thing that keeps a scale-to-zero service awake: a
visible Monitor tab polls `/ui/metrics` every few seconds and counts as
traffic, so it holds the pod whatever the socket settings say; a hidden one
sends nothing (GitHub #329 item 11).
