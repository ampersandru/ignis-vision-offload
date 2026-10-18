# server 14 — A loop the guard sees is counted, exported and shown

GitHub: #330 · ADR 0052, 0053, 0017 · Builds on playground/03 (the Monitor)

## Problem Statement

The loop guard (ADR 0052, tuned by ADR 0053) rewinds a looping reasoning on
every chat-completions and `/v1/responses` request, by default. What it does
is visible only in two places:

- the request log, one `ignis.loop_guard.*` line per loop it saw
  (`rewound`, `detected`, `too_little_kept`, `no_room`, `no_tokenizer`,
  `submit_refused`);
- the Playground's chat, which strikes through a rewind's text after its tag.

Nothing on `/metrics` counts it, and the Monitor shows nothing of it. Both
findings end on the same open question — *watch `ignis.loop_guard.rewound`
in real use; one false rewind reopens the defaults*
(`docs/findings/2026-10-14-reasoning-loop-guard.md`,
`docs/findings/2026-10-16-a-loop-persists.md`) — and the margin it watches is
thin: harmful loops scored at most 0.257, agentic reasoning at least 0.274,
against `loop_guard_ratio` 0.265. Today answering it means grepping logs.

Two more gaps show up next to it:

- **A rewind distorts the request counters.** A rewind cancels the engine
  request and submits a continuation (`loop_guard.rs`, `Guard::rewind`). The
  cancel goes through `Telemetry::on_cancelled` with `CancelCause::Consumer`,
  so `ignis_requests_cancelled_total` counts it, and the continuation counts
  once more in `ignis_requests_accepted_total`. One client request with two
  rewinds reads as 3 accepted, 2 cancelled, 1 completed on the Monitor's
  Request flow card, with nothing to tell those cancels from clients that
  hung up.
- **A guard close is counted nowhere.** After `retries` nudges the guard
  closes the reasoning with the thinking budget's close text, prefilled into
  the continuation's prompt (`starts_in_reasoning = false`, no budget). The
  scheduler never forces it, so neither `thinking_forced` nor
  `ignis_thinking_forced_closes_total` sees it. And that forced-close counter,
  which does exist, is not shown in the Monitor either.

## Solution

- The guard counts what it does on `/metrics`: every loop it sees by what it
  did about it, the requests it watched and those that looped, the tokens it
  took back, and how close each watched request came to the threshold.
- The Monitor gains a **Reasoning** card: loop-guard outcomes, the share of
  watched turns that looped, the tokens taken back, the margin histogram, and
  the thinking budget's forced closes.
- The Request flow card says how many of its cancels were rewinds.
- A turn's figures in the chat show the tokens a rewind took back, from the
  `rewound_tokens` the response already carries.

## User Stories

1. As the owner, I want a counter of rewinds on `/metrics`, so that I see how often the guard acts on real traffic without grepping the request log.
2. As the owner, I want every loop the guard saw counted by its outcome (rewound, closed, logged, held, no room, refused, no tokenizer), so that a guard that sees loops but cannot act on them is visible.
3. As the owner, I want the share of watched requests that looped, so that I can compare real traffic with the findings' "about a third of long enumerations".
4. As the owner, I want the lowest loop score each watched request reached, as a histogram, so that I see how close real agent reasoning comes to `loop_guard_ratio` before a false rewind happens.
5. As the owner, I want the tokens rewinds took back counted, so that I know what the guard costs in generated-and-discarded work.
6. As the owner, I want the guard's closes counted apart from the thinking budget's, so that neither counter hides the other.
7. As the owner, I want all of it in the Playground's Monitor, so that I see it while I work, as I see speculation and memory.
8. As the owner, I want the Request flow card to tell rewinds from cancels, so that a burst of cancels is not misread as clients hanging up.
9. As the owner, I want the thinking budget's forced closes in the same card, so that both ways a reasoning gets closed for it sit side by side.
10. As a Playground user, I want a turn's figures to say how many tokens a rewind took back, so that I can read the struck text against a number.
11. As an operator running `loop_guard: off` (benches, gates), I want no loop-guard series at all, so that the exposition is unchanged for them.
12. As a maintainer, I want each counter to be the exact tally of a log line, so that `/metrics` and the request log never disagree.

## Implementation Decisions

- **The metrics** (ADR 0017's table gains one row each; all lazily created,
  *absent until the first one*, like `ignis_thinking_forced_closes_total`):

  | Name | Type | Labels | Meaning |
  |------|------|--------|---------|
  | `ignis_loop_guard_watched_requests_total` | counter | none | Requests the guard watched: mode `log` or `rewind`, starting in their reasoning (images included: logged, never rewound) |
  | `ignis_loop_guard_looped_requests_total` | counter | none | Of those, the requests with at least one loop, whatever its outcome |
  | `ignis_loop_guard_loops_total` | counter | `outcome` | One per loop the guard saw, labelled by its log line: `rewound`, `closed` (a `rewound` line with `closed=true`), `detected` (log mode), `too_little_kept`, `no_room`, `no_tokenizer`, `submit_refused` |
  | `ignis_loop_guard_rewound_tokens_total` | counter | none | Generated tokens rewinds took back (`generated - kept` on each `rewound` line), closes included |
  | `ignis_loop_guard_min_ratio` | histogram | none | Per watched request that scored at least one window: the lowest normalized ratio any of its checks scored. Buckets 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, +Inf |

- **One tally, one log line.** Every `loops_total` increment sits next to the
  `ignis.loop_guard.*` line it counts, in `loop_guard.rs` and in
  `responses/queue.rs` (the socket's `submit_refused`), and nowhere else. A
  loop that emits no line (the client left, the request ended before its
  continuation was admitted) is not counted: it did nothing.
- **Off the model thread.** The guard is a tokio task; it reaches the
  `Metrics` handle the way `decide.rs` does (`server.metrics`), passed into
  `Guarding` / `EngineOps`. No new model-thread traffic: the existing
  `metrics_leave_the_model_thread_fact_traffic_unchanged` test stays green.
- **`watched` / `looped` / `min_ratio`** are recorded once per watched
  request, when its guard task ends, from state the guard already keeps
  (`Watch` computes every check's `Score`; the guard keeps the minimum across
  rewinds). A request too short to fill one window is watched, not observed.
- **The rewind cancels stay counted (default, owner to confirm).**
  `ignis_requests_cancelled_total` and `_accepted_total` keep counting engine
  requests, as ADR 0017 defines them; a rewind is one cancel and one
  admission, and ADR 0017's rows say so. The Monitor subtracts nothing: the
  Request flow card notes *n of these cancels were rewinds* when the window
  has any. The alternative, a `cause` label on the cancelled counter with a
  new `CancelCause::Rewind`, also changes the logging spec's
  `ignis.request.cancelled` line; it is out of scope unless the owner picks
  it.
- **The guard's close is not a forced close.** `ignis_thinking_forced_closes_total`
  and `thinking_forced` keep meaning the budget's close through the
  constrained decode. The guard's close is `outcome="closed"`.
- **The Monitor card** (`web/src/monitor/`): `exposition`, `snapshot`,
  `derive` know the new names; the mock fixture carries them. The card shows
  only once one of its series exists (the `SpeculationCard` pattern, GitHub
  #307). Inside it, for the selected window:
  - looped / watched turns, and their share;
  - loops by outcome, rewound and closed first, the rest only when non-zero;
  - tokens taken back, and their share of `ignis_decoded_tokens_total`;
  - the `min_ratio` histogram as bars, with the bucket holding the live
    `loop_guard_ratio` marked when the page has the live config (the Server
    view's source); without it, no mark;
  - forced closes by the thinking budget.
- **The per-turn readout** (`web/src/metrics/figures.ts`, `Readout.tsx`):
  `Figures` gains `rewoundTokens` from `usage.completion_tokens_details.rewound_tokens`
  (chat) or `output_tokens_details.rewound_tokens` (Responses), absent when
  the response has none, shown like `thinkingForcedAt`.

## Testing Decisions

- A good test reads what an operator or a user sees: `/metrics`, the request
  log, the rendered card. Never the guard's internals.
- **Server, mock compute** (`crates/server/tests/metrics_http.rs`; prior art:
  the forced-close test at the `ignis_thinking_forced_closes_total` case, and
  `declared_once`), on the loop text of `tests/support/loop_text.rs`:
  - no loop-guard series on a fresh server, nor with `loop_guard: off` after
    a looping request;
  - a prose request: watched 1, looped absent or 0, one `min_ratio` sample in
    a bucket above 0.40;
  - a looping request in `rewind` mode: `loops_total{outcome="rewound"}`,
    `rewound_tokens_total` equal to the response's `rewound_tokens`, looped 1;
  - past `retries`: `outcome="closed"`, and `ignis_thinking_forced_closes_total`
    unchanged;
  - `log` mode: `outcome="detected"`, no rewind;
  - a loop at the reasoning's start: `outcome="too_little_kept"`;
  - the same rewind over `/v1/responses` HTTP and the socket counts the same.
- **Tally = log**: one test captures the request log and `/metrics` for the
  same run and asserts each `outcome` count equals its line count, in its own
  test binary if it installs a tracing subscriber (a scoped `set_default`
  misses a callsite that parallel tests registered first).
- **Web** (vitest): `exposition`/`snapshot` parse the new series from the
  fixture; `derive` computes window rates and the share, and resets on a
  counter going down; `MonitorView` renders no card without the series and the
  card with them; the Request flow note appears only with rewinds in the
  window; `figures` carries `rewoundTokens` from both usage shapes.
- No GPU run: the guard, the counters and the card are all exercised on the
  mock.

## Acceptance

1. **Metrics.** The five series in Implementation Decisions exist with those
   names, types, labels and buckets, absent until their first sample, and
   none with `loop_guard: off`.
2. **Tally.** Every `ignis.loop_guard.*` line increments
   `ignis_loop_guard_loops_total` once, with its outcome; `rewound` with
   `closed=true` counts as `closed`. Asserted against the captured log.
3. **Watched, looped, margin.** Recorded once per watched request at its
   guard's end; `min_ratio` is the lowest normalized check across rewinds.
4. **Model thread untouched.** The model-thread fact-traffic test is green
   and unchanged.
5. **Surfaces.** Chat completions (streaming and not), `/v1/responses` HTTP
   and the socket count a rewind the same.
6. **Closes apart.** A guard close moves `outcome="closed"` and not
   `ignis_thinking_forced_closes_total`.
7. **Reasoning card** in the Monitor as in Implementation Decisions, absent
   without its series, on the mock fixture too.
8. **Request flow note** on rewinds in the window.
9. **Per-turn readout** of `rewoundTokens` on chat and Responses turns.
10. **Docs.** ADR 0017's table gains the five rows and the note that a rewind
    is one cancel plus one admission; ADR 0052's consequences point to the
    counters; CONTEXT.md's **Loop guard** entry names them.
11. `cargo test` passes workspace-wide; in `web/`, `npm run typecheck` and
    `npm test` pass.

## Out of Scope

- A `cause` label on `ignis_requests_cancelled_total` (see the default above).
- Loop guarding of the answer, and any change to detection, thresholds or
  nudges: this spec only observes.
- Per-request loop fields on `ignis.request.done`: the `ignis.loop_guard.*`
  lines already carry the request id.
- Alerts, Grafana dashboards, long-term storage.

## Further Notes

- Evidence: `docs/findings/2026-10-14-reasoning-loop-guard.md`,
  `docs/findings/2026-10-16-a-loop-persists.md`.
- The histogram's buckets bracket the measured margin (0.257 / 0.274) and the
  prose the tests use (0.45-0.49). If `loop_guard_ratio` moves out of
  0.15-0.50, the buckets move with it.

## As built (2026-10-18, GitHub #330)

Where the build departs from, or pins down, the text above:

- **Buckets.** `ignis_loop_guard_min_ratio` reuses the decision family's
  ten-bound ratio histogram: `0.1, 0.15, … 0.5` by 0.05, then `1` and
  `+Inf`. A zlib ratio is at most about 1, so `le="1"` below `_count` would
  be a score that is not a ratio, as it is for the answer mass.
- **The family appears whole** (Acceptance 1), the decision family's way:
  absent until the guard watched its first request, then every series,
  each outcome's zero included. The Monitor reads "no loop in N watched
  turns" off those zeros.
- **`submit_refused` is counted by the guard**, when its continuation's
  submit gives nothing: the line itself is written by the ops that refused
  it (`EngineOps`, the socket's `QueuedOps`), which have no metrics, and
  both write it exactly before giving nothing. `detected` is also what a
  request with images reads, since its loops are only logged.
- **When it is counted.** `watched` when the guard starts, a loop beside
  its log line (before the tag reaches the client), and the lowest score
  before the `Done` does, or when the guard ends without one. A scrape right
  after a response therefore already has all of it.
- **The rewind's engine requests stay counted**, the default above: the
  `accepted` / `cancelled` rows of ADR 0017 say a rewind is one of each.
- **Tests.** `tests/loop_guard_metrics.rs` (chat completions and
  `/v1/responses`, streamed and not; off, prose, rewind, close, log, held),
  the socket's rewind in `responses_socket.rs`, and
  `tests/loop_guard_metrics_log.rs`, one test in its own binary, for the
  tally against the captured log. `no_room`, `no_tokenizer` and
  `submit_refused` are also driven in `loop_guard.rs`'s unit tests;
  `submit_refused` is the one outcome the HTTP tests never reach.
- **The Monitor's Reasoning card** shows the budget's closes alone when only
  that counter exists, and the guard's figures once its family does. The
  live `loop_guard_ratio` is **not marked** on the histogram: no part of the
  page reads it today, and reading it needs new plumbing; the card's note
  names the default (0.265) instead. `npm run dev:mock`'s simulated server
  does not model the guard, as it does not model speculation or the image
  jobs: the card is exercised on `fixture.ts`'s exposition in the tests.
