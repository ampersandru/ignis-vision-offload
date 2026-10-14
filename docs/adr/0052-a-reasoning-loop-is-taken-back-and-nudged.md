# ADR 0052 — a reasoning loop is taken back, the model is nudged, and the client is told

## Status

Proposed (2026-10-14), for the owner to accept. The design is the owner's (2026-10-13: an
engine-level detector off the critical path, an injected sentence, a few retries before
the reasoning is closed, a visible tag the Playground strikes through). The numbers are in
the finding `docs/findings/2026-10-14-reasoning-loop-guard.md`. Code on branch
`experiment/loop-rewind` (`crates/server/src/loop_guard.rs`). **Flags a conflict with ADR
0048**, whose rejected alternatives include re-submitting a turn from the server (below).

## Context

Qwen3.8-27B loops inside its reasoning on long enumerations. On the Playground's "whole
history of Rome" prompt, about a third of first turns loop
(`docs/findings/2026-10-12-reasoning-loops-are-the-models.md`), and about a quarter end
badly: the thinking budget forces the close, the model ends its turn inside the
reasoning, or no answer fits in what is left. The loops come in several shapes:

- verbatim text;
- a single token in every slot;
- counting (`the 321 AD, the 322 AD, …`);
- templates whose slots vary (`the election of Clement XII, the construction of the
  Sistine Chapel (1508–1512)`).

The rate is the same on ninfer with the same artifact. It is the model's, and ignis
cannot fix the model. A turn that loops wastes up to the whole 14,336-token budget, and an
agent that reaches the forced close keeps 2,048 tokens for its answer.

What ignis can do is notice. A loop is text that compresses far better than reasoning
does. A request's reasoning passes through the server on its way to the client, and the
server can score it as it goes without touching the scheduler.

What the server cannot do is step back cheaply. On the hybrid models the GDN state
absorbs every token it is fed, and there is no position to return to without a
checkpoint. The only way back is a new request whose prompt is the kept text.

## Decision

**A guard sits between a request's engine event stream and its consumer.** It is a tokio
task, so it is off the scheduler thread and adds nothing to a round. It watches only the
reasoning of a request that starts inside its reasoning block. Until it acts, it forwards
every event unchanged.

**Detection is a compression score over normalized reasoning.**
- Every 250 raw reasoning characters, the guard takes the last `loop_guard_window`
  characters of the reasoning after three steps:
  - fenced code is removed, and a block still open is never scored;
  - inline code is removed;
  - every run of ASCII digits becomes `0`.
- It computes the zlib-9 compress ratio of that text.
- A check fires when that ratio is under `loop_guard_ratio` **and** the ratio of the last
  `loop_guard_window` raw characters is under `loop_guard_raw_gate`.
- Why each step:
  - the code steps keep legitimate code out of the score;
  - the digit fold turns a counting or template loop into a repetition;
  - the raw gate keeps numeric tables out, where the digits carry the information.

**The action is a rewind: cancel, then a continuation.**
- **The cut.** The guard walks back over its own checks to the last one that scored at
  least `loop_guard_clean`. The cut is the raw start of that check's window, minus
  `loop_guard_margin`, snapped to a line break, and never before the previous cut. If the
  kept text would end inside a ``` block, the cut moves before that block's fence.
- **The continuation.** The engine request is cancelled. The **continuation** is the same
  request input, with its prompt extended by the tokens kept before the cut and by a
  **nudge**: a sentence in the model's own voice ("Wait, I'm starting to repeat myself.
  Let me stop and move on with what I already have."). It gets what is left of the
  generation cap and of the thinking budget, and a fresh seed.
- **Retries and close.** After `loop_guard_retries` nudges, the next fire continues with
  the thinking budget's own close text in place of a nudge. That closes the reasoning,
  and the model answers.
- **One HTTP request.** One HTTP request may run several engine requests. The guard maps
  their events back onto one stream, with its token counts and offsets.

**The client keeps what it received and is told what was taken back.** OpenAI's wire has
no way to retract text, and buffering the reasoning until it is checked would hold it back
by a whole window. So the guard streams a tag into the reasoning channel, followed by the
nudge or the close:

```
\n\n[LOOP DETECTED, REWINDING <n> TOKENS / <m> CHARS]
```

`<m>` counts the Unicode scalar values before the tag, back to the cut. A client that
understands the tag strikes them through: the Playground does
(`web/src/ui/rewinds.ts`). Any other client shows the tag as text. Every standard field
keeps its meaning. This is the extension rule of the Responses spec: ignis extensions are
additive and never change what a standard field means.

**Usage counts what the model generated.**
- `usage.completion_tokens` counts every token generated for the request, the rewound
  tokens and the tags included: they are work done, and they are what the model was
  billed for in time.
- `usage.completion_tokens_details.rewound_tokens` (an additive ignis field) says how many
  of them were taken back. It is present only on a request the guard rewound, and an
  absent field means 0.
- The generation cap bounds what the client keeps, not the work. A rewound request may
  report `completion_tokens` past its `max_tokens`.

**The guard covers every request that reasons, on both surfaces.**
- That means `POST /v1/chat/completions`, and `/v1/responses` over HTTP and over the
  WebSocket.
- The Playground talks over the Responses WebSocket by default, and Codex CLI speaks only
  Responses.
- `/v1/responses` reports the same count as `output_tokens_details.rewound_tokens`.
- A request with images is logged, never rewound, until a continuation can carry its
  media.
- The answer is not watched: a nudge cannot go into content.

**Every threshold is a live config field** (ADR 0046, model group, Visible + Patchable,
no reload):

| field | default |
|---|---|
| `loop_guard` | `off` (also `log`, `rewind`) |
| `loop_guard_normalize` | on |
| `loop_guard_window` | 1,000 characters |
| `loop_guard_ratio` | 200 permille |
| `loop_guard_raw_gate` | 300 permille (0 = off) |
| `loop_guard_clean` | 400 permille |
| `loop_guard_margin` | 0 characters |
| `loop_guard_retries` | 2 |

**`off` until the guard is measured on an agentic prompt.** It has been measured only on a
2,000-token prompt. A run with a 50,000-token-or-larger context decides the default. If it
holds, the default becomes `rewind`, since the data favours it.

## Alternatives rejected

- **Roll back in place** (truncate the KV, restore the GDN state). Attention pages could
  be truncated. The recurrent state, though, would need a checkpoint at every possible cut
  point. A cut can reach 10,000 characters back, and a device snapshot of the mutable
  state is ~148 MiB per lane, so a ring of them costs gigabytes. Host snapshots at
  PCIe Gen 3 cost a copy every check.
- **Re-submit a turn from the server.** This is the alternative ADR 0048 rejected, and
  here it is the chosen one. 0048 rejected it because the leaf could redirect an EOS for
  free, and a resubmit re-prefills the whole reasoning. For a loop there is no free
  in-leaf fix: the damage is thousands of tokens back. A rewind costs one prefill of the
  prompt plus the kept text (measured 0.4-1.7 s at 4-14k tokens). A loop left alone can
  cost the rest of the budget.
- **Close the reasoning at the first loop.** That forfeits the rest of the budget. After
  one nudge, 9 of 16 rewound runs went on reasoning normally.
- **Prevent loops in the sampler.** `presence_penalty` 1.5 halved the loops on 8 seeds but
  lengthened the reasoning into more forced closes. `frequency_penalty` punishes code,
  which legitimately repeats. Neither is a fix for agents.
- **Detect by logit entropy.** Entropy measures how predictable a token is, and counting
  loops and code are both predictable. Logprobs are not served either (GitHub #288).
- **Other text signals:**
  - **Novelty against the reasoning's history** and **the share of repeated lines.**
    Reasoning legitimately re-drafts code it wrote earlier, and both fire on it.
  - **A raw score alone.** It cannot separate counting and template loops from code.
- **Guard in the clients.** Every client would need one, and the Playground, opencode and
  Codex all differ. The server sees every request.
- **Retract the text on the wire.** No OpenAI event does that. A client that does not
  know an invented one would show the loop and the continuation as one text. A tag
  degrades to text in a client that does not understand it.

## Consequences

- **Reference parity.** Output differs from the reference whenever the guard acts. Benches
  and gates run with it `off`, which is the default.
- **Cost of a rewind.** A rewind is one engine request more. Today the continuation
  prefills its whole prompt, because nothing offers the cancelled request's prompt for
  reuse (ADR 0029). Reusing it is the follow-up that makes a rewind cheap on a long
  agentic context.
- **What a continuation does not carry over:**
  - its penalty counts start from zero;
  - a forced tool call (`tool_choice`) is not forced after a close the guard made.
- **Misses.** The guard misses loops that compress no better than legitimate text.
  Measured on the held-out run: 8 episodes at 0.21-0.31 normalized. The lowest legitimate
  texts are at 0.257 (Rome) and 0.273 (agentic coding), so no threshold change recovers
  them.
- **Observability.** The log carries `ignis.loop_guard.detected`, `rewound`, `no_room`,
  `no_tokenizer` and `submit_refused`, each with its scores and offsets. `log` mode
  reports the cut a rewind would have made, without acting.
- **Wiring.** Implementing this ADR means:
  - wiring `/v1/responses`, HTTP and WebSocket, with its own count field;
  - wiring the `rewound_tokens` field;
  - the glossary terms (**Loop guard**, **Rewind**, **Continuation**, **Nudge**) in
    `CONTEXT.md`.
