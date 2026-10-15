# A reasoning-loop guard: a normalized compression score finds the loops, a rewind ends most of them

- Kind: experiment
- Status: current
- Observed: 2026-10-14
- Last verified: 2026-10-15
- Scope: server / reasoning-loop guard on chat completions (`crates/server/src/loop_guard.rs`, config v2 `model.loop_guard*`, branch `experiment/loop-rewind` at `f19d6f4`, not on main); serving / sampled reasoning on qwen3.8-27b; Playground reasoning view
- Related: [Reasoning loops are the model's](2026-10-12-reasoning-loops-are-the-models.md), [ADR 0048](../adr/0048-an-end-of-turn-inside-the-reasoning-closes-it.md), [ADR 0046](../adr/0046-one-declaration-per-config-field.md), [#288](https://github.com/gpillon/ignis/issues/288)
- Superseded by: none

## Question

The [previous finding](2026-10-12-reasoning-loops-are-the-models.md) showed that about a
third of first turns on a long enumeration loop, and that the loop is the model's. The
model cannot be fixed from ignis. Can the server catch a loop while the reasoning
streams, take it back and get the model out of it, without disturbing legitimate
reasoning? And which signal finds the loops, including counting loops
(`the 321 AD, the 322 AD, …`) and template loops whose slots vary
(`the election of Clement XII, the construction of the Sistine Chapel (1508–1512)`)?

## Evidence

**The guard as built** (branch `experiment/loop-rewind`):
- It is a tokio task between a chat request's engine event stream and its consumer, so
  it runs off the scheduler thread. It covers chat completions only, and the
  reasoning only.
- **Detection.** Every 250 raw reasoning characters it scores the last `window`
  characters of the **normalized** reasoning:
  - fenced code is removed, and a block still open is never scored;
  - inline code is removed;
  - every run of ASCII digits becomes `0`.
  The score is the zlib-9 compress ratio (`miniz_oxide`). A check fires when that
  score is under `ratio` **and** the zlib ratio of the last `window` raw characters is
  under `raw_gate`.
- **The cut.** The cut is the raw start of the window of the last earlier check that
  scored at least `clean`, snapped to a line break and never before the previous cut.
  If the kept text would end inside a ``` block, the cut moves before that block's
  fence.
- **The rewind.** The rewind is a cancel and a resubmit: the original prompt, plus the
  kept tokens, plus a nudge in the model's own voice ("Wait, I'm starting to repeat
  myself. …"). A KV truncation is not possible here: the hybrid model's recurrent
  state cannot go back to a position without a snapshot.
- **What the client sees.** The client receives `[LOOP DETECTED, REWINDING n TOKENS /
  m CHARS]` inside the reasoning. The Playground strikes through the `m` characters
  before the tag (`web/src/ui/rewinds.ts`).
- **Retries and close.** After `retries` nudges, the next fire closes the reasoning
  instead.
- **Config.** All knobs are live config v2 fields (`PATCH /v1/config`), and `loop_guard`
  defaults to `off`:

| field | default |
|---|---|
| `loop_guard` | `off` (`log` and `rewind` are the other modes) |
| `loop_guard_normalize` | on |
| `loop_guard_window` | 1,000 characters |
| `loop_guard_ratio` | 200 (permille) |
| `loop_guard_raw_gate` | 300 (0 = off) |
| `loop_guard_clean` | 400 |
| `loop_guard_margin` | 0 |
| `loop_guard_retries` | 2 |

**Labels.** The 98 Rome runs of the previous finding and of the prototype's two arms
were labelled by reading their tails. 40 have an acute loop. 26 are **harmful**: an
acute loop followed by a forced close at the thinking budget, an EOS inside the
reasoning, or `length` with no answer. Those 26 runs are 23 of the 82 unguarded runs
(28%), and 8 of 32 (25%) in the two unguarded arms at the serving default on ignis. The
rules and per-run labels are in `research/labels.json`.

**Negatives:**
- 4,854 Qwen reasoning parts from the owner's opencode sessions (1,420 long enough to
  score);
- 12,936 Qwen thought parts from qwen-code sessions, 47M characters (5,655 scored);
- the 47 Rome runs with no loop.

All three are agentic coding except the Rome runs.

**Signals, simulated online** (check every 250 raw characters, the open fence stripped):

| signal | harmful caught | acute caught | Rome clean min | opencode floor | qwen-code parts that fire |
|---|---:|---:|---:|---:|---:|
| prototype: raw text, 3,000 chars, < 0.12 | 19/25 | 22/38 | 0.338 | 0.163 | 0 |
| normalized, 1,000 chars, < 0.20 | 24/25 | 34/38 | 0.257 | 0.273 | 7 |
| normalized < 0.20 and raw 1,000 < 0.30 (default) | 24/25 | 33/38 | 0.257 | 0.274 | 2 |
| normalized, 1,500 chars, < 0.20 | 24/25 | 34/38 | 0.295 | 0.229 | 7 |
| normalized + capitalized words folded, < 0.20 | 25/25 | 34/38 | 0.177 (fires) | 0.264 | 7 |

The 25 and 38 count runs after merging identical runs: some arms share up to 26k
characters per seed. Of the 7 qwen-code parts that fire ungated, 5 are numeric tables
(Fibonacci, hex/binary, primes, fp16 encodings), where the digits carry the
information. Their raw ratios are 0.307-0.384, and the gate keeps them out. The other 2
are templated text (an issue-status checklist, quoted log lines) at raw 0.201 and
0.218, and they pass the gate.

Tried and dropped:
- **Novelty against the reasoning's own history** (the window compressed with the
  previous 32K characters as a dictionary) and **the share of repeated lines.** Real
  reasoning re-drafts code it wrote earlier, which puts its floor at 0.015-0.08.
- **Adding the old raw rule** (normalized OR raw 3,000 < 0.12). It catches nothing more.
- **Logit entropy was not built.** It measures how predictable the next token is, and
  counting loops and code are both predictable. Logprobs are not served either
  ([#288](https://github.com/gpillon/ignis/issues/288)).

**Parity.** The Rust scorer agrees with an independent Python version, fed prefix by
prefix, on all 4,874 texts: the normalized text, every check window, the cadence.
miniz and zlib differ by at most 0.015 in the 0.15-0.30 band. Rust's cuts are never
later than Python's (median 96 characters earlier), because Rust maps the window start
to the raw text exactly.

**Prototype arm** (2026-10-13, raw 3,000 / 0.12, the old cut rule, 16 seeds). The guard
acted on 3 of 6 loops, and 3 of 3 left the acute loop.

**Confirm run.** It ran on 2026-10-14 with the defaults above, `f19d6f4`, serving
defaults (`yarn:2`, 524,288, `hq-e8-2b`, DFlash2) and the Playground's Rome body: seeds
101-132 non-streaming, then 133-140 streaming. None of its data was used to choose a
threshold.

| | |
|---|---|
| rewinds | 25, **0 on legitimate reasoning**: verbatim 10, template 7, counting 4, "?"-degenerate 4 |
| runs with an acute loop | 20 of 40; 25 episodes caught, 8 missed |
| cut before the acute onset | 25/25; span median 2,144 characters (1,071 tokens), max 8,828 |
| after the first rewind | 9 of 16 escaped for good; 7 relapsed (2 then closed by the guard) |
| harmful | **4/40 (10%)**: 102 and 124 clear; 101 and 123 escaped the loop but met the budget |
| answered or called a tool | 40/40 |
| streaming (8 runs) | well-formed, `[DONE]` last, one usage chunk; tag and nudge in one reasoning delta |
| continuation | prefilled its whole prompt in 25/25 (4,416-13,867 tokens); TTFT 369-1,744 ms |

- **Harmful, compared.** 4/40 here against 23/82 in all unguarded runs: Fisher
  two-sided p = 0.035. Against the 8/32 in the two arms at the same configuration:
  p = 0.12.
- **Misses** score 0.21-0.31 normalized. Run 124's counting loop over papal reign
  ranges ("683–701? 701–709?") scores 0.09 normalized but 0.33 raw, so the gate
  stopped it.
- **Thin margins.** Three caught episodes fired at raw 0.293-0.295, just under the
  0.30 gate.
- **`usage.completion_tokens`** counts the struck tokens and the tags, so it exceeds
  `max_tokens` in 6 runs.
- **Loops in the answer.** 4 of the 18 runs that answered in content end that answer
  in a verbatim loop. The guard does not watch content.

Raw material is in the worktree's `.scratch/loop-rewind/` (not tracked):
- `NOTES.md`: design, calibration, every run's table;
- `research/`: labels, variant and cut scripts, `REPORT.md`;
- `parity/`;
- `confirm/`: responses, SSE captures, server log, `labels.json`, `summary.json`.

The private corpora stay there.

## Finding

Observed:
- On the tuning data, the normalized score with the gate fires falsely on 2 of the
  5,655 scored qwen-code parts, and on no opencode part and no clean Rome run. It
  catches counting and template loops, which the raw score could not separate from
  code.
- On the held-out run (new seeds, same prompt) it made 25 rewinds, none on legitimate
  reasoning.
- The rewind and nudge work as an exit: every cut landed before the acute onset, and
  9 of 16 runs escaped for good after the first nudge. Harmful runs went from about a
  quarter to 10%, though on different runs.
- What is left is in the 0.21-0.31 band. It holds slow templates and numeric loops
  that compression cannot tell from numeric tables (run 124 sits among them at raw
  0.33).
- Every continuation re-prefills its whole prompt.

Inferred:
- No threshold change helps recall any more. The lowest legitimate texts sit at
  0.257 (Rome) and 0.273 (opencode), and the gate's 0.30 cuts through the numeric-loop
  tail. A new signal would be needed. Entropy was not shown to be that signal; the
  reasoning above argues it shares the confusion.
- The resubmit re-prefills because nothing publishes the cancelled request's prompt
  for reuse. On a 50-100k-token agentic prompt, a rewind would cost a re-prefill of
  that size. This is extrapolated from 1.7 s at 14k tokens, not measured.
- Cuts that keep a degenerate lead-in ("?"-hedging) relapsed more often: 5 of 7,
  against 2 of 9 clean cuts. That is suggestive only at this n.

## Implications

- The guard is usable today opt-in (`loop_guard=rewind`) in the Playground, where
  prompts are small. The tag and strike-through make every rewind visible.
- Before it is a default for agents:
  - a run on an agentic prompt (50k+ tokens of context), which is unmeasured;
  - a decision on `usage.completion_tokens`: forwarded tokens, or what the model
    actually has in context.
  The continuation reusing the original prompt lowers a rewind's cost but is not a
  precondition (next point).
- A rewind costs one re-prefill. A loop left alone can burn the rest of a 14,336-token
  budget. When the detection is right, the rewind is the cheaper of the two in wall
  time, even without reuse.
- Loops in the answer are a separate problem of the same size here. A nudge cannot go
  into content, so the action there has to differ, for example truncating at the cut
  or resampling.

## Limits and unknowns

- One prompt (the Rome enumeration) and one labeller. Thresholds were chosen on the
  same 98 runs. The confirm run is new seeds, not a new prompt.
- The arms share long prefixes per seed and use different seeds, so the comparisons
  are unpaired. 4 harmful against 8 is n too small to give a rate beyond its direction.
- The negatives are agentic coding plus one prose prompt. Other languages and other
  long enumerations are untested.
- The gate's 0.30 was set after seeing 7 negatives.
- A loop inside an open code block is never scored.
- Only chat completions are guarded. `/v1/responses` and image requests are not wired,
  and answers are not watched.
- Streaming was checked on 8 runs.

## Follow-ups

- The continuation reuses the cancelled request's prompt: why no `reuse_source`, and
  what makes it one.
- `usage.completion_tokens` under a rewind.
- Loops in the answer: what action, and whether the same score applies.
- If the guard ships: an ADR on rewinding output a client already received, and the
  `loop_guard` default (the data favours `rewind`).

## Addendum (2026-10-15): the nudge's line breaks and the tokenizer

History repair gives the model back the text it saw. Whether that text re-tokenizes
into the tokens the continuation's prompt carried was measured with
`crates/server/examples/loop_retokenize.rs` on the 27B's tokenizer, over the 48 runs
above (`.scratch/loop-rewind/out/rewind/`, `.scratch/loop-rewind/confirm/out/rewind2/`):
4,165 cuts at the end of a line-break run after the first 1,000 characters, and the 18
real cuts. At each cut, `kept tokens ++ encode(sentence)` against
`encode(kept text + sentence)`. The generated tokens are stood in for by the
tokenizer's own split of the reasoning, which was prefix-stable at every cut.

| Sentence after the kept text | Line-break cuts identical | Real cuts identical |
|---|---|---|
| As written, opening with `\n\n` | 0 / 4,165 | 5 / 18 |
| Without its leading line breaks after a kept line break | 4,165 / 4,165 | 18 / 18 |

The figures are the same for both nudges and for the close.

- As written, the kept text's last newline token and the sentence's opening `\n\n`
  re-tokenize as one: `"." "\n\n" | "\n\n" "Wait"` becomes `"." "\n\n\n\n" "Wait"`.
  The next turn's prompt then leaves the retained state at the cut.
- The 5 real cuts that matched are the ones whose kept text does not end in a line
  break.
- The guard now drops the sentence's leading line breaks when the kept text ends in
  one (`loop_guard::sentence_after`). That applies to the model's view and to what
  the client receives after the tag.
- What the model reads changed with it. A kept text ending in one `\n` used to be
  followed by three line breaks before the nudge, and now by that one. Its effect on
  the continuations was not re-measured on the GPU.
- Not measured: the junction at the sentence's end, between its closing `\n\n` and
  the continuation's first token.
