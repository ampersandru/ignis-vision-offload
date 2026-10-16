# A reasoning loop persists and a legitimate enumeration ends: the loop guard acts on four checks in a row

- Kind: experiment
- Status: current
- Observed: 2026-10-16
- Last verified: 2026-10-16
- Scope: server / reasoning-loop guard trigger and cut (`crates/server/src/loop_guard.rs`, config v2 `model.loop_guard*`); sampled reasoning on qwen3.8-27b, the Playground's Rome prompt and the owner's opencode and qwen-code agentic sessions
- Related: [ADR 0053](../adr/0053-a-loop-is-acted-on-once-it-persists.md), [ADR 0052](../adr/0052-a-reasoning-loop-is-taken-back-and-nudged.md), [Reasoning-loop guard](2026-10-14-reasoning-loop-guard.md), [Reasoning loops are the model's](2026-10-12-reasoning-loops-are-the-models.md)
- Superseded by: none

## Question

ADR 0052's guard rewinds at the **first** check whose normalized window compresses under 0.20.
On agentic reasoning it fired only on legitimate work. Does anything separate a loop from a
legitimate enumeration better than the depth of the score, and what does the cut do when the
reasoning opens in one?

## Evidence

**Data.** All offline, CPU only. The private texts stay in `.scratch/nudge/` of the worktree that
produced this finding.
- The 98 labelled Rome runs of the 2026-10-14 finding. After merging aliases:
  - 25 harmful;
  - 38 with an acute loop;
  - 47 with no loop;
  - 9 micro.
- 12,936 unique qwen-code reasoning parts (5,654 long enough to score).
- 4,781 unique opencode Qwen reasoning parts (1,400 scored).

**Scorers.**
- A Python port of the guard's online watch (normalization, cadence, gate, cut) matched the
  step-0 simulation check for check on 400 texts.
- `examples/loop_score.rs` then scored every text with the guard's own Rust code (miniz). Same
  check count on all 7,878 texts; the ratios differ from zlib's by at most 0.022.
- Every rule below was applied to the Rust checks. Today's rule (0.20, first check) reproduces
  the 2026-10-14 numbers: 24/25 harmful, 33/38 acute, 0 Rome no-loop fires, 2 agentic fires.

**What the agentic fires were.** Both were legitimate enumerations, read by eye:
- a per-ticket checklist, with three quarters of the plan still ahead;
- a server log the user had pasted, transcribed line by line. The reasoning opens with it, so
  no check before the fire scored clean. The cut went to the floor (0): the rewind would have
  dropped the whole reasoning and restarted the model from the bare nudge.

**Durations.** The longest run of consecutive low checks per text (normalized < T, raw < 0.30):
- No legitimate text in any corpus has more than 6 at any T ≤ 0.30. The 6 is the log
  transcription; Rome's clean runs reach at most 2.
- 23 of the 25 harmful Rome loops have 9 or more at T = 0.30. 18 are still low when the text
  ends: the loop never left.
- The exceptions are loops that hover around the threshold and break into short runs: a slow
  template (4 at 0.30, 0 at 0.20) and a "Dark Ages" template (1 at 0.20).
- Loops that end on their own are as short as legitimate lists: the 11 acute but not harmful
  Rome loops have a median of 5 checks.

**Persistence alone does not remove the agentic fires.** At T = 0.30, K = 4 (K = checks in a row
needed to act), 2 legitimate agentic texts still fire: the log transcription (6 checks), and a
Fibonacci table computed by hand (5).

**The floor is a separate defect.**
- A cut that keeps nothing:
  - is 0 of the 35 simulated first rewinds on Rome;
  - is 2 of the 25 real rewinds of the confirm run, both second attempts whose floor is the end
    of the first nudge, which keep the earlier text anyway;
  - is the log transcription on the agentic side.
- K cannot move such a cut, only remove the fire.
- Rule tested: **a rewind that would keep no reasoning is only logged**, and that run of low
  checks is held until a check that is not a loop.

**Grid on the Rust checks.** T × raw gate G × K × that rule:
- At G = 0.30, K = 4 with the rule:
  - T between 0.265 and 0.30 catches 25/25 harmful and 34/38 acute;
  - no Rome no-loop or micro fire;
  - every cut before the labelled onset.
- At T = 0.265 and 0.27 no agentic text fires at all. At 0.28 and 0.30 the Fibonacci table
  fires (normalized 0.274).
- At T = 0.26 a breathing loop (a short dip, a partial escape, a relapse) moves its cut into the
  episode.
- K = 3 adds agentic fires; K = 5 loses the slow template and moves 3 cuts into their episode.
- Lowering the raw gate to 0.28 loses 2 harmful runs.
- Fires come a median 500 characters later than today's rule (from 4,250 earlier to 750 later).
  The struck tokens are not charged to the continuation's budget (`rewind()` subtracts the kept
  text and the nudge only).

**Margins at K = 4, G = 0.30.**
- The highest threshold any harmful loop needs is 0.257 (the slow template).
- The lowest at which a legitimate agentic text not taken out by the floor rule fires is 0.274
  (the Fibonacci table).
- 0.265 sits 0.008 and 0.009 from each.

**Agentic loops exist, and they are short.**
- No one had read the 0.20-0.30 band before. qwen-code holds 4 true loop episodes in 5,654
  scored parts:
  - short indecision loops ("let me reconsider…" cycling);
  - 0.26-0.29 normalized, at most 4 checks;
  - all ended on their own.
- Two longer indecision tails, a decision sentence repeated between reconsiderations, never go
  under 0.40: compression does not see them.
- No agentic part shows a persistent loop of the Rome kind.

**Literature.**
- Loops self-reinforce: the more a sentence repeats, the likelier its continuation (Xu et al.
  2022, *Learning to Break the Loop*).
- A reasoning loop, once entered after an impasse, "persists as an inescapable cycle" (Duan et
  al. 2026, arXiv 2601.05693, abstract).
- The Qwen3.8-27B card recommends, in thinking mode, temperature 1.0, top_p 0.95, top_k 20,
  presence_penalty 0.0: the Playground's settings. So the measured loop rate is at the
  producer's recommendation.

## Finding

Observed:
- On these corpora, duration separates harmful loops from legitimate enumerations where depth
  does not. Legitimate text stays at most 6 consecutive low checks; harmful loops mostly 9 and
  more, and most never leave.
- Today's trigger fires on a 17-item list because 17 items fill the 1,000-character window.
- A rewind with no clean window since the floor drops the whole reasoning. It happens on text
  that opens in an enumeration, which agentic reasoning does and the Rome runs do not.
- T 0.265, G 0.30, K 4, with "a rewind keeps something", beats today's rule on every column:
  25/25 harmful and 34/38 acute against 24 and 33, and 0 agentic fires against 2.

Inferred:
- The first nudge's "move on with what I already have" was not the agentic risk itself. The
  trigger put it after legitimate work. With the new trigger, the nudge acts on Rome only after
  a loop has persisted, where an exit is the measured remedy (9 of 16 escaped for good after one).

## Implications

- ADR 0053 adopts `loop_guard_persist` 4, `loop_guard_ratio` 265 and `loop_guard_min_kept` 1.
- The nudge wording is a separate, measurable question: `loop_guard_nudge` is a config field.
- A loop guard on compression cannot see indecision loops whose text varies between repeats. The
  only harmful agentic loop observed (the 2026-10-12 incident's deferral loop) is of that kind.

## Limits and unknowns

- The Rome numbers are in-sample: one prompt, one labeller, the grid chosen on them. 0.265 is
  tuned to 3 decimals between two texts 0.017 apart.
- Harmful durations are censored by the forced close. Only first rewinds were simulated.
- The agentic loop or legitimate calls are one reader's.
- Recall on new Rome seeds: see the held-out run below.

## Held-out run

(Filled after the GPU run.)

## Follow-ups

- A detector for indecision loops (a decision sentence repeated between reconsiderations).
- A/B of the first nudge on Rome, now that it is a config field.
- `.scratch` research tooling: `common.py`'s run path points at a removed worktree and silently
  drops two arms.
