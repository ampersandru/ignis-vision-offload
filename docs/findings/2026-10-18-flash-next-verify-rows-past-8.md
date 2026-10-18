# A Flash-Next verify round past 8 rows agrees with itself but drifts from speculation off past the near-tie rule

- Kind: experiment
- Status: current
- Observed: 2026-10-18
- Last verified: 2026-10-18
- Scope: kernel / the Flash-Next verify round's row budget (`verify::kMaxRows`, `--spec-draft-rows`), MTP speculation's losslessness seam (`crates/core/tests/flash_next_speculative_gpu.rs`)
- Related: https://github.com/gpillon/ignis/issues/329 (item 5), [spec flash-next/07](../specs/flash-next/07-mtp-speculation.md), [the rtxpro6000 profile measured](2026-10-18-rtxpro6000-profile-measured.md), [batched decode width drift](2026-09-14-batched-decode-width-drift.md), `.scratch/verify-rows/` (branch `verify-rows`)
- Superseded by: none

## Question

The verify round carried at most 8 rows (lanes x (drafts + 1)), so MTP drafted
nothing from five lanes on. With the bound raised to 32 (`--spec-draft-rows`
up to 32, default 8 unchanged), does a 16-row round keep MTP lossless as the
repository's seam defines it -- bit for bit across drafters at one round
shape, and within the near-tie rule of speculation off -- and what does it
cost?

## Evidence

**Setup.** RTX 5090 (PCIe Gen 3 x16), branch `verify-rows` (ff02fecf and the
test commit after it), the Flash-Next artifact and its MTP companion
(`qwen3_8_flash_next_mtp_3p0-v2`), 3 draft tokens, 8 decode lanes, 8,192
tokens of context, greedy, 40 tokens a lane; prompts: the four G1 prompts and
test2048 windows 3, 9, 17 and 33 (40 tokens each), lanes taking the first
4, 5 or 8. At 16 rows the windows by width are 3, 3, 3, 3, 2, 1, 1, 1, so
the cells are rounds of 16 (4 lanes), 15 (5) and 16 rows (8).

**What changes past 8 rows.** The MoE decode route keeps its one launch of
8 tokens: a wider round runs it in launches of 8 (`fn_moe_experts_decode`).
Every other route switch at 8 is crossed: the FP8 linear's GEMV (1..8 tokens)
becomes its tensor-core route, the BF16 linear's GEMV its WMMA route, the
hyper-connection mix's decode route its prefill route, the GDN and QSA
grouped GEMVs separate linears, the shared expert's GEMV its tensor-core
route. The sparse attention's split count already varies with the rows below
8 (64 splits at up to 4 rows, 32 at 8, 16 at 16).

**The wide shapes agree with themselves.**
- `flash_next_moe_rows` (CTest): 9, 16, 24 and 32 rows in launches of 8 on
  the register kernel, each accumulator row the bits of that row's own
  one-token launch; a 24-row call replayed from a graph, the eager bits.
- Reject (every draft refused) against none (no drafts) at 4, 5 and 8
  lanes, both KV formats: the same text and the same probe logits bit for bit.
- The oracle (none's own text as drafts, every column of every lane
  committed) in BF16: 120/120, 130/130 and 160/160 drafts accepted, the
  same text and probe bits as none.

**Head against none is not a seam once lanes finish apart.** Replaying the
head's per-lane commit schedule (5 lanes) with a host drafter reproduced the
head's text and probe bits exactly. Non-uniform commit counts on equal width
schedules (two lanes alternating 4 and 1 committed columns, at 8 rows, and
four lanes at 16) left none's state bit for bit. When one lane finishes
rounds before another, the rest run their last rounds narrower -- another
round shape -- and their probe differs, at 8 rows as well (two lanes at
window 3, four at window 1: one lane's probe differs, the text does not).
The existing BF16 head == none assertion
(`the_mtp_head_drafts_and_the_text_is_kept`) holds by construction at one
lane only, and held at three by chance.

**Speculation on against off** (`a_sixteen_row_budget_keeps_...`; a
divergence passes when the prefill route's logits after the agreed text put
the two picks within one BF16 ulp, or side with spec-on):

| budget | format | lanes | divergences (lane @ token: prefill gap off - on) | refused |
|---:|---|---:|---|---|
| 8 | BF16 | 4, 5, 8 | none | -- |
| 8 | hq-e8-2b | 2, 3, 4 | none (oracle against none: none) | -- |
| 16 | BF16 | 4, 5 | none | -- |
| 16 | BF16 | 8 | 4@38: -1.19; 5@7: 0.125 (1 ulp); **7@3: 0.375 (6 ulps)** | 1 |
| 16 | hq-e8-2b | 4 | **0@26: 1.31 (21 ulps)**; 3@29: 0.125 | 1 |
| 16 | hq-e8-2b | 5 | 4@17: 0.125 | -- |
| 16 | hq-e8-2b | 8 | 3@20: 0; 4@7: 0; 5@7: -0.125 | -- |
| 16 | hq-e8-2b | 8 | oracle against none: 3@29, 4@30, 6@8 pass; **5@32: 0.5 (4 ulps)** | 1 |

At 8 rows hq-e8-2b, 5 and 8 lanes run no drafts (window 0), so those cells
equal speculation off trivially. The BF16 5-lane divergence at lane 4 token
38 shows speculation off itself 1.19 logits (9.5 ulps) from the prefill
route's pick there, passing only because the rule takes the prefill route's
side.

**VRAM** (plan lines, `flash_next_bind` at 131,072 tokens hq-e8-2b, 4 lanes x 3
drafts, budget 8 -> 16): verify round 13.2 -> 24.0 MB, sampling 4.6 -> 8.6 MB
(the head's logits, one row a column), decode graph 40.8 -> 75.3 MB (the QSA
listed hq rows of every column); workspace and activations unchanged, the
MoE workspace's decode regions kept at 8 tokens. A budget no round reaches
reserves nothing (3 lanes at 12 rows and at 32 plan alike). The three lines
grow linearly in the rows (1.35, 0.5 and 4.3 MB a row here), so a round of
32 rows would reserve about 150 MB over the default's -- extrapolated from
the plan's linear terms, not measured.

**Cost on the 5090** (`flash_next_mtp_bench mtp --draft-tokens 3 --lanes 8`,
hq-e8-2b, 1,536-token prompts, 128 tokens, 16 GB expert cache; decode is
bound by the experts copied over PCIe):

| lanes | rows 8: tok/s, ms/round, GB moved/round | rows 16: tok/s, ms/round, GB moved/round | x |
|---:|---|---|---:|
| 4 | 62.3, 98.8, 1.16 (window 1) | 46.4, 165.0, 1.63 (window 3) | 0.74 |
| 8 | 61.2, 130.7, 1.55 (no drafts) | 49.2, 266.8, 2.76 (window 1) | 0.80 |

## Finding

Past 8 rows the verify round agrees with itself bit for bit -- across
drafters, and across commit counts on one width schedule, the MoE's launches
of 8 included -- so the commit and the wider records are right. But its
tensor-core routes move its logits further from speculation off's decode
routes than any round of 8 rows or fewer does: at 16 rows three checks out of
the cells above exceed the near-tie rule (4 to 21 BF16 ulps), where the
default's 8 rows show no divergence at all on the same prompts. A budget past
8 is therefore not held lossless by the repository's seam yet. On the 5090 it
is also slower (0.74-0.80x), as a PCIe-bound card predicts.

## Implications

The knob ships with its default unchanged; a load that names no budget plans
and runs as before. Before a budget past 8 is recommended (on the RTX PRO
6000, where decode is not PCIe-bound), one of these has to be chosen:

1. **Launches of 8 rows everywhere a route switches at 8**: the linears
   (`fn_linear`, under a decode-phase flag, since a prefill chunk of 9..32
   tokens must keep its tensor-core route), the shared expert, the
   hyper-connection decode route with its pending inject, the grouped GEMVs.
   Each row then computes as in a round of 8; the dense weights are read once
   per 8 rows (about 2.5 GB more a 16-row round, ~1.4 ms at 1.8 TB/s). The
   attention's split count still varies.
2. **GEMV kernels wider than 8 tokens** (16 or 32), each token's sums in the
   8-token kernel's order, proven bit for bit by CTest, taken only by
   decode-phase calls: the FP8 single, SwiGLU and grouped GEMVs, the BF16
   GEMV, the hyper-connection decode route's accumulators and shared
   activations. The weights read once a round; a kernel redesign.
3. **Accept the drift** and hold budgets past 8 to a looser rule: its size
   is of the class of speculation off's own disagreement with the prefill
   route (9.5 ulps above).

## Limits and unknowns

- One prompt set, 40 tokens a lane, 4/5/8 lanes, one card. Three refusals
  over about 1,400 compared tokens (the head's and the oracle's); their rate
  on longer texts is unmeasured.
- Self-consistency cannot exclude a deterministic error specific to one
  wide shape (for example 8 lanes x 2 columns); the 4-lane x 16-row cell
  being clean in BF16 argues against one. The discriminator is a per-round
  logit comparison: the first column of a 16-row verify pass against a plain
  decode round from the same state.
- The RTX PRO 6000 numbers (MTP 3 at 4 lanes 1.12-1.19x, at 8 lanes 0.97x)
  predate this change; what a 16-row budget pays there is unmeasured.

## Follow-ups

- The owner chooses among the three options above.
- After that: on the RTX PRO 6000, `flash_next_mtp_bench mtp --draft-tokens 3
  --lanes 8 --draft-rows 8|16|24`, then the profile campaign's 4- and 8-lane
  cells at `spec.draft_rows` 16 against 8.
