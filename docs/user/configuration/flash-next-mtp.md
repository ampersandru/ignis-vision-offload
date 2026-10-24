---
title: "Flash-Next speculation (MTP)"
description: "Flash-Next's multi-token-prediction head: when it pays, and the row budget."
sidebar:
  order: 5
---

Qwen3.8-Flash-Next can draft with its own multi-token-prediction head
(`--spec-backend mtp`, `make MODEL=flash-next SPEC=mtp`). It is **off by default**, and
on a 5090 that is the better setting for most uses; the `rtxpro6000` profile
turns it on (below). Before turning it on:

- **It needs its companion container** beside the artifact
  (`qwen3_8_flash_next_mtp_3p0-v2.ninfer`, ~1 GB). `--spec-backend mtp` without it
  refuses the start.
- **It keeps the text.** Greedy output is the same as without it, up to
  near-ties, and sampling keeps its distribution. Past the default row budget
  (`--spec-draft-rows` above 8) a near-tie can go the other way more often:
  the output is numerically equivalent, not token for token (below).
- **It costs ~1.1 GB of VRAM, taken from the expert cache, not from the
  context.** The KV pool, and so `--model-max-context`, is unchanged. The cost is
  reserved at load and stays the same whatever the number of drafts: a
  smaller `--spec-draft-tokens` or `--spec-draft-rows` does not give it back.
- **It pays off at one active request, and costs at two or three.** On an
  RTX 5090 (PCIe Gen 3, 2026-10-07): one lane runs 1.02-1.30x faster (most on
  long prose); two and three lanes run 7-15% slower with the default row
  budget, and 1-2% slower with `--spec-draft-rows 3`, which drafts at one lane
  only. Decode there is bound by the experts it copies in over PCIe, and a
  verify round copies more of them
  ([finding](../../findings/2026-10-07-flash-next-mtp-speculation-is-pcie-bound.md)).
  These figures leave out the 1.1 GB the head takes from the expert cache,
  so the served cost at several lanes is a little higher.
- **Use it** when one user or one agent works at a time: `--spec-backend mtp
  --spec-draft-rows 3`, or serve a single lane outright (`make MODEL=flash-next
  LANES=1 SPEC=mtp`), which also gives the expert cache the ~1 GiB the shared
  pool holds beyond one lane's context. **Leave it off** when several agents decode together.
- A card whose cache holds nearly every expert (96 GB) copies few over PCIe,
  and there MTP pays wherever it drafts. The `rtxpro6000` profile turns it on
  with 3 draft tokens and the default row budget: a verify round carries at
  most 8 rows (lanes × (drafts + 1)), so it drafts 3 at one or two active
  lanes, 1 at three or four, and none from five on. `--spec-draft-rows 16`
  (up to 32) lets five to eight lanes draft too. Past 8 rows the verify
  round's linears run on tensor cores, whose sums run in another order, so
  its output may differ from speculation off's where two tokens are nearly
  tied: the round is held to speculation off's logits within a tolerance,
  not to its text token for token
  ([ADR 0056](../../adr/0056-speculation-past-eight-verify-rows-is-numerically-equivalent.md),
  [finding](../../findings/2026-10-18-flash-next-verify-rows-past-8.md)). On the
  RTX PRO 6000 it does not pay: a verify round past 8 rows takes 64-80 ms against
  ~19.5 ms at 8, so every wider budget runs 0.42-0.84x the default's 8
  ([finding](../../findings/2026-10-18-verify-rows-on-the-rtx-pro-6000.md)), and
  the profile keeps 8. On an RTX PRO 6000 Server Edition
  ([finding](../../findings/2026-10-18-rtxpro6000-profile-measured.md)):
  - with every lane decoding at once (greedy, the 1 GiB n-gram default), one
    lane runs 1.85x speculation off on code and 1.44x on prose, two lanes
    1.24-1.53x, four 1.12-1.19x and eight 0.97-0.98x (no drafts there, only
    the head's fixed cost);
  - sampled, one lane runs 1.9x on code and 1.3x on prose;
  - with simulated agents, whose lanes wait on tools and prefills between
    turns, four agents generate 1.21x the tokens a second and eight 1.05x.
  3 drafts tie with 4 over code and prose together and beat 2 and 5: more
  drafts help code and cost prose. A file's
  `spec: { qwen38flashnext: { backend: off } }` turns it off alone;
  `--spec-backend off` turns off the 27B's DFlash2 with it.
