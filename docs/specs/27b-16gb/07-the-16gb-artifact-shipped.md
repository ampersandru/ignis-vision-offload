# 07 - the 16 GB artifact shipped: the release, and its card

GitHub: #346 (master #334). Added 2026-10-24, when #339 closed. The work is what is left between first light and a user
on a 16 GB card.

ADRs: 0060 (the body and its gates), 0064 (the catalog entry, its card class and the alias).

## Problem Statement

- #338 published the artifact on HF (`gpillon/Qwen3.8-27B-trellis-3bpw-ignis`). #339 made ignis serve it on the 5090
  under a 16 GiB budget. Both are on local main, which is not pushed.
- The released ignis, v0.7.3, has neither the trellis body nor the `qwen3.8-27b-16g` entry, so it cannot load the file.
- The card's "How to run" therefore says "there is no command to run it yet".
- The card's measurements are M1's weights-only PyTorch numbers, not ignis's.
- Its links to `tools/27b-16gb-converter` and to the 2026-10-23 finding resolve only once main is pushed.
- The HF cards are the project's funnel (the 27B card has ~5.3K downloads). A command there that does not run is worse
  than none.

## Solution

1. **Wait for the gates.** #340's gates decide whether this body ships. If it fails them, ADR 0060 falls back to
   3.25 bpw or our own conversion, which means another artifact and another card.
   - A preview release before #340 is the owner's call. If they choose it, the notes and the card say "preview,
     quality gates pending".
2. **Push main, then cut the release** by the repo's release procedure: the version in its four files, the
   changelog, the tag.
   - The notes name the 16 GB card class.
   - They state what was measured and how: on the 5090 under a 16 GiB budget (05a, 05b).
   - They give the speed on a 16 GB card as 05b's estimate.
3. **Rewrite the card from the release:**
   - "How to run": the release's command. On a 16 GB card, `profile auto` picks `rtx50-16gb` and the default id
     `qwen3.8-27b` resolves to `qwen3.8-27b-16g` (ADR 0064's alias), so it can be the 32 GB card's command.
   - Measurements: ignis's own (05a's plan, context, lanes and DFlash2; 05b's gates) beside M1's, each labelled.
   - Caveats:
     - validated on a 5090 under a 16 GiB budget, not on a 16 GB card;
     - the prefill as it stands relative to #345.

## Acceptance

1. Main is pushed, and a release carries the trellis body, the `qwen3.8-27b-16g` entry and `rtx50-16gb`. Its notes say
   what was measured and how, as above.
2. The card's command, run against that release on the 5090 under a 16 GiB budget (`--vram-budget-bytes
   17179869184`, `--profile rtx50-16gb`), downloads the artifact from HF at the catalog's pinned revision and serves
   the canaries.
3. The HF card:
   - gives that command;
   - adds ignis's measurements beside the weights-only ones, labelled;
   - states the caveats above;
   - links only to pages that resolve.
   The owner approves the text before the HF commit.
4. The project README and the user guide name the 16 GB card class with the release's numbers.

## Out of Scope

- A real 16 GB card, its speed gate, and per-card profiles: phase 2 (`phase2-notes.md`).
- #345's prefill work: the card states the prefill as the release has it.
