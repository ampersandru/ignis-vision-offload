# ADR 0064 — a catalog entry states the smallest card class it loads on; aliases resolve within a family; a switch refuses before teardown

## Status

Accepted (2026-10-21, after the owner's review: "accetto tutto"). Proposed the same day. Owner decisions in the 16 GB grill:
- a distinct served id for the 16 GB artifact, plus an alias;
- Flash-Next on 16 GB in phase 2, the image model after it;
- a switch refuses before it tears anything down.

Related:
- [ADR 0047](0047-a-catalog-the-operator-can-extend.md): a catalog the operator can extend;
- [ADR 0060](0060-the-27b-has-a-trellis-body-for-small-cards.md): the 16 GB body;
- [ADR 0063](0063-the-profile-follows-the-card-class.md): the profile follows the card class;
- spec `model-switch/01`.

## Context

- **Served ids are unique.** The catalog refuses two entries with one served id, and an operator entry never reuses a
  built-in one.
- **Clients name `qwen3.8-27b`:** the agents, opencode, the Playground.
- **A request naming a model that is not loaded switches to it implicitly**, if its artifact is on disk.
- **The VRAM check comes after the old model is torn down.** A target that does not fit fails, the old model is
  reloaded, and the request gets `503 model_switch_failed`. On a 16 GB card every request naming Flash-Next or the
  image model would cost a full reload of the 27B. The minimums:
  - Flash-Next needs a 12 GiB expert-cache floor;
  - the image model needs ~17.6 GiB of budget.
- **The 32 GB 27B entry would switch to an artifact that does not fit.**

## Decision

1. **Every catalog entry states the smallest card class it loads on**:
   - the 27B NVFP4: 32 GB;
   - the 27B trellis: 16 GB;
   - Flash-Next: 32 GB, 16 GB in phase 2. This is a policy, not a hardware limit: with
     `vram.allow_expert_cache_below_floor` it starts below its 12 GiB expert-cache floor, with a warning. Phase 1
     does not offer it on 16 GB;
   - the image model: 32 GB.
   An operator entry may state its own.
2. **The 16 GB artifact has its own served id** (proposed `qwen3.8-27b-16g`) **and may answer to its family's
   default id as an alias.**
   - On a card class below the default entry's minimum, a request naming `qwen3.8-27b` is served by the family's
     entry for that class.
   - An alias only ever resolves within one model family, and only to an entry the card class can load. It never
     causes a switch to one it cannot.
3. **A switch, explicit or implicit, and a download check the card class first.** One the card cannot load is refused
   before anything is torn down or fetched, with a message naming the entry's class.
   - **`GET /v1/models` lists such entries as unavailable on this card.**
   - The VRAM plan's own refusal still guards the case the class check cannot see: a budget too small on a card of the
     right class.

## Considered options

- **One served id for both artifacts of the 27B.** Rejected: the catalog's uniqueness rule and the served id's meaning
  ("it tells artifacts of one family apart") both say no.
- **A distinct id and nothing else.** Rejected: every client configured with `qwen3.8-27b` would break on a 16 GB card.
- **Keep refusing after teardown.** Rejected: it costs a reload per request, and the class is known before anything
  moves.

## Consequences

- The catalog entry, the switch, `model download` and `GET /v1/models` gain the card class. The glossary's **Catalog
  entry** and **Served id** say so.
- Phase 2 changes Flash-Next's entry to 16 GB once it runs there. The way in is the owner's (2026-10-21): the
  `rtx50-16gb` profile lowers the expert-cache floor.
  - That floor is a `pub const` today (`EXPERT_CACHE_FLOOR_BYTES`, 12 GiB). Under the owner's rule that a refusal
    threshold is a config field, it becomes one: default 12 GiB, the profile's own value estimated first and tuned on
    hardware in phase 2.
  - First estimate from the plan lines: ~5 GiB. Flash-Next's fixed lines at 1 lane and a 262K pool are ~9.65 GB; that
    leaves ~5.4-7.1 GB on a 16 GB card, depending on what else holds it.
