# 03 — `GET /v1/models` names every model a request can be served by

GitHub: #343. Builds on spec 01 (runtime model switch, its implicit switch)
and spec 02 (idle unload), and on spec playground/05 (`GET
/v1/models/catalog`, switch by id).

## Problem

`GET /v1/models` lists one entry, the loaded model (`list_models`,
`crates/server/src/api.rs`). That was the whole truth before spec 01; since
the implicit switch it is not: a request naming any known model, or any
catalog entry whose artifact is on disk, is switched to and served. A front
end that fills its model picker from `GET /v1/models` — Open WebUI, Hermes,
any OpenAI client — offers only the loaded model, and its users switch by
hand-crafting requests. Reported by a user running both.

## Solution

`GET /v1/models` lists the **servable models**: the active model first,
then every model a request may name and be served by now without a
download — exactly the ids the implicit switch resolves
(`Switcher::known`), filtered to those whose artifact is a file.
`switch.allow_implicit` off: the active model alone, today's answer.

- **One source of truth.** The list is computed at each `GET` by the same
  lookup the implicit switch uses; an id is listed if and only if naming it
  in a request would switch to an artifact that is there. A model fetched
  into `download.path` while the server runs appears on the next `GET`
  (the README's "switchable at once"); a file removed disappears.
- **No snapshot, no reload endpoint, no cache.** Rejected: reading the list
  at start with a `POST /v1/models/reload` to read it again. It breaks the
  "switchable at once" contract, makes the list and the switch two sources
  of truth that diverge (a listed model that 404s, a servable one not
  listed), and adds an endpoint to save about five `stat` calls a `GET`.
  If a measurement ever shows the cost on a slow filesystem, a short-TTL
  cache inside the route may follow, invisible to the API.
- **Never blocks the runtime.** The filesystem work runs on
  `spawn_blocking`, as `get_models_catalog` does; the route stats files and
  never opens an artifact.
- **`data[0]` stays the active model**, in every status. The Playground,
  the bench client, `scripts/agent-swarm.py`, the live-server test helper
  and about forty test assertions read `data[0]` as "the loaded model";
  they keep working. A new per-entry `active` says it explicitly, so new
  readers need not rely on position.

## Seams

- **The route** (`crates/server/src/api.rs`, `list_models`): keeps reading
  `status` before `active` (the ordering comment stays true), builds the
  first entry as today plus `active: true`, then, when a switcher exists,
  appends the servable rest. All of it inside `spawn_blocking`.
- **The servable rest** — a function beside the switcher
  (`crates/server/src/model_switch.rs`), so the rule lives next to
  `Switcher::known`: candidates are the catalog's entry ids in catalog
  order, then `Switcher::known_models()`' ids not in the catalog, in id
  order; drop the active id and duplicates; keep an id when
  `Switcher::known(id)` returns a path that `is_file()` (`known` checks the
  file for a catalog entry, not for a known model). `known` already folds
  `switch.allow_implicit` (an empty known table, the catalog not consulted),
  so the flag needs no second check; a live `PATCH` of
  `switch.allow_implicit` or `switch.known_models` shows on the next `GET`.
- **The entry** (`ModelInfo`): `id`, `object: "model"`, `owned_by:
  "ignis"`, new `active: bool`; `kind` becomes optional — always on the
  first entry, on another only when `ModelFamily::of_model_id(id)` names a
  family (the built-in default ids), else absent: no header read on this
  route; `max_model_len` on the first entry only, as today (it is known
  only once a model is loaded). No path.
- **Envelope**: `status`, `switching`, `reason`, `last_switch` unchanged;
  they speak of `data[0]`.
- **Playground** (`web/src/api/model.ts`, `readModel`): read the entry with
  `active: true`, falling back to `data[0]` for a server without the field.
  The dev mock (`web/mock.ts`, `web/mockServer.ts`, `sim.models`) lists the
  way the server does.
- **Bench** (`crates/bench/src/client.rs`, `HttpEndpoint::list_models`):
  the doc's "v1: a single model" goes; it returns the ids in order, active
  first, which every caller's `models[0]` already assumes.
  `crates/server/tests/ttft_cell_gpu.rs` asserts the first id, not the
  whole list.

## Acceptance

1. Default config (`switch.allow_implicit` on): `GET /v1/models` lists the
   active model, then every catalog entry whose artifact is on disk
   (catalog order), then every known model whose artifact is a file and
   that is not a catalog entry (id order); each id once, the active one not
   repeated.
2. Not listed: a catalog entry `absent` or `partial`; a known model whose
   artifact is not a file; an alias or any id `Switcher::known` does not
   resolve. With `switch.allow_implicit` false only the active entry is
   listed.
3. Listed if and only if servable: for every id in the catalog, in
   `switch.known_models` and a few unknown ones, the id is listed exactly
   when `Switcher::known(id)` returns a path that is a file. A chat request
   naming a listed text model is served (after an implicit switch when it
   is not loaded); one naming an unlisted id is `404 model_not_found`.
4. A catalog artifact created under `download.path` on a running server is
   listed by the next `GET`, and disappears once removed; no restart, no
   other request in between.
5. A live `PATCH /v1/config` of `switch.allow_implicit` (on → off → on) or
   `switch.known_models` changes the next `GET`'s list accordingly.
6. `data[0]` is the active model with `active: true` in every status —
   `serving`, `switching` (still the old model until the new one serves),
   `loading`, `unloaded`, `failed` — and every other entry has `active:
   false`. During a switch the rest of the list is still listed (it is not
   filtered by `switchable`).
7. Entry fields: `kind` and `max_model_len` on the first entry as today;
   on another entry `kind` only when `ModelFamily::of_model_id(id)` names a
   family, `max_model_len` never; no entry carries a path.
8. The route runs its filesystem work on `spawn_blocking` and opens no
   artifact (a test loader that counts `kind_of` calls sees none from a
   `GET`).
9. Polling `GET /v1/models` with servable models listed still neither
   resets the idle clock nor loads an unloaded model (spec 02 AC 5); a
   missing API key is still `401`, the warm-up gate still `503
   server_not_ready`.
10. With a text model loaded and the image model on disk, the image model
    is listed; a chat request naming it is `409 model_kind_mismatch` and
    starts no switch.
11. Readers of `data[0]` keep working: the Playground reads the `active`
    entry (fallback `data[0]`) and its header and context bar are
    unchanged with three models listed; the web mock lists like the
    server; the bench's `list_models` returns the active id first; every
    existing test passes, `ttft_cell_gpu.rs` asserting the first id.
12. OpenAPI: `list_models`' summary and description say what is listed
    and that `data[0]` is the active model; `ModelInfo` documents `active`
    and the optional `kind`.
13. Docs: `docs/user/README.md` — the API table row for `/v1/models`, the
    `--model-id` and `--switch-allow-implicit` rows, and the paragraph on
    two known models ping-ponging (opencode's `small_model`), extended to a
    front end that picks ids from `/v1/models` (Open WebUI's task model,
    Hermes): two different ids reload the server on alternate requests,
    13–33 s each; point both at the same id or start with
    `--switch-allow-implicit false`. `CONTEXT.md`: **Servable model**
    (the active model plus every model the implicit switch resolves to an
    artifact on disk; what `GET /v1/models` lists).
14. Tests run on the mock loader, no GPU; `cargo test` workspace-wide and,
    in `web/`, the typecheck and `npm test` pass.

## Defaults taken (owner may revisit)

- The image model is listed like any other: OpenAI lists image models too,
  and a chat request naming it is refused `409` without a switch (AC 10).
- `kind` is left out rather than read from the artifact's header when the
  id is not a family's own (`qwen3.8-27b-abliterated`, an operator's id):
  the route stays at one `stat` per candidate.
- Aliases (ADR 0064) are not listed: the implicit switch does not resolve
  them today; when it does, AC 2–3 decide whether they are.
