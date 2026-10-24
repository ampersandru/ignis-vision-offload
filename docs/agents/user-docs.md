# User documentation

How a change reaches the people who run ignis. One rule governs everything
below: every user-visible change ships with its user-guide update in the same
commit, and a task is not complete until `docs/user/` describes what the code
now does — every user-visible behaviour the diff changes, found where a user
would look for it.

## User-visible

A change is user-visible when someone running, configuring or calling the
server can observe it without reading the code:

- **Configuration**: a field added, renamed, removed or re-defaulted; a
  profile's values; a subcommand (`config`, `model`, `help`); a start that is
  now refused or now accepted.
- **HTTP surface**: a `/v1` route, a request or response field, a status code,
  a header, an error body, a deadline or limit.
- **Models**: a catalog entry, an alias, a card class, what a download fetches
  and from where.
- **Playground**: a view, a control, what a screen shows or does.
- **Telemetry**: a metric, a log line an operator reads (`ignis.config.source`,
  `ignis.runtime.vram_plan`), a gauge's meaning.
- **Running it**: a `make` target, the container image, the release archives,
  a supported card or a hardware limit (VRAM, context, lanes).
- **Behaviour with no new knob**: a default, a refusal, a fallback, a limit
  that moved.

Internal work — a refactor, tests, CI, a kernel swap with no observable
effect — leaves the guide as it is. A measured speed change updates a figure
only where the guide or the root README quotes one.

## Where it goes

- **`docs/user/README.md`** is the user reference. A behaviour lives in one
  section, the one a user would open for it; every other mention links there.
- **`docs/user/<topic>.md`** holds a topic that outgrows a section
  (`dgx-spark.md`), linked from the reference.
- **`docs/user/images/`** holds the screenshots; a Playground change that
  alters a pictured screen replaces its image.
- **The root `README.md`** is the overview: it changes when a headline
  feature, the quick start or a headline number changes, and links into the
  guide for the rest. It links no individual ADR.
- **The Hugging Face model cards** carry run commands: when a flag in one is
  renamed, or a number a card quotes changes, the card is updated too
  (`hf upload <repo> README.md README.md`).

`docs/user/` is the published tree: the GitHub Pages site builds from it
alone, through a workflow (owner decision 2026-10-24; the site is not built
yet). Specs, ADRs, findings and these agent docs stay in their own trees; the
guide links to them for the reasons behind a behaviour.

## How it is written

- **What a user does and sees**: the command, the field, the request, the
  answer, the log line. The reason behind it is one sentence and a link to its
  ADR.
- **Generated references stay the reference.** `ignis-server help --fields`
  lists every config field with its default and rule, `/v1/openapi.json` and
  `/v1/docs/` describe the HTTP surface (ADR 0036), `make help` lists the
  targets. The guide adds the why and the worked example; where its prose
  names a default or a field, that value moves with the code.
- **Every command runs as written** on the build the commit produces, with
  current flag names.
- **Public by construction**: examples use placeholder hosts, keys and paths,
  and public model names only.

## Specs and review

- A spec whose work is user-visible names the guide sections it changes in its
  last acceptance criterion.
- `/code-review`'s Standards axis applies this document: a user-visible change
  with no matching `docs/user/` hunk is a violation.
