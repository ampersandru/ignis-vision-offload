# 01 - The agent path: a way in for an AI agent working for a user

GitHub: #357

## Problem Statement

More and more, the first reader of `github.com/gpillon/ignis` is not a person but the person's agent: a user
asks their coding agent "what is https://github.com/gpillon/ignis?", "will it run here?" or "install ignis from
https://github.com/gpillon/ignis", and the agent fetches the repository page and acts on what it reads. Today
nothing there is written for it:

- **The README's first screen** is a logo, badges and a table of numbers; the install route (TL;DR) starts at
  line 107. An agent reading a fetched page often sees only its start.
- **`AGENTS.md`** is what Claude Code, Codex and others read by themselves in a clone, and it is written for agents
  *developing* ignis (issue tracker, GPU lock, `cargo test`). An agent cloning to install ignis is sent the wrong way.
- **The guide** has four reading paths, all for people. The steps an agent needs between the pages are written
  nowhere: which checks rule a machine out, which choices are the user's, that the server started from an agent's
  shell (no terminal on stdin) downloads tens of GiB without asking, that the server holds the shell until stopped.
- **The site** has no `llms.txt` (llmstxt.org), the index crawlers and agents look for.

Two audiences share the word "agent" here, and stay apart: agents working **on** ignis (`AGENTS.md`,
`docs/agents/`, unchanged) and agents working **for a user** of ignis (this spec).

## Solution

- **`docs/user/paths/agent.md`**, a fifth reading path, for an agent acting for a user, organised by task: what
  ignis is (one paragraph it can quote, and what it is not), the checks before installing (card, driver, system,
  disk, RAM, a card already in use) with the command and what passes, the choices the user makes, the install and
  the model fetch, starting it without holding the shell and seeing it serve (`/health`, `/ready`, `/v1/models`),
  connecting a client (itself included), where each question is answered, and what waits for the user's word.
  Mostly links into the existing pages; the page holds the glue, not a second copy of them.
- **`docs/user/start/install-with-an-agent.md`**, for the person: what to tell their agent, what it checks, what it
  asks them, what it leaves to them. `start/install.md` links it.
- **The README** carries a two-line pointer right below its table of numbers, before any prose: the agent path in the
  repository, its raw Markdown URL, the site's `llms.txt`, and `AGENTS.md` for contributors. Its Documentation table
  gains the agent path.
- **`AGENTS.md`** opens with one line routing an agent that is here for a user to the agent path.
- **The site publishes `llms.txt`**, generated at build from the sidebar: one link per page, in the sidebar's order,
  to the page's Markdown source at the ref being published, with its front matter's description.
- **The Hugging Face cards** carry the same pointer as the README (uploaded by the owner's go-ahead, after the push).

## Implementation Decisions

- **The tone is information, not instruction.** Agents treat fetched text as data, and a line that orders "AI agent:
  do X" reads as prompt injection. Pages say what is there and why ("an agent acting for a user: the steps are here"),
  and never ask for anything beyond the user's request.
- **Raw Markdown first.** An agent may never see the rendered site: every link the README and `llms.txt` give an
  agent resolves to plain Markdown (`raw.githubusercontent.com`), and the guide's pages read as Markdown already.
- **`llms.txt` is ours, not `starlight-llms-txt`.** The plugin (0.12.0, peers `astro ^7`, `starlight >=0.41`)
  renders pages to text and lists `llms-full.txt` / `llms-small.txt`, not a page-by-page index; the index with each
  page's description is what an agent navigates by, and it is a few lines over `sectionItems`. The Changelog is left
  out (generated and git-ignored: no Markdown source at the ref); the releases page stands for it.
- **One place per fact.** The agent path links the pages that own a fact (sizes, flags, probes) instead of copying
  them, except the few commands its checks run. `docs/agents/user-docs.md` names it, so a change to the install, the
  first start or connecting a client updates it in the same commit.

## Testing Decisions

- `docs/site/src/llms.test.mjs`: the generator over a temporary guide (title, summary, sections in sidebar order, a
  section's index page, `.mdx` pages, a single-link entry, raw URLs at the ref, a missing page fails), and over the
  real guide (the agent path is listed).
- `make docs-test` and `make docs-build` green: the new pages' links and the landing page's cards resolve.
- `cargo test` workspace-wide.

## Acceptance

1. `docs/user/paths/agent.md` exists with the sections above; every command in it runs as written on v0.8.0; every
   link resolves (`make docs-build`).
2. `docs/user/start/install-with-an-agent.md` exists and `start/install.md` links it.
3. The README's pointer sits between the table of numbers and the first paragraph of prose, links the agent path
   (repository and raw URL), `llms.txt` and `AGENTS.md`; the Documentation table lists the agent path.
4. `AGENTS.md`'s first lines route an agent acting for a user to the agent path.
5. A site build writes `dist/llms.txt` in llmstxt.org's shape, every page but the Changelog's listed with its
   description and a raw Markdown URL at the ref; `llms.test.mjs` holds it.
6. The guide's own indexes name the fifth path: `docs/user/README.md`, the landing page's paths, and
   `docs/agents/user-docs.md` (its `paths/` row and the rule that keeps the agent path current).
7. The Hugging Face cards' pointer text is drafted; uploading it is the owner's call.
8. Guide sections changed: `paths/`, `start/`, the landing page, `docs/user/README.md`; `make docs-test`,
   `make docs-build` and `cargo test` workspace-wide green.

## Out of Scope

- `llms-full.txt` and Markdown routes on the site (`/page.md`): raw GitHub URLs give the same text today.
- An agent-run installer script (`curl | sh`): the agent runs the documented commands, with the user watching.
- Changing `AGENTS.md` or `docs/agents/` beyond the routing line: they stay the contributors' agents' rules.
