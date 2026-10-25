# AGENTS.md

This file is for agents working **on** ignis: its code, its guide, its issues.
An agent here for a user of ignis (to explain it, install it, or connect a
client to it) wants [the agent path](docs/user/paths/agent.md) instead.

## Agent skills

### Issue tracker

GitHub Issues on `gpillon/ignis`, through the `gh` CLI: a spec is an issue and
its tickets are sub-issues with native blocking. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles and the two categories as same-name labels, plus the
repo's `idea` and `quick` markers. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` and `docs/adr/`. See `docs/agents/domain.md`.

## Repo conventions

### Scratch

`.scratch/` is **not tracked by git**: it is the scratchpad for temporary
artifacts, experiments, raw logs and workflow output, local to each clone.
Nothing durable lives there, and it is never an issue tracker.

### Findings

When recording or consulting durable findings from discovery, research, or
experiments, start from `docs/findings/README.md`.

### Testing

Every code change ships with a test, and the task is not complete until
`cargo test` passes workspace-wide. Before launching **anything** on the GPU
(the profile, a bench run, a gate leg, one `--ignored` test), always check
that no other test, bench or agent session is already using the card — in
any worktree. The 5090 fits one run at a time and the loser dies with no
diagnostic. See `docs/agents/testing.md`.

### User documentation

Every user-visible change ships with its user-guide update in the same
commit, and the task is not complete until `docs/user/` describes what the
code now does. What counts as user-visible, where each change goes and how
the guide is written: `docs/agents/user-docs.md`.

### Make

`make help` lists the targets; `make config` prints the exact server command
a run will use. An agent runs the server as a daemon — `make start`, then
`make stop` — since `run`, `dev`, `dev-ui` and `watch` hold the terminal
until Ctrl+C. `run` and `start` only ever launch the last build: a stale
binary fails naming the changed files, and `make build` is the fix.
`make gpu-status` is the one-command check before GPU work. Why it is
shaped this way: ADR 0027.

### Worktrees

All git worktrees for this repo live under `../.inference-qwen-worktrees/`
(sibling of the repo root), one subdirectory per issue/feature. Do not create
worktrees anywhere else (e.g. no separate `.inference-claude-worktrees/`) —
one shared location for every agent.
