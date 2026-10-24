---
title: "How work is tracked"
description: "GitHub issues and their triage labels, specs, ADRs, findings, the guide, worktrees and commit messages."
sidebar:
  order: 5
---

Each kind of record has one home, and nothing is said twice.

| What | Where |
|---|---|
| Status, blocking, labels, closure | the [GitHub issue](https://github.com/gpillon/ignis/issues) |
| Acceptance criteria and the seam a change goes in | its spec, `docs/specs/<feature>/NN-name.md` |
| A decision hard to reverse, and its reasons | an ADR, `docs/adr/NNNN-title.md` |
| A measured or researched result worth reusing | a finding, `docs/findings/` |
| What a user can see and do | this guide, `docs/user/` |
| Temporary logs, experiments, raw output | `.scratch/`, untracked |

## Issues

Issues live only on GitHub, and an issue's body is short: a few lines of
context, a link to its spec and the issues that block it. Five labels carry
triage ([`docs/agents/triage-labels.md`](../../agents/triage-labels.md)):

- `needs-triage`: a maintainer has to evaluate it;
- `needs-info`: waiting on the reporter;
- `ready-for-agent`: fully specified, an agent can take it;
- `ready-for-human`: needs a human to implement;
- `wontfix`: will not be actioned.

A bug report says what you ran (the command, or `make config`'s output), what
you expected, what happened, and the server's log lines around it. The log
file of a server `make start` launched is under `.scratch/serve/`.

## Specs, ADRs and findings

- **A spec** holds a work item's acceptance criteria, numbered, and the seam
  the change goes in; the issue links it and does not repeat it
  ([`docs/agents/issue-tracker.md`](../../agents/issue-tracker.md)).
- **An ADR** records a decision, the options it rejected and its consequences.
  It is written when the decision is taken, and amended rather than rewritten
  when it changes. The [list](../../adr/) is the project's memory of why.
- **A finding** preserves evidence another reader can inspect or reproduce:
  a discovery, a research result or an experiment, with its question, its
  evidence, its limits ([`docs/agents/findings.md`](../../agents/findings.md)).

## The guide

Every user-visible change — a flag, a default, a route, a response field, a
Playground screen, a metric, a refusal — updates `docs/user/` in the same
commit. What counts, which page a change goes in and how the guide is written
are in [`docs/agents/user-docs.md`](../../agents/user-docs.md). The reference
pages under `docs/user/reference/` are generated: change the code, then run
`make docs-reference`.

## Branches, worktrees and commits

- Work happens on a branch per issue, merged into `main` with a merge commit
  that names the issue.
- Agents keep every git worktree under `../.inference-qwen-worktrees/`, one
  directory per issue, beside the repository.
- A commit message says what the change does and why, in full sentences, and
  ends with the issue it belongs to, `(#347)`.
- Diffs stay semantic: no reformatting of lines a change does not touch.
