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
| A feature's spec, and the tickets that build it | the spec's GitHub issue and its sub-issues |
| A decision hard to reverse, and its reasons | an ADR, `docs/adr/NNNN-title.md` |
| A measured or researched result worth reusing | a finding, `docs/findings/` |
| What a user can see and do | this guide, `docs/user/` |
| Temporary logs, experiments, raw output | `.scratch/`, untracked |

## Issues

Issues live only on GitHub. A feature's spec is an issue, and the tickets
that build it are its sub-issues, each with its acceptance criteria and with
the tickets that block it as GitHub's own blocking links
([`docs/agents/issue-tracker.md`](../../agents/issue-tracker.md)). An issue
opened before 2026-10-25 has a short body instead, which links its spec: a
file under `docs/specs/`, kept as it was and no longer added to.

Each triaged issue carries `bug` or `enhancement`, and one of five state
labels ([`docs/agents/triage-labels.md`](../../agents/triage-labels.md)):

- `needs-triage`: a maintainer has to evaluate it;
- `needs-info`: waiting on the reporter;
- `ready-for-agent`: fully specified, an agent can take it;
- `ready-for-human`: needs a human to implement;
- `wontfix`: will not be actioned.

A bug report says what you ran (the command, or `make config`'s output), what
you expected, what happened, and the server's log lines around it. The log
file of a server `make start` launched is under `.scratch/serve/`.

## Specs, ADRs and findings

- **A spec** is a GitHub issue: the problem, the solution, the user stories
  and the implementation and testing decisions. Its tickets, the sub-issues,
  carry the acceptance criteria
  ([`docs/agents/issue-tracker.md`](../../agents/issue-tracker.md)).
- **An ADR** records a decision and why it was taken, in the short format: a
  title as its heading and one to three sentences of context, decision and
  reason, with a status, the options it rejected or its consequences only when
  they add something. ADRs 0001 to 0069 keep their longer form with the number
  in the heading; the release notes name both kinds "ADR NNNN — title",
  taking a short one's number from its file name. An ADR takes the next free number when it
  merges into `main`, and is renumbered, with every line that cites it, when
  another ADR took that number first. It is written when the decision is
  taken, and amended rather than rewritten when it changes. The
  [list](../../adr/) is the project's memory of why; ADR 0070 lists the places
  where the repo differs from the agent skills it follows.
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
