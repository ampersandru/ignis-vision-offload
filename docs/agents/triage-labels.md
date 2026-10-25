# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

Edit the right-hand column to match whatever vocabulary you actually use.

## Repo-only labels

Two labels this repo adds to the five roles above. Both are added beside the
issue's other labels: neither replaces its state role or category.

| Label   | Kind     | Meaning |
| ------- | -------- | ------- |
| `idea`  | marker   | Parked: kept open, not specified, not scheduled. Add it to a proposal nobody will build soon, and keep the state role it has (often `ready-for-human`). When picked up, remove `idea` and take the issue into `/grill-with-docs`; close it with a link to the spec or tickets it became. |
| `quick` | marker   | The owner's decision is fast and the work after it is small. It combines with any state and with `idea`. When showing what needs attention, list open `quick` issues as a fourth bucket. |