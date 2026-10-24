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

Two labels this repo adds to the five roles above.

| Label   | Kind     | Meaning |
| ------- | -------- | ------- |
| `idea`  | state    | Parked: kept open, not specified, not scheduled. Use it instead of `ready-for-human` for a proposal nobody will build soon; it counts as the issue's state role. When picked up it goes back to `needs-triage` and into `/grill-with-docs`; close it with a link to the spec or tickets it became. |
| `quick` | modifier | The owner's decision is fast and the work after it is small. It combines with any state (e.g. `idea` + `quick`) and never replaces one. When showing what needs attention, list open `quick` issues as a fourth bucket. |