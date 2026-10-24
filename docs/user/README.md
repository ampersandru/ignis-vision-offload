# The ignis user guide

This folder is the source of the guide published at
**https://gpillon.github.io/ignis/**, where it has search, navigation and
reading paths. Every page is plain Markdown and reads here as well.

| Section | Start with |
|---|---|
| Start here | [Install](start/install.md) |
| Reading paths | [User](paths/user.md), [advanced](paths/advanced.md), [developer](paths/developer.md), [contributor](paths/contributor.md) |
| Guides | [Get a model](guides/models.md), [air-gapped machines](guides/air-gapped.md), [Knative](guides/knative.md), [telemetry](guides/telemetry.md) |
| Concepts | [How ignis serves a request](concepts/how-ignis-serves.md) |
| Configuration | [Fields, precedence, profiles](configuration/index.md) |
| Playground | [The Playground](playground/index.md) |
| API | [Endpoints](api/index.md), [chat completions](api/chat-completions.md), [Responses](api/responses.md) |
| Reference | [Every config field](reference/configuration.md), [profiles](reference/profiles.md), [models](reference/models.md), [metrics](reference/metrics.md), [HTTP routes](reference/http-routes.md) |
| Troubleshooting | [Troubleshooting](troubleshooting.md) |
| Contributing | [Contributing](contributing/index.md) |

The pages under `reference/` are generated from the code (`make
docs-reference`); the rest is written by hand, and every user-visible change
updates it in the same commit ([`docs/agents/user-docs.md`](../agents/user-docs.md)).
