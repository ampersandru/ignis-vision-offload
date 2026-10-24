# The ignis user guide

This folder is the source of the guide published at
**https://gpillon.github.io/ignis/**, where it has search, navigation and
reading paths. Every page is plain Markdown and reads here as well.

| Section | Start with |
|---|---|
| Start here | [Install](start/install.md) |
| Why ignis | [What sets it apart](why-ignis/index.mdx), a topic per page with its numbers and sources |
| Reading paths | [User](paths/user.md), [advanced](paths/advanced.md), [developer](paths/developer.md), [contributor](paths/contributor.md) |
| Guides | [Get a model](guides/models.md), [air-gapped machines](guides/air-gapped.md), [Knative](guides/knative.md), [telemetry](guides/telemetry.md) |
| Concepts | [How ignis serves a request](concepts/how-ignis-serves.md) |
| Configuration | [Fields, precedence, profiles](configuration/index.md) |
| Playground | [Overview](playground/index.md), [chat](playground/chat.md), [settings](playground/settings.md), [tools and agents](playground/tools-and-agents.md), [Decide](playground/decide.md), [image studio](playground/image-studio.md), [Monitor](playground/monitor.md), [Server](playground/server.md) |
| API | [Endpoints](api/index.md), [chat completions](api/chat-completions.md), [Responses](api/responses.md) |
| Reference | [Every config field](reference/configuration.md), [profiles](reference/profiles.md), [models](reference/models.md), [metrics](reference/metrics.md), [HTTP routes](reference/http-routes.md) |
| Troubleshooting | [Troubleshooting](troubleshooting.md) |
| Changelog | On the site, written from the [GitHub releases](https://github.com/gpillon/ignis/releases) |
| Contributing | [Contributing](contributing/index.md) |

The pages under `reference/` are generated from the code (`make
docs-reference`), the pictures under `images/playground/` by `make
docs-screenshots`, and the site's Changelog from the releases; the rest is
written by hand, and every user-visible change
updates it in the same commit ([`docs/agents/user-docs.md`](../agents/user-docs.md)).
