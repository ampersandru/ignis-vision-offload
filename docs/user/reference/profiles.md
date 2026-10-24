---
title: "Built-in profiles"
description: "The hardware profiles built into ignis-server and every value each one sets, general and per model."
sidebar:
  order: 2
  badge:
    text: generated
    variant: note
editUrl: false
---

:::note
Generated from the code by `make docs-reference` (`crates/server/src/user_docs.rs`); an edit here is overwritten, and `cargo test` fails while this page and the code disagree.
:::

A profile is a layer of defaults for one kind of card, between the config file and the built-in defaults: any flag, env var or file value still wins over it. An unset `--profile` is `rtx5090`; `--profile none` skips the layer. A config file's `profiles:` section defines more, or shadows these ([Profiles and defaults](../concepts/profiles.md)).

## rtx5090

| Field | Model | Value |
|---|---|---|
| [`model.max_context`](configuration.md#modelmax_context) | Qwen3.8-27B | `917504` |
| [`model.max_context`](configuration.md#modelmax_context) | Qwen3.8-Flash-Next | `589824` |
| [`model.prefill_chunk`](configuration.md#modelprefill_chunk) | Qwen3.8-27B | `2048` |
| [`spec.backend`](configuration.md#specbackend) | Qwen3.8-27B | `dflash2` |
| [`spec.decode_lanes`](configuration.md#specdecode_lanes) | every model | `3` |
| [`spec.draft_tokens`](configuration.md#specdraft_tokens) | Qwen3.8-27B | `7` |
| [`vision.enabled`](configuration.md#visionenabled) | every model | `true` |
| [`vram.headroom_bytes`](configuration.md#vramheadroom_bytes) | Qwen3.8-27B | `1200M` |
| [`vram.headroom_bytes`](configuration.md#vramheadroom_bytes) | every model | `1536M` |
| [`vram.pinned_expert_bytes`](configuration.md#vrampinned_expert_bytes) | every model | `2G` |

## rtxpro6000

| Field | Model | Value |
|---|---|---|
| [`model.max_context`](configuration.md#modelmax_context) | Qwen3.8-27B | `1048576` |
| [`model.max_context`](configuration.md#modelmax_context) | Qwen3.8-Flash-Next | `1048576` |
| [`model.prefill_chunk`](configuration.md#modelprefill_chunk) | Qwen3.8-27B | `2048` |
| [`ngram.hot_bytes`](configuration.md#ngramhot_bytes) | every model | `auto` |
| [`spec.backend`](configuration.md#specbackend) | Qwen3.8-27B | `dflash2` |
| [`spec.backend`](configuration.md#specbackend) | Qwen3.8-Flash-Next | `mtp` |
| [`spec.decode_lanes`](configuration.md#specdecode_lanes) | every model | `8` |
| [`spec.draft_tokens`](configuration.md#specdraft_tokens) | Qwen3.8-27B | `7` |
| [`spec.draft_tokens`](configuration.md#specdraft_tokens) | Qwen3.8-Flash-Next | `3` |
| [`vision.enabled`](configuration.md#visionenabled) | every model | `true` |
