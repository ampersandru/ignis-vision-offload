---
title: "Built-in models"
description: "The catalog compiled into ignis-server: every model it can fetch, the repository and revision it is pinned at, and every file's size and SHA-256."
sidebar:
  order: 3
  badge:
    text: generated
    variant: note
editUrl: false
---

:::note
Generated from the code by `make docs-reference` (`crates/server/src/user_docs.rs`); an edit here is overwritten, and `cargo test` fails while this page and the code disagree.
:::

The models the server knows how to fetch without an operator catalog. Each is pinned file by file, byte count and SHA-256, at a fixed revision of its Hugging Face repository, and nothing outside the binary can change a pin ([Get a model](../guides/models.md), [an operator catalog](../guides/operator-catalog.md)).

| Id | Repository | Size | Artifact |
|---|---|---|---|
| [`qwen3.8-27b`](#qwen38-27b) | [`gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer) | 18.1 GiB | `qwen3_8_27b_nvfp4full-v2.ninfer` |
| [`qwen3.8-27b-abliterated`](#qwen38-27b-abliterated) | [`gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer) | 18.1 GiB | `qwen3_8_27b_nvfp4full-v2-huihui-abliterated.ninfer` |
| [`qwen3.8-flash-next`](#qwen38-flash-next) | [`gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis`](https://huggingface.co/gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis) | 67.8 GiB | `qwen3_8_flash_next_trellis_a25-v2.ninfer` |
| [`ignis-image-2.1-turbo`](#ignis-image-21-turbo) | [`gpillon/ignis-image-2.1-turbo`](https://huggingface.co/gpillon/ignis-image-2.1-turbo) | 15.6 GiB | `ignis_image_2_1_turbo-v2.ninfer` |

## qwen3.8-27b

[`gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer) at revision `e961b419b672e183aa55df8c4b975abc82006e8a`, 19,406,949,449 bytes in 2 files.

| File | Bytes | SHA-256 |
|---|---|---|
| `qwen3_8_27b_nvfp4full-v2.ninfer.graft.json` | 6,981 | `b87e3c005fb1daf7d6b208da52790f95a3f478732f767f87417a5d7fc23491d1` |
| `qwen3_8_27b_nvfp4full-v2.ninfer` | 19,406,942,468 | `abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af` |

## qwen3.8-27b-abliterated

[`gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer) at revision `ef3216949e42c42fc713feffb2e05849d57f0f3b`, 19,406,984,175 bytes in 2 files.

| File | Bytes | SHA-256 |
|---|---|---|
| `qwen3_8_27b_nvfp4full-v2-huihui-abliterated.ninfer.graft.json` | 41,707 | `09b0df892c314563192b0a4f102139939d4ef6261fcd210439d4e2636d12e6fe` |
| `qwen3_8_27b_nvfp4full-v2-huihui-abliterated.ninfer` | 19,406,942,468 | `18954280c794cb2ff1fc24ada8158de1df11bf0a0a2ea63f109045af48905ef1` |

## qwen3.8-flash-next

[`gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis`](https://huggingface.co/gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis) at revision `8d94db83cd8e58d6be31e796723e5e35b960638c`, 72,799,980,571 bytes in 4 files.

| File | Bytes | SHA-256 |
|---|---|---|
| `qwen3_8_flash_next_mtp_3p0-v2.ninfer.conversion.json` | 179,029 | `a5b2c10fe0745cfe6b8e2914135e44903cfb80acb3f0db21e549e76641f14127` |
| `qwen3_8_flash_next_trellis_a25-v2.ninfer.conversion.json` | 1,183,942 | `9858d9fb03dd66fa2558b5dc6aa6463ee70f00024ffbd746545b231c9514f7a0` |
| `qwen3_8_flash_next_mtp_3p0-v2.ninfer` | 1,037,905,920 | `432fe5c0838027047c5b46c4fde526f3d3ef3d126f00fed1213921f8705988d2` |
| `qwen3_8_flash_next_trellis_a25-v2.ninfer` | 71,760,711,680 | `12e3b265b0678ba935a4b05d1b612ba4b68b3a96c3026d818ad65838985fd864` |

## ignis-image-2.1-turbo

[`gpillon/ignis-image-2.1-turbo`](https://huggingface.co/gpillon/ignis-image-2.1-turbo) at revision `d6912417aa47320cd71c78e19d99872faa5cb63b`, 16,722,262,468 bytes in 2 files.

| File | Bytes | SHA-256 |
|---|---|---|
| `ignis_image_2_1_turbo-v2.ninfer.conversion.json` | 617,924 | `a97f04912b9a6d76e828efc02b7e458d0f45d00bc00d794cca39d703ad0f33ca` |
| `ignis_image_2_1_turbo-v2.ninfer` | 16,721,644,544 | `4205dd1932f0801804b0a544fe1340697e4e0a0dea4a276996bcb981006050af` |
