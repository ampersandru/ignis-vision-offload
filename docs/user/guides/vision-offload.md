---
title: "Vision offload (experimental fork)"
description: "Encode the 27B's images on CPU, or stage its GPU tower per image."
sidebar:
  order: 30
---

## Placement

This fork adds `vision.residency` for Qwen3.8-27B, including its 16 GB trellis
artifact. GPU-resident vision is unchanged by default. These options do not
add vision to Flash-Next or change the image-generation model's tower.

| Value | Tower between images | Encoding | Persistent device cost |
| --- | --- | --- | --- |
| `resident` | GPU | GPU | Tower, shared workspace and embedding pool |
| `cpu` | FP32 RAM plus pinned stored weights | CPU | Finished embedding pool and text resources |
| `on-demand` | Stored-format pinned RAM | GPU | Finished embedding pool and text resources |

CPU mode runs this fork's Rust encoder with SIMD GEMMs and segmented attention.
It is not guaranteed bit-identical to CUDA. It competes for CPU/RAM bandwidth
and synchronously holds the engine's compute call while encoding, which can
delay other lanes. Without an explicitly configured `vision.max_tokens`, CPU
mode uses 256 merged tokens (roughly a square 512-pixel image). A higher limit
is allowed, but attention grows quadratically. Zero CPU threads selects the
process's available parallelism. FP32 tower storage is included in the host
safety allowance before loading, in addition to pinned stored weights.

On-demand mode allocates a device copy of the whole tower and encoder workspace
per image. The stream drains before both allocations are freed, including on
errors. The resulting embedding remains on the GPU for text prefill and normal
cache reuse; this does not unload the text model or its conversation KV.

The memory plan leaves the peak temporary allocation free in its workspace
budget so the KV pool cannot consume it. The bytes are not physically allocated
while idle. Other GPU applications can consume this allowance and cause an
image encode allocation to fail. This is not an overlay over live KV or text
weights. Every uncached image incurs the tower upload cost.

## Build on Unraid

The upstream image does not contain these options. Reuse your existing model
directory: its bundled vision objects require no new model conversion.

```sh
git clone https://github.com/ampersandru/ignis-vision-offload.git
cd ignis-vision-offload
git checkout vision-offload
docker build -f Containerfile --build-arg KERNEL_BUILD_JOBS=4 \
  -t ignis:vision-offload .
```

Change the existing Ignis service to `image: ignis:vision-offload` with
`pull_policy: never`. Preserve the model volume, GPU selection, API key,
driver-library search path and port. Merge these into its existing environment:

```yaml
IGNIS_VISION_ENABLED: "true"
IGNIS_VISION_RESIDENCY: cpu
IGNIS_VISION_CPU_THREADS: "4"
IGNIS_VISION_MAX_TOKENS: "256"
```

For temporary GPU encoding, use `IGNIS_VISION_RESIDENCY: on-demand` and start
with `IGNIS_VISION_MAX_TOKENS: "1024"`. Recreate after changing environment
settings. Neither mode changes context, speculation or lane count by itself.
If a router sends images to another server, enabling vision here does not
change that routing policy.

## Validation

Keep the working image for rollback. CPU tests and compilation cannot establish
image accuracy, real-world latency or GPU leak freedom. The GPU profile's
`offloaded_towers_answer_images_and_release_encode_memory` test exercises both
modes against the image canary with repeated fresh encodes. Set
`IGNIS_TEST_ARTIFACT` to the absolute path of a real 27B artifact when testing
outside the original developer's model directory. Run only on an idle GPU
through the repository's GPU preflight/profile workflow.
