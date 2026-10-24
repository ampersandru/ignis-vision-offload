---
title: "The uncensored variant"
description: "qwen3.8-27b-abliterated: the 27B with the huihui-ai abliteration, and what it changes."
sidebar:
  order: 4
---

`qwen3.8-27b-abliterated` is the same image with the
[huihui-ai abliteration](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)
applied. It is the same container: 1,255 of the 1,325 objects are
byte-identical to the default image, and only the 70 matrices the abliteration
changed are re-encoded. **It does not refuse**, so put the guardrails in the
tool layer.

```
ignis-server model download qwen3.8-27b-abliterated
ignis-server --model-id qwen3.8-27b-abliterated   # found under --download-path, fetched when it is not there
make dev UNCENSORED=1   # the same, from a checkout
```

It is served as `qwen3.8-27b-abliterated`, so a client can tell it from the
27B. The pointing heads and the DFlash2 drafter load unchanged. They were
fitted to the default weights, and the target still verifies every draft, so
the drafter can only change the acceptance rate.
