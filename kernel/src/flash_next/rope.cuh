// ignis kernel leaf -- one Flash-Next rope pair's rotation (ADR 0059): OURS (ADR 0043), no port
// claim. Both of Flash-Next's rope sites rotate through it -- the attention's q and k heads
// (qsa.cu prepare_kernel) and the indexer's queries and pooled block keys (indexer.cu rope_row) --
// so the two round alike by construction. Device-only: indexer.h, which declares Rope, is also
// read by host-only tests.

#pragma once

#include "indexer.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ignis::flash_next::indexer {

// Rotates pair (x1, x2) -- dims (i, i + 32) of a head, i = `pair` -- to `position` in torch's BF16
// arithmetic: phi = fp32(position) * inv_freq in fp32 (ADR 0059 D2); cos and sin times the table's
// scale rounded to BF16 (D1: YaRN's attention factor on queries and keys alike, 1.0f without
// scaling); each product rounded to BF16, then their sum.
__device__ __forceinline__ void rope_pair(float &x1, float &x2, const Rope &rope, int32_t position, int pair) {
  const auto bf16 = [](float v) { return __bfloat162float(__float2bfloat16_rn(v)); };
  const float phi = static_cast<float>(position) * rope.inv_freq[pair];
  const float c = bf16(cosf(phi) * rope.scale);
  const float s = bf16(sinf(phi) * rope.scale);
  const float a = x1, b = x2;
  x1 = bf16(bf16(a * c) + bf16(-b * s));
  x2 = bf16(bf16(b * c) + bf16(a * s));
}

}  // namespace ignis::flash_next::indexer
