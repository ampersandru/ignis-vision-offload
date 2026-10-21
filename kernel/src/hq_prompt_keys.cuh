// Where a prefill chunk's hq-e8-2b keys come from, as ignis's prompt attention reads them
// (kernel/src/hq_prompt_attention.cu, GitHub #335) -- shared with the attention readout and the
// test-only tap, which must read every key exactly as the attention did. OURS: the rule is the
// vendored gqa_attention_prefill_hq_scratch_kernel's (gqa_attention_prefill_hq.cuh), restated
// key by key, and the decode, the side-plane copy and the rotated store are the vendored
// helpers', called.
#pragma once

#include "ignis_hq_prompt_attention.h"

#include "ninfer/ops/gqa_attention.h"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/gqa_attention_prefill_hq.cuh"
#include "ops/kernel/hq_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ignis_hq_prompt {

using Geometry = ninfer::ops::Gqa27Geometry;

enum class KeySource { Zero, Fresh, Side, Codec };

// The chunk's own rows when it attends first, the sinks and the ring before it from the side
// planes, every other key the codec; keys past the history are zero.
__device__ __forceinline__ KeySource key_source(const IgnisHqPromptKeys &s, int key) {
  const int keys_total = s.start + s.width;
  const int fresh_from = s.new_k != nullptr ? s.start : -1;
  if (key < 0 || key >= keys_total) return KeySource::Zero;
  if (fresh_from >= 0 && key >= fresh_from) return KeySource::Fresh;
  if (s.residual_k != nullptr) {
    const int ring_end = fresh_from >= 0 ? fresh_from : keys_total;
    if (key < static_cast<int>(ninfer::ops::kGqaHqSinkKeys) ||
        (key >= ring_end - static_cast<int>(ninfer::ops::kGqaHqRecentKeys) && key < ring_end &&
         ninfer::ops::hq_ring_slot_valid(s.ring_valid, key))) {
      return KeySource::Side;
    }
  }
  return KeySource::Codec;
}

// A whole converged warp fetches up to four rows of one KV head and role, one per 8-lane group:
// `key` is this lane's group's key (negative or past the history: a zero row) and `row` its
// destination, 256 BF16 in the codec's rotated frame, 16-byte aligned. `signs` holds the engine
// signs (hq_engine_signs_fill), visible to the warp. The rows are written when it returns.
__device__ __forceinline__ void fetch_rows_warp(const IgnisHqPromptKeys &s, int head, bool role_v,
                                                int key, __nv_bfloat16 *row,
                                                const std::int8_t *signs) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int lane8 = lane & 7;
  const KeySource source = key_source(s, key);
  if (source == KeySource::Zero) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      *reinterpret_cast<uint4 *>(row + lane8 * 32 + j * 8) = make_uint4(0u, 0u, 0u, 0u);
    }
  } else if (source == KeySource::Side) {
    const auto *plane = static_cast<const __nv_bfloat16 *>(role_v ? s.residual_v : s.residual_k);
    const __nv_bfloat16 *side = ninfer::ops::hq_residual_row<Geometry>(plane, 0, head, key);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      *reinterpret_cast<uint4 *>(row + lane8 * 32 + j * 8) =
          *reinterpret_cast<const uint4 *>(side + lane8 * 32 + j * 8);
    }
  }
  const bool codec = source == KeySource::Codec;
  const auto *codes_plane = static_cast<const std::uint8_t *>(role_v ? s.codes_v : s.codes_k);
  const auto *meta_plane = static_cast<const std::uint8_t *>(role_v ? s.meta_v : s.meta_k);
  ninfer::ops::hq_decode_row_group_fast(
      codec ? ninfer::ops::hq_row_codes<Geometry>(codes_plane, s.block_table, head, key) : nullptr,
      codec ? ninfer::ops::hq_row_meta<Geometry>(meta_plane, s.block_table, head, key) : nullptr,
      row, lane8, 0, ninfer::ops::hq_dither_row_seed(head, key, role_v), codec);
  // The chunk's own rows: the whole warp rotates each, with the fill side's single rounding.
  const unsigned fresh = __ballot_sync(0xffffffffu, source == KeySource::Fresh && lane8 == 0);
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    if ((fresh >> (8 * g)) & 1u) {
      const int fresh_key = __shfl_sync(0xffffffffu, key, 8 * g);
      auto *dst = reinterpret_cast<__nv_bfloat16 *>(
          __shfl_sync(0xffffffffu, reinterpret_cast<unsigned long long>(row), 8 * g));
      const auto *chunk = static_cast<const __nv_bfloat16 *>(role_v ? s.new_v : s.new_k);
      ninfer::ops::hq_store_rotated_row_warp(
          chunk + ninfer::ops::gqa_kv_quant_src_index<Geometry>(head, 0, fresh_key - s.start),
          signs, dst);
    }
  }
  __syncwarp();
}

} // namespace ignis_hq_prompt
