// The hq-e8-2b prompt attention without prompt planes (ADR 0061, spec 27b-16gb/03, GitHub
// #335): an ADAPTED op (ADR 0051), kernel/src/hq_prompt_attention.cu.
//
// What it replaces: the vendored A1 Prompt route under hq-e8-2b, which decodes the layer's whole
// visible history into BF16 prompt planes (4,096 bytes per context token, up to 262,144 keys)
// before its FA2 kernel attends. This op reads the history in two passes, both on the tensor
// cores, and its workspace stops growing with the context at one band:
//   - the residual window -- the 32 sinks, the 512-key ring before the chunk and the chunk
//     itself -- in BF16, its rows exact as before (the ring slots a rollback cleared through the
//     codec, as before);
//   - every other key, which only the codec holds, decoded once per chunk and layer into a fixed
//     band of `band_keys` keys in E4M3 (2,048 bytes per key) and attended on the block-scaled
//     FP8 MMAs, band after band.
// Without the residual window every key takes the FP8 pass.
//
// The call is A1's for one sequence (B = 1) on an hq-e8-2b cache, with the same order of append
// and attention: with the residual window on, the chunk attends first (its own rows exact from
// `k`/`v`) and is appended after (ADR 0037, GitHub #258); without it the chunk is appended first
// and read back through the codec. The output takes A1's sigmoid gate.
//
// Not part of the flat C ABI: `run_gqa_layer` (kernel/src/gqa_layer.cu) and the leaf's own tests
// call it.
#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>

// The keys one band holds: 2,048 bytes each (E4M3 K and V over 4 KV heads), so 64 MiB.
inline constexpr std::uint32_t kIgnisHqPromptBandKeys = 32768;
// The most key splits a launch runs; each keeps its own online-softmax state across bands.
inline constexpr std::int32_t kIgnisHqPromptMaxSplits = 5;
// The keys one tile holds: the band, the splits and the scales are cut at this grain.
inline constexpr std::uint32_t kIgnisHqPromptTileKeys = 32;

struct IgnisHqPromptOptions {
  // Keys per band, a multiple of `kIgnisHqPromptTileKeys`. Tests narrow it to walk several
  // bands at a small context; the engine keeps the default.
  std::uint32_t band_keys = kIgnisHqPromptBandKeys;
  // Key splits, 1..kIgnisHqPromptMaxSplits; 0 lets the op pick them to fill the card.
  std::int32_t splits = 0;
  // Runs on the host after the attention is enqueued and before an attend-first append, while
  // every key the attention read is still where it read it (the attention readout, GitHub #260).
  std::function<void()> after_attention;
};

// The transient arena one call of `width` tokens bumps when `visible_keys` keys (the chunk's
// included) are visible, in bytes. It depends on the context only up to one band.
std::size_t ignis_hq_prompt_workspace_bytes(std::int32_t width, std::uint32_t visible_keys,
                                            std::uint32_t band_keys = kIgnisHqPromptBandKeys);

// A1 at B = 1 on an hq-e8-2b cache: appends `k`/`v` at `positions`, attends `q` over keys
// [0, start_position + T), and writes `out` = attention * sigmoid(gate).
//
// q, gate, out: BF16 [256, 24, T]; k, v: BF16 [256, 4, T]; positions: I32 [T], holding
// start_position .. start_position + T - 1; `cache`: one sequence's layer view
// (ignis_kv_layer_view). `workspace` must hold `ignis_hq_prompt_workspace_bytes(T,
// start_position + T, options.band_keys)`. Throws std::invalid_argument on a shape or format it
// does not serve.
void ignis_hq_prompt_attention(const ninfer::Tensor &q, const ninfer::Tensor &k,
                               const ninfer::Tensor &v, const ninfer::Tensor &positions,
                               std::int32_t start_position, const ninfer::Tensor &gate,
                               float scale, const ninfer::PagedKVLayerView &cache,
                               ninfer::WorkspaceArena &workspace, ninfer::Tensor &out,
                               cudaStream_t stream, const IgnisHqPromptOptions &options = {});
