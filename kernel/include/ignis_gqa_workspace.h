/* ignis kernel leaf: how much transient arena one GQA attention call needs --
 * A1 (`gqa_attention`), or ignis's prompt attention, which serves every
 * hq-e8-2b prefill chunk (GitHub #335) -- per call and over a load's widths
 * (P4-05, GitHub #123).
 *
 * Not part of the public flat C ABI. It lives in `kernel/include` rather than
 * beside `kernel/src/gqa_layer.cu` for the same reason `ignis_seq_internal.h`
 * does: the leaf's own CTest
 * (kernel/tests/test_hq_route_agreement.cu) must size its workspace through
 * the function the engine actually ships, or the narrow-width arm there would
 * be testing a second copy of the arithmetic and the shipped one could break
 * unseen.
 */
#ifndef IGNIS_GQA_WORKSPACE_H
#define IGNIS_GQA_WORKSPACE_H

#include "ignis_hq_prompt_attention.h"
#include "ninfer/ops/gqa_attention.h"
/* The vendored route resolver, read-only: which op a width takes is A1's own
 * answer about its route, rather than a width threshold restated here. */
#include "ops/launcher/gqa_attention.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

/* The Q-head geometry the 27B text backbone runs: 24 query heads over 4 KV
 * heads of 256. */
inline constexpr std::int32_t kIgnisGqaQHeads = 24;

/* One past the vendored capacity query's own verify cap
 * (`kMaximumVerifyTokens` = 16, kernel/vendor/src/ops/wrapper/
 * gqa_attention.cpp): the narrowest width at which that query accounts for an
 * hq prompt call's two transient riders together rather than separately. */
inline constexpr std::int32_t kIgnisGqaQueryVerifyCapWidth = 17;

/* The transient arena A1 itself needs for one B=1 call at `width` over
 * `envelope` -- since GitHub #335 the engine no longer calls A1 on an hq-e8-2b
 * Prompt width (see below), but the route agreement test does, as the oracle
 * the prompt attention is held against, and it sizes A1 here.
 *
 * An hq-e8-2b Prompt call holds two transient riders at once: the
 * rotated-frame BF16 span planes it materializes the visible history into,
 * and the key-split partials. `gqa_attention_workspace_capacity_bytes` adds
 * them together only for widths above its own 16-token verify cap; at or
 * below that cap it combines them with `max` instead, and the answer comes
 * back short by one split-partial set. Asked per call, at exactly the width
 * about to run, a 9..16-token prefill chunk -- a short prompt, or the tail of
 * a longer span -- would get an arena the op then overruns, and the op's own
 * bump throws `bad allocation` mid-layer. (The reference's own engine does not
 * meet this: it queries once over a whole width interval, where the widths
 * above the cap dominate the answer.)
 *
 * The correction is to ask over an interval wide enough that the answer
 * bounds this width, which is the query's documented contract -- it "returns
 * the transient arena capacity required for every W in the inclusive
 * interval". The split partials grow with the width at a fixed split count
 * across this range, so the answer at `kIgnisGqaQueryVerifyCapWidth` covers
 * any narrower width, and the envelope is raised with it because the query
 * refuses an interval its envelope cannot hold. Only a narrow Prompt width
 * pays the difference, and the difference is one split-partial set (under
 * 2 MB at this geometry).
 *
 * Deliberately a correction at this caller rather than a patch to the
 * vendored query (ADR 0010): it changes no kernel, no numerics and no route
 * -- only how many bytes this caller reserves before making the call. */
inline std::size_t ignis_gqa_a1_workspace_bytes(ninfer::DType cache_dtype,
                                                ninfer::ops::GqaExecutionEnvelope envelope,
                                                std::int32_t width) {
  ninfer::ops::GqaExecutionEnvelope query_envelope = envelope;
  std::int32_t query_width = width;
  const bool prompt_route = ninfer::ops::detail::gqa_attention_resolve_route(
                                kIgnisGqaQHeads, width, /*batch_size=*/1, cache_dtype, envelope) ==
                            ninfer::ops::detail::GqaAttentionRoute::Prompt;
  if (cache_dtype == ninfer::DType::U8 && prompt_route &&
      width < kIgnisGqaQueryVerifyCapWidth) {
    query_width = kIgnisGqaQueryVerifyCapWidth;
    query_envelope.max_visible_keys =
        std::max(query_envelope.max_visible_keys, static_cast<std::uint32_t>(query_width));
  }
  return ninfer::ops::gqa_attention_workspace_capacity_bytes(
      kIgnisGqaQHeads, cache_dtype, query_envelope, /*batch_size=*/1, width, query_width);
}

/* The transient arena one B=1 attention call at `width` over `envelope` needs:
 * what `run_gqa_layer` (kernel/src/gqa_layer.cu) hands the op it dispatches.
 *
 * An hq-e8-2b Prompt width takes ignis's prompt attention (GitHub #335,
 * ADR 0061), and its own query answers: the band, the split states, no prompt
 * planes, the same bytes at any context past one band. Every other call is
 * A1's (`ignis_gqa_a1_workspace_bytes` above).
 *
 * A decode round's own query is not this one: it runs B lanes of one token,
 * which the resolver sends to the small-T route, so kernel/src/gqa_layer.cu's
 * graph path asks the vendored query directly at its own batch size. */
inline std::size_t ignis_gqa_attention_workspace_bytes(ninfer::DType cache_dtype,
                                                       ninfer::ops::GqaExecutionEnvelope envelope,
                                                       std::int32_t width) {
  const bool prompt_route = ninfer::ops::detail::gqa_attention_resolve_route(
                                kIgnisGqaQHeads, width, /*batch_size=*/1, cache_dtype, envelope) ==
                            ninfer::ops::detail::GqaAttentionRoute::Prompt;
  if (cache_dtype == ninfer::DType::U8 && prompt_route) {
    return ignis_hq_prompt_workspace_bytes(width, envelope.max_visible_keys);
  }
  return ignis_gqa_a1_workspace_bytes(cache_dtype, envelope, width);
}

/* The transient arena one GQA layer's attention needs for every width in
 * [min_width, max_width] at `batch_size` rows over `envelope`: what the load
 * reserves (kernel/src/model.cu `gqa_layer_scratch_bytes`). A1's own query,
 * except where the layer runs ignis's prompt attention instead -- the hq-e8-2b
 * Prompt widths of one row (GitHub #335) -- whose answer is that op's at the
 * widest such width (every buffer it bumps grows with the width). */
inline std::size_t ignis_gqa_attention_capacity_bytes(ninfer::DType cache_dtype,
                                                      ninfer::ops::GqaExecutionEnvelope envelope,
                                                      std::int32_t batch_size,
                                                      std::int32_t min_width,
                                                      std::int32_t max_width) {
  if (cache_dtype != ninfer::DType::U8 || batch_size != 1) {
    return ninfer::ops::gqa_attention_workspace_capacity_bytes(
        kIgnisGqaQHeads, cache_dtype, envelope, batch_size, min_width, max_width);
  }
  std::size_t bytes = 0;
  std::int32_t vendored_last = min_width - 1;
  while (vendored_last < max_width &&
         ninfer::ops::detail::gqa_attention_resolve_route(kIgnisGqaQHeads, vendored_last + 1, 1,
                                                          cache_dtype, envelope) !=
             ninfer::ops::detail::GqaAttentionRoute::Prompt) {
    ++vendored_last;
  }
  if (vendored_last >= min_width) {
    bytes = ninfer::ops::gqa_attention_workspace_capacity_bytes(
        kIgnisGqaQHeads, cache_dtype, envelope, 1, min_width, vendored_last);
  }
  if (vendored_last < max_width) {
    bytes = std::max(bytes, ignis_hq_prompt_workspace_bytes(max_width, envelope.max_visible_keys));
  }
  return bytes;
}

/* Whether A1 must be handed a zero-initialized transient workspace.
 *
 * A1 owns its workspace and suballocates from it, but nothing in
 * `ninfer/ops/gqa_attention.h` promises that every byte is written before it
 * is read, and one route says out loud that it is not: the BF16 small-T
 * partial kernel leaves the split slots it does not use untouched and
 * "relies on the engine's zero-initialized partial workspace instead"
 * (kernel/vendor/src/ops/kernel/gqa_attention_decode_bf16.cuh, the
 * `split >= active_split_count` arm). So zeroing is the default, and the
 * question this answers is the narrow one: which format carries an in-kernel
 * guarantee strong enough to skip it.
 *
 * Deliberately enumerates the format that may skip the zeroing rather than
 * the formats that need it. A KV format added later then keeps the zeroing
 * until someone reads its kernels and says otherwise, instead of silently
 * inheriting an exemption nobody checked it against.
 *
 * hq-e8-2b (U8) is that format, on both routes the GQA layer dispatches:
 *
 *   - small-T (every decode round, and the verify widths): the same
 *     `split >= active_split_count` arm calls `write_neutral()` under hq --
 *     the route's "public partials contract neutralizes inactive splits" --
 *     so every slot the reducer can read is written by the launch in front
 *     of it. The reducer's own bound is device-side
 *     (`gqa_small_t_active_splits` over `last_pos + 1`,
 *     kernel/vendor/src/ops/kernel/gqa_attention_decode.cuh) and never
 *     exceeds the launch capacity this caller reserved for.
 *   - Prompt (every prefill chunk), which ignis's prompt attention serves
 *     (GitHub #335, kernel/src/hq_prompt_attention.cu): its passes write
 *     every tile of the window and the band before they read it, zero rows
 *     included, and every split writes its state on the first pass, the
 *     empty state when it has no keys, before any later pass or the merge
 *     reads it.
 *
 * What it buys: the band and the split states are ~130 MB at a 1,024-token
 * chunk, so zeroing them would be the largest single write in a prefill
 * chunk outside attention itself, once per GQA layer -- and on the decode
 * round it would clear slots `write_neutral()` neutralizes again a moment
 * later.
 *
 * kernel/tests/test_hq_route_agreement.cu runs every hq shape this engine
 * dispatches twice, over a zeroed and over a hostile workspace, and requires
 * the two outputs to be bit-identical: the claim has a test rather than this
 * comment. */
inline bool ignis_gqa_workspace_needs_zeroing(ninfer::DType cache_dtype) {
  return cache_dtype != ninfer::DType::U8;
}

#endif /* IGNIS_GQA_WORKSPACE_H */
