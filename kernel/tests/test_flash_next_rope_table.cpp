// ADR 0059 (spec rope-scaling/02, ACs 3-6): the rotary table a Flash-Next load rotates at --
// OURS, not vendored.
//
// Host-only: the load's builder (flash_next::text_rope) and the kernels' fp32 form of its table
// (indexer::rope_from), no device work anywhere.
//
// 1. YaRN at f in {2, 2.25, 3, 4}: rope_from(rope_yarn_frequencies(1e7, 64, 262144, f)) within 2
//    fp32 ulps on every pair of HF's _compute_yarn_parameters (transformers d4804f9d,
//    modeling_rope_utils.py), evaluated here in fp64 from its own formula: the correction
//    dimension, truncated bounds clamped to [0, dim - 1], the linear ramp, the blend. Its scale
//    is float(0.1 ln f + 1), HF's get_mscale, and its ramp bounds read off the table are 14/22.
// 2. The load's builder gives that table for a YaRN scaling, and without scaling (factor 0 or
//    1) the table every Flash-Next load rotated at before ADR 0059, bit for bit (the literals
//    below: rope_from's 1.0f / (float)(1 / theta^(-2i/64)), evaluated with a correctly rounded
//    pow), at a scale of exactly 1.0f.
//
// ADR 0006: no SKIP_RETURN_CODE -- but nothing here needs a GPU in the first place.

#include "flash_next/flash_next_internal.h"
#include "flash_next/indexer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

namespace fn = ignis::flash_next;

int g_failed = 0;

void check(bool ok, const std::string &label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label.c_str());
    ++g_failed;
  }
}

// The fp32 table of every Flash-Next load before ADR 0059.
constexpr float kLinear[32] = {
    0x1p+0F,         0x1.356656p-1F,  0x1.75f036p-2F,  0x1.c3f06ap-3F,  0x1.111aeep-3F,  0x1.4a12acp-4F,
    0x1.8eec7ep-5F,  0x1.e222eep-6F,  0x1.235a72p-6F,  0x1.6020a4p-7F,  0x1.a9942ap-8F,  0x1.012cfap-8F,
    0x1.36d21ap-9F,  0x1.77a7dap-10F, 0x1.c603c4p-11F, 0x1.125c04p-11F, 0x1.4b96cp-12F,  0x1.90c182p-13F,
    0x1.e459c4p-14F, 0x1.24b0fcp-14F, 0x1.61bea4p-15F, 0x1.ab8882p-16F, 0x1.025b56p-16F, 0x1.383f88p-17F,
    0x1.796182p-18F, 0x1.c8198cp-19F, 0x1.139e94p-19F, 0x1.4d1c98p-20F, 0x1.9298aep-21F, 0x1.e6933ap-22F,
    0x1.26091cp-22F, 0x1.635e88p-23F,
};

// HF's YaRN inv_freq for pair i at the checkpoint's constants, in fp64.
struct HfYarn {
  double low = 0.0, high = 0.0;
  double inv_freq[32] = {};
};

HfYarn hf_yarn(double factor) {
  constexpr double kPi = 3.14159265358979323846;
  const double dim = 64.0, base = 1.0e7, original = 262144.0;
  const auto correction_dim = [&](double rotations) {
    return dim * std::log(original / (rotations * 2.0 * kPi)) / (2.0 * std::log(base));
  };
  HfYarn out;
  out.low = std::max(std::floor(correction_dim(32.0)), 0.0);
  out.high = std::min(std::ceil(correction_dim(1.0)), dim - 1.0);
  const double high = out.low == out.high ? out.high + 0.001 : out.high;
  for (int i = 0; i < 32; ++i) {
    const double pos_freqs = std::pow(base, 2.0 * i / dim);
    const double extrapolation = 1.0 / pos_freqs;
    const double interpolation = 1.0 / (factor * pos_freqs);
    const double ramp = std::clamp((i - out.low) / (high - out.low), 0.0, 1.0);
    const double extrapolation_factor = 1.0 - ramp;
    out.inv_freq[i] = interpolation * (1.0 - extrapolation_factor) + extrapolation * extrapolation_factor;
  }
  return out;
}

// The fp32 ulps between two positive floats.
int64_t ulps(float a, float b) {
  int32_t ia = 0, ib = 0;
  std::memcpy(&ia, &a, 4);
  std::memcpy(&ib, &b, 4);
  return ia > ib ? int64_t{ia} - ib : int64_t{ib} - ia;
}

ignis::RopeScaling yarn(float factor) {
  ignis::RopeScaling scaling;
  scaling.factor = factor;
  return scaling;
}

// Flash-Next's rotary geometry, as the topology Rust hands the leaf carries it.
ignis_topology topology() {
  ignis_topology t{};
  t.head_dim = 256;
  t.rotary_dim = 64;
  t.rope_theta = 1e7;
  return t;
}

bool same_table(const fn::indexer::Rope &a, const fn::indexer::Rope &b) {
  return std::memcmp(a.inv_freq, b.inv_freq, sizeof(a.inv_freq)) == 0 && a.scale == b.scale;
}

}  // namespace

int main() {
  // --- 1. YaRN against HF's formula -----------------------------------------
  for (const float f : {2.0F, 2.25F, 3.0F, 4.0F}) {
    const std::string label = "yarn:" + std::to_string(f);
    const HfYarn hf = hf_yarn(f);
    check(hf.low == 14.0 && hf.high == 22.0, label + ": HF's own bounds are 14/22");
    const fn::indexer::Rope rope =
        fn::indexer::rope_from(ignis::rope_yarn_frequencies(1e7F, 64, fn::kOriginalPositions, yarn(f)));
    int64_t worst = 0;
    for (int i = 0; i < 32; ++i) {
      const int64_t d = ulps(rope.inv_freq[i], static_cast<float>(hf.inv_freq[i]));
      worst = std::max(worst, d);
      check(d <= 2, label + ": pair " + std::to_string(i) + " is " + std::to_string(d) + " ulps from HF's");
    }
    std::printf("  %s: worst pair %lld fp32 ulps from HF's fp64 table\n", label.c_str(), static_cast<long long>(worst));
    const float mscale = static_cast<float>(0.1 * std::log(static_cast<double>(f)) + 1.0);
    check(rope.scale == mscale, label + ": the scale is HF's get_mscale, 0.1 ln f + 1, got " +
                                    std::to_string(rope.scale));
    check(same_table(fn::indexer::rope_from(fn::text_rope(topology(), yarn(f))), rope),
          label + ": the load's builder gives this table");
  }

  // The ramp bounds, read off the table: below 14 and at 14 (blend 0) the linear frequency, at
  // 22 (blend 1) and past it the linear one over the factor -- exactly, at a power-of-two factor
  // -- and in between neither.
  {
    const fn::indexer::Rope four = fn::indexer::rope_from(fn::text_rope(topology(), yarn(4.0F)));
    check(four.inv_freq[13] == kLinear[13], "pair 13 is below the ramp and extrapolates");
    check(four.inv_freq[14] == kLinear[14], "pair 14 opens the ramp at blend 0, still linear");
    for (int i : {15, 18, 21}) {
      check(four.inv_freq[i] != kLinear[i] && four.inv_freq[i] != kLinear[i] / 4.0F,
            "pair " + std::to_string(i) + " is inside the ramp");
    }
    check(four.inv_freq[22] == kLinear[22] / 4.0F, "pair 22 closes the ramp at blend 1, interpolated");
    check(four.inv_freq[31] == kLinear[31] / 4.0F, "pair 31 is past the ramp and interpolates");
  }

  // --- 2. no scaling is the table before ADR 0059 ---------------------------
  for (const float factor : {0.0F, 1.0F}) {
    const std::string label = "factor " + std::to_string(factor);
    const ninfer::ops::RopeFrequencies table = fn::text_rope(topology(), yarn(factor));
    check(table.attention_factor == 1.0F, label + ": the attention factor stays exactly 1");
    const fn::indexer::Rope rope = fn::indexer::rope_from(table);
    check(std::memcmp(rope.inv_freq, kLinear, sizeof(kLinear)) == 0, label + ": the linear table, bit for bit");
    check(rope.scale == 1.0F, label + ": the scale is exactly 1, an exact multiply");
  }

  if (g_failed != 0) {
    std::fprintf(stderr, "flash-next rope table: %d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("flash-next rope table: all checks passed\n");
  return EXIT_SUCCESS;
}
