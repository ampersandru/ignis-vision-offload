// GitHub #339 (spec 27b-16gb/05a): the trellis body's own plumbing in the layer
// programs (kernel/src/trellis_body.h, a leaf-internal header) -- OURS. The op's
// numerics are test_trellis_linear's; this holds what the layer bodies add:
//
//   split    the GQA q_proj's rows are 24 heads of [q 256 | gate 256] (transformers'
//            order), and the attention reads the query and the gate head-major: every
//            element of a [tokens][24][2][256] pattern lands in its own head and half,
//            bit for bit, so a swapped half or a wrong pitch fails here and not three
//            layers later in a logit.
//   q/gate   query_gate on a trellis q_proj is exactly one ignis_trellis_linear into the
//            interleaved rows followed by that split, on both of the op's routes.
//   head     output_head on a trellis load is the op into the logits at ldy = vocab,
//            and on the served body the vendored linear (not run here: the op tests).
//   epilogue a GDN input projection writes its columns into the conv input at its own
//            ldy, and the output projection adds into the residual: the layer bodies'
//            two uses of the op's ldy and ADD, against the op's own STORE.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "model_internal.h"
#include "moe_fixture.h"
#include "trellis_body.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

using namespace moe_test;
namespace tb = ignis::trellis_body;

namespace {

#define TL_RC(expr)                                                                                \
  do {                                                                                             \
    const int32_t rc_ = (expr);                                                                    \
    if (rc_ != 0) {                                                                                \
      std::fprintf(stderr, "FATAL: %s returned %d: %s\n", #expr, rc_, ignis_trellis_last_error()); \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

// The 27B's GQA heads.
constexpr int kHeads = 24, kHeadDim = 256;

void split_lands_every_half_in_its_head() {
  const int tokens = 5;
  const std::size_t n = static_cast<std::size_t>(tokens) * kHeads * 2 * kHeadDim;
  // Element (t, h, half, d) holds its own index: 5 * 24 * 2 * 256 < 65536.
  std::vector<uint16_t> pattern(n);
  for (std::size_t i = 0; i < n; ++i) pattern[i] = static_cast<uint16_t>(i);
  DeviceBytes interleaved(n * 2), query(n), gate(n);
  upload(interleaved, pattern);
  MOE_CUDA(tb::split_query_gate(interleaved.p, kHeads, kHeadDim, tokens, query.p, gate.p, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const auto q = download<uint16_t>(query.p, n / 2);
  const auto g = download<uint16_t>(gate.p, n / 2);
  int wrong = 0;
  for (int t = 0; t < tokens; ++t) {
    for (int h = 0; h < kHeads; ++h) {
      for (int d = 0; d < kHeadDim; ++d) {
        const std::size_t dense = (static_cast<std::size_t>(t) * kHeads + h) * kHeadDim + d;
        const std::size_t row = (static_cast<std::size_t>(t) * kHeads + h) * 2 * kHeadDim;
        wrong += q[dense] != pattern[row + d];
        wrong += g[dense] != pattern[row + kHeadDim + d];
      }
    }
  }
  check(wrong == 0, "split: " + std::to_string(wrong) + " elements outside their head's half");
}

// A synthetic trellis weight: counter-hash words, suh and svh of +-1 (fp16).
struct Trellis {
  DeviceBytes words, suh, svh;
  ignis_trellis_weight w{};
  Trellis(uint32_t seed, uint32_t in, uint32_t out, uint32_t k2)
      : words(static_cast<std::size_t>(in) * out / 16 * k2), suh(in * 2u), svh(out * 2u) {
    std::vector<uint16_t> v(words.bytes / 2);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<uint16_t>(hash_u32(seed, i));
    upload(words, v);
    const auto signs = [seed](uint32_t n, uint32_t s) {
      std::vector<uint16_t> out(n);
      for (uint32_t i = 0; i < n; ++i) out[i] = (hash_u32(seed + s, i) & 1u) ? 0x3C00u : 0xBC00u;
      return out;
    };
    upload(suh, signs(in, 1));
    upload(svh, signs(out, 2));
    w.trellis = words.p;
    w.suh = suh.p;
    w.svh = svh.p;
    w.k2 = k2;
    w.in = in;
    w.out = out;
  }
};

std::vector<uint16_t> activations(uint32_t seed, std::size_t n) {
  std::vector<uint16_t> x(n);
  for (std::size_t i = 0; i < n; ++i) x[i] = f32_to_bf16(hash_uniform(seed, i, 0.5f));
  return x;
}

// A model handle with nothing bound but the trellis workspace (and, where an
// arm sets it, the head): the layer bodies' helpers read no more of it.
struct Model {
  ignis_model m;
  DeviceBytes workspace;
  explicit Model(uint64_t bytes) : workspace(bytes) {
    m.trellis_body = true;
    m.trellis_workspace.base = workspace.p;
    m.trellis_workspace.bytes = bytes;
    TL_RC(ignis_trellis_workspace_init(&m.trellis_workspace, nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
  }
};

uint64_t workspace_for(uint32_t in, uint32_t out, std::initializer_list<uint32_t> tokens) {
  uint64_t bytes = 0;
  for (uint32_t t : tokens) bytes = std::max(bytes, ignis_trellis_workspace_bytes(in, out, 1, t));
  return bytes;
}

void query_gate_is_the_op_then_the_split() {
  // Two heads' worth of a smaller q_proj (4 heads of 2 x 128 over 256 inputs),
  // at a decode width and a prefill one.
  const int heads = 4, head_dim = 128;
  const uint32_t in = 256, out = 2 * heads * head_dim;
  Trellis q_proj(11, in, out, 6);
  Model model(workspace_for(in, out, {3, 80}));
  ninfer::DeviceArena arena(16u << 20);
  for (const int tokens : {3, 80}) {
    const auto xs = activations(21 + tokens, static_cast<std::size_t>(tokens) * in);
    DeviceBytes x(xs.size() * 2), full(static_cast<std::size_t>(tokens) * out * 2),
        query(static_cast<std::size_t>(tokens) * out), gate(static_cast<std::size_t>(tokens) * out);
    upload(x, xs);
    TL_RC(ignis_trellis_linear(&q_proj.w, x.p, tokens, full.p, out, IGNIS_TRELLIS_STORE, &model.m.trellis_workspace,
                               nullptr));
    const ninfer::Tensor xt(x.p, ninfer::DType::BF16, {static_cast<int32_t>(in), tokens, 1, 1});
    ninfer::Tensor qt(query.p, ninfer::DType::BF16, {heads * head_dim, tokens, 1, 1});
    ninfer::Tensor gt(gate.p, ninfer::DType::BF16, {heads * head_dim, tokens, 1, 1});
    {
      ninfer::DeviceArena::Scope scope = arena.scope();
      tb::query_gate(model.m, q_proj.w, xt, heads, head_dim, arena, qt, gt, nullptr);
    }
    MOE_CUDA(cudaDeviceSynchronize());
    const auto f = download<uint16_t>(full.p, static_cast<std::size_t>(tokens) * out);
    const auto q = download<uint16_t>(query.p, static_cast<std::size_t>(tokens) * out / 2);
    const auto g = download<uint16_t>(gate.p, static_cast<std::size_t>(tokens) * out / 2);
    int wrong = 0;
    for (int t = 0; t < tokens; ++t) {
      for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < head_dim; ++d) {
          const std::size_t dense = (static_cast<std::size_t>(t) * heads + h) * head_dim + d;
          const std::size_t row = static_cast<std::size_t>(t) * out + static_cast<std::size_t>(h) * 2 * head_dim;
          wrong += q[dense] != f[row + d];
          wrong += g[dense] != f[row + head_dim + d];
        }
      }
    }
    check(wrong == 0, "query/gate at " + std::to_string(tokens) + " tokens: " + std::to_string(wrong) +
                          " elements differ from the op's rows");
  }
}

void head_and_epilogues_are_the_op_at_the_bodies_widths() {
  const uint32_t in = 256, vocab = 1024, wide = 1536;
  Trellis head(31, in, vocab, 8);
  Model model(workspace_for(in, vocab, {1, 2, 5, 80}));
  model.m.trellis_head = head.w;
  for (const int tokens : {2, 80}) {
    const auto xs = activations(41 + tokens, static_cast<std::size_t>(tokens) * in);
    DeviceBytes x(xs.size() * 2), direct(static_cast<std::size_t>(tokens) * vocab * 2),
        logits(static_cast<std::size_t>(tokens) * vocab * 2), into(static_cast<std::size_t>(tokens) * wide * 2),
        added(static_cast<std::size_t>(tokens) * vocab * 2);
    upload(x, xs);
    const auto residual = activations(51 + tokens, static_cast<std::size_t>(tokens) * vocab);
    upload(added, residual);
    MOE_CUDA(cudaMemset(into.p, 0, into.bytes));
    TL_RC(ignis_trellis_linear(&head.w, x.p, tokens, direct.p, vocab, IGNIS_TRELLIS_STORE, &model.m.trellis_workspace,
                               nullptr));
    const ninfer::Tensor xt(x.p, ninfer::DType::BF16, {static_cast<int32_t>(in), tokens, 1, 1});
    ninfer::Tensor lt(logits.p, ninfer::DType::BF16, {static_cast<int32_t>(vocab), tokens, 1, 1});
    tb::output_head(model.m, xt, lt, nullptr);
    // A GDN input projection's columns in a wider buffer, and an output
    // projection's residual add.
    tb::linear(model.m, head.w, xt, into.as<uint16_t>() + 256, wide, IGNIS_TRELLIS_STORE, nullptr);
    tb::linear(model.m, head.w, xt, added.p, vocab, IGNIS_TRELLIS_ADD, nullptr);
    MOE_CUDA(cudaDeviceSynchronize());
    const auto d = download<uint16_t>(direct.p, static_cast<std::size_t>(tokens) * vocab);
    const auto l = download<uint16_t>(logits.p, d.size());
    const auto w = download<uint16_t>(into.p, static_cast<std::size_t>(tokens) * wide);
    const auto a = download<uint16_t>(added.p, d.size());
    int head_wrong = 0, strided_wrong = 0, outside = 0, add_wrong = 0;
    for (int t = 0; t < tokens; ++t) {
      for (uint32_t c = 0; c < wide; ++c) {
        const uint16_t got = w[static_cast<std::size_t>(t) * wide + c];
        if (c < 256 || c >= 256 + vocab) {
          outside += got != 0;
          continue;
        }
        strided_wrong += got != d[static_cast<std::size_t>(t) * vocab + (c - 256)];
      }
      for (uint32_t c = 0; c < vocab; ++c) {
        const std::size_t i = static_cast<std::size_t>(t) * vocab + c;
        head_wrong += l[i] != d[i];
        // The op rounds r + y once, y unrounded; the stored y is one BF16
        // rounding from it, so the two sums are within two.
        const float sum = bf16_to_f32(d[i]) + bf16_to_f32(residual[i]);
        const float diff = std::fabs(bf16_to_f32(a[i]) - sum);
        add_wrong += diff > (std::fabs(bf16_to_f32(d[i])) + std::fabs(sum)) * 0x1p-7f + 0x1p-12f;
      }
    }
    const std::string at = " at " + std::to_string(tokens) + " tokens";
    check(head_wrong == 0, "the trellis head is the op" + at);
    check(strided_wrong == 0 && outside == 0, "the columns land at their ldy and offset, nothing else" + at);
    check(add_wrong == 0, "the residual add" + at + ": " + std::to_string(add_wrong) + " outside one rounding");
  }
  // The head refuses logits that are not its vocab wide.
  const auto xs = activations(61, in);
  DeviceBytes x(xs.size() * 2), logits(vocab * 2);
  upload(x, xs);
  const ninfer::Tensor xt(x.p, ninfer::DType::BF16, {static_cast<int32_t>(in), 1, 1, 1});
  ninfer::Tensor short_logits(logits.p, ninfer::DType::BF16, {static_cast<int32_t>(vocab / 2), 1, 1, 1});
  bool refused = false;
  try {
    tb::output_head(model.m, xt, short_logits, nullptr);
  } catch (const std::exception &) {
    refused = true;
  }
  check(refused, "the head refuses logits narrower than its vocab");
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "FAIL: no CUDA device\n");
    return 1;
  }
  TL_RC(ignis_trellis_prepare());
  split_lands_every_half_in_its_head();
  query_gate_is_the_op_then_the_split();
  head_and_epilogues_are_the_op_at_the_bodies_widths();
  if (g_failed != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
