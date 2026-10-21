// The 27B's nine BF16 exception matrices at prefill width, timed on the vendored BF16 routes and on
// the FP8 W8A8 GEMM (GitHub #341, spec prefill-5090/01 lever 1) -- OURS, and NOT a CTest test (a
// timing is a finding, not a pass/fail; docs/agents/testing.md).
//
//   qkgv  attention/query_key_gate_value [14336][5120] (GQA layers 3, 7, 11, 15, 19, 23): the
//         vendored BF16 attn_input_proj, against one per-token E4M3 quantization of x and
//         ignis_image_gemm_w8a8 on four row-scale payloads (q 6144 | k 1024 | gate 6144 | v 1024),
//         in four launches, in one grouped launch (ignis_image_gemm_w8a8_grouped, what the dispatch
//         runs), and on one [14336] payload (the fused shape's bound).
//   out   attention/output (GQA 3, 7) and gdn/output (GDN 4), [5120][6144]: the vendored BF16
//         linear_add, against a quantization and the W8A8 GEMM's residual add.
//
// Prints the median time per call over --iters launches after a --warmup-ms warm-up, the TFLOPS,
// and the nine matrices' total per token: 6 x qkgv + 3 x out.
//
//   ignis_fp8_exception_bench [--tokens 2048] [--iters 50] [--warmup-ms 1500]

#include "ignis_fp8_linear.h"
#include "ignis_image_gemm.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/linear_add.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

#define CUDA_FATAL(expr)                                                                           \
  do {                                                                                             \
    const cudaError_t err_ = (expr);                                                               \
    if (err_ != cudaSuccess) {                                                                     \
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, cudaGetErrorString(err_));                    \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

#define RC_FATAL(expr, err)                                                                        \
  do {                                                                                             \
    if ((expr) != 0) {                                                                             \
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, err());                                       \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

constexpr int kHidden = 5120, kQRows = 6144, kKvRows = 1024, kQkgvRows = 14336, kOutCols = 6144;

struct DeviceBytes {
  void *p = nullptr;
  explicit DeviceBytes(std::size_t n) { CUDA_FATAL(cudaMalloc(&p, std::max<std::size_t>(n, 256))); }
  ~DeviceBytes() { cudaFree(p); }
  DeviceBytes(const DeviceBytes &) = delete;
  DeviceBytes &operator=(const DeviceBytes &) = delete;
};

// A deterministic BF16 fill in [-amp, amp) (lowbias32 of the index).
__global__ void fill_bf16(__nv_bfloat16 *p, std::size_t n, uint32_t seed, float amp) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    uint32_t x = static_cast<uint32_t>(i) * 0x9E3779B9u + seed * 0x85EBCA6Bu;
    x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
    p[i] = __float2bfloat16((static_cast<float>(x >> 8) * 0x1p-23f - 1.0f) * amp);
  }
}

void fill(void *p, std::size_t n, uint32_t seed, float amp) {
  fill_bf16<<<1024, 256>>>(static_cast<__nv_bfloat16 *>(p), n, seed, amp);
  CUDA_FATAL(cudaGetLastError());
}

ninfer::Weight bf16_weight(const void *data, int n, int k) {
  ninfer::Weight w{};
  w.payload = data;
  w.payload_bytes = static_cast<std::uint64_t>(n) * k * 2;
  w.qtype = ninfer::QType::BF16_CTRL;
  w.shape[0] = w.padded_shape[0] = n;
  w.shape[1] = w.padded_shape[1] = k;
  w.ndim = 2;
  w.qdata = data;
  w.n = n;
  w.k = k;
  w.layout = ninfer::QuantLayout::Contiguous;
  return w;
}

int arg(int argc, char **argv, const char *name, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
  }
  return fallback;
}

}  // namespace

int main(int argc, char **argv) {
  const int tokens = arg(argc, argv, "--tokens", 2048);
  const int iters = arg(argc, argv, "--iters", 50);
  const int warmup_ms = arg(argc, argv, "--warmup-ms", 1500);
  RC_FATAL(ignis_image_gemm_prepare(), ignis_image_gemm_last_error);

  // Weights and activations.
  DeviceBytes wq(static_cast<std::size_t>(kQkgvRows) * kHidden * 2), wo(static_cast<std::size_t>(kHidden) * kOutCols * 2);
  DeviceBytes x(static_cast<std::size_t>(kHidden) * tokens * 2), xo(static_cast<std::size_t>(kOutCols) * tokens * 2);
  fill(wq.p, static_cast<std::size_t>(kQkgvRows) * kHidden, 1, 0.02f);
  fill(wo.p, static_cast<std::size_t>(kHidden) * kOutCols, 2, 0.02f);
  fill(x.p, static_cast<std::size_t>(kHidden) * tokens, 3, 2.0f);
  fill(xo.p, static_cast<std::size_t>(kOutCols) * tokens, 4, 2.0f);

  // FP8 payloads: the qkgv parent as four row blocks and as one, the out projection as one.
  const int seg_rows[4] = {kQRows, kKvRows, kQRows, kKvRows};
  std::size_t seg_at[5] = {0};
  for (int i = 0; i < 4; ++i) seg_at[i + 1] = seg_at[i] + (ignis_fp8_payload_bytes(seg_rows[i], kHidden) + 255) / 256 * 256;
  DeviceBytes pq(seg_at[4]), pfused(ignis_fp8_payload_bytes(kQkgvRows, kHidden)), po(ignis_fp8_payload_bytes(kHidden, kOutCols));
  std::size_t row0 = 0;
  for (int i = 0; i < 4; ++i) {
    RC_FATAL(ignis_fp8_quantize_rows(static_cast<const char *>(wq.p) + row0 * kHidden * 2, seg_rows[i], kHidden,
                                     static_cast<char *>(pq.p) + seg_at[i], nullptr),
             ignis_fp8_linear_last_error);
    row0 += seg_rows[i];
  }
  RC_FATAL(ignis_fp8_quantize_rows(wq.p, kQkgvRows, kHidden, pfused.p, nullptr), ignis_fp8_linear_last_error);
  RC_FATAL(ignis_fp8_quantize_rows(wo.p, kHidden, kOutCols, po.p, nullptr), ignis_fp8_linear_last_error);

  // Outputs and the quantized activations.
  DeviceBytes yq(static_cast<std::size_t>(kQRows) * tokens * 2), yg(static_cast<std::size_t>(kQRows) * tokens * 2),
      yk(static_cast<std::size_t>(kKvRows) * tokens * 2), yv(static_cast<std::size_t>(kKvRows) * tokens * 2),
      yfused(static_cast<std::size_t>(kQkgvRows) * tokens * 2), res(static_cast<std::size_t>(kHidden) * tokens * 2);
  fill(res.p, static_cast<std::size_t>(kHidden) * tokens, 5, 1.0f);
  DeviceBytes codes(static_cast<std::size_t>(kOutCols) * tokens), scales(static_cast<std::size_t>(tokens) * 4);
  CUDA_FATAL(cudaDeviceSynchronize());

  const ninfer::Tensor tx(x.p, ninfer::DType::BF16, {kHidden, tokens});
  const ninfer::Tensor txo(xo.p, ninfer::DType::BF16, {kOutCols, tokens});
  ninfer::Tensor tq(yq.p, ninfer::DType::BF16, {kQRows, tokens}), tg(yg.p, ninfer::DType::BF16, {kQRows, tokens}),
      tk(yk.p, ninfer::DType::BF16, {kKvRows, tokens}), tv(yv.p, ninfer::DType::BF16, {kKvRows, tokens}),
      tres(res.p, ninfer::DType::BF16, {kHidden, tokens});
  const ninfer::Weight bq = bf16_weight(wq.p, kQkgvRows, kHidden), bo = bf16_weight(wo.p, kHidden, kOutCols);
  ninfer::DeviceArena arena(1u << 20);

  cudaEvent_t e0, e1;
  CUDA_FATAL(cudaEventCreate(&e0));
  CUDA_FATAL(cudaEventCreate(&e1));
  auto time = [&](const char *name, double flop, auto &&call) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(warmup_ms);
    while (std::chrono::steady_clock::now() < until) {
      call();
      CUDA_FATAL(cudaDeviceSynchronize());
    }
    std::vector<double> us;
    for (int i = 0; i < iters; ++i) {
      CUDA_FATAL(cudaEventRecord(e0, nullptr));
      call();
      CUDA_FATAL(cudaEventRecord(e1, nullptr));
      CUDA_FATAL(cudaEventSynchronize(e1));
      float ms = 0.0F;
      CUDA_FATAL(cudaEventElapsedTime(&ms, e0, e1));
      us.push_back(1000.0 * ms);
    }
    std::sort(us.begin(), us.end());
    const double t = us[us.size() / 2];
    std::printf("%-34s %9.1f us  %6.1f TFLOPS\n", name, t, flop / t * 1e-6);
    return t;
  };

  std::printf("T = %d tokens, %d timed launches each\n", tokens, iters);
  const double qkgv_flop = 2.0 * tokens * kQkgvRows * kHidden, out_flop = 2.0 * tokens * kHidden * kOutCols;
  const double bf16_qkgv = time("bf16 qkgv (vendored)", qkgv_flop, [&] {
    ninfer::ops::attn_input_proj(tx, bq, tq, tg, tk, tv, nullptr);
  });
  const double bf16_out = time("bf16 out + residual (vendored)", out_flop, [&] {
    ninfer::ops::linear_add(txo, bo, tres, ninfer::ops::LinearPolicy::A16Only, arena, nullptr);
  });
  const double quant_x = time("quantize x [T][5120]", 0, [&] {
    RC_FATAL(ignis_image_quantize_tokens(x.p, tokens, kHidden, codes.p, static_cast<float *>(scales.p), nullptr),
             ignis_image_gemm_last_error);
  });
  void *ys[4] = {yq.p, yk.p, yg.p, yv.p};
  const double fp8_qkgv4 = time("fp8 qkgv, four GEMMs", qkgv_flop, [&] {
    for (int i = 0; i < 4; ++i) {
      RC_FATAL(ignis_image_gemm_w8a8(static_cast<char *>(pq.p) + seg_at[i], seg_rows[i], kHidden, codes.p,
                                     static_cast<float *>(scales.p), tokens, ys[i], 0, nullptr),
               ignis_image_gemm_last_error);
    }
  });
  ignis_fp8_segment segs[4];
  for (int i = 0; i < 4; ++i) segs[i] = ignis_fp8_segment{static_cast<char *>(pq.p) + seg_at[i], static_cast<uint32_t>(seg_rows[i]), ys[i]};
  const double fp8_qkgvg = time("fp8 qkgv, one grouped launch", qkgv_flop, [&] {
    RC_FATAL(ignis_image_gemm_w8a8_grouped(segs, 4, kHidden, codes.p, static_cast<float *>(scales.p), tokens, 0, nullptr),
             ignis_image_gemm_last_error);
  });
  const double fp8_qkgv1 = time("fp8 qkgv, one fused GEMM", qkgv_flop, [&] {
    RC_FATAL(ignis_image_gemm_w8a8(pfused.p, kQkgvRows, kHidden, codes.p, static_cast<float *>(scales.p), tokens,
                                   yfused.p, 0, nullptr),
             ignis_image_gemm_last_error);
  });
  const double quant_xo = time("quantize x [T][6144]", 0, [&] {
    RC_FATAL(ignis_image_quantize_tokens(xo.p, tokens, kOutCols, codes.p, static_cast<float *>(scales.p), nullptr),
             ignis_image_gemm_last_error);
  });
  const double fp8_out = time("fp8 out + residual", out_flop, [&] {
    RC_FATAL(ignis_image_gemm_w8a8_add(po.p, kHidden, kOutCols, codes.p, static_cast<float *>(scales.p), tokens, res.p,
                                       nullptr),
             ignis_image_gemm_last_error);
  });

  const double bf16_total = 6 * bf16_qkgv + 3 * bf16_out;
  const double fp8_total4 = 6 * (quant_x + fp8_qkgv4) + 3 * (quant_xo + fp8_out);
  const double fp8_total1 = 6 * (quant_x + fp8_qkgv1) + 3 * (quant_xo + fp8_out);
  const double fp8_totalg = 6 * (quant_x + fp8_qkgvg) + 3 * (quant_xo + fp8_out);
  std::printf("nine matrices per token: bf16 %.3f us; fp8 four GEMMs %.3f us, one grouped launch %.3f us, fused "
              "bound %.3f us (saves %.3f / %.3f / %.3f us per token)\n",
              bf16_total / tokens, fp8_total4 / tokens, fp8_totalg / tokens, fp8_total1 / tokens,
              (bf16_total - fp8_total4) / tokens, (bf16_total - fp8_totalg) / tokens, (bf16_total - fp8_total1) / tokens);
  return 0;
}
