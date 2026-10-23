// The dense trellis linear's microbenchmark (spec 27b-16gb/02 Acceptance 4 and 5, GitHub #337) --
// OURS. A report, not a pass/fail: a tool and not a CTest. Run it with the card to itself.
//
// For every matrix the 27B's 16 GB artifact serves (the repack's split matrices at the quant's K),
// the fused shapes of spec 02's table and the head:
//   decode   device time per call at 1, 8, 16 and 64 tokens, from a CUDA graph of calls cycling
//            over enough copies of the weight that none is in the L2 when it is read; the achieved
//            DRAM bandwidth for the call's bytes (trellis, suh, svh) against the roofline; the
//            width-cost ratios 8 / 1 and 64 / 1.
//   prefill  TFLOP/s (2 tokens in out per call) at 1,024 and 8,192 tokens, against the 5090's FP16
//            accumulate peak (whitepaper Table 3: 419 TFLOPS dense).
//   body     one decode token through the whole body (48 GDN and 16 GQA layers' matrices, each at
//            the K the public SC_3.00bpw_H4 quant gives it): its bytes over its time, the figure
//            Acceptance 4 asks to reach 65% of the roofline; and eight tokens against one.
//   peak     the card's own fp16-input, fp16-accumulate m16n8k16 rate from a register-only loop
//            (eight independent chains a warp), with the SM clock it ran at: the measured
//            denominator beside the whitepaper's 419 TFLOPS.
// Every time includes the call's activation rotation. The whole report repeats under the debug SM
// cap at each of --caps (default 0,36,70,84; 0 is uncapped).
//
//   ignis_kernel_trellis_linear_bench [--caps 0,36,70,84] [--calls N] [--reps N] [--no-prefill]

#include "trellis_test_common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace trellis_test;

namespace {

constexpr double kRooflineGBs = 1792.0;  // 512-bit, 28 Gbps (whitepaper)
constexpr double kFp16AccTflops = 419.0;
constexpr uint64_t kPoolBytes = 512ull << 20;  // > 5 x the 96 MB L2

struct Shape {
  const char *name;
  uint32_t in, out, k2;
  int swiglu;  // the gate/up pair, one call
};

// The repack's matrices (spec 01) at the quant's dominant K.
const Shape kBody[] = {
    {"gdn qkv", 5120, 10240, 6, 0}, {"gdn z", 5120, 6144, 6, 0}, {"gdn out", 6144, 5120, 6, 0},
    {"gqa q", 5120, 12288, 6, 0},   {"gqa k", 5120, 1024, 8, 0},  {"gqa v", 5120, 1024, 8, 0},
    {"gqa o", 6144, 5120, 6, 0},    {"mlp gate+up", 5120, 17408, 6, 1}, {"mlp down", 17408, 5120, 6, 0},
};
const Shape kOther[] = {
    {"fused gate_up", 5120, 34816, 6, 0}, {"fused qkvz", 5120, 16384, 6, 0}, {"fused qkgv", 5120, 14336, 6, 0},
    {"head", 5120, 248320, 8, 0},
    {"floor 128x128", 128, 128, 6, 0},  // a call's fixed cost: two launches, one unit, no bytes to speak of
};

// One decode token through the body: every (matrix, K) the public quant has and how many of its
// 64 layers have it (its quantization_config.json; gate and up, and k and v, share a K in every
// layer, so the pair is one call at that K).
struct MixEntry {
  Shape shape;
  int calls;
};
const MixEntry kMix[] = {
    {{"gdn qkv", 5120, 10240, 4, 0}, 4},  {{"gdn qkv", 5120, 10240, 6, 0}, 42}, {{"gdn qkv", 5120, 10240, 8, 0}, 2},
    {{"gdn z", 5120, 6144, 4, 0}, 3},     {{"gdn z", 5120, 6144, 6, 0}, 40},    {{"gdn z", 5120, 6144, 8, 0}, 5},
    {{"gdn out", 6144, 5120, 6, 0}, 40},  {{"gdn out", 6144, 5120, 8, 0}, 8},
    {{"gqa q", 5120, 12288, 4, 0}, 6},    {{"gqa q", 5120, 12288, 6, 0}, 10},
    {{"gqa k, v", 5120, 1024, 8, 0}, 28}, {{"gqa k, v", 5120, 1024, 10, 0}, 4},
    {{"gqa o", 6144, 5120, 6, 0}, 14},    {{"gqa o", 6144, 5120, 8, 0}, 2},
    {{"mlp gate+up", 5120, 17408, 4, 1}, 9}, {{"mlp gate+up", 5120, 17408, 6, 1}, 49}, {{"mlp gate+up", 5120, 17408, 8, 1}, 6},
    {{"mlp down", 17408, 5120, 4, 0}, 3}, {{"mlp down", 17408, 5120, 6, 0}, 53}, {{"mlp down", 17408, 5120, 8, 0}, 8},
};

int g_calls = 64;
int g_reps = 5;

uint64_t weight_bytes(const Shape &s) {
  const uint64_t one = static_cast<uint64_t>(s.in) * s.out / 16 * s.k2 + 2ull * s.in + 2ull * s.out;
  return one * (s.swiglu ? 2 : 1);
}

// Copies of a shape's weight (the pair, for SwiGLU) so that cycling through them misses the L2.
struct Pool {
  std::vector<DeviceBytes> mem;
  std::vector<ignis_trellis_weight> w;  // [copy][0 or 0, 1]
  int copies = 0;
};

void make_pool(Pool &pool, const Shape &s) {
  const int per = s.swiglu ? 2 : 1;
  pool.copies = static_cast<int>(std::max<uint64_t>(1, std::min<uint64_t>(32, kPoolBytes / weight_bytes(s) + 1)));
  const uint64_t n_words = static_cast<uint64_t>(s.in / 16) * (s.out / 16) * tile_words16(s.k2);
  for (int c = 0; c < pool.copies * per; ++c) {
    pool.mem.emplace_back(n_words * 2 + 2ull * s.in + 2ull * s.out + 256);
    char *base = static_cast<char *>(pool.mem.back().p);
    device_hash_words(base, n_words, 90000 + c);
    std::vector<uint16_t> scales(s.in + s.out, moe_test::f32_to_f16(0.03f));
    MOE_CUDA(cudaMemcpy(base + n_words * 2, scales.data(), scales.size() * 2, cudaMemcpyHostToDevice));
    pool.w.push_back(ignis_trellis_weight{base, base + n_words * 2, base + n_words * 2 + 2ull * s.in, s.k2, s.in, s.out, 0});
  }
}

// Device microseconds per call of `tokens` tokens through the pool, graph-replayed.
double time_call(const Pool &pool, const Shape &s, int tokens, cudaStream_t stream) {
  const int per = s.swiglu ? 2 : 1;
  const uint64_t ws_bytes = ignis_trellis_workspace_bytes(s.in, s.out, per, tokens);
  DeviceBytes ws_mem(ws_bytes), x(static_cast<size_t>(tokens) * s.in * 2), y(static_cast<size_t>(tokens) * s.out * 2);
  ignis_trellis_workspace ws{ws_mem.p, ws_bytes};
  TL_RC(ignis_trellis_workspace_init(&ws, stream));
  std::vector<uint16_t> xh(static_cast<size_t>(tokens) * s.in);
  for (size_t i = 0; i < xh.size(); ++i) xh[i] = moe_test::f32_to_bf16(moe_test::hash_uniform(91000, i, 1.0f));
  MOE_CUDA(cudaMemcpy(x.p, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
  const int calls = tokens > IGNIS_TRELLIS_DECODE_MAX_TOKENS ? std::max(4, g_calls / 8) : g_calls;
  auto issue = [&](int i) {
    const int c = i % pool.copies;
    if (s.swiglu) {
      TL_RC(ignis_trellis_linear_swiglu(&pool.w[2 * c], &pool.w[2 * c + 1], x.p, tokens, y.p, s.out, &ws, stream));
    } else {
      TL_RC(ignis_trellis_linear(&pool.w[c], x.p, tokens, y.p, s.out, IGNIS_TRELLIS_STORE, &ws, stream));
    }
  };
  for (int i = 0; i < 2 * pool.copies; ++i) issue(i);  // warm the clocks
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  MOE_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < calls; ++i) issue(i);
  MOE_CUDA(cudaStreamEndCapture(stream, &graph));
  MOE_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  cudaEvent_t a, b;
  MOE_CUDA(cudaEventCreate(&a));
  MOE_CUDA(cudaEventCreate(&b));
  MOE_CUDA(cudaGraphLaunch(exec, stream));
  double best = 1e30;
  for (int r = 0; r < g_reps; ++r) {
    MOE_CUDA(cudaEventRecord(a, stream));
    MOE_CUDA(cudaGraphLaunch(exec, stream));
    MOE_CUDA(cudaEventRecord(b, stream));
    MOE_CUDA(cudaEventSynchronize(b));
    float ms = 0.0f;
    MOE_CUDA(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, 1000.0 * ms / calls);
  }
  MOE_CUDA(cudaEventDestroy(a));
  MOE_CUDA(cudaEventDestroy(b));
  MOE_CUDA(cudaGraphExecDestroy(exec));
  MOE_CUDA(cudaGraphDestroy(graph));
  return best;
}

// ---- the fp16-accumulate tensor peak ------------------------------------------------------------

constexpr int kPeakChains = 8;

__global__ void __launch_bounds__(256) peak_f16acc_kernel(int iters, uint32_t seed, uint32_t *sink,
                                                         unsigned long long *clocks) {
  uint32_t a[4], b0 = 0x3C003C00u ^ seed, b1 = b0 ^ 0x00010001u;
  for (int i = 0; i < 4; ++i) a[i] = 0x3C003C00u ^ (seed + i);  // ~1.0 halves: no overflow in fp16
  uint32_t acc[kPeakChains][2] = {};
  const unsigned long long c0 = clock64();
  unsigned long long t0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int c = 0; c < kPeakChains; ++c) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
          : "+r"(acc[c][0]), "+r"(acc[c][1])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
  }
  const unsigned long long c1 = clock64();
  unsigned long long t1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
  uint32_t x = 0;
  for (int c = 0; c < kPeakChains; ++c) x ^= acc[c][0] ^ acc[c][1];
  if (x == 0x12345678u) *sink = x;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    clocks[0] = c1 - c0;
    clocks[1] = t1 - t0;
  }
}

// TFLOPS and MHz of the loop over the whole card.
void peak_f16acc(double *tflops, double *mhz) {
  int sms = 0;
  MOE_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  const int blocks = 4 * sms, iters = 20000;
  DeviceBytes sink(4), clocks(16);
  cudaEvent_t a, b;
  MOE_CUDA(cudaEventCreate(&a));
  MOE_CUDA(cudaEventCreate(&b));
  peak_f16acc_kernel<<<blocks, 256>>>(iters, 1, static_cast<uint32_t *>(sink.p), static_cast<unsigned long long *>(clocks.p));
  float best = 1e30f;
  for (int r = 0; r < 5; ++r) {
    MOE_CUDA(cudaEventRecord(a));
    peak_f16acc_kernel<<<blocks, 256>>>(iters, 2 + r, static_cast<uint32_t *>(sink.p), static_cast<unsigned long long *>(clocks.p));
    MOE_CUDA(cudaEventRecord(b));
    MOE_CUDA(cudaEventSynchronize(b));
    float ms = 0.0f;
    MOE_CUDA(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, ms);
  }
  const std::vector<unsigned long long> c = download<unsigned long long>(clocks.p, 2);
  MOE_CUDA(cudaEventDestroy(a));
  MOE_CUDA(cudaEventDestroy(b));
  const double flop = 2.0 * 16 * 8 * 16 * kPeakChains * static_cast<double>(iters) * blocks * (256 / 32);
  *tflops = flop / (best * 1e-3) / 1e12;
  *mhz = c[1] > 0 ? static_cast<double>(c[0]) / static_cast<double>(c[1]) * 1e3 : 0.0;
}

__global__ void stream_read_kernel(const uint4 *p, size_t n, unsigned *sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const uint4 v = __ldcs(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) *sink = acc;
}

double stream_read_gbs() {
  const size_t bytes = 1ull << 30;
  DeviceBytes buf(bytes), sink(4);
  MOE_CUDA(cudaMemset(buf.p, 1, bytes));
  cudaEvent_t a, b;
  MOE_CUDA(cudaEventCreate(&a));
  MOE_CUDA(cudaEventCreate(&b));
  stream_read_kernel<<<2048, 512>>>(static_cast<const uint4 *>(buf.p), bytes / 16, static_cast<unsigned *>(sink.p));
  float best = 1e30f;
  for (int r = 0; r < 5; ++r) {
    MOE_CUDA(cudaEventRecord(a));
    stream_read_kernel<<<2048, 512>>>(static_cast<const uint4 *>(buf.p), bytes / 16, static_cast<unsigned *>(sink.p));
    MOE_CUDA(cudaEventRecord(b));
    MOE_CUDA(cudaEventSynchronize(b));
    float ms = 0.0f;
    MOE_CUDA(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, ms);
  }
  MOE_CUDA(cudaEventDestroy(a));
  MOE_CUDA(cudaEventDestroy(b));
  return bytes / (best * 1e-3) / 1e9;
}

void set_cap(int cap) {
  const std::string v = std::to_string(cap);
#ifdef _WIN32
  _putenv_s("IGNIS_TRELLIS_SM_CAP", v.c_str());
#else
  setenv("IGNIS_TRELLIS_SM_CAP", v.c_str(), 1);
#endif
  TL_RC(ignis_trellis_prepare());
}

void report(int cap, bool prefill, cudaStream_t stream) {
  set_cap(cap);
  std::printf("\n== SM cap %s ==\n", cap == 0 ? "none (all SMs)" : std::to_string(cap).c_str());
  std::printf("%-15s %6s %7s %3s | %9s %6s | %9s %9s %9s | %6s %6s | %8s %8s\n", "matrix", "in", "out", "K",
              "1 tok us", "%roof", "8 tok us", "16 tok us", "64 tok us", "8/1", "64/1", "1K TF/s", "8K TF/s");
  auto row = [&](const Shape &s) {
    Pool pool;
    make_pool(pool, s);
    const double t1 = time_call(pool, s, 1, stream);
    const double t8 = time_call(pool, s, 8, stream);
    const double t16 = time_call(pool, s, 16, stream);
    const double t64 = time_call(pool, s, 64, stream);
    const double gbs = weight_bytes(s) / (t1 * 1e-6) / 1e9;
    double tf1 = 0.0, tf8 = 0.0;
    if (prefill) {
      const double flop = 2.0 * s.in * s.out * (s.swiglu ? 2 : 1);
      tf1 = flop * 1024 / (time_call(pool, s, 1024, stream) * 1e-6) / 1e12;
      tf8 = flop * 8192 / (time_call(pool, s, 8192, stream) * 1e-6) / 1e12;
    }
    std::printf("%-15s %6u %7u %3.1f | %9.2f %5.1f%% | %9.2f %9.2f %9.2f | %6.3f %6.3f | %8.1f %8.1f\n", s.name, s.in,
                s.out, s.k2 / 2.0, t1, 100.0 * gbs / kRooflineGBs, t8, t16, t64, t8 / t1, t64 / t1, tf1, tf8);
  };
  for (const Shape &s : kBody) row(s);
  for (const Shape &s : kOther) row(s);
  double body_bytes = 0.0, body_us = 0.0, body_us8 = 0.0;
  for (const MixEntry &e : kMix) {
    Pool pool;
    make_pool(pool, e.shape);
    body_bytes += static_cast<double>(weight_bytes(e.shape)) * e.calls;
    body_us += time_call(pool, e.shape, 1, stream) * e.calls;
    body_us8 += time_call(pool, e.shape, 8, stream) * e.calls;
  }
  const double body_gbs = body_bytes / (body_us * 1e-6) / 1e9;
  std::printf("body at the quant's K mix, one token: %.1f MB in %.1f us = %.0f GB/s, %.1f%% of the %.0f GB/s roofline; 8 tokens %.1f us (%.3fx)\n",
              body_bytes / 1e6, body_us, body_gbs, 100.0 * body_gbs / kRooflineGBs, kRooflineGBs, body_us8, body_us8 / body_us);
  if (prefill) {
    double tflops = 0.0, mhz = 0.0;
    peak_f16acc(&tflops, &mhz);
    std::printf("prefill TF/s against the whitepaper's %.0f TFLOPS fp16-accumulate peak; this card's register loop: "
                "%.1f TFLOPS at %.0f MHz\n", kFp16AccTflops, tflops, mhz);
  }
}

}  // namespace

int main(int argc, char **argv) {
  std::vector<int> caps = {0, 36, 70, 84};
  bool prefill = true;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--caps" && i + 1 < argc) {
      caps.clear();
      std::string list = argv[++i];
      size_t pos = 0;
      while (pos <= list.size()) {
        const size_t comma = list.find(',', pos);
        caps.push_back(std::atoi(list.substr(pos, comma - pos).c_str()));
        if (comma == std::string::npos) break;
        pos = comma + 1;
      }
    } else if (a == "--calls" && i + 1 < argc) {
      g_calls = std::atoi(argv[++i]);
    } else if (a == "--reps" && i + 1 < argc) {
      g_reps = std::atoi(argv[++i]);
    } else if (a == "--no-prefill") {
      prefill = false;
    } else {
      std::fprintf(stderr, "usage: %s [--caps 0,36,70,84] [--calls N] [--reps N] [--no-prefill]\n", argv[0]);
      return 2;
    }
  }
  cudaDeviceProp prop{};
  MOE_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::printf("dense trellis linear microbenchmark on %s (%d SMs); streaming read %.0f GB/s, roofline %.0f GB/s\n",
              prop.name, prop.multiProcessorCount, stream_read_gbs(), kRooflineGBs);
  cudaStream_t stream;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (const int cap : caps) report(cap, prefill, stream);
  MOE_CUDA(cudaStreamDestroy(stream));
  return 0;
}
