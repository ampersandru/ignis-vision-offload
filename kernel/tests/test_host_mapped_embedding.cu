// ADR 0062: the 27B's token embedding lives in mapped pinned host memory, and
// the vendored W8 gather reads it there unchanged -- OURS: the allocation
// (ignis_device_host_alloc_mapped, kernel/src/device.cu); the gather is the
// vendored ninfer::ops::embedding.
//
// - The allocation is host memory the runtime knows as such (the load counts
//   it apart from VRAM by asking the same question, kernel/src/model.cu).
// - A W8G32 table at the 27B's width (5120, the generic gather route) gives
//   the same output bit for bit read from the mapped allocation as from
//   device memory, for a decode step's 1 and 8 tokens and a prefill chunk's
//   1,024, and both equal bf16(code * scale) computed on the host.
// - The gather captured in a CUDA graph (the decode graphs) and replayed reads
//   the mapped table the same.
//
// GPU test (ADR 0006): no SKIP_RETURN_CODE, a missing device fails.

#include "ignis_device.h"

#include "ninfer/ops/embedding.h"

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void check(bool ok, const std::string &label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label.c_str());
    ++g_failed;
  }
}

void cuda_ok(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "FATAL: %s: %s\n", what, cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

uint32_t lcg(uint32_t &state) {
  state = state * 1664525U + 1013904223U;
  return state;
}

uint16_t bf16_bits(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  const uint32_t rounding = 0x7FFFU + ((u >> 16) & 1U);
  return static_cast<uint16_t>((u + rounding) >> 16);
}

// The 27B's width; any row count serves the gather.
constexpr int32_t kVocab = 97;
constexpr int32_t kD = 5120;
constexpr int32_t kGroups = kD / 32;
constexpr std::size_t kCodeBytes = static_cast<std::size_t>(kVocab) * kD;
constexpr std::size_t kScaleOffset = (kCodeBytes + 255) / 256 * 256;
constexpr std::size_t kTableBytes = kScaleOffset + static_cast<std::size_t>(kVocab) * kGroups * 2;

// The table at `base` as the leaf binds `text/token_embedding`
// (kernel/src/model.cu, to_weight): row-split, the scale plane 256-aligned
// past the codes.
ninfer::Weight w8_table(const void *base) {
  ninfer::Weight w{};
  w.qtype = ninfer::QType::W8G32_F16S;
  w.layout = ninfer::QuantLayout::RowSplit;
  w.payload = base;
  w.payload_bytes = kTableBytes;
  w.ndim = 2;
  w.shape[0] = kVocab;
  w.shape[1] = kD;
  w.padded_shape[0] = kVocab;
  w.padded_shape[1] = kD;
  w.qdata = base;
  w.scales = static_cast<const uint8_t *>(base) + kScaleOffset;
  w.n = kVocab;
  w.k = kD;
  w.group_size = 32;
  w.group = 32;
  w.scale_dtype = ninfer::DType::FP16;
  return w;
}

// The gather of `ids` (on the device) from the table at `base` into `out`.
void gather(const void *base, int32_t *ids, int32_t tokens, __nv_bfloat16 *out, cudaStream_t stream) {
  const ninfer::Tensor id_tensor(ids, ninfer::DType::I32, {tokens});
  ninfer::Tensor out_tensor(out, ninfer::DType::BF16, {kD, tokens});
  ninfer::ops::embedding(id_tensor, w8_table(base), out_tensor, stream);
}

std::vector<uint16_t> read_back(const __nv_bfloat16 *out, int32_t tokens) {
  std::vector<uint16_t> host(static_cast<std::size_t>(kD) * tokens);
  cuda_ok(cudaMemcpy(host.data(), out, host.size() * 2, cudaMemcpyDeviceToHost), "read back");
  return host;
}

}  // namespace

int main() {
  struct ignis_device *device = ignis_device_create(0);
  if (device == nullptr) {
    std::fprintf(stderr, "FATAL: no CUDA device\n");
    return EXIT_FAILURE;
  }

  // Random codes and FP16 scales, as the artifact stores them.
  std::vector<uint8_t> table(kTableBytes, 0);
  std::vector<float> scales(static_cast<std::size_t>(kVocab) * kGroups);
  uint32_t state = 62;
  for (std::size_t i = 0; i < kCodeBytes; ++i) {
    table[i] = static_cast<uint8_t>(lcg(state) >> 24);
  }
  for (std::size_t g = 0; g < scales.size(); ++g) {
    const __half scale = __float2half(1e-3F + static_cast<float>(lcg(state) >> 8) * 1e-9F);
    scales[g] = __half2float(scale);
    std::memcpy(&table[kScaleOffset + g * 2], &scale, 2);
  }

  void *host = nullptr;
  void *mapped = nullptr;
  if (ignis_device_host_alloc_mapped(device, kTableBytes, &host, &mapped) != 0) {
    std::fprintf(stderr, "FATAL: ignis_device_host_alloc_mapped(%zu)\n", kTableBytes);
    return EXIT_FAILURE;
  }
  check(reinterpret_cast<uintptr_t>(host) % 4096 == 0, "the host side is page-aligned for a direct read");
  std::memcpy(host, table.data(), kTableBytes);
  cudaPointerAttributes attributes{};
  cuda_ok(cudaPointerGetAttributes(&attributes, mapped), "cudaPointerGetAttributes");
  check(attributes.type == cudaMemoryTypeHost, "the mapped table is host memory");
  check(ignis_device_host_alloc_mapped(device, 0, &host, &mapped) == -1, "a zero size is refused");

  void *on_device = nullptr;
  cuda_ok(cudaMalloc(&on_device, kTableBytes), "cudaMalloc table");
  cuda_ok(cudaMemcpy(on_device, table.data(), kTableBytes, cudaMemcpyHostToDevice), "upload table");

  constexpr int32_t kMaxTokens = 1024;
  int32_t *ids = nullptr;
  __nv_bfloat16 *out_device = nullptr;
  __nv_bfloat16 *out_mapped = nullptr;
  cuda_ok(cudaMalloc(&ids, kMaxTokens * sizeof(int32_t)), "cudaMalloc ids");
  cuda_ok(cudaMalloc(&out_device, static_cast<std::size_t>(kD) * kMaxTokens * 2), "cudaMalloc out");
  cuda_ok(cudaMalloc(&out_mapped, static_cast<std::size_t>(kD) * kMaxTokens * 2), "cudaMalloc out");
  cudaStream_t stream = nullptr;
  cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");

  for (int32_t tokens : {1, 8, kMaxTokens}) {
    std::vector<int32_t> host_ids(tokens);
    for (int32_t &id : host_ids) {
      id = static_cast<int32_t>(lcg(state) % kVocab);
    }
    cuda_ok(cudaMemcpy(ids, host_ids.data(), tokens * sizeof(int32_t), cudaMemcpyHostToDevice), "ids");
    try {
      gather(on_device, ids, tokens, out_device, stream);
      gather(mapped, ids, tokens, out_mapped, stream);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "FATAL: embedding (T=%d): %s\n", tokens, e.what());
      return EXIT_FAILURE;
    }
    cuda_ok(cudaStreamSynchronize(stream), "gather");
    const std::vector<uint16_t> from_device = read_back(out_device, tokens);
    const std::vector<uint16_t> from_mapped = read_back(out_mapped, tokens);
    const std::string cell = "T=" + std::to_string(tokens);
    check(from_device == from_mapped, cell + ": the mapped table gathers what the device one does");
    std::size_t wrong = 0;
    for (int32_t t = 0; t < tokens; ++t) {
      const std::size_t row = static_cast<std::size_t>(host_ids[t]);
      for (int32_t k = 0; k < kD; ++k) {
        const float code = static_cast<float>(static_cast<int8_t>(table[row * kD + k]));
        const uint16_t want = bf16_bits(code * scales[row * kGroups + k / 32]);
        wrong += from_mapped[static_cast<std::size_t>(t) * kD + k] != want;
      }
    }
    check(wrong == 0, cell + ": " + std::to_string(wrong) + " elements differ from bf16(code * scale)");
  }

  // The decode graphs capture the gather: a replay reads the mapped table.
  {
    constexpr int32_t kTokens = 8;
    cuda_ok(cudaMemsetAsync(out_device, 0, static_cast<std::size_t>(kD) * kTokens * 2, stream), "clear");
    gather(mapped, ids, kTokens, out_mapped, stream);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    gather(mapped, ids, kTokens, out_device, stream);
    cuda_ok(cudaStreamEndCapture(stream, &graph), "end capture");
    cuda_ok(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    cuda_ok(cudaGraphLaunch(exec, stream), "launch");
    cuda_ok(cudaStreamSynchronize(stream), "replay");
    check(read_back(out_device, kTokens) == read_back(out_mapped, kTokens),
          "a graph replay gathers from the mapped table as an eager launch does");
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
  }

  cudaStreamDestroy(stream);
  cudaFree(out_mapped);
  cudaFree(out_device);
  cudaFree(ids);
  cudaFree(on_device);
  ignis_device_host_free(device, host);
  ignis_device_destroy(device);

  if (g_failed > 0) {
    std::fprintf(stderr, "test_host_mapped_embedding: %d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("test_host_mapped_embedding: ok\n");
  return EXIT_SUCCESS;
}
