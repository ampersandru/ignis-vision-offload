// ADR 0062: what the vendored W8 gather costs reading the 27B's token
// embedding from mapped pinned host memory against device memory -- the
// whole [248320, 5120] table, ids spread over the vocabulary, at a decode
// step's 1 and 8 tokens, a verify round's 64 and a prefill chunk's 1,024.
// A tool, NOT a CTest: it prints microseconds per gather (the median of
// 200 launches, cudaEvent-timed on one stream).
//
//   ignis_kernel_host_mapped_embedding_bench.exe

#include "ignis_device.h"

#include "ninfer/ops/embedding.h"

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void cuda_ok(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "FATAL: %s: %s\n", what, cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

constexpr int32_t kVocab = 248320;
constexpr int32_t kD = 5120;
constexpr std::size_t kCodeBytes = static_cast<std::size_t>(kVocab) * kD;
constexpr std::size_t kScaleOffset = (kCodeBytes + 255) / 256 * 256;
constexpr std::size_t kTableBytes = kScaleOffset + static_cast<std::size_t>(kVocab) * (kD / 32) * 2;

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

// The median microseconds of `reps` gathers of `tokens` ids from `base`.
double median_micros(const void *base, int32_t *ids, int32_t tokens, __nv_bfloat16 *out, cudaStream_t stream,
                     int reps) {
  const ninfer::Tensor id_tensor(ids, ninfer::DType::I32, {tokens});
  ninfer::Tensor out_tensor(out, ninfer::DType::BF16, {kD, tokens});
  const ninfer::Weight table = w8_table(base);
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cuda_ok(cudaEventCreate(&start), "event");
  cuda_ok(cudaEventCreate(&stop), "event");
  std::vector<float> micros;
  for (int r = 0; r < reps + 10; ++r) {
    cuda_ok(cudaEventRecord(start, stream), "record");
    ninfer::ops::embedding(id_tensor, table, out_tensor, stream);
    cuda_ok(cudaEventRecord(stop, stream), "record");
    cuda_ok(cudaEventSynchronize(stop), "sync");
    float ms = 0.0F;
    cuda_ok(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    if (r >= 10) {
      micros.push_back(ms * 1000.0F);
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::sort(micros.begin(), micros.end());
  return micros[micros.size() / 2];
}

}  // namespace

int main() {
  struct ignis_device *device = ignis_device_create(0);
  if (device == nullptr) {
    std::fprintf(stderr, "FATAL: no CUDA device\n");
    return EXIT_FAILURE;
  }
  void *host = nullptr;
  void *mapped = nullptr;
  if (ignis_device_host_alloc_mapped(device, kTableBytes, &host, &mapped) != 0) {
    std::fprintf(stderr, "FATAL: ignis_device_host_alloc_mapped\n");
    return EXIT_FAILURE;
  }
  uint32_t state = 62;
  auto *bytes = static_cast<uint8_t *>(host);
  for (std::size_t i = 0; i < kTableBytes; i += 4096) {
    state = state * 1664525U + 1013904223U;
    bytes[i] = static_cast<uint8_t>(state >> 24);
  }
  void *on_device = nullptr;
  cuda_ok(cudaMalloc(&on_device, kTableBytes), "cudaMalloc table");
  cuda_ok(cudaMemcpy(on_device, host, kTableBytes, cudaMemcpyHostToDevice), "upload");

  constexpr int32_t kMaxTokens = 1024;
  std::vector<int32_t> host_ids(kMaxTokens);
  for (int32_t &id : host_ids) {
    state = state * 1664525U + 1013904223U;
    id = static_cast<int32_t>(state % kVocab);
  }
  int32_t *ids = nullptr;
  __nv_bfloat16 *out = nullptr;
  cuda_ok(cudaMalloc(&ids, kMaxTokens * sizeof(int32_t)), "ids");
  cuda_ok(cudaMemcpy(ids, host_ids.data(), kMaxTokens * sizeof(int32_t), cudaMemcpyHostToDevice), "ids");
  cuda_ok(cudaMalloc(&out, static_cast<std::size_t>(kD) * kMaxTokens * 2), "out");
  cudaStream_t stream = nullptr;
  cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");

  std::printf("tokens  device_us  mapped_us  mapped_bytes\n");
  for (int32_t tokens : {1, 8, 64, 1024}) {
    const double device_us = median_micros(on_device, ids, tokens, out, stream, 200);
    const double mapped_us = median_micros(mapped, ids, tokens, out, stream, 200);
    std::printf("%6d  %9.1f  %9.1f  %12zu\n", tokens, device_us, mapped_us,
                static_cast<std::size_t>(tokens) * (kD + kD / 32 * 2));
  }

  cudaStreamDestroy(stream);
  cudaFree(out);
  cudaFree(ids);
  cudaFree(on_device);
  ignis_device_host_free(device, host);
  ignis_device_destroy(device);
  return EXIT_SUCCESS;
}
