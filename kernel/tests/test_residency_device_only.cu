// Expert residency with no host pool -- OURS (ADR 0045's resident load: a card that holds every
// expert keeps none in RAM).
//
// A desc with host_pool_bytes 0 is device-only. Checked here, on a random catalog of synthetic
// records:
//   - creation refuses a class with fewer slots than projections, and needs no pool offsets;
//   - a step, and the warm start, are refused until the admission is finished;
//   - ignis_residency_admit refuses a key admitted twice; ignis_residency_admit_finish refuses
//     while any projection is missing, and succeeds once all are;
//   - every slot-table entry then points at a slot of its own class holding its record's bytes,
//     its K, and no two entries share a slot; occupancy is every projection;
//   - random decode and prefill steps, with and without a lookahead (logits or ranked), and a
//     captured decode round, all hit: status 0, no miss, eviction or prefetch, no byte moved,
//     no stall;
//   - the counters agree at the end.
// Same no-SKIP_RETURN_CODE rule as every GPU test here (ADR 0006).

#include "ignis_residency.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

#define CUDA_OK(expr)                                                                            \
  do {                                                                                           \
    const cudaError_t status_ = (expr);                                                          \
    if (status_ != cudaSuccess) {                                                                \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #expr,                         \
                   cudaGetErrorString(status_));                                                 \
      std::exit(1);                                                                              \
    }                                                                                            \
  } while (0)

int g_failures = 0;

void check(bool ok, const std::string &what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

void ok(int32_t rc, const std::string &what) {
  if (rc != 0) {
    std::fprintf(stderr, "FAIL: %s: %s\n", what.c_str(), ignis_residency_last_error());
    ++g_failures;
  }
}

void refused(int32_t rc, const std::string &what, const char *needle) {
  if (rc == 0) {
    std::fprintf(stderr, "FAIL: %s was accepted\n", what.c_str());
    ++g_failures;
  } else if (std::strstr(ignis_residency_last_error(), needle) == nullptr) {
    std::fprintf(stderr, "FAIL: %s refused as \"%s\", not naming \"%s\"\n", what.c_str(),
                 ignis_residency_last_error(), needle);
    ++g_failures;
  }
}

struct Rng {
  uint64_t s;
  uint64_t next() {
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  uint32_t below(uint32_t n) { return static_cast<uint32_t>(next() % n); }
};

constexpr uint32_t kLayers = 3;
constexpr uint32_t kExperts = 32;
constexpr uint32_t kKeys = kLayers * kExperts * 2;
constexpr uint32_t kMaxRows = 8;
constexpr uint32_t kWidth = 3;
constexpr uint64_t kRecord[IGNIS_RESIDENCY_CLASSES] = {4096, 8192, 8192, 12288, 4096, 4096, 8192, 8192};
constexpr uint8_t kK2[4] = {4, 5, 6, 8};

uint32_t class_of(uint32_t key, uint8_t k2) {
  const uint32_t k = k2 == 4 ? 0 : k2 == 5 ? 1 : k2 == 6 ? 2 : 3;
  return (key & 1) * 4 + k;
}

uint8_t byte_of(uint32_t key, uint64_t i) { return static_cast<uint8_t>((key * 131u + i * 7u + 3u) & 0xFF); }

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::fprintf(stderr, "no CUDA device\n");
    return 1;
  }
  Rng rng{2026};
  std::vector<uint8_t> k2(kKeys);
  uint32_t count[IGNIS_RESIDENCY_CLASSES] = {};
  for (uint32_t key = 0; key < kKeys; ++key) {
    k2[key] = kK2[rng.below(4)];
    ++count[class_of(key, k2[key])];
  }
  uint64_t heaviest = 0;
  for (uint32_t l = 0; l < kLayers; ++l) {
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < 2 * kExperts; ++i) {
      const uint32_t key = l * 2 * kExperts + i;
      bytes += kRecord[class_of(key, k2[key])];
    }
    heaviest = bytes > heaviest ? bytes : heaviest;
  }
  ignis_residency_desc desc{};
  desc.layers = kLayers;
  desc.experts = kExperts;
  for (uint32_t c = 0; c < IGNIS_RESIDENCY_CLASSES; ++c) {
    desc.capacity[c] = count[c];
    desc.record_bytes[c] = kRecord[c];
  }
  desc.max_tokens = kMaxRows;
  desc.lookahead_width = kWidth;
  desc.prefill_lookahead_width = kWidth;
  desc.prefetch_budget_one_row_bytes = IGNIS_RESIDENCY_NO_BUDGET;
  desc.staging_half_bytes = heaviest;
  desc.host_pool_bytes = 0;
  desc.copy_blocks = 4;
  desc.report = 1;

  // A class short of one slot is refused: there would be a projection with nowhere to live.
  {
    ignis_residency_desc short_desc = desc;
    uint32_t c = 0;
    while (count[c] == 0) ++c;
    short_desc.capacity[c] = count[c] - 1;
    ignis_residency *r = nullptr;
    refused(ignis_residency_create(&short_desc, k2.data(), nullptr, &r), "a class short of a slot",
            "needs a slot for every projection");
    check(r == nullptr, "a refused create hands out nothing");
  }

  ignis_residency *r = nullptr;
  ok(ignis_residency_create(&desc, k2.data(), nullptr, &r), "create device-only");
  if (r == nullptr) return 1;
  check(ignis_residency_host_pool(r) == nullptr, "no host pool");

  int32_t *ids = nullptr;
  float *logits = nullptr;
  int32_t *ranked = nullptr;
  CUDA_OK(cudaMalloc(&ids, kMaxRows * IGNIS_MOE_TOP_K * 4));
  CUDA_OK(cudaMalloc(&logits, kMaxRows * kExperts * 4));
  CUDA_OK(cudaMalloc(&ranked, kMaxRows * (kWidth + 1) * 4));
  std::vector<int32_t> host_ids(kMaxRows * IGNIS_MOE_TOP_K);
  for (uint32_t i = 0; i < host_ids.size(); ++i) host_ids[i] = static_cast<int32_t>(i % kExperts);
  CUDA_OK(cudaMemcpy(ids, host_ids.data(), host_ids.size() * 4, cudaMemcpyHostToDevice));

  refused(ignis_residency_step(r, 0, IGNIS_RESIDENCY_DECODE, ids, 1, nullptr, nullptr), "a step before admission",
          "admitted");
  uint32_t warmed = 0;
  const uint32_t one = 0;
  refused(ignis_residency_warm_start(r, &one, 1, &warmed), "a warm start", "no host pool");

  // The source: every record back to back, as a loader's layer buffers would hold them.
  std::vector<uint64_t> offset(kKeys);
  uint64_t total = 0;
  for (uint32_t key = 0; key < kKeys; ++key) {
    offset[key] = total;
    total += kRecord[class_of(key, k2[key])];
  }
  std::vector<uint8_t> src(total);
  for (uint32_t key = 0; key < kKeys; ++key) {
    const uint64_t bytes = kRecord[class_of(key, k2[key])];
    for (uint64_t i = 0; i < bytes; ++i) src[offset[key] + i] = byte_of(key, i);
  }
  // Admitted a layer at a time, in an order that is not key order, minus one key at first.
  std::vector<uint32_t> keys;
  std::vector<uint64_t> offs;
  const uint32_t held_back = kKeys / 2 + 1;
  for (uint32_t l : {2u, 0u, 1u}) {
    keys.clear();
    offs.clear();
    for (uint32_t i = 0; i < 2 * kExperts; ++i) {
      const uint32_t key = l * 2 * kExperts + (2 * kExperts - 1 - i);
      if (key == held_back) continue;
      keys.push_back(key);
      offs.push_back(offset[key]);
    }
    ok(ignis_residency_admit(r, keys.data(), static_cast<uint32_t>(keys.size()), src.data(), offs.data()),
       "admit layer " + std::to_string(l));
  }
  const uint32_t again = 5;
  refused(ignis_residency_admit(r, &again, 1, src.data(), &offset[again]), "a key admitted twice", "twice");
  refused(ignis_residency_admit_finish(r), "a finish with a projection missing", "never admitted");
  refused(ignis_residency_step(r, 0, IGNIS_RESIDENCY_DECODE, ids, 1, nullptr, nullptr),
          "a step after a refused finish", "admitted");
  ok(ignis_residency_admit(r, &held_back, 1, src.data(), &offset[held_back]), "admit the last key");
  ok(ignis_residency_admit_finish(r), "finish");
  refused(ignis_residency_admit_finish(r), "a second finish", "finished");

  // Every entry: its own class's slot, its K, its bytes; no slot shared.
  ignis_residency_layout layout{};
  ok(ignis_residency_get_layout(r, &layout), "layout");
  std::set<const void *> seen;
  std::vector<uint8_t> back;
  for (uint32_t l = 0; l < kLayers; ++l) {
    std::vector<ignis_moe_slot> table(2 * kExperts);
    CUDA_OK(cudaMemcpy(table.data(), ignis_residency_slot_table(r, l), table.size() * sizeof(ignis_moe_slot),
                       cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < 2 * kExperts; ++i) {
      const uint32_t key = l * 2 * kExperts + i;
      const uint32_t c = class_of(key, k2[key]);
      const auto *base = static_cast<const uint8_t *>(layout.pool[c]);
      const auto *at = static_cast<const uint8_t *>(table[i].record);
      const std::string where = "key " + std::to_string(key);
      check(at != nullptr, where + " is resident");
      if (at == nullptr) continue;
      check(at >= base && at + kRecord[c] <= base + layout.pool_bytes[c] && (at - base) % kRecord[c] == 0,
            where + " sits on a slot of its class");
      check(table[i].k2 == k2[key], where + " carries its K");
      check(seen.insert(at).second, where + "'s slot is its own");
      back.resize(kRecord[c]);
      CUDA_OK(cudaMemcpy(back.data(), at, kRecord[c], cudaMemcpyDeviceToHost));
      check(std::memcmp(back.data(), src.data() + offset[key], kRecord[c]) == 0, where + " holds its bytes");
    }
  }
  uint32_t used[IGNIS_RESIDENCY_CLASSES] = {};
  ok(ignis_residency_read_occupancy(r, used), "occupancy");
  for (uint32_t c = 0; c < IGNIS_RESIDENCY_CLASSES; ++c) check(used[c] == count[c], "class " + std::to_string(c) + " full");

  // Steps: everything hits, nothing moves.
  cudaStream_t stream;
  CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  uint64_t selections = 0;
  std::vector<uint32_t> entries(IGNIS_RESIDENCY_LISTS * 4 * kExperts);
  for (uint32_t round = 0; round < 40; ++round) {
    const bool prefill = round % 4 == 3;
    const uint32_t rows = prefill ? 4 + rng.below(5) : 1 + rng.below(3);
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      std::set<uint32_t> distinct;
      for (uint32_t row = 0; row < rows; ++row) {
        std::set<int32_t> picked;
        while (picked.size() < IGNIS_MOE_TOP_K) picked.insert(static_cast<int32_t>(rng.below(kExperts)));
        uint32_t j = 0;
        for (int32_t e : picked) {
          host_ids[row * IGNIS_MOE_TOP_K + j++] = e;
          distinct.insert(static_cast<uint32_t>(e));
        }
      }
      CUDA_OK(cudaMemcpy(ids, host_ids.data(), rows * IGNIS_MOE_TOP_K * 4, cudaMemcpyHostToDevice));
      const uint32_t phase = prefill ? IGNIS_RESIDENCY_PREFILL : IGNIS_RESIDENCY_DECODE;
      const uint32_t look = rng.below(3);
      int32_t rc;
      if (look == 0) {
        rc = ignis_residency_step(r, layer, phase, ids, rows, nullptr, stream);
      } else if (look == 1) {
        std::vector<float> lg(rows * kExperts);
        for (auto &v : lg) v = static_cast<float>(rng.below(9)) - 4.0f;
        CUDA_OK(cudaMemcpy(logits, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice));
        rc = ignis_residency_step(r, layer, phase, ids, rows, logits, stream);
      } else {
        std::vector<int32_t> rk(rows * (kWidth + 1));
        for (auto &v : rk) v = rng.below(5) == 0 ? -1 : static_cast<int32_t>(rng.below(kExperts));
        CUDA_OK(cudaMemcpy(ranked, rk.data(), rk.size() * 4, cudaMemcpyHostToDevice));
        rc = ignis_residency_step_ranked(r, layer, phase, ids, rows, ranked, rows, kWidth + 1, stream);
      }
      const std::string at = "round " + std::to_string(round) + " layer " + std::to_string(layer);
      ok(rc, at + " step");
      CUDA_OK(cudaStreamSynchronize(stream));
      ignis_residency_report head{};
      ok(ignis_residency_last_report(r, layer, &head, entries.data(), 4 * kExperts), at + " report");
      check(head.status == 0, at + " status 0");
      check(head.count[0] == 2 * distinct.size(), at + " every selected projection hits");
      check(head.count[2] == 0 && head.count[3] == 0 && head.count[4] == 0, at + " no miss, eviction or prefetch");
      check(head.bytes_moved == 0, at + " no byte moved");
      selections += 2 * distinct.size();
    }
  }
  // A captured decode round, replayed: the graph path hits as well.
  {
    ok(ignis_residency_join(r, stream), "join before capture");
    for (uint32_t i = 0; i < IGNIS_MOE_TOP_K; ++i) host_ids[i] = static_cast<int32_t>(i);
    CUDA_OK(cudaMemcpy(ids, host_ids.data(), IGNIS_MOE_TOP_K * 4, cudaMemcpyHostToDevice));
    cudaGraph_t graph;
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      ok(ignis_residency_step(r, layer, IGNIS_RESIDENCY_DECODE, ids, 1, layer + 1 < kLayers ? logits : nullptr, stream),
         "captured step");
    }
    CUDA_OK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec;
    CUDA_OK(cudaGraphInstantiate(&exec, graph, 0));
    for (int replay = 0; replay < 3; ++replay) CUDA_OK(cudaGraphLaunch(exec, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      ignis_residency_report head{};
      ok(ignis_residency_last_report(r, layer, &head, entries.data(), 4 * kExperts), "graph report");
      check(head.status == 0 && head.count[2] == 0 && head.bytes_moved == 0, "the graph's steps hit");
    }
    selections += 3ull * kLayers * 2 * IGNIS_MOE_TOP_K;
    CUDA_OK(cudaGraphExecDestroy(exec));
    CUDA_OK(cudaGraphDestroy(graph));
  }
  ignis_residency_counters counters{};
  ok(ignis_residency_read_counters(r, &counters), "counters");
  uint64_t hits = 0, misses = 0;
  for (uint32_t c = 0; c < IGNIS_RESIDENCY_CLASSES; ++c) {
    for (uint32_t p = 0; p < 2; ++p) {
      hits += counters.hits[c][p];
      misses += counters.misses[c][p];
    }
  }
  check(hits == selections, "counted hits " + std::to_string(hits) + " of " + std::to_string(selections));
  check(misses == 0 && counters.prefetch_issued == 0, "no miss or prefetch counted");
  check(counters.bytes_moved[0] == 0 && counters.bytes_moved[1] == 0, "no byte counted");
  check(counters.stall_nanos[0] == 0 && counters.stall_nanos[1] == 0, "no stall");

  CUDA_OK(cudaStreamDestroy(stream));
  CUDA_OK(cudaFree(ids));
  CUDA_OK(cudaFree(logits));
  CUDA_OK(cudaFree(ranked));
  ignis_residency_free(r);
  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("device-only residency: ok (%u keys, %llu selections)\n", kKeys,
              static_cast<unsigned long long>(selections));
  return 0;
}
