// Expert residency with a pinned hot set -- OURS (GitHub #329 item 12: the calibration-hottest
// experts held in VRAM for the load's life, outside the LRU, with no copy in the host pool).
//
// A desc naming `pinned` slots per class, beside a host pool whose offsets mark the pinned keys
// IGNIS_RESIDENCY_NO_HOST_COPY. Checked here, on a random catalog of synthetic records:
//   - creation refuses pinned counts that do not match the marked keys, and pins on a device-only
//     desc; the plan's and the layout's pools hold the pinned slots past the LRU's;
//   - a step and the warm start are refused until the pinned keys are admitted; an unpinned key's
//     admission and a second one of a pinned key are refused, and so is a finish with one missing;
//   - every pinned key then sits on a slot of its class past the LRU's capacity, holding its
//     record's bytes, its K, no slot shared; the warm start skips them and fills the LRU alone;
//   - random decode and prefill steps (whole, with logits or a ranked lookahead; split, plain or
//     routed; a captured decode round) that keep selecting pinned experts: a selected pinned key
//     is always a hit, and never a prefetch hit, a miss, an eviction, a prefetch or a drop; every
//     other resident key stays in the LRU's slots; and at the end every pinned entry still points
//     at its slot and its bytes, the LRU's occupancy never counted them, and the counters agree.
// The default path, no key pinned, is what test_residency_trace.cu replays against the policy
// model's committed outcomes. What a step reads goes up on the step's stream (GitHub #324). Same
// no-SKIP_RETURN_CODE rule as every GPU test here (ADR 0006).

#include "ignis_residency.h"

#include <cuda_runtime.h>

#include <algorithm>
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

void upload(void *dst, const void *src, size_t bytes, cudaStream_t stream) {
  CUDA_OK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
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
constexpr uint32_t kExperts = 64;
constexpr uint32_t kKeys = kLayers * kExperts * 2;
constexpr uint32_t kMaxRows = 8;
constexpr uint32_t kWidth = 3;
constexpr uint32_t kLru = 24;            // LRU slots of every populated class
constexpr uint32_t kHotExperts = 4;      // experts 0..3 of every layer are pinned whole
constexpr uint64_t kRecord[IGNIS_RESIDENCY_CLASSES] = {4096, 8192, 8192, 12288, 4096, 4096, 8192, 8192};
constexpr uint8_t kK2[4] = {4, 5, 6, 8};

uint32_t class_of(uint32_t key, uint8_t k2) {
  const uint32_t k = k2 == 4 ? 0 : k2 == 5 ? 1 : k2 == 6 ? 2 : 3;
  return (key & 1) * 4 + k;
}

uint8_t byte_of(uint32_t key, uint64_t i) { return static_cast<uint8_t>((key * 131u + i * 7u + 3u) & 0xFF); }

uint64_t round_up(uint64_t v) { return (v + 255) / 256 * 256; }

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::fprintf(stderr, "no CUDA device\n");
    return 1;
  }
  Rng rng{329};
  std::vector<uint8_t> k2(kKeys);
  std::vector<uint8_t> pinned(kKeys, 0);
  uint32_t count[IGNIS_RESIDENCY_CLASSES] = {};
  uint32_t pins[IGNIS_RESIDENCY_CLASSES] = {};
  for (uint32_t key = 0; key < kKeys; ++key) {
    k2[key] = kK2[rng.below(4)];
    ++count[class_of(key, k2[key])];
    // Whole hot experts, and a few single projections besides.
    pinned[key] = ((key / 2) % kExperts) < kHotExperts || rng.below(40) == 0;
    if (pinned[key]) ++pins[class_of(key, k2[key])];
  }
  // The host pool: every unpinned record back to back; a pinned one has no place there.
  std::vector<uint64_t> offsets(kKeys);
  uint64_t pool_bytes = 0;
  uint64_t heaviest = 0;
  for (uint32_t l = 0; l < kLayers; ++l) {
    uint64_t layer_bytes = 0;
    for (uint32_t i = 0; i < 2 * kExperts; ++i) {
      const uint32_t key = l * 2 * kExperts + i;
      const uint64_t bytes = kRecord[class_of(key, k2[key])];
      layer_bytes += bytes;
      if (pinned[key]) {
        offsets[key] = IGNIS_RESIDENCY_NO_HOST_COPY;
      } else {
        offsets[key] = pool_bytes;
        pool_bytes += bytes;
      }
    }
    heaviest = std::max(heaviest, layer_bytes);
  }
  ignis_residency_desc desc{};
  desc.layers = kLayers;
  desc.experts = kExperts;
  for (uint32_t c = 0; c < IGNIS_RESIDENCY_CLASSES; ++c) {
    desc.capacity[c] = count[c] == 0 ? 0 : kLru;
    desc.record_bytes[c] = kRecord[c];
    desc.pinned[c] = pins[c];
  }
  desc.max_tokens = kMaxRows;
  desc.lookahead_width = kWidth;
  desc.prefill_lookahead_width = kWidth;
  desc.prefetch_budget_one_row_bytes = 3 * 8192;
  desc.prefetch_budget_per_row_bytes = 8192;
  desc.staging_half_bytes = heaviest;
  desc.host_pool_bytes = pool_bytes;
  desc.copy_blocks = 4;
  desc.report = 1;

  uint32_t some = 0;
  while (pins[some] == 0) ++some;
  // Pinned counts that are not the marked keys' are refused.
  {
    ignis_residency_desc wrong = desc;
    ++wrong.pinned[some];
    ignis_residency *r = nullptr;
    refused(ignis_residency_create(&wrong, k2.data(), offsets.data(), &r), "a pinned count past the marked keys",
            "pinned");
    check(r == nullptr, "a refused create hands out nothing");
    std::vector<uint64_t> unmarked = offsets;
    for (uint32_t key = 0; key < kKeys; ++key) {
      if (pinned[key]) unmarked[key] = 0;
    }
    refused(ignis_residency_create(&desc, k2.data(), unmarked.data(), &r), "pinned slots with no key marked",
            "pinned");
  }
  // A device-only residency already holds every projection: it pins nothing.
  {
    ignis_residency_desc device_only = desc;
    device_only.host_pool_bytes = 0;
    for (uint32_t c = 0; c < IGNIS_RESIDENCY_CLASSES; ++c) device_only.capacity[c] = count[c];
    ignis_residency *r = nullptr;
    refused(ignis_residency_create(&device_only, k2.data(), nullptr, &r), "pins on a device-only residency",
            "pinned");
  }

  ignis_residency_plan plan{};
  ok(ignis_residency_plan_bytes(&desc, &plan), "plan bytes");
  uint64_t pools = 0;
  for (uint32_t c = 0; c < IGNIS_RESIDENCY_CLASSES; ++c) pools += round_up((desc.capacity[c] + pins[c]) * kRecord[c]);
  check(plan.pools == pools, "the plan's pools hold the pinned slots: " + std::to_string(plan.pools) + " of " +
                                 std::to_string(pools));

  ignis_residency *r = nullptr;
  ok(ignis_residency_create(&desc, k2.data(), offsets.data(), &r), "create with pinned keys");
  if (r == nullptr) return 1;
  ignis_residency_layout layout{};
  ok(ignis_residency_get_layout(r, &layout), "layout");
  for (uint32_t c = 0; c < IGNIS_RESIDENCY_CLASSES; ++c) {
    check(layout.pool_bytes[c] == (desc.capacity[c] + pins[c]) * kRecord[c], "class " + std::to_string(c) + "'s pool");
  }
  {
    auto *host = static_cast<uint8_t *>(ignis_residency_host_pool(r));
    for (uint32_t key = 0; key < kKeys; ++key) {
      if (pinned[key]) continue;
      for (uint64_t i = 0; i < kRecord[class_of(key, k2[key])]; ++i) host[offsets[key] + i] = byte_of(key, i);
    }
  }

  cudaStream_t stream;
  CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  int32_t *ids = nullptr;
  float *weights = nullptr;
  float *logits = nullptr;
  float *router_logits = nullptr;
  int32_t *ranked = nullptr;
  CUDA_OK(cudaMalloc(&ids, kMaxRows * IGNIS_MOE_TOP_K * 4));
  CUDA_OK(cudaMalloc(&weights, kMaxRows * IGNIS_MOE_TOP_K * 4));
  CUDA_OK(cudaMalloc(&logits, kMaxRows * kExperts * 4));
  CUDA_OK(cudaMalloc(&router_logits, kMaxRows * IGNIS_MOE_EXPERTS * 4));
  CUDA_OK(cudaMalloc(&ranked, kMaxRows * (kWidth + 1) * 4));
  std::vector<int32_t> host_ids(kMaxRows * IGNIS_MOE_TOP_K);
  for (uint32_t i = 0; i < host_ids.size(); ++i) host_ids[i] = static_cast<int32_t>(i % kExperts);
  upload(ids, host_ids.data(), host_ids.size() * 4, stream);

  // Nothing steps or warms before the pinned keys are in.
  refused(ignis_residency_step(r, 0, IGNIS_RESIDENCY_DECODE, ids, 1, nullptr, stream), "a step before admission",
          "admitted");
  std::vector<uint32_t> hottest;
  for (uint32_t key = 0; key < kKeys; ++key) hottest.push_back(key);
  uint32_t warmed = 0;
  refused(ignis_residency_warm_start(r, hottest.data(), kKeys, &warmed), "a warm start before admission", "admitted");

  // The pinned records, back to back in a buffer of their own, as the loader streams them.
  std::vector<uint64_t> src_offset(kKeys, 0);
  std::vector<uint8_t> src;
  for (uint32_t key = 0; key < kKeys; ++key) {
    if (!pinned[key]) continue;
    src_offset[key] = src.size();
    for (uint64_t i = 0; i < kRecord[class_of(key, k2[key])]; ++i) src.push_back(byte_of(key, i));
  }
  uint32_t unpinned = 0;
  while (pinned[unpinned]) ++unpinned;
  refused(ignis_residency_admit(r, &unpinned, 1, src.data(), &src_offset[unpinned]), "an unpinned key's admission",
          "host copy");
  uint32_t held_back = kKeys - 1;
  while (!pinned[held_back]) --held_back;
  for (uint32_t l : {1u, 2u, 0u}) {
    std::vector<uint32_t> keys;
    std::vector<uint64_t> offs;
    for (uint32_t i = 0; i < 2 * kExperts; ++i) {
      const uint32_t key = l * 2 * kExperts + (2 * kExperts - 1 - i);
      if (!pinned[key] || key == held_back) continue;
      keys.push_back(key);
      offs.push_back(src_offset[key]);
    }
    ok(ignis_residency_admit(r, keys.data(), static_cast<uint32_t>(keys.size()), src.data(), offs.data()),
       "admit layer " + std::to_string(l) + "'s pinned keys");
  }
  {
    uint32_t again = 0;
    while (!pinned[again] || again == held_back) ++again;
    refused(ignis_residency_admit(r, &again, 1, src.data(), &src_offset[again]), "a pinned key admitted twice",
            "twice");
  }
  refused(ignis_residency_admit_finish(r), "a finish with a pinned key missing", "never admitted");
  refused(ignis_residency_step(r, 0, IGNIS_RESIDENCY_DECODE, ids, 1, nullptr, stream), "a step after a refused finish",
          "admitted");
  ok(ignis_residency_admit(r, &held_back, 1, src.data(), &src_offset[held_back]), "admit the last pinned key");
  ok(ignis_residency_admit_finish(r), "finish");
  refused(ignis_residency_admit_finish(r), "a second finish", "finished");

  // Every pinned entry: a slot of its class past the LRU's, its K, its bytes; no slot shared.
  std::vector<const void *> pinned_at(kKeys, nullptr);
  const auto read_tables = [&]() {
    std::vector<ignis_moe_slot> all(kKeys);
    for (uint32_t l = 0; l < kLayers; ++l) {
      CUDA_OK(cudaMemcpy(all.data() + l * 2 * kExperts, ignis_residency_slot_table(r, l),
                         2 * kExperts * sizeof(ignis_moe_slot), cudaMemcpyDeviceToHost));
    }
    return all;
  };
  const auto holds_its_bytes = [&](uint32_t key, const void *at) {
    const uint32_t c = class_of(key, k2[key]);
    std::vector<uint8_t> back(kRecord[c]);
    CUDA_OK(cudaMemcpy(back.data(), at, kRecord[c], cudaMemcpyDeviceToHost));
    for (uint64_t i = 0; i < kRecord[c]; ++i) {
      if (back[i] != byte_of(key, i)) return false;
    }
    return true;
  };
  // Where an entry sits in its class's pool: its slot index, or -1 outside it.
  const auto slot_of = [&](uint32_t key, const void *at) -> int64_t {
    const uint32_t c = class_of(key, k2[key]);
    const auto *base = static_cast<const uint8_t *>(layout.pool[c]);
    const auto *p = static_cast<const uint8_t *>(at);
    if (p < base || p + kRecord[c] > base + layout.pool_bytes[c] || (p - base) % kRecord[c] != 0) return -1;
    return (p - base) / static_cast<int64_t>(kRecord[c]);
  };
  {
    const std::vector<ignis_moe_slot> table = read_tables();
    std::set<const void *> seen;
    for (uint32_t key = 0; key < kKeys; ++key) {
      const std::string where = "key " + std::to_string(key);
      if (!pinned[key]) {
        check(table[key].record == nullptr, where + " is not resident before the warm start");
        continue;
      }
      const uint32_t c = class_of(key, k2[key]);
      pinned_at[key] = table[key].record;
      const int64_t slot = slot_of(key, table[key].record);
      check(slot >= static_cast<int64_t>(desc.capacity[c]), where + " sits past its class's LRU slots");
      check(table[key].k2 == k2[key], where + " carries its K");
      check(seen.insert(table[key].record).second, where + "'s slot is its own");
      check(holds_its_bytes(key, table[key].record), where + " holds its bytes");
    }
  }

  // The warm start fills the LRU alone: a pinned key is resident already.
  ok(ignis_residency_warm_start(r, hottest.data(), kKeys, &warmed), "warm start");
  uint32_t used[IGNIS_RESIDENCY_CLASSES] = {};
  ok(ignis_residency_read_occupancy(r, used), "occupancy");
  uint32_t lru_total = 0;
  for (uint32_t c = 0; c < IGNIS_RESIDENCY_CLASSES; ++c) {
    check(used[c] == std::min(desc.capacity[c], count[c] - pins[c]), "class " + std::to_string(c) + "'s LRU is full");
    lru_total += used[c];
  }
  check(warmed == lru_total, "the warm start admitted only unpinned keys: " + std::to_string(warmed));

  uint64_t pinned_selections = 0;
  std::vector<uint32_t> entries(IGNIS_RESIDENCY_LISTS * 4 * kExperts);
  // `rows` rows of distinct ids, a pinned expert in most rows; the distinct experts they name.
  const auto random_ids = [&](uint32_t rows) {
    std::set<uint32_t> distinct;
    for (uint32_t row = 0; row < rows; ++row) {
      std::set<int32_t> picked;
      if (rng.below(4) != 0) picked.insert(static_cast<int32_t>(rng.below(kHotExperts)));
      while (picked.size() < IGNIS_MOE_TOP_K) picked.insert(static_cast<int32_t>(rng.below(kExperts)));
      uint32_t j = 0;
      for (int32_t e : picked) {
        host_ids[row * IGNIS_MOE_TOP_K + j++] = e;
        distinct.insert(static_cast<uint32_t>(e));
      }
    }
    upload(ids, host_ids.data(), rows * IGNIS_MOE_TOP_K * 4, stream);
    return distinct;
  };
  const auto random_logits = [&](float *dst, uint32_t rows, uint32_t experts) {
    std::vector<float> lg(rows * experts);
    for (uint32_t i = 0; i < lg.size(); ++i) {
      const uint32_t e = i % experts;
      // Hot experts rank high now and then, so the lookahead names pinned keys too.
      lg[i] = e >= kExperts ? -100.0f : static_cast<float>(rng.below(9)) - 4.0f + (e < kHotExperts ? 2.0f : 0.0f);
    }
    upload(dst, lg.data(), lg.size() * 4, stream);
  };
  const auto random_ranked = [&](uint32_t rows) {
    std::vector<int32_t> rk(rows * (kWidth + 1));
    for (auto &v : rk) v = rng.below(5) == 0 ? -1 : static_cast<int32_t>(rng.below(rng.below(2) ? kHotExperts : kExperts));
    upload(ranked, rk.data(), rk.size() * 4, stream);
  };
  // The layer's report after its step: a selected pinned key is a hit and nothing else; nothing
  // else touches a pinned key.
  const auto pins_held = [&](uint32_t layer, const std::set<uint32_t> &distinct, const std::string &at) {
    ignis_residency_report head{};
    ok(ignis_residency_last_report(r, layer, &head, entries.data(), 4 * kExperts), at + " report");
    if (head.status != 0) return;  // a refused decode step changed nothing
    std::set<uint32_t> hits;
    for (uint32_t i = 0; i < head.count[0]; ++i) hits.insert(entries[i]);
    for (uint32_t e : distinct) {
      for (uint32_t p = 0; p < 2; ++p) {
        const uint32_t key = (layer * kExperts + e) * 2 + p;
        if (!pinned[key]) continue;
        check(hits.count(key) == 1, at + " pinned key " + std::to_string(key) + " hits");
        ++pinned_selections;
      }
    }
    for (uint32_t l = 1; l < IGNIS_RESIDENCY_LISTS; ++l) {
      for (uint32_t i = 0; i < head.count[l]; ++i) {
        const uint32_t key = entries[l * 4 * kExperts + i] & ~IGNIS_RESIDENCY_STAGING_BIT;
        check(!pinned[key], at + " list " + std::to_string(l) + " names pinned key " + std::to_string(key));
      }
    }
  };

  // Whole steps.
  for (uint32_t round = 0; round < 60; ++round) {
    const bool prefill = round % 4 == 3;
    const uint32_t rows = prefill ? 4 + rng.below(5) : 1 + rng.below(3);
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      const std::set<uint32_t> distinct = random_ids(rows);
      const uint32_t phase = prefill ? IGNIS_RESIDENCY_PREFILL : IGNIS_RESIDENCY_DECODE;
      const uint32_t look = rng.below(3);
      int32_t rc;
      if (look == 0) {
        rc = ignis_residency_step(r, layer, phase, ids, rows, nullptr, stream);
      } else if (look == 1) {
        random_logits(logits, rows, kExperts);
        rc = ignis_residency_step(r, layer, phase, ids, rows, logits, stream);
      } else {
        random_ranked(rows);
        rc = ignis_residency_step_ranked(r, layer, phase, ids, rows, ranked, rows, kWidth + 1, stream);
      }
      const std::string at = "round " + std::to_string(round) + " layer " + std::to_string(layer);
      ok(rc, at + " step");
      CUDA_OK(cudaStreamSynchronize(stream));
      pins_held(layer, distinct, at);
    }
  }

  // Split steps: the demand half (from ids, or routed from the router's logits over every expert
  // of IGNIS_MOE_EXPERTS, the hot ones and this catalog's only), then a prefetch half.
  for (uint32_t round = 0; round < 24; ++round) {
    const bool prefill = round % 4 == 3;
    const uint32_t rows = prefill ? 4 + rng.below(5) : 1 + rng.below(3);
    const uint32_t phase = prefill ? IGNIS_RESIDENCY_PREFILL : IGNIS_RESIDENCY_DECODE;
    const bool routed = round % 2 == 1;
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      std::set<uint32_t> distinct;
      const bool ranked_look = rng.below(2) == 0;
      if (ranked_look) {
        random_ranked(rows);
      } else {
        random_logits(logits, rows, kExperts);
      }
      void *look = nullptr;
      int32_t rc;
      if (routed) {
        random_logits(router_logits, rows, IGNIS_MOE_EXPERTS);
        rc = ignis_residency_step_demand_routed(r, layer, phase, router_logits, ids, weights, rows, stream, &look);
      } else {
        distinct = random_ids(rows);
        rc = ignis_residency_step_demand(r, layer, phase, ids, rows, stream, &look);
      }
      const std::string at = std::string(routed ? "routed" : "split") + " round " + std::to_string(round) +
                             " layer " + std::to_string(layer);
      ok(rc, at + " demand");
      if (look != nullptr) {
        ok(ranked_look ? ignis_residency_step_prefetch_ranked(r, ranked, rows, kWidth + 1)
                       : ignis_residency_step_prefetch(r, logits, rows),
           at + " prefetch");
      }
      ok(ignis_residency_join(r, stream), at + " join");
      CUDA_OK(cudaStreamSynchronize(stream));
      if (routed) {
        std::vector<int32_t> selected(rows * IGNIS_MOE_TOP_K);
        CUDA_OK(cudaMemcpy(selected.data(), ids, selected.size() * 4, cudaMemcpyDeviceToHost));
        for (int32_t e : selected) distinct.insert(static_cast<uint32_t>(e));
      }
      pins_held(layer, distinct, at);
    }
  }

  // A captured decode round over the hot experts, replayed.
  {
    ok(ignis_residency_join(r, stream), "join before capture");
    for (uint32_t i = 0; i < IGNIS_MOE_TOP_K; ++i) host_ids[i] = static_cast<int32_t>(i);
    upload(ids, host_ids.data(), IGNIS_MOE_TOP_K * 4, stream);
    random_logits(logits, 1, kExperts);
    CUDA_OK(cudaStreamSynchronize(stream));
    cudaGraph_t graph;
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      ok(ignis_residency_step(r, layer, IGNIS_RESIDENCY_DECODE, ids, 1, layer + 1 < kLayers ? logits : nullptr, stream),
         "captured step");
    }
    CUDA_OK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec;
    CUDA_OK(cudaGraphInstantiate(&exec, graph, 0));
    std::set<uint32_t> distinct;
    for (uint32_t i = 0; i < IGNIS_MOE_TOP_K; ++i) distinct.insert(i);
    for (int replay = 0; replay < 3; ++replay) {
      CUDA_OK(cudaGraphLaunch(exec, stream));
      CUDA_OK(cudaStreamSynchronize(stream));
      for (uint32_t layer = 0; layer < kLayers; ++layer) pins_held(layer, distinct, "graph replay " + std::to_string(replay));
    }
    CUDA_OK(cudaGraphExecDestroy(exec));
    CUDA_OK(cudaGraphDestroy(graph));
  }

  // The pins never moved; everything else resident lives in the LRU's slots.
  {
    const std::vector<ignis_moe_slot> table = read_tables();
    for (uint32_t key = 0; key < kKeys; ++key) {
      const std::string where = "after the steps, key " + std::to_string(key);
      const uint32_t c = class_of(key, k2[key]);
      if (pinned[key]) {
        check(table[key].record == pinned_at[key], where + " is on its pinned slot");
        check(holds_its_bytes(key, table[key].record), where + " holds its bytes");
      } else if (table[key].record != nullptr) {
        const int64_t slot = slot_of(key, table[key].record);
        const bool staged = slot < 0;  // a prefill's staging, released at the next step
        check(staged || slot < static_cast<int64_t>(desc.capacity[c]), where + " sits on an LRU slot");
      }
    }
  }
  ok(ignis_residency_read_occupancy(r, used), "occupancy");
  for (uint32_t c = 0; c < IGNIS_RESIDENCY_CLASSES; ++c) {
    check(used[c] <= desc.capacity[c], "class " + std::to_string(c) + "'s occupancy counts no pin");
  }
  ignis_residency_counters counters{};
  ok(ignis_residency_read_counters(r, &counters), "counters");
  uint64_t hits = 0;
  for (uint32_t c = 0; c < IGNIS_RESIDENCY_CLASSES; ++c) hits += counters.hits[c][0] + counters.hits[c][1];
  check(hits >= pinned_selections, "every pinned use counted as a hit");
  check(pinned_selections > 0, "the steps selected pinned keys");

  CUDA_OK(cudaStreamDestroy(stream));
  CUDA_OK(cudaFree(ids));
  CUDA_OK(cudaFree(weights));
  CUDA_OK(cudaFree(logits));
  CUDA_OK(cudaFree(router_logits));
  CUDA_OK(cudaFree(ranked));
  ignis_residency_free(r);
  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("pinned residency: ok (%u keys, %llu pinned selections)\n", kKeys,
              static_cast<unsigned long long>(pinned_selections));
  return 0;
}
