// GitHub #336 (spec 27b-16gb/04 solution 1): the 27B's decode lanes as a
// load option -- OURS, not vendored.
//
// Two things are pinned here, on a topology with no decoder layer:
//
// 1. Host-only, through `ignis_model_plan_reservations`: the option is
//    validated (0 is the leaf's 8 and plans byte for byte as 8, 9 is refused
//    by name), and the round lines a load without a draft window has --
//    sampling and the decode round -- are, at every lane count, the formula
//    the load allocates by. With no layer, each is the round's own staging
//    and activations. A draft window needs a GDN layer (its replay records),
//    so the verify and drafter lines, the layers' scratch and the lanes'
//    state are a real load's: crates/core/src/speculation.rs states the
//    drafter's line at every lane count, and the artifact's plan at 1, 2 and
//    8 lanes is pinned by crates/runtime/tests/vram_plan_gpu.rs.
// 2. On the device, three BF16 tensors loaded: a load of L lanes holds
//    what its plan laid out, captures its decode graphs at widths 1..L and
//    no wider, and refuses a round of L+1 by name -- its round buffers are L
//    lanes wide, and a wider round would run past them. The 27B's verify
//    round is refused the same way (the check comes first in
//    `ignis_program_decode`), on the artifact in crates/core/tests/
//    decode_lanes_gpu.rs.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so nothing here can
// read as a skip; a host without a device fails part 2.

#include "ignis_model.h"
#include "ignis_seq.h"
#include "ignis_step.h"

#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
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

bool contains(const std::string &haystack, const std::string &needle) {
  return haystack.find(needle) != std::string::npos;
}

ignis_topology no_layer_topology(uint64_t hidden, uint64_t vocab) {
  ignis_topology topology{};
  topology.num_layers = 0;
  topology.layer_kinds = nullptr;
  topology.hidden = hidden;
  topology.vocab = vocab;
  topology.ffn_intermediate = 512;
  topology.gdn_num_layers = 0;
  topology.rms_norm_eps = 1e-6F;
  return topology;
}

ignis_model_load_options lanes_option(uint32_t lanes) {
  ignis_model_load_options options{};
  options.size = sizeof(options);
  options.decode_lanes = lanes;
  return options;
}

// --- 1. the plan, host-only, at the 27B's own widths -------------------------

constexpr uint64_t kHidden = 5120;
constexpr uint64_t kVocab = 248320;

struct Named {
  std::string name;
  std::vector<int32_t> shape;
};

// Plans the 27B's three text tensors (descriptors only, no data) at `lanes`:
// "" and `*out` on success, the leaf's message otherwise.
std::string plan_27b(uint32_t lanes, ignis_model_reservations *out) {
  const auto h = static_cast<int32_t>(kHidden);
  const auto v = static_cast<int32_t>(kVocab);
  const std::vector<Named> named{
      {"text/token_embedding", {v, h}}, {"text/final_norm", {h}}, {"text/output_head", {v, h}}};
  std::vector<ignis_bound_tensor> tensors;
  for (const Named &n : named) {
    ignis_bound_tensor t{};
    t.name = n.name.c_str();
    t.ndim = static_cast<uint32_t>(n.shape.size());
    for (std::size_t i = 0; i < 4; ++i) {
      t.shape[i] = i < n.shape.size() ? n.shape[i] : 1;
      t.padded_shape[i] = t.shape[i];
    }
    tensors.push_back(t);
  }
  const ignis_topology topology = no_layer_topology(kHidden, kVocab);
  const ignis_model_load_options options = lanes_option(lanes);
  *out = ignis_model_reservations{};
  const int32_t rc = ignis_model_plan_reservations(tensors.data(), tensors.size(), &topology,
                                                   /*prefill_chunk_tokens=*/128, /*max_context_tokens=*/4096,
                                                   IGNIS_KV_FORMAT_BF16, &options, out);
  return rc == 0 ? std::string() : std::string(ignis_model_last_error());
}

// kernel/src/model.cu's arena rounding: every allocation from an arena is a
// whole number of 256-byte blocks.
uint64_t aligned(uint64_t bytes) {
  return (bytes + 255) / 256 * 256;
}

// What the two round lines are at `lanes`, as the load allocates them
// (kernel/src/model.cu `plan_load_sizes`, `LoadSizes::reservations`).
uint64_t sampling_line(uint64_t lanes) {
  const uint64_t configs = sizeof(ninfer::ops::SamplingConfig) * (1 + lanes);
  const uint64_t positions_and_out = 2 * sizeof(int32_t) * (1 + lanes);
  const uint64_t permitted = sizeof(int32_t) * lanes * IGNIS_MAX_PERMITTED_TOKENS + sizeof(int32_t) * lanes +
                             sizeof(float) * lanes;
  const uint64_t logits = kVocab * lanes * 2;
  const uint64_t workspace = std::max<uint64_t>(
      ninfer::ops::sampling_workspace_capacity_bytes(static_cast<int32_t>(kVocab), 1, static_cast<int32_t>(lanes)),
      256);
  return configs + positions_and_out + permitted + logits + workspace;
}

uint64_t decode_graph_line(uint64_t lanes) {
  // The round scratch at one column a lane (program_outer_scratch_bytes: the
  // ids, both residuals, the normalized head input, the logits, the argmax;
  // no layer adds a peak here), then the token ids and the slots.
  const uint64_t scratch = aligned(lanes * 4) + 2 * aligned(kHidden * lanes * 2) + aligned(kHidden * lanes * 2) +
                           aligned(kVocab * lanes * 2) + aligned(4);
  return scratch + 2 * sizeof(int32_t) * lanes;
}

void plan_lines_follow_the_lanes() {
  ignis_model_reservations eight{};
  ignis_model_reservations zero{};
  check(plan_27b(8, &eight).empty(), "eight lanes plan");
  check(plan_27b(0, &zero).empty(), "lanes 0 plan");
  // 0 is the leaf's 8, byte for byte: a load that names no lanes is today's.
  check(zero.workspace_bytes == eight.workspace_bytes && zero.sampling_bytes == eight.sampling_bytes &&
            zero.decode_graph_bytes == eight.decode_graph_bytes &&
            zero.verify_round_bytes == eight.verify_round_bytes &&
            zero.drafter_round_bytes == eight.drafter_round_bytes &&
            zero.media_embedding_bytes == eight.media_embedding_bytes && zero.activation_bytes == 0,
        "decode_lanes 0 plans exactly as 8");

  ignis_model_reservations refused{};
  const std::string nine = plan_27b(9, &refused);
  check(contains(nine, "decode_lanes 9") && contains(nine, "the 8 a Qwen 3.8-27B round serves"),
        "nine lanes are refused by name: got \"" + nine + "\"");

  for (uint32_t lanes = 1; lanes <= IGNIS_DECODE_MAX_BATCH; ++lanes) {
    ignis_model_reservations at{};
    const std::string error = plan_27b(lanes, &at);
    const std::string label = std::to_string(lanes) + " lanes";
    check(error.empty(), label + " plan: " + error);
    check(at.sampling_bytes == sampling_line(lanes),
          label + ": sampling " + std::to_string(at.sampling_bytes) + ", formula " + std::to_string(sampling_line(lanes)));
    check(at.decode_graph_bytes == decode_graph_line(lanes),
          label + ": decode round " + std::to_string(at.decode_graph_bytes) + ", formula " +
              std::to_string(decode_graph_line(lanes)));
    check(at.verify_round_bytes == 0 && at.drafter_round_bytes == 0, label + ": no window, no verify round");
    // The prefill chunk's arena is one span's, never a round's.
    check(at.workspace_bytes == eight.workspace_bytes, label + ": the workspace does not move");
  }
}

// --- 2. a load on the device --------------------------------------------------

// The one BF16 linear shape the vendored op serves beside the drafter's
// (`select_bf16_a16_launch`): a 14,336-row head over the 27B's 5,120-wide
// residual, ~147 MB a table.
constexpr uint64_t kSmallHidden = 5120;
constexpr uint64_t kSmallVocab = 14336;

struct SmallWeights {
  void *embedding = nullptr;
  void *norm = nullptr;
  void *head = nullptr;
  ~SmallWeights() {
    cudaFree(embedding);
    cudaFree(norm);
    cudaFree(head);
  }
};

ignis_bound_tensor bf16_tensor(const char *name, void *data, std::initializer_list<int32_t> shape) {
  ignis_bound_tensor t{};
  t.name = name;
  t.qtype = IGNIS_QTYPE_BF16_CTRL;
  t.layout = IGNIS_LAYOUT_CONTIGUOUS;
  t.qdata = data;
  t.ndim = static_cast<uint32_t>(shape.size());
  uint64_t elements = 1;
  std::size_t i = 0;
  for (int32_t d : shape) {
    t.shape[i] = d;
    t.padded_shape[i] = d;
    elements *= static_cast<uint64_t>(d);
    ++i;
  }
  for (; i < 4; ++i) {
    t.shape[i] = 1;
    t.padded_shape[i] = 1;
  }
  t.bytes = elements * 2;
  return t;
}

// A load of `lanes` lanes, its graphs captured over a pool with more slots
// than any load's lanes, and its round asked one wider than the lanes and at
// them.
void a_load_serves_its_lanes_and_no_wider(uint32_t lanes, const SmallWeights &w, ignis_seq_pool *pool) {
  const std::string label = std::to_string(lanes) + " lanes";
  const auto h = static_cast<int32_t>(kSmallHidden);
  const auto v = static_cast<int32_t>(kSmallVocab);
  const ignis_bound_tensor tensors[] = {bf16_tensor("text/token_embedding", w.embedding, {v, h}),
                                        bf16_tensor("text/final_norm", w.norm, {h}),
                                        bf16_tensor("text/output_head", w.head, {v, h})};
  const ignis_topology topology = no_layer_topology(kSmallHidden, kSmallVocab);
  const ignis_model_load_options options = lanes_option(lanes);

  ignis_model_reservations planned{};
  check(ignis_model_plan_reservations(tensors, 3, &topology, 128, 1024, IGNIS_KV_FORMAT_BF16, &options, &planned) == 0,
        label + ": plan: " + ignis_model_last_error());
  ignis_model *model = nullptr;
  if (ignis_model_load(tensors, 3, &topology, 128, 1024, IGNIS_KV_FORMAT_BF16, &options, &model) != 0 ||
      model == nullptr) {
    check(false, label + ": load: " + ignis_model_last_error());
    return;
  }
  struct ignis_model_stats stats{};
  check(ignis_model_stats(model, &stats) == 0, label + ": stats");
  const ignis_model_reservations &held = stats.reserved;
  check(held.sampling_bytes == planned.sampling_bytes && held.decode_graph_bytes == planned.decode_graph_bytes &&
            held.workspace_bytes == planned.workspace_bytes,
        label + ": the load holds what its plan laid out");

  uint32_t ready = 0;
  check(ignis_decode_graph_capture(model, pool, nullptr, &ready) == 0, label + ": capture");
  const uint32_t widths = (1u << lanes) - 1;
  check(ready == widths, label + ": decode graphs at widths 1.." + std::to_string(lanes) + ", mask " +
                             std::to_string(ready) + " (" + ignis_decode_graph_last_error() + ")");
  struct ignis_program_stats program{};
  check(ignis_program_stats(model, pool, &program) == 0, label + ": program stats");
  check(program.decode_graph_ready_mask == widths && program.verify_graph_ready_mask == 0,
        label + ": the stats say the same, mask " + std::to_string(program.decode_graph_ready_mask));

  // A round past the lanes is refused before any lane is read; one at the
  // lanes gets past that check and fails on the null sequence instead.
  std::vector<ignis_seq *> sequences(IGNIS_DECODE_MAX_BATCH + 1, nullptr);
  std::vector<ignis_sampling_params> sampling(IGNIS_DECODE_MAX_BATCH + 1);
  for (ignis_sampling_params &s : sampling) {
    s = ignis_sampling_params{};
    s.size = sizeof(s);
    s.greedy = 1;
  }
  std::vector<int32_t> out(IGNIS_DECODE_MAX_BATCH + 1, -1);
  check(ignis_program_decode(model, pool, sequences.data(), lanes + 1, sampling.data(), out.data(), nullptr) != 0,
        label + ": a round of " + std::to_string(lanes + 1) + " is refused");
  const std::string wider = ignis_step_last_error();
  check(contains(wider, "exceeds the load's decode lanes (" + std::to_string(lanes) + ")"),
        label + ": the refusal names the lanes: got \"" + wider + "\"");
  check(ignis_program_decode(model, pool, sequences.data(), lanes, sampling.data(), out.data(), nullptr) != 0,
        label + ": a round of null sequences fails");
  const std::string at = ignis_step_last_error();
  check(!contains(at, "decode lanes"), label + ": a round at the lanes is not refused for its width: got \"" + at + "\"");
  ignis_model_free(model);
}

void loads_on_the_device() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    check(false, "no CUDA device: the device half cannot run");
    return;
  }
  SmallWeights w;
  const std::size_t table = kSmallVocab * kSmallHidden * 2;
  if (cudaMalloc(&w.embedding, table) != cudaSuccess || cudaMalloc(&w.norm, kSmallHidden * 2) != cudaSuccess ||
      cudaMalloc(&w.head, table) != cudaSuccess) {
    check(false, "allocating the small weights");
    return;
  }
  cudaMemset(w.embedding, 0, table);
  cudaMemset(w.norm, 0, kSmallHidden * 2);
  cudaMemset(w.head, 0, table);

  // The pool's own geometry is test_seq_alloc.cpp's; nothing here reads a
  // sequence of it, and its slots outnumber every load's lanes.
  ignis_seq_pool_spec spec{};
  spec.num_kv_heads = 2;
  spec.head_dim = 8;
  spec.kv_page_group_count = 4;
  spec.max_context_tokens = 128;
  spec.slot_count = IGNIS_DECODE_MAX_BATCH + 1;
  spec.gdn_num_layers = 2;
  spec.kv_num_layers = 16;
  spec.gdn_conv_channels = 6;
  spec.gdn_value_heads = 2;
  spec.gdn_head_dim = 4;
  spec.vocab = static_cast<uint32_t>(kSmallVocab);
  ignis_seq_pool *pool = nullptr;
  if (ignis_seq_pool_create(&spec, &pool) != 0 || pool == nullptr) {
    check(false, "pool create");
    return;
  }
  for (const uint32_t lanes : {1u, 2u, 8u}) {
    a_load_serves_its_lanes_and_no_wider(lanes, w, pool);
  }
  ignis_seq_pool_free(pool);
}

} // namespace

int main() {
  plan_lines_follow_the_lanes();
  loads_on_the_device();
  if (g_failed != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("ok\n");
  return EXIT_SUCCESS;
}
