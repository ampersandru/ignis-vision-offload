// GitHub #329 item 5: a Flash-Next verify round wider than the MoE decode route's one launch --
// OURS (kernel/src/flash_next/program.cu's fn_moe_experts_decode).
//
// Past IGNIS_MOE_DECODE_MAX_TOKENS rows a decode-phase call runs kern's decode route in launches of
// the workspace's decode tokens, each adding its rows into the routed accumulator. That is exact
// only if a row's sums do not depend on the rows sharing its launch: the register kernel's units
// run one fp32 chain per token (the tokens are MMA rows, each scaled by its own power of two) and
// the cross-unit sums are exact fixed point (ignis_moe.h). Held here, on the route Flash-Next
// loads (IGNIS_MOE_DECODE_REGISTERS):
//
//   per row      9, 16, 24 and 32 rows through fn_moe_experts_decode, routed as a verify round's
//                columns are (each row sharing experts with the one before): every accumulator
//                row is the bits of that row's own one-token launch;
//   one launch   8 rows: fn_moe_experts_decode is the one launch ignis_moe_experts_decode makes,
//                bit for bit, and a call of no rows is refused as that op refuses it;
//   graph        a 24-row call captured in a CUDA graph and replayed twice: the eager bits.
//
// Every call runs on one workspace in sequence, so each also proves the launches before it left
// the workspace as they found it. ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a
// missing GPU fails.

#include "flash_next/flash_next_internal.h"

#include "ignis_moe.h"
#include "moe_experts_common.h"
#include "moe_fixture.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

using namespace moe_test;
namespace fn = ignis::flash_next;

namespace {

constexpr uint32_t kWidest = 32;
constexpr uint32_t kPrefillTokens = 64;

struct Workspace {
  DeviceBytes base{ignis_moe_workspace_bytes(IGNIS_MOE_DECODE_MAX_TOKENS, kPrefillTokens)};
  DeviceBytes acc{static_cast<std::size_t>(kPrefillTokens) * kH * 8};
  ignis_moe_workspace ws{base.p, IGNIS_MOE_DECODE_MAX_TOKENS, kPrefillTokens, IGNIS_MOE_DECODE_REGISTERS};
  Workspace() {
    MOE_RC(ignis_moe_workspace_init(&ws, acc.as<int64_t>(), nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
  }
};

struct Call {
  std::vector<uint16_t> x;
  std::vector<int32_t> ids;
  std::vector<float> weights;
  DeviceBytes dx, dids, dw;
  Call(int tokens, uint32_t stream)
      : x(make_tokens(tokens, stream, 2.0f)), dx(x.size() * 2), dids(static_cast<std::size_t>(tokens) * kTop * 4),
        dw(static_cast<std::size_t>(tokens) * kTop * 4) {
    make_routing(tokens, stream + 1, 3, ids, weights);
    upload(dx, x);
    upload(dids, ids);
    upload(dw, weights);
  }
};

std::vector<int64_t> wide(Workspace &w, const ExpertSet &set, const Call &c, uint32_t rows) {
  MOE_CUDA(cudaMemset(w.acc.p, 0, static_cast<std::size_t>(rows) * kH * 8));
  check(fn::fn_moe_experts_decode(c.dx.p, rows, c.dids.as<int32_t>(), c.dw.as<float>(), set.d_slots.as<ignis_moe_slot>(),
                                  w.ws, w.acc.as<int64_t>(), nullptr) == 0,
        std::to_string(rows) + " rows: fn_moe_experts_decode runs: " + ignis_moe_last_error());
  MOE_CUDA(cudaDeviceSynchronize());
  return download<int64_t>(w.acc.p, static_cast<std::size_t>(rows) * kH);
}

// Row `t` of the call, alone in its own launch.
std::vector<int64_t> alone(Workspace &w, const ExpertSet &set, const Call &c, uint32_t t) {
  MOE_CUDA(cudaMemset(w.acc.p, 0, static_cast<std::size_t>(kH) * 8));
  const auto *x = static_cast<const __nv_bfloat16 *>(c.dx.p) + static_cast<std::size_t>(t) * kH;
  MOE_RC(ignis_moe_experts_decode(x, 1, c.dids.as<int32_t>() + t * kTop, c.dw.as<float>() + t * kTop,
                                  set.d_slots.as<ignis_moe_slot>(), &w.ws, w.acc.as<int64_t>(), nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  return download<int64_t>(w.acc.p, static_cast<std::size_t>(kH));
}

void per_row_arm(Workspace &w, const ExpertSet &set) {
  for (uint32_t rows : {9u, 16u, 24u, kWidest}) {
    const Call c(static_cast<int>(rows), 1200 + rows);
    const std::vector<int64_t> got = wide(w, set, c, rows);
    int differing = 0;
    for (uint32_t t = 0; t < rows; ++t) {
      const std::vector<int64_t> own = alone(w, set, c, t);
      differing += std::equal(own.begin(), own.end(), got.begin() + static_cast<std::ptrdiff_t>(t) * kH) ? 0 : 1;
    }
    check(differing == 0, std::to_string(rows) + " rows: " + std::to_string(differing) +
                              " row(s) differ from their own one-token launch");
    std::printf("  %u rows in launches of %u: every row the bits of its own launch\n", rows, IGNIS_MOE_DECODE_MAX_TOKENS);
  }
}

void one_launch_arm(Workspace &w, const ExpertSet &set) {
  const uint32_t rows = IGNIS_MOE_DECODE_MAX_TOKENS;
  const Call c(static_cast<int>(rows), 1300);
  const std::vector<int64_t> got = wide(w, set, c, rows);
  MOE_CUDA(cudaMemset(w.acc.p, 0, static_cast<std::size_t>(rows) * kH * 8));
  MOE_RC(ignis_moe_experts_decode(c.dx.p, rows, c.dids.as<int32_t>(), c.dw.as<float>(), set.d_slots.as<ignis_moe_slot>(),
                                  &w.ws, w.acc.as<int64_t>(), nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  check(download<int64_t>(w.acc.p, static_cast<std::size_t>(rows) * kH) == got,
        "8 rows: the one launch ignis_moe_experts_decode makes, bit for bit");
  check(fn::fn_moe_experts_decode(c.dx.p, 0, c.dids.as<int32_t>(), c.dw.as<float>(), set.d_slots.as<ignis_moe_slot>(),
                                  w.ws, w.acc.as<int64_t>(), nullptr) != 0 &&
            std::string(ignis_moe_last_error()).find("tokens must be 1..") != std::string::npos,
        "a call of no rows is refused as ignis_moe_experts_decode refuses it");
  std::printf("  8 rows: one launch, ignis_moe_experts_decode's bits; no rows refused\n");
}

void graph_arm(Workspace &w, const ExpertSet &set) {
  const uint32_t rows = 24;
  const Call c(static_cast<int>(rows), 1400);
  const std::vector<int64_t> eager = wide(w, set, c, rows);
  cudaStream_t stream;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  MOE_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  MOE_CUDA(cudaMemsetAsync(w.acc.p, 0, static_cast<std::size_t>(rows) * kH * 8, stream));
  MOE_RC(fn::fn_moe_experts_decode(c.dx.p, rows, c.dids.as<int32_t>(), c.dw.as<float>(), set.d_slots.as<ignis_moe_slot>(),
                                   w.ws, w.acc.as<int64_t>(), stream));
  MOE_CUDA(cudaStreamEndCapture(stream, &graph));
  MOE_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  for (int replay = 0; replay < 2; ++replay) {
    MOE_CUDA(cudaGraphLaunch(exec, stream));
    MOE_CUDA(cudaStreamSynchronize(stream));
    check(download<int64_t>(w.acc.p, static_cast<std::size_t>(rows) * kH) == eager,
          "24 rows: graph replay " + std::to_string(replay + 1) + " agrees with the eager call bit for bit");
  }
  MOE_CUDA(cudaGraphExecDestroy(exec));
  MOE_CUDA(cudaGraphDestroy(graph));
  MOE_CUDA(cudaStreamDestroy(stream));
  std::printf("  24 rows captured: two graph replays, the eager bits\n");
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  std::printf("flash_next moe rows: a verify round's MoE past the decode route's one launch\n");
  ExpertSet set;
  std::vector<int> identity(2 * kRecords);
  std::iota(identity.begin(), identity.end(), 0);
  set.place(identity);
  Workspace w;
  per_row_arm(w, set);
  one_launch_arm(w, set);
  graph_arm(w, set);
  if (g_failed != 0) {
    std::fprintf(stderr, "flash_next moe rows test: %d check(s) FAILED\n", g_failed);
    return 1;
  }
  std::printf("flash_next moe rows test: ok\n");
  return 0;
}
