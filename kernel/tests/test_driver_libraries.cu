// The NVIDIA driver's libraries, reached at run time (GitHub #329).
//
// Ours, not vendored. The leaf links no driver library, so a binary built on
// it starts on a machine without the driver: the two driver API calls in the
// leaf go through kernel/src/cuda_driver_shim.cu, which forwards them to the
// driver the CUDA runtime loaded, and NVML is opened by device.cu through
// kernel/src/shared_library.cu. A GPU path asks ignis_device_driver_version
// first.
//
// As CTest runs it, on the GPU host: the driver is found, NVML reads device 0,
// and both forwarders reach the driver. With --expect-no-driver, on a machine
// without one (the Linux container proof of #329): the version is 0, NVML does
// not load, and the forwarders fail cleanly. Both ways, an absent library opens
// as nullptr. It launches and allocates nothing, so it runs while another
// process holds the card.

#include "ignis_device.h"
#include "shared_library.h"

#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

int failures = 0;

void expect(bool ok, const char *label) {
  std::printf("%s: %s\n", ok ? "ok" : "FAIL", label);
  if (!ok) {
    ++failures;
  }
}

}  // namespace

int main(int argc, char **argv) {
  const bool no_driver = argc > 1 && std::strcmp(argv[1], "--expect-no-driver") == 0;

  expect(ignis::open_shared_library("ignis-no-such-library.so.1") == nullptr,
         "an absent library opens as nullptr");
  expect(ignis::shared_library_symbol(nullptr, "nvmlInit_v2") == nullptr,
         "no library, no symbol");

  int32_t version = -1;
  const int32_t version_rc = ignis_device_driver_version(&version);
  std::printf("ignis_device_driver_version: rc %d, version %d\n", version_rc, version);
  expect(ignis_device_driver_version(nullptr) == -1, "a NULL version is refused");

  uint64_t free_bytes = 0;
  uint64_t total_bytes = 0;
  const int32_t nvml_rc = ignis_device_nvml_mem_info(0, &free_bytes, &total_bytes);
  std::printf("ignis_device_nvml_mem_info: rc %d, free %llu of %llu\n", nvml_rc,
              static_cast<unsigned long long>(free_bytes),
              static_cast<unsigned long long>(total_bytes));

  // Rank 0 is outside the 1..5 the driver accepts: whatever error it gives,
  // it is the driver's, not the forwarder's CUDA_ERROR_NOT_FOUND.
  CUtensorMap map{};
  const cuuint64_t dims[] = {64, 64};
  const cuuint64_t strides[] = {64};
  const cuuint32_t box[] = {16, 16};
  const cuuint32_t element_strides[] = {1, 1};
  const CUresult encoded = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 0, nullptr, dims, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  const char *name = nullptr;
  const CUresult named = cuGetErrorName(CUDA_ERROR_INVALID_VALUE, &name);
  std::printf("cuTensorMapEncodeTiled(rank 0): %d; cuGetErrorName: %d, \"%s\"\n",
              static_cast<int>(encoded), static_cast<int>(named), name != nullptr ? name : "(null)");

  if (no_driver) {
    expect(version_rc != 0 || version == 0, "no driver: no driver version");
    expect(nvml_rc == -1, "no driver: NVML reads nothing");
    expect(encoded == CUDA_ERROR_NOT_FOUND, "no driver: cuTensorMapEncodeTiled has no driver to reach");
    expect(named == CUDA_SUCCESS && name != nullptr && std::strstr(name, "unavailable") != nullptr,
           "no driver: cuGetErrorName still names why");
  } else {
    expect(version_rc == 0 && version > 0, "the driver's version");
    expect(nvml_rc == 0 && total_bytes > 0 && free_bytes <= total_bytes, "NVML reads device 0");
    expect(encoded != CUDA_SUCCESS && encoded != CUDA_ERROR_NOT_FOUND,
           "cuTensorMapEncodeTiled reaches the driver");
    expect(named == CUDA_SUCCESS && name != nullptr &&
               std::strcmp(name, "CUDA_ERROR_INVALID_VALUE") == 0,
           "cuGetErrorName reaches the driver");
  }

  if (failures != 0) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  return 0;
}
