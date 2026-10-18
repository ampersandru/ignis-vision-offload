// ignis kernel leaf: the CUDA driver API entry points the leaf calls, resolved
// at run time (GitHub #329).
//
// The driver's library (libcuda.so.1, nvcuda.dll) ships with the NVIDIA
// driver, not with the toolkit: a binary that names it on its link line does
// not start where there is no driver -- not even for `--help` or `model
// download`, which never touch the GPU (on Linux the dynamic loader refuses
// before `main`). So the leaf links only the CUDA runtime, and the two driver
// functions anything in it calls are defined here, each forwarding to the
// driver's own entry point, which the runtime looks up in the driver it has
// loaded (`cudaGetDriverEntryPointByVersion`).
//
// The callers are vendored, and stay verbatim (ADR 0010):
// kernel/vendor/src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh calls
// `cuTensorMapEncodeTiled` for the W4A4 GEMM's TMA descriptors and
// `cuGetErrorName` for its error text. Their references resolve to the
// definitions below at link time, as they resolved to CUDA::cuda_driver
// before. A new direct driver call anywhere in the leaf is a link error
// (undefined reference), which is the cue to add its forwarder here.

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

namespace {

// The driver's `symbol` at the ABI of CUDA `version` -- the version its PFN_
// typedef in cudaTypedefs.h names -- or nullptr when the runtime finds no
// driver, or a driver without it.
void *driver_entry_point(const char *symbol, unsigned int version) {
  void *fn = nullptr;
  cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
  if (cudaGetDriverEntryPointByVersion(symbol, &fn, version, cudaEnableDefault, &status) !=
          cudaSuccess ||
      status != cudaDriverEntryPointSuccess) {
    return nullptr;
  }
  return fn;
}

}  // namespace

extern "C" CUresult CUDAAPI cuTensorMapEncodeTiled(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank,
    void *globalAddress, const cuuint64_t *globalDim, const cuuint64_t *globalStrides,
    const cuuint32_t *boxDim, const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
  static const auto fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(
      driver_entry_point("cuTensorMapEncodeTiled", 12000));
  if (fn == nullptr) {
    return CUDA_ERROR_NOT_FOUND;
  }
  return fn(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, boxDim,
            elementStrides, interleave, swizzle, l2Promotion, oobFill);
}

extern "C" CUresult CUDAAPI cuGetErrorName(CUresult error, const char **pStr) {
  static const auto fn =
      reinterpret_cast<PFN_cuGetErrorName_v6000>(driver_entry_point("cuGetErrorName", 6000));
  if (fn == nullptr) {
    // Still a name, so the caller's error message says why rather than
    // printing nothing: the only callers turn it into an exception's text.
    if (pStr != nullptr) {
      *pStr = "CUDA driver entry point unavailable";
    }
    return CUDA_SUCCESS;
  }
  return fn(error, pStr);
}
