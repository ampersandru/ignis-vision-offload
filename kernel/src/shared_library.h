// ignis kernel leaf: a shared library opened at run time (GitHub #329).
//
// For the NVIDIA driver's own libraries (NVML: libnvidia-ml.so.1, nvml.dll),
// which ship with the driver rather than the toolkit: a binary that names one
// on its link line does not start on a machine without a driver, so the leaf
// opens them when it first needs them instead. dlopen/dlsym on Linux,
// LoadLibraryA/GetProcAddress on Windows; never closed, as an import would
// not be.
#pragma once

namespace ignis {

// Opens `name` by the platform's own search, the one an import would use.
// nullptr when the library is absent or does not load.
void *open_shared_library(const char *name);

// The address of `symbol` in `library` (from open_shared_library), nullptr
// when it has none.
void *shared_library_symbol(void *library, const char *symbol);

}  // namespace ignis
