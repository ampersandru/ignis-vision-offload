// ignis kernel leaf: a shared library opened at run time (see shared_library.h).
//
// Its own translation unit so <windows.h> reaches nothing else in the leaf.

#include "shared_library.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ignis {

void *open_shared_library(const char *name) {
#ifdef _WIN32
  return reinterpret_cast<void *>(LoadLibraryA(name));
#else
  return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void *shared_library_symbol(void *library, const char *symbol) {
  if (library == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(library), symbol));
#else
  return dlsym(library, symbol);
#endif
}

}  // namespace ignis
