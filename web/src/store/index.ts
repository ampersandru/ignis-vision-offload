import type { ImageStore } from "./imageStore.ts";
import { openIndexedDbStore } from "./indexeddb.ts";
import { createMemoryStore } from "./memory.ts";

// The page's one store: IndexedDB where the browser has it, memory otherwise.
// Chosen at the first call, never at import, so a module that imports this
// one loads where there is no IndexedDB (the tests, in Node).

let store: ImageStore | null = null;

export function pageStore(): ImageStore {
  store ??= typeof indexedDB === "undefined" ? createMemoryStore() : openIndexedDbStore(indexedDB);
  return store;
}

/** Puts `replacement` in the page's place: what a test of a hook that reads the store does first. */
export function setPageStore(replacement: ImageStore | null) {
  store = replacement;
}
