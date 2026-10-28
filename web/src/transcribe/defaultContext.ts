import { useSyncExternalStore } from "react";
import { type CallContext, isBlank, readContext } from "./context.ts";

// The context a Transcript never started begins from (spec speech/07, the
// owner's "Save as default"): this browser's, in localStorage, which can be
// absent or throw, so every read and write is guarded and a save lasts the
// page then. A blank one saved clears it. Never applied by itself: the first
// Start still asks (Deck.tsx).

export const DEFAULT_CONTEXT_KEY = "ignis.transcribeContext";

const read = (): CallContext | null => {
  try {
    const stored = globalThis.localStorage?.getItem(DEFAULT_CONTEXT_KEY);
    const c = stored ? readContext(JSON.parse(stored)) : undefined;
    return c && !isBlank(c) ? { ...c, appliedAt: null } : null;
  } catch {
    return null;
  }
};

let current = read();
const listeners = new Set<() => void>();
const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

/** The context saved as this browser's default, `null` when none is. */
export const savedContext = (): CallContext | null => current;

/** Saves `c` as the default, what it tells alone; a blank one clears it. */
export function saveContext(c: CallContext) {
  current = isBlank(c) ? null : { languages: c.languages, topic: c.topic, terms: c.terms, other: c.other, appliedAt: null };
  try {
    if (current === null) globalThis.localStorage?.removeItem(DEFAULT_CONTEXT_KEY);
    else globalThis.localStorage?.setItem(DEFAULT_CONTEXT_KEY, JSON.stringify(current));
  } catch {
    // No storage: the default lasts this page.
  }
  for (const listener of listeners) listener();
}

/** Forgets the page's copy so the next read goes to storage again (a test's). */
export function reloadSavedContext() {
  current = read();
}

/** The default, for a component that shows it. */
export const useSavedContext = (): CallContext | null => useSyncExternalStore(subscribe, savedContext, savedContext);
