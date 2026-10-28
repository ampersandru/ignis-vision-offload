import { useSyncExternalStore } from "react";

// A value the viewer picks among a few, kept by this browser: the Transcribe
// view's file pace, its silence and the kind of its notes. As the dictation's silence is
// (speech/silence.ts), it lives in localStorage, which can be absent or throw,
// so every read and write is guarded and a pick simply lasts the page then;
// a stored value the view no longer offers reads as the default.

export type ViewerChoice<T extends string | number> = {
  get: () => T;
  /** The viewer's pick, remembered by this browser. */
  choose: (value: T) => void;
  /** Forgets the page's copy so the next read goes to storage again (a test's). */
  reload: () => void;
  /** The pick, for a component that shows it. */
  use: () => T;
};

export function viewerChoice<T extends string | number>({ key, choices, fallback }: { key: string; choices: readonly T[]; fallback: T }): ViewerChoice<T> {
  const read = (): T => {
    try {
      const stored = globalThis.localStorage?.getItem(key);
      return choices.find((choice) => String(choice) === stored) ?? fallback;
    } catch {
      return fallback;
    }
  };
  let current = read();
  const listeners = new Set<() => void>();
  const tell = () => {
    for (const listener of listeners) listener();
  };
  const subscribe = (listener: () => void) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  };
  const get = () => current;
  return {
    get,
    choose(value) {
      current = value;
      try {
        globalThis.localStorage?.setItem(key, String(value));
      } catch {
        // No storage: the pick lasts this page.
      }
      tell();
    },
    reload() {
      current = read();
      tell();
    },
    use: () => useSyncExternalStore(subscribe, get, get),
  };
}
