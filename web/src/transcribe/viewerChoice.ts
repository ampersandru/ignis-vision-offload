import { useSyncExternalStore } from "react";

// A number the viewer picks among a few, kept by this browser: the Transcribe
// view's file pace and its silence. As the dictation's silence is
// (speech/silence.ts), it lives in localStorage, which can be absent or throw,
// so every read and write is guarded and a pick simply lasts the page then;
// a stored value the view no longer offers reads as the default.

export type ViewerChoice = {
  get: () => number;
  /** The viewer's pick, remembered by this browser. */
  choose: (value: number) => void;
  /** Forgets the page's copy so the next read goes to storage again (a test's). */
  reload: () => void;
  /** The pick, for a component that shows it. */
  use: () => number;
};

export function viewerChoice({ key, choices, fallback }: { key: string; choices: number[]; fallback: number }): ViewerChoice {
  const read = (): number => {
    try {
      const value = Number(globalThis.localStorage?.getItem(key));
      return choices.includes(value) ? value : fallback;
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
