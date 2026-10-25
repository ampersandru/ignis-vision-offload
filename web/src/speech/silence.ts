import { useSyncExternalStore } from "react";

// How long a silence ends a dictation by itself (Q80, spec speech/04): the
// server's own `server_vad` ends the item after that much silence, and the
// page treats it as Done. A per-viewer setting, this browser's, like the
// transport: kept in localStorage, which can be absent or throw, so every read
// and write is guarded and the setting simply lasts the page then.

/** What the setting holds: the silence in ms, or `null` for Off, the manual Done. */
export type SilenceMs = number | null;

export const SILENCE_KEY = "ignis.dictationSilence";

/** The choices the settings panel offers, in order. */
export const SILENCE_CHOICES: { value: SilenceMs; label: string }[] = [
  { value: null, label: "Off" },
  { value: 2_000, label: "2 s" },
  { value: 3_000, label: "3 s" },
  { value: 5_000, label: "5 s" },
];

/** On at 3 s until the viewer says otherwise (owner, 2026-10-25). */
export const DEFAULT_SILENCE_MS = 3_000;

function read(): SilenceMs {
  try {
    const stored = globalThis.localStorage?.getItem(SILENCE_KEY);
    if (stored === "off") return null;
    const ms = Number(stored);
    return SILENCE_CHOICES.some((c) => c.value === ms) ? ms : DEFAULT_SILENCE_MS;
  } catch {
    return DEFAULT_SILENCE_MS;
  }
}

let current: SilenceMs = read();
const listeners = new Set<() => void>();

/** The silence the next dictation asks for. */
export function dictationSilence(): SilenceMs {
  return current;
}

/** The viewer's pick, remembered by this browser. */
export function chooseSilence(ms: SilenceMs) {
  current = ms;
  try {
    globalThis.localStorage?.setItem(SILENCE_KEY, ms === null ? "off" : String(ms));
  } catch {
    // No storage: the choice lasts this page.
  }
  for (const listener of listeners) listener();
}

/** Forgets the page's copy so the next read goes to storage again (a test's). */
export function reloadSilence() {
  current = read();
  for (const listener of listeners) listener();
}

export function useSilence(): SilenceMs {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    dictationSilence,
    dictationSilence,
  );
}
