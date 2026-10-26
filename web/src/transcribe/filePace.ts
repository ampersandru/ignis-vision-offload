import { useSyncExternalStore } from "react";

// N: how far ahead of its transcript a file is sent, in seconds of audio not
// yet transcribed (spec speech/07, Q99; pacing.ts). A per-viewer setting, this
// browser's, as the dictation's silence is (speech/silence.ts): kept in
// localStorage, which can be absent or throw, so every read and write is
// guarded and the setting simply lasts the page then. The server's own limit,
// `speech.max_queued_audio_seconds` (120 by default), refuses an append past
// it, so no choice goes beyond it.

export const FILE_PACE_KEY = "ignis.transcribeFilePace";

/** The choices the view offers, in seconds. */
export const FILE_PACE_CHOICES = [30, 60, 120];

export const DEFAULT_FILE_PACE_SECONDS = 60;

function read(): number {
  try {
    const seconds = Number(globalThis.localStorage?.getItem(FILE_PACE_KEY));
    return FILE_PACE_CHOICES.includes(seconds) ? seconds : DEFAULT_FILE_PACE_SECONDS;
  } catch {
    return DEFAULT_FILE_PACE_SECONDS;
  }
}

let current = read();
const listeners = new Set<() => void>();

/** N for the next file. */
export function filePace(): number {
  return current;
}

/** The viewer's pick, remembered by this browser. */
export function chooseFilePace(seconds: number) {
  current = seconds;
  try {
    globalThis.localStorage?.setItem(FILE_PACE_KEY, String(seconds));
  } catch {
    // No storage: the choice lasts this page.
  }
  for (const listener of listeners) listener();
}

/** Forgets the page's copy so the next read goes to storage again (a test's). */
export function reloadFilePace() {
  current = read();
  for (const listener of listeners) listener();
}

export function useFilePace(): number {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    filePace,
    filePace,
  );
}
