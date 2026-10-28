import { endDictation } from "./dictation.ts";

// Whether the Transcribe view is listening (spec speech/07, "While
// listening", Q85): the page's speech is its own then. The other views' mics
// are struck, saying "Listening in Transcribe" (availability.ts), and open no
// second session; a dictation under way when a listening starts ends as it
// does when another mic starts, its text landing in its own field.

let on = false;
const listeners = new Set<() => void>();

export const listening = (): boolean => on;

export function subscribeListening(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function setListening(next: boolean) {
  if (next === on) return;
  on = next;
  if (on) endDictation();
  for (const listener of listeners) listener();
}
