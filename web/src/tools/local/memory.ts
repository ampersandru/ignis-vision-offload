import { useSyncExternalStore } from "react";

// Memory: notes the model keeps across sessions, in this browser's
// localStorage. A note is a short title and a body. Only the titles reach the
// model, in the turn notice after the history (ADR 0072), so memory stays
// cheap however much it holds; the model reads a body with memory_read when a
// title looks relevant.

export type MemoryNote = { id: string; title: string; body: string; savedAt: string };

const STORAGE_KEY = "ignis.memory";

/** The titles a turn notice lists, newest kept first, cut to this many characters in all. */
export const MEMORY_PROMPT_LIMIT = 4000;
export const MEMORY_TITLE_LIMIT = 80;

export type MemoryStore = {
  list(): MemoryNote[];
  /** An empty title is taken from the body's first line. */
  save(title: string, body: string, now?: Date): MemoryNote;
  /** Rewrites a note; an edited note counts as saved now. */
  update(id: string, title: string, body: string, now?: Date): boolean;
  remove(id: string): boolean;
  /** Puts back a note that was removed (undo). */
  restore(note: MemoryNote): void;
  clear(): void;
  subscribe(listener: () => void): () => void;
};

type Storage = { getItem(key: string): string | null; setItem(key: string, value: string): void };

/** A title from `text`: its first line, cut to MEMORY_TITLE_LIMIT. */
export function titleOf(text: string): string {
  const line = text.trim().split("\n")[0].trim();
  return line.length > MEMORY_TITLE_LIMIT ? `${line.slice(0, MEMORY_TITLE_LIMIT - 1).trimEnd()}…` : line;
}

/** A stored note, including one saved before notes had titles (its first line becomes the title). */
function fromStored(value: unknown): MemoryNote | null {
  if (typeof value !== "object" || value === null) return null;
  const n = value as { id?: unknown; title?: unknown; body?: unknown; note?: unknown; savedAt?: unknown };
  if (typeof n.id !== "string") return null;
  const savedAt = typeof n.savedAt === "string" ? n.savedAt : new Date(0).toISOString();
  if (typeof n.title === "string") return { id: n.id, title: n.title, body: typeof n.body === "string" ? n.body : "", savedAt };
  if (typeof n.note === "string") return { id: n.id, title: titleOf(n.note), body: n.note, savedAt };
  return null;
}

export function createMemoryStore(storage: Storage | null): MemoryStore {
  let notes = load();
  const listeners = new Set<() => void>();

  function load(): MemoryNote[] {
    try {
      const parsed = JSON.parse(storage?.getItem(STORAGE_KEY) ?? "[]") as unknown;
      return Array.isArray(parsed) ? parsed.map(fromStored).filter((n): n is MemoryNote => n !== null) : [];
    } catch {
      return [];
    }
  }

  function commit(next: MemoryNote[]) {
    notes = next;
    try {
      storage?.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      // No storage (private window, blocked site data): the notes last this page.
    }
    for (const listener of listeners) listener();
  }

  const clean = (title: string, body: string) => ({ title: titleOf(title.trim() || body), body: body.trim() });

  return {
    list: () => notes,
    save(title, body, now = new Date()) {
      const n = notes.reduce((max, x) => Math.max(max, Number(x.id.slice(1)) || 0), 0) + 1;
      const saved = { id: `m${n}`, ...clean(title, body), savedAt: now.toISOString() };
      commit([...notes, saved]);
      return saved;
    },
    update(id, title, body, now = new Date()) {
      if (!notes.some((x) => x.id === id) || (title.trim() === "" && body.trim() === "")) return false;
      commit(notes.map((x) => (x.id === id ? { ...x, ...clean(title, body), savedAt: now.toISOString() } : x)));
      return true;
    },
    remove(id) {
      if (!notes.some((x) => x.id === id)) return false;
      commit(notes.filter((x) => x.id !== id));
      return true;
    },
    restore(note) {
      if (notes.some((x) => x.id === note.id)) return;
      commit([...notes, note].sort((a, b) => Date.parse(a.savedAt) - Date.parse(b.savedAt)));
    },
    clear() {
      commit([]);
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}

function browserStorage(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

export const browserMemory = createMemoryStore(browserStorage());

/** A memory note as a chat was last told of it: what a later turn's notice compares the notes with. */
export type NoteMark = Pick<MemoryNote, "id" | "title" | "savedAt">;

export const markOf = ({ id, title, savedAt }: MemoryNote): NoteMark => ({ id, title, savedAt });

/** The lines that fit in MEMORY_PROMPT_LIMIT, the newest (the last) kept first, and how many did not fit. */
function newestFitting(lines: string[]): { listed: string[]; left: number } {
  const listed: string[] = [];
  let used = 0;
  for (const line of [...lines].reverse()) {
    if (used + line.length > MEMORY_PROMPT_LIMIT) break;
    listed.unshift(line);
    used += line.length + 1;
  }
  return { listed, left: lines.length - listed.length };
}

/** The memory section's text: its heading, what the lines are, the lines, and what was left out, if anything. */
const section = (what: string, lines: string[], leftOut: string) => ["# Memory notes", what, ...lines, ...(leftOut ? [leftOut] : [])].join("\n");

/**
 * The memory section of a turn notice (ADR 0072), and the notes the chat
 * knows once it has read it. A chat told nothing yet (`known` undefined)
 * hears every title, as many as fit; with no note saved there is nothing to
 * tell, but the chat now knows that. A chat told before hears only the notes
 * saved, changed or deleted since, by anyone but itself — its own calls are
 * already in `known` — and nothing at all when there are none.
 */
export function memoryNotice(known: NoteMark[] | undefined, notes: MemoryNote[]): { text?: string; known?: NoteMark[] } {
  const marks = notes.map(markOf);
  if (known === undefined) {
    if (notes.length === 0) return { known: marks };
    const { listed, left } = newestFitting(notes.map((n) => `- [${n.id}] ${n.title}`));
    return { text: section("Saved notes, by title:", listed, left > 0 ? `(${left} older notes not listed.)` : ""), known: marks };
  }
  const before = new Map(known.map((n) => [n.id, n]));
  const kept = new Set(notes.map((n) => n.id));
  const changes = [
    ...known.filter((n) => !kept.has(n.id)).map((n) => `- deleted [${n.id}] ${n.title}`),
    ...notes.flatMap((n) => {
      const was = before.get(n.id);
      if (!was) return [`- saved [${n.id}] ${n.title}`];
      return was.savedAt !== n.savedAt || was.title !== n.title ? [`- changed [${n.id}] ${n.title}`] : [];
    }),
  ];
  if (changes.length === 0) return {};
  const { listed, left } = newestFitting(changes);
  const what = "Changed by the user or another chat since the last notice:";
  return { text: section(what, listed, left > 0 ? `(${left} older changes not listed.)` : ""), known: marks };
}

export function useMemoryNotes(store: MemoryStore = browserMemory): MemoryNote[] {
  return useSyncExternalStore(store.subscribe, store.list, store.list);
}
