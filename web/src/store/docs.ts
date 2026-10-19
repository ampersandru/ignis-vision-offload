// What the chat and the Decide tab share in keeping their sessions as
// documents of the page's store (imageStore.ts): a stamp per session, the
// session each view had open, and the order a list comes back in.
//
// A session's stamp is given once, when the store first hears of it, and kept
// for good. Its key is a UUID, not the page's numeric id: two tabs of one
// origin each count their ids from the same place, and a key they shared
// would write one tab's session over the other's. Its order is the moment it
// was stamped, not the id either: the ids are handed out again at each load,
// and a session written since would sort by its new one, above the newer ones
// not written since.

/** Where a session is written, and where it sorts. */
export type Stamp = { key: string; order: number };

/** A session's key in the store. */
export const newKey = () => globalThis.crypto?.randomUUID?.() ?? `k-${Date.now()}-${Math.random().toString(36).slice(2)}`;

let lastOrder = 0;

/** A stamp for a session the store has not heard of: it sorts after every one stamped before it, here or on an earlier page. */
export function newStamp(): Stamp {
  lastOrder = Math.max(Date.now(), lastOrder + 1);
  return { key: newKey(), order: lastOrder };
}

/** The collection naming, by view (`chat`, `decide`), the key of the session it had open. */
export const ACTIVE = "active-session";

/**
 * Stored sessions in the order the list showed them, newest first — a new
 * session or a fork opens at the top, so the last stamped is the newest — each
 * with a new id counted up from `firstId`. The next stamp sorts after all of
 * them, whatever the clock says.
 */
export function restoreOrder<T extends Stamp>(stored: T[], firstId: number): { items: (T & { id: number })[]; nextId: number } {
  let next = firstId;
  const items = [...stored].sort((a, b) => a.order - b.order).map((s) => ({ ...s, id: next++ }));
  lastOrder = Math.max(lastOrder, ...stored.map((s) => s.order));
  return { items: items.reverse(), nextId: next };
}

/**
 * What changed between two looks at a list: the ids of the sessions that are
 * new or another object — every change to a session makes a new one — and the
 * sessions no longer there.
 */
export function listChanges<T extends { id: number }>(before: ReadonlyMap<number, T>, after: T[]): { changed: number[]; removed: T[] } {
  const ids = new Set(after.map((item) => item.id));
  return {
    changed: after.filter((item) => before.get(item.id) !== item).map((item) => item.id),
    removed: [...before.values()].filter((item) => !ids.has(item.id)),
  };
}

/**
 * The list a page shows once its stored sessions are read: what was done
 * before they came — any session worth keeping — on top, the stored ones
 * below. The session open stays open when it holds something; otherwise the
 * one the view last had open (`loadedActive`) is, else the newest.
 */
export function mergeLoaded<T extends { id: number }>(
  current: T[],
  activeId: number,
  loaded: T[],
  loadedActive: number | undefined,
  worthKeeping: (item: T) => boolean,
): { items: T[]; activeId: number } {
  const kept = current.filter(worthKeeping);
  const items = [...kept, ...loaded];
  if (items.length === 0) return { items: current, activeId };
  const stays = kept.some((item) => item.id === activeId);
  const reopened = loaded.some((item) => item.id === loadedActive) ? loadedActive : undefined;
  return { items, activeId: stays ? activeId : (reopened ?? items[0].id) };
}
