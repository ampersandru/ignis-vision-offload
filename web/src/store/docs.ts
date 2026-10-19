// What the chat and the Decide tab share in keeping their sessions as
// documents of the page's store (imageStore.ts): a key per session, the
// session each view had open, and the order a list comes back in.
//
// A session's key is a UUID, not the page's numeric id: two tabs of one origin
// each count their ids from the same place, and a key they shared would write
// one tab's session over the other's. The numeric id stays the page's own and
// is handed out again at each load.

/** A session's key in the store. */
export const newKey = () => globalThis.crypto?.randomUUID?.() ?? `k-${Date.now()}-${Math.random().toString(36).slice(2)}`;

/** The collection naming, by view (`chat`, `decide`), the key of the session it had open. */
export const ACTIVE = "active-session";

/** A stored session as `restoreOrder` sorts it: its key, and the id it had when written. */
export type Ordered = { key: string; id: number };

/**
 * Stored sessions in the order the list showed them, newest first, each with a
 * new id counted up from `firstId` — older sessions get the smaller ids, so the
 * order survives the next load too. The id each had when written orders them:
 * a page's ids only grow, and a new session or a fork opens at the top.
 */
export function restoreOrder<T extends Ordered>(stored: T[], firstId: number): { items: T[]; nextId: number } {
  let next = firstId;
  const items = [...stored].sort((a, b) => a.id - b.id).map((s) => ({ ...s, id: next++ }));
  return { items: items.reverse(), nextId: next };
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
