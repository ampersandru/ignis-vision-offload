import { useEffect, useRef, useState } from "react";
import { ACTIVE, newKey } from "./docs.ts";
import type { ImageStore } from "./imageStore.ts";
import { createSaver, type Saver } from "./saver.ts";

// A view's sessions kept in the page's store (spec playground-enhancements/03):
// the chat's and the Decide tab's, each a document of its own collection under
// a key of its own (docs.ts). The view keeps its list as it always did; this
// hook watches it. Read once, when the view mounts, and handed to the view to
// take in (`adopt`); from then on a session that changed — a new object in the
// list, since every change makes one — is written a moment later (saver.ts), a
// session the list no longer holds is deleted, and the one open is noted under
// ACTIVE. What the store refuses is said in `error`, and the page goes on in
// memory.

const SAVE_DELAY_MS = 400;
/** A reply streaming never pauses: this is the most of it a reload can lose. */
const SAVE_MAX_WAIT_MS = 1000;

export type StoredSessionsOptions<T> = {
  store: ImageStore;
  collection: string;
  /** The view's name under ACTIVE. */
  view: string;
  items: T[];
  activeId: number;
  toStored: (item: T, key: string) => unknown;
  /** A session that is not is never written, and deleted if it was. */
  worthKeeping: (item: T) => boolean;
  /**
   * The stored values and the key of the session the view last had open, for
   * the view to read and put in its list; it hands back the sessions it added,
   * as they are in the list, and the key of each by its id.
   */
  adopt: (values: unknown[], activeKey: string | null) => Promise<{ loaded: T[]; keys: Map<number, string> }>;
  /** A session the list no longer holds, gone from the store too, and the sessions left. */
  onRemoved?: (gone: T, left: T[]) => void;
};

export function useStoredSessions<T extends { id: number }>(options: StoredSessionsOptions<T>): { loaded: boolean; error: string | null } {
  const now = useRef(options);
  now.current = options;
  const { store, collection, view, items, activeId } = options;
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const keys = useRef(new Map<number, string>());
  // The sessions as last written or read, by id: null until the store has answered.
  const seen = useRef<Map<number, T> | null>(null);
  const saver = useRef<Saver | null>(null);

  const refuse = (err: unknown) => setError(`Not saved in this browser: ${err instanceof Error ? err.message : String(err)}`);
  function keyOf(id: number) {
    let key = keys.current.get(id);
    if (!key) keys.current.set(id, (key = newKey()));
    return key;
  }
  saver.current ??= createSaver({
    delayMs: SAVE_DELAY_MS,
    maxWaitMs: SAVE_MAX_WAIT_MS,
    write: (key) => {
      const { store, collection, items, worthKeeping, toStored } = now.current;
      const item = items.find((i) => keys.current.get(i.id) === key);
      if (!item) return;
      const written = worthKeeping(item) ? store.putDoc(collection, key, toStored(item, key)) : store.deleteDoc(collection, key);
      void written.then(() => setError(null), refuse);
    },
  });

  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const [values, active] = await Promise.all([store.listDocs<unknown>(collection), store.getDoc<string>(ACTIVE, view)]);
        if (!live) return;
        const taken = await now.current.adopt(values, active);
        for (const [id, key] of taken.keys) keys.current.set(id, key);
        seen.current = new Map(taken.loaded.map((item) => [item.id, item]));
      } catch (err) {
        refuse(err);
        seen.current = new Map();
      }
      if (live) setLoaded(true);
    })();
    return () => {
      live = false;
    };
  }, [store, collection, view]);

  useEffect(() => {
    const before = seen.current;
    if (!loaded || !before) return;
    const after = new Map(items.map((item) => [item.id, item]));
    for (const [id, item] of after) if (before.get(id) !== item) saver.current?.schedule(keyOf(id));
    for (const [id, item] of before) {
      if (after.has(id)) continue;
      const key = keys.current.get(id);
      if (key) {
        saver.current?.forget(key);
        keys.current.delete(id);
        void store.deleteDoc(collection, key).catch(refuse);
      }
      now.current.onRemoved?.(item, items);
    }
    seen.current = after;
  }, [items, loaded, store, collection]);

  useEffect(() => {
    if (loaded) void store.putDoc(ACTIVE, view, keyOf(activeId)).catch(refuse);
  }, [loaded, activeId, store, view]);

  // A page going away writes what was waiting: a reload mid-reply keeps the reply up to there.
  useEffect(() => {
    const flush = () => saver.current?.flush();
    globalThis.addEventListener?.("pagehide", flush);
    return () => {
      globalThis.removeEventListener?.("pagehide", flush);
      flush();
    };
  }, []);

  return { loaded, error };
}
