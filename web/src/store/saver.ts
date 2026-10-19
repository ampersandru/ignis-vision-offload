// When a session goes back to the page's store. A change is written a moment
// after the last one (`delayMs`), so typing or a picture landing is one write
// and not one per keystroke; while changes never pause — a reply streaming a
// token per frame — it is written by `maxWaitMs` from the first one waiting,
// so a reload mid-reply loses at most that much of it. Each key keeps its own
// clock. The saver holds keys, not values: `write` reads the session as it is
// when the write happens, and a session gone by then is the caller's to skip.

export type Saver = {
  /** A change to `key`: written soon. */
  schedule(key: string): void;
  /** Nothing waiting for `key` is written: the session is gone. */
  forget(key: string): void;
  /** Everything waiting, written now: the page is going away. */
  flush(): void;
};

export function createSaver({ delayMs, maxWaitMs, write }: { delayMs: number; maxWaitMs: number; write: (key: string) => void }): Saver {
  const pending = new Map<string, { timer: ReturnType<typeof setTimeout>; first: number }>();
  const drop = (key: string) => {
    const waiting = pending.get(key);
    if (waiting) clearTimeout(waiting.timer);
    pending.delete(key);
  };
  const writeNow = (key: string) => {
    drop(key);
    write(key);
  };
  return {
    schedule(key) {
      const at = Date.now();
      const first = pending.get(key)?.first ?? at;
      drop(key);
      const wait = Math.max(0, Math.min(delayMs, first + maxWaitMs - at));
      pending.set(key, { timer: setTimeout(() => writeNow(key), wait), first });
    },
    forget: drop,
    flush() {
      for (const key of [...pending.keys()]) writeNow(key);
    },
  };
}
