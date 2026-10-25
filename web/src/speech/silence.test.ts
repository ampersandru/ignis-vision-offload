import { afterEach, describe, expect, it } from "vitest";
import { chooseSilence, DEFAULT_SILENCE_MS, dictationSilence, reloadSilence, SILENCE_KEY } from "./silence.ts";

// The auto-stop silence (Q80): on at 3 s, Off / 2 s / 3 s / 5 s, per viewer.

class Storage {
  items = new Map<string, string>();
  getItem(key: string) {
    return this.items.get(key) ?? null;
  }
  setItem(key: string, value: string) {
    this.items.set(key, value);
  }
}

function withStorage(storage: unknown) {
  Object.defineProperty(globalThis, "localStorage", { value: storage, configurable: true, writable: true });
}

afterEach(() => {
  withStorage(undefined);
  reloadSilence();
});

describe("the dictation silence", () => {
  it("is on at 3 s when the browser stored nothing", () => {
    withStorage(new Storage());
    reloadSilence();
    expect(dictationSilence()).toBe(DEFAULT_SILENCE_MS);
    expect(DEFAULT_SILENCE_MS).toBe(3_000);
  });

  it("keeps the pick across a reload, Off included", () => {
    const storage = new Storage();
    withStorage(storage);
    for (const pick of [2_000, 5_000, null, 3_000]) {
      chooseSilence(pick);
      reloadSilence();
      expect(dictationSilence()).toBe(pick);
    }
    chooseSilence(null);
    expect(storage.getItem(SILENCE_KEY)).toBe("off");
  });

  it("falls back to 3 s on a value it does not offer", () => {
    const storage = new Storage();
    storage.setItem(SILENCE_KEY, "1234");
    withStorage(storage);
    reloadSilence();
    expect(dictationSilence()).toBe(3_000);
  });

  it("lasts the page where storage throws", () => {
    withStorage({
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
    });
    chooseSilence(5_000);
    expect(dictationSilence()).toBe(5_000);
  });
});
