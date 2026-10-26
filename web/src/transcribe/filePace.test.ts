import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { chooseFilePace, DEFAULT_FILE_PACE_SECONDS, FILE_PACE_KEY, filePace, reloadFilePace } from "./filePace.ts";

// N, the most audio not yet transcribed a file is sent ahead by (spec speech/07, Q99): a viewer setting, this
// browser's, 60 s until the viewer says otherwise.

const storage = new Map<string, string>();

beforeEach(() => {
  const local = { getItem: (key: string) => storage.get(key) ?? null, setItem: (key: string, value: string) => void storage.set(key, value) };
  Object.defineProperty(globalThis, "localStorage", { value: local, configurable: true, writable: true });
  reloadFilePace();
});

afterEach(() => {
  storage.clear();
  Object.defineProperty(globalThis, "localStorage", { value: undefined, configurable: true, writable: true });
  reloadFilePace();
});

describe("the file pace", () => {
  it("is 60 s by default", () => {
    expect(DEFAULT_FILE_PACE_SECONDS).toBe(60);
    expect(filePace()).toBe(60);
  });

  it("is the viewer's pick, kept by this browser", () => {
    chooseFilePace(30);
    expect(storage.get(FILE_PACE_KEY)).toBe("30");
    reloadFilePace();
    expect(filePace()).toBe(30);
  });

  it("falls back to the default on a value it does not offer", () => {
    storage.set(FILE_PACE_KEY, "45");
    reloadFilePace();
    expect(filePace()).toBe(60);
  });
});
