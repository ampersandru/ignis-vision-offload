import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { chooseUtteranceSilence, DEFAULT_UTTERANCE_SILENCE_MS, reloadUtteranceSilence, UTTERANCE_SILENCE_CHOICES, UTTERANCE_SILENCE_KEY, utteranceSilence } from "./silence.ts";

// The silence that ends an utterance (Q122): 1.5 s until the viewer picks another, kept by this browser.

const storage = new Map<string, string>();

beforeEach(() => {
  const local = { getItem: (key: string) => storage.get(key) ?? null, setItem: (key: string, value: string) => void storage.set(key, value) };
  Object.defineProperty(globalThis, "localStorage", { value: local, configurable: true, writable: true });
  reloadUtteranceSilence();
});

afterEach(() => {
  storage.clear();
  Object.defineProperty(globalThis, "localStorage", { value: undefined, configurable: true, writable: true });
  reloadUtteranceSilence();
});

describe("the utterance silence", () => {
  it("is 1.5 s by default, among 0.5, 1, 1.5 and 2 s", () => {
    expect(DEFAULT_UTTERANCE_SILENCE_MS).toBe(1_500);
    expect(UTTERANCE_SILENCE_CHOICES).toEqual([500, 1_000, 1_500, 2_000]);
    expect(utteranceSilence()).toBe(1_500);
  });

  it("is the viewer's pick, kept by this browser, and the default for a value it does not offer", () => {
    chooseUtteranceSilence(2_000);
    expect(storage.get(UTTERANCE_SILENCE_KEY)).toBe("2000");
    reloadUtteranceSilence();
    expect(utteranceSilence()).toBe(2_000);
    storage.set(UTTERANCE_SILENCE_KEY, "750");
    reloadUtteranceSilence();
    expect(utteranceSilence()).toBe(1_500);
  });

  it("lasts the page where the browser keeps nothing", () => {
    Object.defineProperty(globalThis, "localStorage", { value: undefined, configurable: true, writable: true });
    chooseUtteranceSilence(1_000);
    expect(utteranceSilence()).toBe(1_000);
  });
});
