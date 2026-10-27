import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { EMPTY_CONTEXT } from "./context.ts";
import { DEFAULT_CONTEXT_KEY, reloadSavedContext, savedContext, saveContext } from "./defaultContext.ts";

// The context saved as this browser's default (spec speech/07, the owner's "Save as default").

const storage = new Map<string, string>();

beforeEach(() => {
  const local = {
    getItem: (key: string) => storage.get(key) ?? null,
    setItem: (key: string, value: string) => void storage.set(key, value),
    removeItem: (key: string) => void storage.delete(key),
  };
  Object.defineProperty(globalThis, "localStorage", { value: local, configurable: true, writable: true });
  reloadSavedContext();
});

afterEach(() => {
  storage.clear();
  Object.defineProperty(globalThis, "localStorage", { value: undefined, configurable: true, writable: true });
  reloadSavedContext();
});

const call = { languages: { mode: "free" as const, codes: ["it"] }, topic: "Rilascio", terms: ["Forgejo", "Tekton"], other: "", appliedAt: 1_700 };

describe("the default context", () => {
  it("is none until one is saved", () => {
    expect(savedContext()).toBeNull();
  });

  it("keeps what a context tells, never when it was applied, in this browser", () => {
    saveContext(call);
    reloadSavedContext();
    expect(savedContext()).toEqual({ ...call, appliedAt: null });
    expect(JSON.parse(storage.get(DEFAULT_CONTEXT_KEY)!).appliedAt).toBeNull();
  });

  it("is cleared by a blank one saved", () => {
    saveContext(call);
    saveContext(EMPTY_CONTEXT);
    expect(storage.has(DEFAULT_CONTEXT_KEY)).toBe(false);
    reloadSavedContext();
    expect(savedContext()).toBeNull();
  });

  it("reads as none from what this page did not write", () => {
    storage.set(DEFAULT_CONTEXT_KEY, "{not json");
    reloadSavedContext();
    expect(savedContext()).toBeNull();
    storage.set(DEFAULT_CONTEXT_KEY, JSON.stringify({ topic: 3 }));
    reloadSavedContext();
    expect(savedContext()).toBeNull();
  });
});
