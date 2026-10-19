import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createSaver } from "./saver.ts";

// When the page writes a session back: a moment after the last change, no
// later than a bound while changes keep coming (a reply streaming), all at
// once as the page goes away, and never after the session is gone.

describe("the saver", () => {
  beforeEach(() => void vi.useFakeTimers());
  afterEach(() => void vi.useRealTimers());

  it("writes once, a moment after the last change", () => {
    const write = vi.fn();
    const saver = createSaver({ delayMs: 400, maxWaitMs: 1000, write });
    saver.schedule("a");
    vi.advanceTimersByTime(300);
    saver.schedule("a");
    vi.advanceTimersByTime(300);
    expect(write).not.toHaveBeenCalled();
    vi.advanceTimersByTime(100);
    expect(write).toHaveBeenCalledTimes(1);
    expect(write).toHaveBeenCalledWith("a");
  });

  it("writes by the bound while changes never pause, then waits again", () => {
    const write = vi.fn();
    const saver = createSaver({ delayMs: 400, maxWaitMs: 1000, write });
    for (let t = 0; t < 2500; t += 100) {
      saver.schedule("a");
      vi.advanceTimersByTime(100);
    }
    // At 1000 and 2000 ms from the first change of each run.
    expect(write).toHaveBeenCalledTimes(2);
    vi.advanceTimersByTime(400);
    expect(write).toHaveBeenCalledTimes(3);
  });

  it("keeps each key's clock apart", () => {
    const write = vi.fn();
    const saver = createSaver({ delayMs: 400, maxWaitMs: 1000, write });
    saver.schedule("a");
    vi.advanceTimersByTime(200);
    saver.schedule("b");
    vi.advanceTimersByTime(200);
    expect(write.mock.calls).toEqual([["a"]]);
    vi.advanceTimersByTime(200);
    expect(write.mock.calls).toEqual([["a"], ["b"]]);
  });

  it("forgets a key: what was pending for it is never written", () => {
    const write = vi.fn();
    const saver = createSaver({ delayMs: 400, maxWaitMs: 1000, write });
    saver.schedule("a");
    saver.forget("a");
    vi.advanceTimersByTime(2000);
    expect(write).not.toHaveBeenCalled();
  });

  it("flushes every pending key, as the page goes away", () => {
    const write = vi.fn();
    const saver = createSaver({ delayMs: 400, maxWaitMs: 1000, write });
    saver.schedule("a");
    saver.schedule("b");
    saver.flush();
    expect(write.mock.calls.map(([k]) => k).sort()).toEqual(["a", "b"]);
    vi.advanceTimersByTime(2000);
    expect(write).toHaveBeenCalledTimes(2);
  });
});
