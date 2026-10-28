import { describe, expect, it } from "vitest";
import { allowance, backlogMs, estimate } from "./pacing.ts";

// A file fed faster than real time (spec speech/07, "Sources", Q99): the page
// sends while the audio not yet transcribed stays under N seconds.

describe("the audio not yet transcribed", () => {
  it("runs from the start of the oldest utterance not completed yet to the audio sent", () => {
    expect(backlogMs(90_000, [{ startMs: 41_000 }, { startMs: 30_500 }])).toBe(59_500);
  });

  it("is none while no utterance is under way or waiting for its text: silence is transcribed as it is sent", () => {
    expect(backlogMs(90_000, [])).toBe(0);
  });
});

describe("what the page sends now", () => {
  const at = { limitMs: 60_000, sinceLastMs: 100, burstMs: 5_000 };

  it("is a burst while the backlog is under N, never past N", () => {
    expect(allowance({ ...at, backlogMs: 0, open: false })).toBe(5_000);
    expect(allowance({ ...at, backlogMs: 57_000, open: true })).toBe(3_000);
  });

  it("stops at N while utterances wait for their text", () => {
    expect(allowance({ ...at, backlogMs: 60_000, open: false })).toBe(0);
  });

  it("goes on at real time past N while an utterance is under way: only more audio can end it", () => {
    expect(allowance({ ...at, backlogMs: 75_000, open: true })).toBe(100);
  });
});

describe("the progress and the time left", () => {
  it("is the share transcribed, and the time left at the pace so far", () => {
    expect(estimate({ doneMs: 30_000, totalMs: 120_000, elapsedMs: 10_000 })).toEqual({ fraction: 0.25, leftMs: 30_000 });
  });

  it("gives no time left before there is a pace to read", () => {
    expect(estimate({ doneMs: 0, totalMs: 120_000, elapsedMs: 4_000 })).toEqual({ fraction: 0, leftMs: null });
    expect(estimate({ doneMs: 120_000, totalMs: 120_000, elapsedMs: 9_000 })).toEqual({ fraction: 1, leftMs: 0 });
  });
});
