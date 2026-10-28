import { describe, expect, it } from "vitest";
import { AWAY_MS, AwayWatch, escapeClears, missedSince, pickFrom } from "./missed.ts";
import type { Utterance } from "./transcript.ts";

// Where "What did I miss?" reads from (spec speech/07, the owner's pick): the last five minutes, from where the user
// left, or from a moment picked on the time spine.

const said = (startMs: number, endMs: number, id = startMs): Utterance => ({ id, startMs, endMs, text: "x", speaker: null });
/** 04:00, 06:00, 08:00 and 10:00, each 10 s long. */
const call = [said(240_000, 250_000), said(360_000, 370_000), said(480_000, 490_000), said(600_000, 610_000)];

describe("what What did I miss? reads", () => {
  it("is the last five minutes until the user leaves or picks", () => {
    // 610 s less five minutes is 310 s: the first utterance that ended after it started at 06:00.
    expect(missedSince(call, null)).toEqual({ index: 1, startMs: 360_000, detail: "last 5 min" });
  });

  it("is what came after what the user had seen, after an absence", () => {
    expect(missedSince(call, { kind: "away", seenUntilMs: 370_000 })).toEqual({ index: 2, startMs: 480_000, detail: "since 08:00 · 2 min" });
    // Nothing came yet: the moment they left, and nothing to tint.
    expect(missedSince(call, { kind: "away", seenUntilMs: 610_000 })).toEqual({ index: 4, startMs: 610_000, detail: "since 10:10" });
  });

  it("is from the moment picked on the spine, which a second press on it puts back", () => {
    expect(missedSince(call, { kind: "picked", startMs: 240_000 })).toEqual({ index: 0, startMs: 240_000, detail: "since 04:00 · 6 min" });
    expect(pickFrom(null, 240_000)).toEqual({ kind: "picked", startMs: 240_000 });
    expect(pickFrom({ kind: "picked", startMs: 240_000 }, 240_000)).toBeNull();
    expect(pickFrom({ kind: "away", seenUntilMs: 1 }, 240_000)).toEqual({ kind: "picked", startMs: 240_000 });
  });
});

describe("Escape", () => {
  const page = { closest: () => null };
  const field = { closest: (selector: string) => (selector.includes("input") ? {} : null) };

  it("puts back the last five minutes, but not from a field, nor while a drawer is out", () => {
    expect(escapeClears("Escape", page, false)).toBe(true);
    expect(escapeClears("Escape", null, false)).toBe(true);
    expect(escapeClears("Escape", field, false)).toBe(false);
    expect(escapeClears("Escape", page, true)).toBe(false);
    expect(escapeClears("Enter", page, false)).toBe(false);
  });
});

describe("an absence", () => {
  it("counts from a minute, from the first leave, once", () => {
    const watch = new AwayWatch();
    watch.leave(1_000, 3, 370_000);
    watch.leave(5_000, 3, 490_000);
    expect(watch.back(1_000 + AWAY_MS)).toEqual({ transcript: 3, from: { kind: "away", seenUntilMs: 370_000 } });
    expect(watch.back(1_000 + 2 * AWAY_MS)).toBeNull();
    watch.leave(10_000, 3, 490_000);
    expect(watch.back(10_000 + AWAY_MS - 1)).toBeNull();
  });
});
