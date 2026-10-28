import { describe, expect, it } from "vitest";
import { clockMs, jumpOnClick, landing, linkTimes, onJump } from "./jump.ts";

// A moment named [mm:ss] in the notes or an answer, as a link to it (spec
// speech/07, Q141).

describe("a moment's link", () => {
  it("is made of every [mm:ss] or [h:mm:ss] that is not already a link's text", () => {
    expect(linkTimes("Deciso [03:12], poi [1:02:03] e [~00:15]; [link](https://x) e [04:00](#t=04:00).")).toBe(
      "Deciso [03:12](#t=03:12), poi [1:02:03](#t=1:02:03) e [~00:15](#t=~00:15); [link](https://x) e [04:00](#t=04:00).",
    );
  });

  it("reads its clock, and lands on the last utterance that starts within that second", () => {
    expect(clockMs("03:12")).toBe(192_000);
    expect(clockMs("1:02:03")).toBe(3_723_000);
    expect(clockMs("~00:15")).toBe(15_000);
    expect(clockMs("soon")).toBeNull();
    const utterances = [
      { id: 1, startMs: 0 },
      { id: 2, startMs: 12_400 },
      { id: 3, startMs: 15_900 },
      { id: 4, startMs: 20_000 },
    ];
    expect(landing(utterances, 15_000)).toBe(3);
    expect(landing(utterances, 13_000)).toBe(2);
    expect(landing([], 1)).toBeNull();
  });

  it("clicked, jumps instead of following the link", () => {
    const jumps: number[] = [];
    const off = onJump((ms) => jumps.push(ms));
    let prevented = false;
    const link = { getAttribute: () => "#t=00:42" };
    jumpOnClick({ target: { closest: () => link } as unknown as EventTarget, preventDefault: () => (prevented = true) });
    off();
    expect(prevented).toBe(true);
    expect(jumps).toEqual([42_000]);
  });
});
