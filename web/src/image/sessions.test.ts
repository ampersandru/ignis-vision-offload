import { describe, expect, it } from "vitest";
import {
  addMarked,
  addToPool,
  coverOf,
  endStep,
  fromStore,
  type ImageSession,
  seedClash,
  seedsBehind,
  imagesOf,
  inputNumbers,
  keepFor,
  landOutput,
  newSession,
  reusePicture,
  reuseStep,
  setDraft,
  settleRunning,
  startStep,
  type StepSettings,
  titleOf,
  toggleInput,
  UNTITLED,
  unmark,
  wireImages,
} from "./sessions.ts";

// An image session (image/sessions.ts): the pool numbered for good, steps
// sent with inputs from it, the next step going on from the last picture,
// marked pictures as new pool entries, and what a request sends for them.

const settings: StepSettings = { size: "auto", count: 1, seed: null, background: "auto", enhance: false };
const pic = (id: string) => ({ id, width: 64, height: 48 });

describe("an image session", () => {
  it("starts empty and untitled, and numbers its pool on from the last, never reusing a number", () => {
    let s = newSession("s1", 1);
    expect(s.title).toBe(UNTITLED);
    let added: number[];
    [s, added] = addToPool(s, [{ ...pic("a"), origin: "upload", name: "a.png" }, { ...pic("b"), origin: "upload" }], 2);
    expect(added).toEqual([1, 2]);
    s = { ...s, pool: s.pool.filter((p) => p.n !== 2) };
    [s, added] = addToPool(s, [{ ...pic("c"), origin: "upload" }], 3);
    expect(added).toEqual([2]);
    [s, added] = addToPool(s, [{ ...pic("d"), origin: "upload" }], 4, true);
    expect(added).toEqual([3]);
    expect(s.draft.inputs).toEqual([3]);
    expect(s.updated).toBe(4);
  });

  it("toggles a pool picture in and out of the inputs, the newest last", () => {
    let [s] = addToPool(newSession("s", 0), [{ ...pic("a"), origin: "upload" }, { ...pic("b"), origin: "upload" }], 0);
    s = toggleInput(s, 2, 1);
    s = toggleInput(s, 1, 1);
    expect(s.draft.inputs).toEqual([2, 1]);
    s = toggleInput(s, 2, 1);
    expect(s.draft.inputs).toEqual([1]);
    expect(toggleInput(s, 9, 1)).toBe(s);
  });

  it("names itself after its first prompt, and lands each picture in the pool and on its step", () => {
    let s = newSession("s", 0);
    let step: number;
    [s, step] = startStep(s, "a fox in the snow, dusk", [], settings, 1);
    expect(step).toBe(1);
    expect(s.title).toBe("a fox in the snow, dusk");
    expect(s.steps[0].status).toBe("running");
    let n: number;
    [s, n] = landOutput(s, step, { ...pic("out1"), seed: "41" }, 2);
    [s] = landOutput(s, step, { ...pic("out2"), seed: "42" }, 2);
    expect(n).toBe(1);
    expect(s.steps[0].outputs).toEqual([1, 2]);
    expect(s.pool.map((p) => [p.n, p.origin, p.step])).toEqual([
      [1, "made", 1],
      [2, "made", 1],
    ]);
    [s] = startStep(s, "make it night", [1], settings, 3);
    expect(s.title).toBe("a fox in the snow, dusk");
  });

  it("goes on from the step's first picture: it becomes the only input, and the prompt and the seed clear", () => {
    let s = setDraft(newSession("s", 0), { prompt: "a fox", seedText: "41", inputs: [] }, 0);
    let step: number;
    [s, step] = startStep(s, "a fox", [], { ...settings, seed: "41" }, 1);
    [s] = landOutput(s, step, { ...pic("o1"), seed: "41" }, 2);
    [s] = landOutput(s, step, { ...pic("o2"), seed: "42" }, 2);
    s = endStep(s, step, { status: "done", notes: ["scaled"], revised: "a red fox" }, 3);
    expect(s.draft).toMatchObject({ prompt: "", inputs: [1], seedText: "" });
    expect(s.steps[0]).toMatchObject({ status: "done", notes: ["scaled"], revised: "a red fox" });
  });

  it("with the same seed chosen, keeps the last picture's seed in the field after each step", () => {
    let s = setDraft(newSession("s", 0), { sameSeed: true }, 0);
    let step: number;
    [s, step] = startStep(s, "a fox", [], settings, 1);
    [s] = landOutput(s, step, { ...pic("o1"), seed: "41" }, 2);
    expect(endStep(s, step, { status: "done" }, 3).draft).toMatchObject({ seedText: "41", sameSeed: true });
    expect(newSession("n", 0).draft.sameSeed).toBe(false);
  });

  it("never sends, for an edit, a seed behind its inputs: the picture's own, or one it was made or marked from", () => {
    // #1 made with seed 7; #2 made from #1 with seed 8; #3 #2 marked; #4 an upload.
    let s = newSession("s", 0);
    let step: number;
    [s, step] = startStep(s, "a fox", [], { ...settings, seed: "7" }, 1);
    [s] = landOutput(s, step, { ...pic("o1"), seed: "7" }, 1);
    s = endStep(s, step, { status: "done" }, 1);
    [s, step] = startStep(s, "at night", [1], { ...settings, seed: "8" }, 2);
    [s] = landOutput(s, step, { ...pic("o2"), seed: "8" }, 2);
    s = endStep(s, step, { status: "done" }, 2);
    [s] = addMarked(s, 2, pic("o2-drawn"), { marks: [], mode: "drawn", keepOutside: false, mask: "m", region: "r" }, 3);
    [s] = addToPool(s, [{ ...pic("up"), origin: "upload" }], 4);
    expect([...seedsBehind(s, 3)].sort()).toEqual(["7", "8"]);
    expect(seedClash(s, [3], "8")).toBe(3);
    expect(seedClash(s, [4, 2], "7")).toBe(2);
    expect(seedClash(s, [4], "7")).toBeUndefined();
    expect(seedClash(s, [2], "9")).toBeUndefined();
    expect(seedClash(s, [2], null)).toBeUndefined();
  });

  it("blanks a seed field that the old 'Keep the last seed' filled, and forgets the option", () => {
    const old = { ...newSession("s", 0), draft: { ...newSession("s", 0).draft, seedText: "41", keepSeed: true } } as ImageSession;
    const read = fromStore(old, 1);
    expect(read.draft.seedText).toBe("");
    expect(read.draft).not.toHaveProperty("keepSeed");
    const typed = setDraft(newSession("t", 0), { seedText: "41" }, 0);
    expect(fromStore(typed, 1).draft.seedText).toBe("41");
  });

  it("empties the prompt box when a step is sent, and gives the prompt back when the step made nothing", () => {
    let s = setDraft(newSession("s", 0), { prompt: "a fox", inputs: [] }, 0);
    let step: number;
    [s, step] = startStep(s, "a fox", [], settings, 1);
    expect(s.draft.prompt).toBe("");
    s = endStep(s, step, { status: "failed", error: "The queue is full" }, 2);
    expect(s.draft.prompt).toBe("a fox");
    // Not over what the owner has typed since.
    let t = setDraft(newSession("t", 0), { prompt: "a fox" }, 0);
    [t, step] = startStep(t, "a fox", [], settings, 1);
    t = setDraft(t, { prompt: "a wolf" }, 1);
    expect(endStep(t, step, { status: "stopped" }, 2).draft.prompt).toBe("a wolf");
    expect(s.steps[0]).toMatchObject({ status: "failed", error: "The queue is full" });
  });

  it("reads a session the store kept from before seeds were digits: its seeds as digits, a running step stopped", () => {
    let [s, step] = startStep(newSession("s", 0), "x", [], { ...settings, seed: 7 as unknown as string }, 1);
    [s] = landOutput(s, step, { ...pic("o"), seed: 5936113633719656000 as unknown as string }, 1);
    const read = fromStore(s, 2);
    expect(read.steps[0]).toMatchObject({ status: "stopped", settings: { seed: "7" } });
    expect(read.pool[0].seed).toBe("5936113633719656000");
  });

  it("ends a step a page left running as stopped", () => {
    let [s] = startStep(newSession("s", 0), "x", [], settings, 1);
    s = settleRunning(s, 5);
    expect(s.steps[0].status).toBe("stopped");
    expect(settleRunning(s, 6)).toBe(s);
  });

  it("puts a step, or a picture's making, back in the draft", () => {
    let s = newSession("s", 0);
    let step: number;
    [s, step] = startStep(s, "a fox", [], { ...settings, size: "2752x1536", seed: "7", background: "transparent" }, 1);
    [s] = landOutput(s, step, { ...pic("o1"), seed: "9" }, 1);
    s = endStep(s, step, { status: "done" }, 2);
    expect(reuseStep(s, 1, 3).draft).toMatchObject({ prompt: "a fox", size: "2752x1536", seedText: "7", background: "transparent" });
    expect(reusePicture(s, 1, 3).draft).toMatchObject({ prompt: "a fox", seedText: "9", size: "2752x1536", inputs: [], count: 1 });
  });
});

describe("marked pictures", () => {
  function marked(mode: "drawn" | "mask", keepOutside = true) {
    let [s] = addToPool(newSession("s", 0), [{ ...pic("photo"), origin: "upload" }, { ...pic("jacket"), origin: "upload" }], 0, true);
    let n: number;
    [s, n] = addMarked(s, 1, pic("photo-drawn"), { marks: [], mode, keepOutside, mask: "photo-mask", region: "photo-region" }, 1);
    return { s, n };
  }

  it("are new pool pictures that take their source's place among the inputs", () => {
    const { s, n } = marked("drawn");
    expect(n).toBe(3);
    expect(s.draft.inputs).toEqual([3, 2]);
    expect(s.pool[2]).toMatchObject({ origin: "marked", markup: { source: 1, mode: "drawn" } });
  });

  it("marked again, replace their earlier marking among the inputs; unmarked, give their source its place back", () => {
    const { s, n } = marked("drawn");
    const [again, m] = addMarked(s, 1, pic("photo-drawn-2"), { marks: [], mode: "drawn", keepOutside: true, mask: "m2", region: "r2" }, 2, n);
    expect(again.draft.inputs).toEqual([m, 2]);
    expect(again.pool[m - 1].markup?.source).toBe(1);
    expect(unmark(again, m, 3).draft.inputs).toEqual([1, 2]);
  });

  it("send their drawn picture, or in mask mode the source then the mask, and number the prompt's images so", () => {
    const drawn = marked("drawn");
    expect(wireImages(drawn.s, drawn.s.draft.inputs).map((w) => w.id)).toEqual(["photo-drawn", "jacket"]);
    expect(inputNumbers(drawn.s, drawn.s.draft.inputs)).toEqual([
      { n: 3, picture: 1 },
      { n: 2, picture: 2 },
    ]);
    const mask = marked("mask");
    expect(wireImages(mask.s, mask.s.draft.inputs).map((w) => w.id)).toEqual(["photo", "photo-mask", "jacket"]);
    expect(inputNumbers(mask.s, mask.s.draft.inputs)).toEqual([
      { n: 3, picture: 1, mask: 2 },
      { n: 2, picture: 3 },
    ]);
  });

  it("keep a result to their area, laid over their source", () => {
    expect(keepFor(marked("drawn").s, [3, 2])).toEqual({ source: "photo", region: "photo-region" });
    expect(keepFor(marked("drawn", false).s, [3, 2])).toBeNull();
  });

  it("name their masks and areas as images of the session", () => {
    expect([...imagesOf(marked("mask").s)].sort()).toEqual(["jacket", "photo", "photo-drawn", "photo-mask", "photo-region"]);
  });
});

describe("the list", () => {
  it("shows a session by its last picture made, else its last one added", () => {
    let [s] = addToPool(newSession("s", 0), [{ ...pic("up"), origin: "upload" }], 0);
    expect(coverOf(s)?.id).toBe("up");
    let step: number;
    [s, step] = startStep(s, "x", [1], settings, 1);
    [s] = landOutput(s, step, pic("made"), 1);
    [s] = addToPool(s, [{ ...pic("later"), origin: "upload" }], 2);
    expect(coverOf(s)?.id).toBe("made");
    expect(coverOf(newSession("e", 0))).toBeUndefined();
  });

  it("titles a session by its prompt's first line, cut", () => {
    expect(titleOf("  a fox\nin the snow")).toBe("a fox");
    expect(titleOf("x".repeat(60))).toHaveLength(48);
    expect(titleOf("  ")).toBe(UNTITLED);
  });
});
