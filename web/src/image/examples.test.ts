import { describe, expect, it } from "vitest";
import { EDIT_IDEAS, exampleDraft, ideasFor, PICTURE_EXAMPLES } from "./examples.ts";
import { DEFAULT_LIMITS } from "./studio.ts";

// The image view's starting points (image/examples.ts): each example on one of
// the model's presets, picked into the draft whole; edit ideas by the inputs
// they need.

describe("the examples", () => {
  it("each sit on one of the model's presets, with an id of their own", () => {
    for (const example of PICTURE_EXAMPLES) expect(DEFAULT_LIMITS.sizes).toContain(example.size);
    expect(new Set(PICTURE_EXAMPLES.map((e) => e.id)).size).toBe(PICTURE_EXAMPLES.length);
    expect(new Set(PICTURE_EXAMPLES.map((e) => e.kind)).size).toBeGreaterThanOrEqual(10);
  });

  it("fill the prompt, the shape and the background; a shape the server does not serve falls back to auto", () => {
    const sticker = PICTURE_EXAMPLES.find((e) => e.id === "sticker")!;
    expect(exampleDraft(sticker, DEFAULT_LIMITS.sizes)).toEqual({ prompt: sticker.prompt, size: "2048x2048", background: "transparent" });
    const slide = PICTURE_EXAMPLES.find((e) => e.id === "slide")!;
    expect(exampleDraft(slide, ["1024x1024"])).toMatchObject({ size: "auto", background: "auto" });
  });

  it("offer edit ideas only with inputs, and the two-picture ones only with two", () => {
    expect(ideasFor(0)).toEqual([]);
    expect(ideasFor(1).every((idea) => idea.inputs === 1)).toBe(true);
    expect(ideasFor(2)).toHaveLength(EDIT_IDEAS.length);
  });
});
