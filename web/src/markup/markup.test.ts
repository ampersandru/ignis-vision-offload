import { describe, expect, it } from "vitest";
import { colorName, ellipseOf, extendStroke, featherOf, isMark, promptStub, toImagePoint } from "./markup.ts";

// The markup editor's geometry (markup/markup.ts): marks in the picture's own
// pixels whatever it is shown at, and the prompt's first words for them.

describe("the marks' geometry", () => {
  it("maps a pointer on the shown picture to the picture's own pixels, held inside it", () => {
    const rect = { left: 100, top: 50, width: 500, height: 250 };
    const natural = { width: 2000, height: 1000 };
    expect(toImagePoint(350, 175, rect, natural)).toEqual([1000, 500]);
    expect(toImagePoint(50, 400, rect, natural)).toEqual([0, 1000]);
  });

  it("adds a point to a stroke only once the pointer has moved far enough", () => {
    const start: [number, number][] = [[10, 10]];
    expect(extendStroke(start, [10.5, 10.5], 2)).toBe(start);
    expect(extendStroke(start, [13, 10], 2)).toEqual([
      [10, 10],
      [13, 10],
    ]);
  });

  it("reads an ellipse from either corner of its drag, and drops one too small to mean anything", () => {
    expect(ellipseOf([300, 100], [100, 300])).toEqual({ cx: 200, cy: 200, rx: 100, ry: 100 });
    expect(isMark({ kind: "ellipse", color: "#fff", width: 4, from: [0, 0], to: [3, 30] })).toBe(false);
    expect(isMark({ kind: "ellipse", color: "#fff", width: 4, from: [0, 0], to: [30, 30] })).toBe(true);
    expect(isMark({ kind: "stroke", color: "#fff", width: 4, points: [[1, 1]] })).toBe(true);
  });

  it("feathers the kept edge by about half a percent of the short side", () => {
    expect(featherOf(2048, 2048)).toBe(12);
    expect(featherOf(100, 80)).toBe(2);
  });
});

describe("the prompt's first words for the marks", () => {
  it("match how the marks went", () => {
    expect(promptStub("drawn", { picture: 2 }, "#ff2a2a")).toBe("In image 2, change what is marked in red: ");
    expect(promptStub("mask", { picture: 1, mask: 2 }, "#ff2a2a")).toBe("In image 1, change only the area that is white in image 2: ");
    expect(colorName("#123456")).toBe("the marked colour");
  });
});
