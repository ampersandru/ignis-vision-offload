import { describe, expect, it } from "vitest";
import { insertAt, wordCount } from "./insert.ts";

// Where a dictation lands (spec speech/04, Q17): at the cursor, selected, with
// a space added on a side that needs one.

describe("insertAt", () => {
  it("puts the text at the cursor and selects it, adding a space only where a word would touch it", () => {
    expect(insertAt("Prima di toccare il piano: ", 27, "apri vram.rs")).toEqual({ value: "Prima di toccare il piano: apri vram.rs", start: 27, end: 39 });
    expect(insertAt("Prima di toccare il piano:", 26, "apri vram.rs")).toEqual({ value: "Prima di toccare il piano: apri vram.rs", start: 27, end: 39 });
    expect(insertAt("helloworld", 5, "there")).toEqual({ value: "hello there world", start: 6, end: 11 });
    expect(insertAt("hello\nworld", 6, "there")).toEqual({ value: "hello\nthere world", start: 6, end: 11 });
    expect(insertAt("", 0, "ciao")).toEqual({ value: "ciao", start: 0, end: 4 });
  });

  it("trims what the model gave, and clamps a cursor past the end", () => {
    expect(insertAt("a", 9, "  per il  ")).toEqual({ value: "a per il", start: 2, end: 8 });
  });
});

describe("wordCount", () => {
  it("counts the words the hint line reports", () => {
    expect(wordCount(" Apri vram.rs e controlla\nperché ")).toBe(5);
    expect(wordCount("   ")).toBe(0);
  });
});
