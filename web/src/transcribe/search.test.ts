import { describe, expect, it } from "vitest";
import { fold, hitsIn, matchesOf, partsOf } from "./search.ts";
import { addUtterance, createTranscript, type Transcript } from "./transcript.ts";

// Search in the Transcribe view (spec speech/07, Q136): every occurrence, in
// order, whatever the case and the accents; a Transcript's matches for the
// list; the parts an utterance is drawn in.

function said(texts: string[]): Transcript {
  let t = createTranscript(1);
  texts.forEach((text, i) => (t = addUtterance(t, { stream: "s", startMs: i * 1_000, endMs: i * 1_000 + 500, text, voice: null })));
  return t;
}

describe("a search", () => {
  it("folds case and accents one for one, so its offsets are the text's", () => {
    expect(fold("Perché È così")).toBe("perche e cosi");
    expect(fold("Perché È così")).toHaveLength("Perché È così".length);
  });

  it("finds every occurrence in order, none for a blank query", () => {
    const t = said(["Il budget, poi il BUDGET.", "Nessuno.", "Perché il budgét?"]);
    expect(hitsIn(t, "budget")).toEqual([
      { utterance: 1, start: 3, end: 9 },
      { utterance: 1, start: 18, end: 24 },
      { utterance: 3, start: 10, end: 16 },
    ]);
    expect(hitsIn(t, "  ")).toEqual([]);
  });

  it("tells the list a Transcript's matches, and whether its title holds the query", () => {
    const t = said(["Il budget."]);
    expect(matchesOf(t, "Budget review", "budget")).toEqual({ hits: 1, title: true });
    expect(matchesOf(t, "Call", "server")).toEqual({ hits: 0, title: false });
  });

  it("draws an utterance in parts: hits, fixed words, both, plain", () => {
    expect(partsOf("Apri PostgreSQL ora", ["PostgreSQL"], [{ start: 5, end: 12, current: true }])).toEqual([
      { text: "Apri ", fixed: false, hit: null },
      { text: "Postgre", fixed: true, hit: "current" },
      { text: "SQL", fixed: true, hit: null },
      { text: " ora", fixed: false, hit: null },
    ]);
  });
});
