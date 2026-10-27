import { describe, expect, it } from "vitest";
import {
  accept,
  analysisOffers,
  autoTitled,
  dismiss,
  dropTerm,
  dueForAnalysis,
  languageOffer,
  openSuggestions,
  readAnalysis,
  readSuggestions,
  suggest,
} from "./suggestions.ts";
import { addUtterance, beginListening, createTranscript, type Transcript } from "./transcript.ts";

// What the Transcribe view suggests for a Transcript (spec speech/07, Q132):
// the languages from the server's word on each utterance, the topic, a title
// and the terms from the text model; each waits for Accept, Always or Dismiss.

const STARTED = new Date(2026, 9, 27, 15, 42);

/** A call heard as `[language, seconds]` per utterance. */
function heard(langs: [string, number][]): Transcript {
  let t = beginListening(createTranscript(1), { source: "call", at: STARTED });
  let at = 0;
  for (const [language, seconds] of langs) {
    t = addUtterance(t, { stream: "s", startMs: at, endMs: at + seconds * 1_000, text: `Frase a ${at}.`, voice: { kind: "others" }, language });
    at += seconds * 1_000 + 500;
  }
  return t;
}

const italian = (n: number): [string, number][] => Array.from({ length: n }, () => ["it", 4] as [string, number]);

describe("the languages suggested (Q130, Q132)", () => {
  it("wait for six utterances with a language", () => {
    expect(languageOffer(heard(italian(5)))).toBeNull();
  });

  it("force the language spoken when the others came only on short utterances, as on the owner's first call", () => {
    const offer = languageOffer(heard([...italian(8), ["th", 1.2], ["hi", 2], ["it", 4]]));
    expect(offer).toEqual({ kind: "languages", languages: { mode: "force", codes: ["it"] }, why: "2 short utterances came out as Thai and Hindi, among 9 in Italian." });
  });

  it("detect the languages spoken when they hold their share, and say nothing once the context says the same", () => {
    const t = heard([...italian(6), ["en", 4], ["en", 5], ["en", 3]]);
    const offer = languageOffer(t);
    expect(offer).toEqual({ kind: "languages", languages: { mode: "free", codes: ["it", "en"] }, why: "Heard Italian (6) and English (3)." });
    const told = { ...t, context: { languages: { mode: "free" as const, codes: ["en", "it"] }, topic: "", terms: [], other: "", appliedAt: 1 } };
    expect(languageOffer(told)).toBeNull();
  });
});

describe("a suggestion", () => {
  const offer = { kind: "topic" as const, topic: "Supporto IT" };

  it("waits open, is offered once, and its acceptance tells the Transcript", () => {
    const t = suggest(heard(italian(1)), offer, 5);
    expect(openSuggestions(t)).toEqual([{ ...offer, id: 1, status: "open" }]);
    expect(suggest(t, offer, 6)).toBe(t);
    const accepted = accept(t, 1, 7);
    expect(accepted.context?.topic).toBe("Supporto IT");
    expect(accepted.context?.appliedAt).toBe(7);
    expect(openSuggestions(accepted)).toEqual([]);
    expect(suggest(accepted, offer, 8)).toBe(accepted);
  });

  it("dismissed, is never offered again; another of its kind takes the place of an open one", () => {
    const t = suggest(heard(italian(1)), offer, 5);
    expect(suggest(dismiss(t, 1), offer, 6).suggestions?.items).toEqual([{ ...offer, id: 1, status: "dismissed" }]);
    const replaced = suggest(t, { kind: "topic", topic: "Assistenza remota" }, 6);
    expect(openSuggestions(replaced).map((s) => s.kind === "topic" && s.topic)).toEqual(["Assistenza remota"]);
  });

  it("accepted always, applies every later one of its kind in this Transcript as it comes", () => {
    const t = accept(suggest(heard(italian(1)), { kind: "terms", terms: ["AnyDesk"] }, 5), 1, 6, { always: true });
    expect(t.suggestions?.always).toEqual(["terms"]);
    const later = suggest(t, { kind: "terms", terms: ["VPN", "anydesk"] }, 7);
    expect(later.context?.terms).toEqual(["AnyDesk", "VPN"]);
    expect(openSuggestions(later)).toEqual([]);
    expect(later.suggestions?.items.at(-1)).toMatchObject({ kind: "terms", terms: ["VPN"], status: "accepted", auto: true });
  });

  it("of terms gathers new ones into the open one, and a term taken out is never offered again", () => {
    let t = suggest(heard(italian(1)), { kind: "terms", terms: ["AnyDesk", "VPN"] }, 5);
    t = suggest(t, { kind: "terms", terms: ["vpn", "Jira"] }, 6);
    const [open] = openSuggestions(t);
    expect(open).toMatchObject({ kind: "terms", terms: ["AnyDesk", "VPN", "Jira"] });
    t = dropTerm(t, open.id, "Jira");
    expect(openSuggestions(t)).toMatchObject([{ kind: "terms", terms: ["AnyDesk", "VPN"] }]);
    expect(openSuggestions(suggest(t, { kind: "terms", terms: ["Jira"] }, 7))).toMatchObject([{ terms: ["AnyDesk", "VPN"] }]);
  });

  it("of languages applies them as what the Transcript is told", () => {
    const t = suggest(heard(italian(1)), { kind: "languages", languages: { mode: "force", codes: ["it"] }, why: "x" }, 5);
    expect(accept(t, 1, 9).context?.languages).toEqual({ mode: "force", codes: ["it"] });
  });
});

describe("a look's findings", () => {
  it("are read leniently: bullets and bold are fine, NONE and lines in no form are dropped", () => {
    const text = ["- **TOPIC:** Supporto IT informale.", "TITLE: Assistenza AnyDesk", "TERM: AnyDesk", "* TERM: VPN", "TERM: NONE", "Here are the items:", "TERM: <b>"].join("\n");
    expect(readAnalysis(text)).toEqual({ topic: "Supporto IT informale", title: "Assistenza AnyDesk", terms: ["AnyDesk", "VPN"] });
    expect(readAnalysis("NONE")).toEqual({ topic: null, title: null, terms: [] });
  });

  it("offer the topic while none is told, a title while the listening's stands, and the terms", () => {
    const t = heard(italian(1));
    expect(autoTitled(t)).toBe(true);
    expect(analysisOffers(t, { topic: "Supporto IT", title: "Assistenza AnyDesk", terms: ["AnyDesk"] })).toEqual([
      { kind: "topic", topic: "Supporto IT" },
      { kind: "title", title: "27 Oct 15:42 · Assistenza AnyDesk" },
      { kind: "terms", terms: ["AnyDesk"] },
    ]);
    const renamed = { ...t, title: "Con Marco", context: { languages: { mode: "free" as const, codes: [] }, topic: "IT", terms: [], other: "", appliedAt: 1 } };
    expect(analysisOffers(renamed, { topic: "Supporto IT", title: "Assistenza", terms: [] })).toEqual([]);
  });

  it("come about every minute and a half while something new was said, the first at six utterances, once more at the end", () => {
    expect(dueForAnalysis({ utterances: 5, analyzed: 0, sinceLastMs: null })).toBe(false);
    expect(dueForAnalysis({ utterances: 6, analyzed: 0, sinceLastMs: null })).toBe(true);
    expect(dueForAnalysis({ utterances: 9, analyzed: 6, sinceLastMs: 90_000 })).toBe(true);
    expect(dueForAnalysis({ utterances: 8, analyzed: 6, sinceLastMs: 200_000 })).toBe(false);
    expect(dueForAnalysis({ utterances: 9, analyzed: 6, sinceLastMs: 89_999 })).toBe(false);
    expect(dueForAnalysis({ utterances: 7, analyzed: 6, sinceLastMs: 1, final: true })).toBe(true);
  });
});

describe("the suggestions as kept", () => {
  it("read back those this page can read", () => {
    const kept = { items: [{ id: 1, kind: "topic", topic: "IT", status: "open" }, { id: 2, kind: "party", status: "open" }], always: ["terms", "jokes"] };
    expect(readSuggestions(kept)).toEqual({ items: [{ id: 1, kind: "topic", topic: "IT", status: "open" }], always: ["terms"] });
    expect(readSuggestions("x")).toBeUndefined();
  });
});
