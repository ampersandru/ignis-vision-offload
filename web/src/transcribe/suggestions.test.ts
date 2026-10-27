import { describe, expect, it } from "vitest";
import {
  accept,
  analysisOffers,
  autoTitled,
  dismiss,
  dropTerm,
  dueForAnalysis,
  followFixes,
  languageOffer,
  openSuggestions,
  readAnalysis,
  readSuggestions,
  suggest,
} from "./suggestions.ts";
import { addUtterance, beginListening, createTranscript, type Transcript, undoFixes } from "./transcript.ts";

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
    const t = accept(suggest(heard(italian(1)), { kind: "terms", terms: ["PostgreSQL"] }, 5), 1, 6, { always: true });
    expect(t.suggestions?.always).toEqual(["terms"]);
    const later = suggest(t, { kind: "terms", terms: ["VPN", "postgresql"] }, 7);
    expect(later.context?.terms).toEqual(["PostgreSQL", "VPN"]);
    expect(openSuggestions(later)).toEqual([]);
    expect(later.suggestions?.items.at(-1)).toMatchObject({ kind: "terms", terms: ["VPN"], status: "accepted", auto: true });
  });

  it("of terms gathers new ones into the open one, and a term taken out is never offered again", () => {
    let t = suggest(heard(italian(1)), { kind: "terms", terms: ["PostgreSQL", "VPN"] }, 5);
    t = suggest(t, { kind: "terms", terms: ["vpn", "Kubernetes"] }, 6);
    const [open] = openSuggestions(t);
    expect(open).toMatchObject({ kind: "terms", terms: ["PostgreSQL", "VPN", "Kubernetes"] });
    t = dropTerm(t, open.id, "Kubernetes");
    expect(openSuggestions(t)).toMatchObject([{ kind: "terms", terms: ["PostgreSQL", "VPN"] }]);
    expect(openSuggestions(suggest(t, { kind: "terms", terms: ["Kubernetes"] }, 7))).toMatchObject([{ terms: ["PostgreSQL", "VPN"] }]);
  });

  it("of languages applies them as what the Transcript is told", () => {
    const t = suggest(heard(italian(1)), { kind: "languages", languages: { mode: "force", codes: ["it"] }, why: "x" }, 5);
    expect(accept(t, 1, 9).context?.languages).toEqual({ mode: "force", codes: ["it"] });
  });
});

describe("a look's findings", () => {
  it("are read leniently: bullets and bold are fine, NONE and lines in no form are dropped", () => {
    const text = ["- **TOPIC:** Supporto IT informale.", "TITLE: Assistenza PostgreSQL", "TERM: PostgreSQL", "* TERM: VPN", "TERM: NONE", "Here are the items:", "TERM: <b>"].join("\n");
    expect(readAnalysis(text)).toEqual({ topic: "Supporto IT informale", title: "Assistenza PostgreSQL", terms: ["PostgreSQL", "VPN"], fixes: [] });
    expect(readAnalysis("NONE")).toEqual({ topic: null, title: null, terms: [], fixes: [] });
  });

  it("offer the topic while none is told, a title while the listening's stands, and the terms", () => {
    const t = heard(italian(1));
    expect(autoTitled(t)).toBe(true);
    expect(analysisOffers(t, { topic: "Supporto IT", title: "Assistenza PostgreSQL", terms: ["PostgreSQL"], fixes: [] })).toEqual([
      { kind: "topic", topic: "Supporto IT" },
      { kind: "title", title: "27 Oct 15:42 · Assistenza PostgreSQL" },
      { kind: "terms", terms: ["PostgreSQL"] },
    ]);
    const renamed = { ...t, title: "Con Marco", context: { languages: { mode: "free" as const, codes: [] }, topic: "IT", terms: [], other: "", appliedAt: 1 } };
    expect(analysisOffers(renamed, { topic: "Supporto IT", title: "Assistenza", terms: [], fixes: [] })).toEqual([]);
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

describe("a fix (Q133)", () => {
  function it_call(): Transcript {
    let t = beginListening(createTranscript(1), { source: "call", at: STARTED });
    t = addUtterance(t, { stream: "s", startMs: 27_000, endMs: 31_000, text: "Aspetta che mi ricollego a Postgress.", voice: { kind: "others" } });
    t = addUtterance(t, { stream: "s", startMs: 40_000, endMs: 44_000, text: "Su postgress vedo i dati, non su Postgressx.", voice: { kind: "others" } });
    t = addUtterance(t, { stream: "s", startMs: 50_000, endMs: 52_000, text: "Ok.", voice: { kind: "me" } });
    return t;
  }

  it("is read from a FIX line, its time, the words as heard and as meant; a rephrasing or no change is dropped", () => {
    const found = readAnalysis(["FIX: [00:27] Postgress => PostgreSQL", "- FIX: 01:05 \"kuber netis\" → Kubernetes.", "FIX: [00:40] non su questo schermo vedo => vedo lo schermo bene ora", "FIX: [00:41] VPN => vpn"].join("\n"));
    expect(found.fixes).toEqual([
      { at: "00:27", from: "Postgress", to: "PostgreSQL" },
      { at: "01:05", from: "kuber netis", to: "Kubernetes" },
    ]);
  });

  it("covers every utterance that holds the words as words, whatever their case, and none holding them inside another word", () => {
    const t = it_call();
    const [offer] = analysisOffers(t, { topic: null, title: null, terms: [], fixes: [{ at: "00:27", from: "Postgress", to: "PostgreSQL" }] });
    expect(offer).toEqual({ kind: "fix", from: "Postgress", to: "PostgreSQL", utterances: [1, 2] });
    expect(analysisOffers(t, { topic: null, title: null, terms: [], fixes: [{ at: "00:27", from: "Teamviewer", to: "TeamViewer" }] })).toEqual([]);
  });

  it("accepted, fixes the words, keeps the text as heard, marks what it put in, and makes the right spelling a term", () => {
    const t = suggest(it_call(), { kind: "fix", from: "Postgress", to: "PostgreSQL", utterances: [1, 2] }, 1);
    const fixed = accept(t, 1, 5);
    expect(fixed.utterances.map((u) => u.text)).toEqual(["Aspetta che mi ricollego a PostgreSQL.", "Su PostgreSQL vedo i dati, non su Postgressx.", "Ok."]);
    expect(fixed.utterances[0]).toMatchObject({ heard: "Aspetta che mi ricollego a Postgress.", fixed: ["PostgreSQL"] });
    expect(fixed.utterances[2]).not.toHaveProperty("heard");
    expect(fixed.context?.terms).toEqual(["PostgreSQL"]);
  });

  it("reaches the utterances heard after it was accepted: a card for them alone, or applied when accepted always; never once dismissed or undone", () => {
    const later = (t: Transcript) => addUtterance(t, { stream: "s", startMs: 60_000, endMs: 62_000, text: "Riavvio Postgress.", voice: { kind: "others" } });
    const offered = suggest(it_call(), { kind: "fix", from: "Postgress", to: "PostgreSQL", utterances: [1, 2] }, 1);
    const accepted = accept(offered, 1, 2);
    const asked = followFixes(later(accepted), 3);
    expect(openSuggestions(asked)).toMatchObject([{ kind: "fix", from: "Postgress", to: "PostgreSQL", utterances: [4] }]);
    expect(followFixes(asked, 4)).toBe(asked);
    const always = followFixes(later(accept(offered, 1, 2, { always: true })), 3);
    expect(always.utterances.at(-1)).toMatchObject({ text: "Riavvio PostgreSQL.", heard: "Riavvio Postgress." });
    expect(openSuggestions(always)).toEqual([]);
    const undone = undoFixes(accepted, 1);
    expect(followFixes(undone, 3)).toBe(undone);
    const dismissed = dismiss(offered, 1);
    expect(suggest(later(dismissed), { kind: "fix", from: "Postgress", to: "PostgreSQL", utterances: [4] }, 3).suggestions).toEqual(dismissed.suggestions);
  });

  it("offered again while its card is open, gathers the new utterances into that card", () => {
    const t = suggest(it_call(), { kind: "fix", from: "Postgress", to: "PostgreSQL", utterances: [1] }, 1);
    const again = suggest(t, { kind: "fix", from: "Postgress", to: "PostgreSQL", utterances: [1, 2] }, 2);
    expect(openSuggestions(again)).toMatchObject([{ id: 1, utterances: [1, 2] }]);
  });

  it("stands beside the other fixes, one card each", () => {
    let t = suggest(it_call(), { kind: "fix", from: "Postgress", to: "PostgreSQL", utterances: [1] }, 1);
    t = suggest(t, { kind: "fix", from: "Ok", to: "Okay", utterances: [3] }, 2);
    expect(openSuggestions(t).map((s) => s.kind)).toEqual(["fix", "fix"]);
    expect(suggest(t, { kind: "fix", from: "postgress", to: "postgresql", utterances: [1] }, 3)).toBe(t);
  });
});

describe("the suggestions as kept", () => {
  it("read back those this page can read", () => {
    const fix = { id: 3, kind: "fix", from: "Postgress", to: "PostgreSQL", utterances: [1], status: "accepted" };
    const kept = { items: [{ id: 1, kind: "topic", topic: "IT", status: "open" }, { id: 2, kind: "party", status: "open" }, fix, { ...fix, id: 4, utterances: ["1"] }], always: ["terms", "jokes", "fix"] };
    expect(readSuggestions(kept)).toEqual({ items: [{ id: 1, kind: "topic", topic: "IT", status: "open" }, fix], always: ["terms", "fix"] });
    expect(readSuggestions("x")).toBeUndefined();
  });
});
