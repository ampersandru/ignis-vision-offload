import { describe, expect, it } from "vitest";
import {
  addTerms,
  asksFirst,
  type CallContext,
  contextAtStart,
  contextOf,
  contextSummary,
  contextTokensOf,
  copiedDraft,
  EMPTY_CONTEXT,
  fromDefault,
  isBlank,
  languagesLine,
  languagesUnder,
  mergeDraft,
  namedSpeakers,
  pickLanguage,
  readContext,
  sameContext,
  termProblem,
  told,
  toldTokens,
} from "./context.ts";
import type { Speaker } from "./transcript.ts";

// What a Transcript's transcription is told (spec speech/07, Q128, Q130): the
// topic, the languages and anything else as the prompt, the terms and the
// names the user gave as keywords, a forced language as the language.

const it_support: CallContext = {
  languages: { mode: "free", codes: ["it", "en"] },
  topic: "Supporto IT informale",
  terms: ["PostgreSQL", "VPN"],
  other: "Si parla di un portatile aziendale.",
  appliedAt: 1_000,
};

const speakers: Speaker[] = [
  { id: 1, name: "Me", role: "me" },
  { id: 2, name: "Marco", role: "voice" },
  { id: 3, name: "Speaker 2", role: "voice" },
  { id: 4, name: "Others", role: "others" },
];

describe("what the transcription is told", () => {
  it("is the free languages as a hint, first, in the model's names, as measured, then the topic and anything else, as the prompt; the terms and the names given as keywords", () => {
    expect(told(it_support, speakers)).toEqual({
      prompt: "Languages: Italian, English.\nTopic: Supporto IT informale\nSi parla di un portatile aziendale.",
      keywords: ["PostgreSQL", "VPN", "Marco"],
      language: null,
    });
  });

  it("forces the first language alone under Force, and leaves it out of the prompt", () => {
    const forced = told({ ...it_support, languages: { mode: "force", codes: ["it", "en"] } }, []);
    expect(forced.language).toBe("it");
    expect(forced.prompt).not.toContain("Languages");
  });

  it("is nothing for a Transcript with no context, and never a speaker the page named", () => {
    expect(told(undefined, speakers)).toEqual({ prompt: "", keywords: ["Marco"], language: null });
    expect(namedSpeakers(speakers)).toEqual(["Marco"]);
    expect(told(EMPTY_CONTEXT, [])).toEqual({ prompt: "", keywords: [], language: null });
  });

  it("counts about a token for three characters, against the server's 512", () => {
    expect(toldTokens({ prompt: "a".repeat(30), keywords: ["bb", "cc"], language: null })).toBe(12);
  });
});

describe("the languages (Q130)", () => {
  it("are several under Detect and one under Force, the first kept when Force is picked", () => {
    const detect = pickLanguage(pickLanguage({ mode: "free", codes: [] }, "it"), "en");
    expect(detect).toEqual({ mode: "free", codes: ["it", "en"] });
    expect(pickLanguage(detect, "it")).toBe(detect);
    expect(languagesUnder(detect, "force")).toEqual({ mode: "force", codes: ["it"] });
    expect(pickLanguage({ mode: "force", codes: ["it"] }, "de")).toEqual({ mode: "force", codes: ["de"] });
    expect(pickLanguage(detect, "xx")).toBe(detect);
  });

  it("read as the model's names, detected or forced", () => {
    expect(languagesLine({ mode: "free", codes: ["it", "en"] })).toBe("Italian, English (detected)");
    expect(languagesLine({ mode: "force", codes: ["it"] })).toBe("Italian (forced)");
    expect(languagesLine({ mode: "free", codes: [] })).toBeNull();
  });
});

describe("the terms", () => {
  it("are added comma by comma, trimmed, each once whatever its case", () => {
    expect(addTerms(["PostgreSQL"], " vpn, postgresql ,Kubernetes,, ")).toEqual(["PostgreSQL", "vpn", "Kubernetes"]);
  });

  it("refuse what the server refuses in a keyword", () => {
    expect(termProblem("<b>")).toBe("A term cannot hold < or >.");
    expect(termProblem("a\nb")).toBe("A term is one line.");
    expect(termProblem("PostgreSQL")).toBeNull();
    expect(addTerms([], "<b>, ok")).toEqual(["ok"]);
  });
});

describe("the line of what is told", () => {
  it("says what is told, in short, or that nothing is", () => {
    expect(contextSummary(it_support)).toBe("Italian, English · Supporto IT informale · 2 terms · more");
    expect(contextSummary({ ...EMPTY_CONTEXT, languages: { mode: "force", codes: ["it"] } })).toBe("Italian only");
    expect(contextSummary(undefined)).toBe("Nothing told: the model guesses the languages and spells the names as it hears them.");
    expect(isBlank({ ...EMPTY_CONTEXT, appliedAt: 5 })).toBe(true);
  });
});

describe("a draft applied (Q128)", () => {
  it("changes only what it changed since Edit, keeping a suggestion accepted or a fix's term added meanwhile", () => {
    const base: CallContext = { languages: { mode: "free", codes: ["it"] }, topic: "IT", terms: ["VPN", "SSH"], other: "", appliedAt: 1 };
    const draft: CallContext = { ...base, topic: "Supporto IT", terms: ["VPN", "Kubernetes"] };
    const meanwhile: CallContext = { ...base, languages: { mode: "free", codes: ["it", "en"] }, terms: ["VPN", "SSH", "PostgreSQL"], other: "Da un fix." };
    expect(mergeDraft(base, draft, meanwhile, 9)).toEqual({
      languages: { mode: "free", codes: ["it", "en"] },
      topic: "Supporto IT",
      terms: ["VPN", "PostgreSQL", "Kubernetes"],
      other: "Da un fix.",
      appliedAt: 9,
    });
    expect(mergeDraft(base, { ...base, languages: { mode: "force", codes: ["it"] } }, meanwhile, 9).languages).toEqual({ mode: "force", codes: ["it"] });
  });
});

describe("the default and the first Start (the owner's word)", () => {
  const saved: CallContext = { ...it_support, appliedAt: null };
  const fresh = { source: null };

  it("a Transcript never started tells the default, one of its own or once started its own alone", () => {
    expect(contextOf(fresh, saved)).toBe(saved);
    expect(fromDefault(fresh, saved)).toBe(true);
    expect(contextOf(fresh, null)).toEqual(EMPTY_CONTEXT);
    const own: CallContext = { ...EMPTY_CONTEXT, topic: "Altro", appliedAt: 4 };
    expect(contextOf({ source: null, context: own }, saved)).toBe(own);
    expect(fromDefault({ source: null, context: own }, saved)).toBe(false);
    expect(contextOf({ source: "call" }, saved)).toEqual(EMPTY_CONTEXT);
  });

  it("a Start keeps what it tells, applied then, or the draft open merged onto it", () => {
    expect(contextAtStart(fresh, null, saved, 7)).toEqual({ ...saved, appliedAt: 7 });
    expect(contextAtStart(fresh, null, null, 7)).toEqual({ ...EMPTY_CONTEXT, appliedAt: 7 });
    expect(contextAtStart({ source: "call", context: it_support }, null, saved, 7)).toEqual(it_support);
    const draft = { ...saved, topic: "Rilascio" };
    expect(contextAtStart(fresh, { base: saved, context: draft }, saved, 7)).toEqual({ ...saved, topic: "Rilascio", appliedAt: 7 });
  });

  it("asks at the first Start alone: nothing applied, nothing being edited", () => {
    expect(asksFirst(fresh, null)).toBe(true);
    expect(asksFirst({ source: null, context: { ...saved, appliedAt: null } }, null)).toBe(true);
    expect(asksFirst(fresh, saved)).toBe(false);
    expect(asksFirst({ source: null, context: it_support }, null)).toBe(false);
    expect(asksFirst({ source: "microphone" }, null)).toBe(false);
  });

  it("opens another Transcript's context as a draft never applied, over what this one tells", () => {
    const other: CallContext = { ...EMPTY_CONTEXT, topic: "Rilascio", terms: ["Tekton"], appliedAt: 3 };
    expect(copiedDraft(fresh, other, saved)).toEqual({ base: saved, context: { ...other, appliedAt: null } });
    expect(copiedDraft(fresh, other, null).base).toEqual(EMPTY_CONTEXT);
  });

  it("tells two contexts the same whenever each was applied", () => {
    expect(sameContext(it_support, saved)).toBe(true);
    expect(sameContext(it_support, { ...saved, terms: ["VPN"] })).toBe(false);
    expect(sameContext(it_support, { ...saved, languages: { mode: "force", codes: ["it"] } })).toBe(false);
  });
});

describe("the server's context", () => {
  it("is its speech.context_max_tokens, 512 until it is read or when it says nothing usable", () => {
    expect(contextTokensOf({ speech: { context_max_tokens: 1_024 } })).toBe(1_024);
    expect(contextTokensOf({ speech: { context_max_tokens: "lots" } })).toBe(512);
    expect(contextTokensOf(null)).toBe(512);
  });
});

describe("the context as kept", () => {
  it("reads back as written, and a malformed one as none", () => {
    expect(readContext(it_support)).toEqual(it_support);
    expect(readContext({ ...it_support, languages: { mode: "auto", codes: [] } })).toBeUndefined();
    expect(readContext({ ...it_support, terms: [3] })).toBeUndefined();
    expect(readContext("Standup.")).toBeUndefined();
  });
});
