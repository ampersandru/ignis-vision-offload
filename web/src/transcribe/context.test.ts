import { describe, expect, it } from "vitest";
import { addTerms, type CallContext, contextSummary, EMPTY_CONTEXT, isBlank, namedSpeakers, readContext, termProblem, told, toldTokens } from "./context.ts";
import type { Speaker } from "./transcript.ts";

// What a Transcript's transcription is told (spec speech/07, Q128, Q130): the
// topic, the languages and anything else as the prompt, the terms and the
// names the user gave as keywords, a forced language as the language.

const it_support: CallContext = {
  languages: { mode: "free", codes: ["it", "en"] },
  topic: "Supporto IT informale",
  terms: ["AnyDesk", "VPN"],
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
  it("is the topic, the free languages as a hint and anything else as the prompt; the terms and the names given as keywords", () => {
    expect(told(it_support, speakers)).toEqual({
      prompt: "Topic: Supporto IT informale\nLanguages: Italian, English\nSi parla di un portatile aziendale.",
      keywords: ["AnyDesk", "VPN", "Marco"],
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

describe("the terms", () => {
  it("are added comma by comma, trimmed, each once whatever its case", () => {
    expect(addTerms(["AnyDesk"], " vpn, anydesk ,Jira,, ")).toEqual(["AnyDesk", "vpn", "Jira"]);
  });

  it("refuse what the server refuses in a keyword", () => {
    expect(termProblem("<b>")).toBe("A term cannot hold < or >.");
    expect(termProblem("a\nb")).toBe("A term is one line.");
    expect(termProblem("AnyDesk")).toBeNull();
    expect(addTerms([], "<b>, ok")).toEqual(["ok"]);
  });
});

describe("the deck's line", () => {
  it("says what is told, in short, or that nothing is", () => {
    expect(contextSummary(it_support)).toBe("Italian, English · Supporto IT informale · 2 terms · more");
    expect(contextSummary({ ...EMPTY_CONTEXT, languages: { mode: "force", codes: ["it"] } })).toBe("Italian only");
    expect(contextSummary(undefined)).toBe("Nothing told: the model guesses the languages and spells the names as it hears them.");
    expect(isBlank({ ...EMPTY_CONTEXT, appliedAt: 5 })).toBe(true);
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
