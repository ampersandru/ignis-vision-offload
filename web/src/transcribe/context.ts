import { isSpeechLanguage, languageName } from "../speech/languages.ts";
import type { Told } from "../speech/realtime.ts";
import type { Speaker } from "./transcript.ts";

// What a Transcript's transcription is told (spec speech/07, Q128, Q130): the
// call's context, kept with the Transcript and applied, not sent as it is
// typed. The topic, the expected languages and anything else become the
// session's `prompt`; the terms, and the speakers' names once the user gave
// them, its `keywords`; a forced language its `language`. An Apply sends one
// `session.update`, which the server prefills at once and holds from the
// next utterance.

/** Free: the model detects each utterance's language, the expected ones a hint. Force: one language, always. */
export type LanguageMode = "free" | "force";

export type CallContext = {
  /** The languages expected, by ISO code; under Force the first alone. */
  languages: { mode: LanguageMode; codes: string[] };
  topic: string;
  /** Words to spell right: names, products, jargon. */
  terms: string[];
  /** Anything else the transcription should know, as written. */
  other: string;
  /** When it was last applied, in epoch ms; `null` before. */
  appliedAt: number | null;
};

export const EMPTY_CONTEXT: CallContext = { languages: { mode: "free", codes: [] }, topic: "", terms: [], other: "", appliedAt: null };

/** What the server reads of a prompt and its keywords (`speech.context_max_tokens`); past it, the last tokens. */
export const CONTEXT_MAX_TOKENS = 512;
/** Characters a token is taken to hold, low for safety. */
const CHARS_PER_TOKEN = 3;

/**
 * Whether the free languages go into the prompt as a hint (Q130): measured on cut Italian speech, `Languages:
 * Italian.` took the shortest pieces from 91 % to 100 % heard as Italian, as forcing does, without forcing's cost on
 * English speech (docs/findings/2026-10-27-asr-language-hint.md). In the model's English names, first, with its
 * period, as it was measured; worded in the call's language it did harm.
 */
export const LANGUAGE_HINT = true;

/** The languages with `code` picked: under Force it is the one, under Detect one more. */
export function pickLanguage(languages: CallContext["languages"], code: string): CallContext["languages"] {
  if (!isSpeechLanguage(code)) return languages;
  if (languages.mode === "force") return { mode: "force", codes: [code] };
  return languages.codes.includes(code) ? languages : { mode: "free", codes: [...languages.codes, code] };
}

/** The languages under `mode`: Force keeps the first alone. */
export const languagesUnder = (languages: CallContext["languages"], mode: LanguageMode): CallContext["languages"] => ({
  mode,
  codes: mode === "force" ? languages.codes.slice(0, 1) : languages.codes,
});

/** "Italian, English (detected)" or "Italian (forced)"; `null` with none. */
export function languagesLine(languages: CallContext["languages"]): string | null {
  const codes = languages.codes.filter(isSpeechLanguage);
  if (codes.length === 0) return null;
  return languages.mode === "force" ? `${languageName(codes[0])} (forced)` : `${codes.map(languageName).join(", ")} (detected)`;
}

/** The names the page gives a speaker: never a term. */
const GIVEN_NAME = /^(Speaker \d+|Me|Others)$/;

/** The speakers' names the user gave, which the transcription should spell. */
export const namedSpeakers = (speakers: Speaker[]): string[] => speakers.map((s) => s.name.trim()).filter((name) => name !== "" && !GIVEN_NAME.test(name));

/** Why `term` cannot be a keyword, or `null`: the server refuses `<`, `>` and a line break in one. */
export function termProblem(term: string): string | null {
  if (/[<>]/.test(term)) return "A term cannot hold < or >.";
  if (/[\r\n]/.test(term)) return "A term is one line.";
  if (term.length > 60) return "A term is at most 60 characters.";
  return null;
}

const unique = (words: string[]) => {
  const seen = new Set<string>();
  return words.filter((word) => {
    const key = word.toLocaleLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
};

/** `terms` with what was typed added: comma-separated, trimmed, each once whatever its case, none the server refuses. */
export function addTerms(terms: string[], typed: string): string[] {
  const added = typed
    .split(",")
    .map((term) => term.trim())
    .filter((term) => term !== "" && termProblem(term) === null);
  return unique([...terms, ...added]);
}

/** What the session is told: the prompt, the keywords and a forced language. */
export function told(context: CallContext | undefined, speakers: Speaker[]): Told {
  const c = context ?? EMPTY_CONTEXT;
  const codes = c.languages.codes.filter(isSpeechLanguage);
  const lines: string[] = [];
  if (c.languages.mode === "free" && LANGUAGE_HINT && codes.length > 0) lines.push(`Languages: ${codes.map(languageName).join(", ")}.`);
  if (c.topic.trim() !== "") lines.push(`Topic: ${c.topic.trim()}`);
  if (c.other.trim() !== "") lines.push(c.other.trim());
  const keywords = unique([...c.terms, ...namedSpeakers(speakers)].filter((term) => termProblem(term) === null));
  return { prompt: lines.join("\n"), keywords, language: c.languages.mode === "force" ? (codes[0] ?? null) : null };
}

/** About how many of the server's 512 tokens it takes. */
export const toldTokens = (t: Told): number => Math.ceil((t.prompt.length + t.keywords.join(" ").length) / CHARS_PER_TOKEN);

/** Nothing in it: no topic, language, term or other word. */
export const isBlank = (c: CallContext | undefined): boolean =>
  !c || (c.topic.trim() === "" && c.languages.codes.length === 0 && c.terms.length === 0 && c.other.trim() === "");

/** The deck's line: what the transcription is told, in short. */
export function contextSummary(c: CallContext | undefined): string {
  if (!c || isBlank(c)) return "Nothing told: the model guesses the languages and spells the names as it hears them.";
  const codes = c.languages.codes.filter(isSpeechLanguage);
  const parts: string[] = [];
  if (codes.length > 0) parts.push(c.languages.mode === "force" ? `${languageName(codes[0])} only` : codes.map(languageName).join(", "));
  if (c.topic.trim() !== "") parts.push(c.topic.trim());
  if (c.terms.length > 0) parts.push(c.terms.length === 1 ? "1 term" : `${c.terms.length} terms`);
  if (c.other.trim() !== "") parts.push("more");
  return parts.join(" · ");
}

const strings = (value: unknown): value is string[] => Array.isArray(value) && value.every((item) => typeof item === "string");

/** A stored Transcript's context, or `undefined` when it is not one this page wrote. */
export function readContext(value: unknown): CallContext | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const c = value as Record<string, unknown>;
  const languages = c.languages as Record<string, unknown> | undefined;
  const ok =
    typeof languages === "object" &&
    languages !== null &&
    (languages.mode === "free" || languages.mode === "force") &&
    strings(languages.codes) &&
    typeof c.topic === "string" &&
    strings(c.terms) &&
    typeof c.other === "string" &&
    (c.appliedAt === null || (typeof c.appliedAt === "number" && Number.isFinite(c.appliedAt)));
  return ok ? (c as CallContext) : undefined;
}
