import { isSpeechLanguage, languageName } from "../speech/languages.ts";
import type { Told } from "../speech/realtime.ts";
import type { ConfigDocument } from "../server/routes.ts";
import { CHARS_PER_TOKEN } from "./assist.ts";
import type { Speaker, Transcript } from "./transcript.ts";

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

/** What the server reads of a prompt and its keywords, `speech.context_max_tokens`, unless it says otherwise. */
export const DEFAULT_CONTEXT_TOKENS = 512;

/** The running server's `speech.context_max_tokens` (`GET /v1/config`); past it, the server keeps the last tokens. */
export function contextTokensOf(config: ConfigDocument | null): number {
  const value = config?.speech?.context_max_tokens;
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : DEFAULT_CONTEXT_TOKENS;
}

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
  // Under Detect, the languages are a hint (Q130): measured on cut Italian speech, `Languages: Italian.` took the
  // shortest pieces from 91 % to 100 % heard as Italian, as forcing does, without forcing's cost on English speech
  // (docs/findings/2026-10-27-asr-language-hint.md). In the model's English names, first, with its period, as it was
  // measured; worded in the call's language it did harm.
  if (c.languages.mode === "free" && codes.length > 0) lines.push(`Languages: ${codes.map(languageName).join(", ")}.`);
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

/** What it is told, in short: the line of the first Start's question. */
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

/** The same told: the languages, the topic, the terms and the rest, whenever each was applied. */
export const sameContext = (a: CallContext, b: CallContext): boolean =>
  a.languages.mode === b.languages.mode &&
  a.languages.codes.join() === b.languages.codes.join() &&
  a.topic.trim() === b.topic.trim() &&
  a.terms.join("\n") === b.terms.join("\n") &&
  a.other.trim() === b.other.trim();

/** The Transcript is told this browser's default, `saved`: it was never started and holds no context of its own. */
export const fromDefault = (t: Pick<Transcript, "context" | "source">, saved: CallContext | null): boolean =>
  t.context === undefined && t.source === null && saved !== null;

/** What the Transcript tells: its own context, else the default while it was never started, else nothing. */
export const contextOf = (t: Pick<Transcript, "context" | "source">, saved: CallContext | null): CallContext =>
  t.context ?? (t.source === null ? saved : null) ?? EMPTY_CONTEXT;

/**
 * What a Start tells and the Transcript keeps: a draft open merged onto what it tells, else what it tells, applied
 * now if it never was, so the default a new Transcript shows is the one sent.
 */
export function contextAtStart(
  t: Pick<Transcript, "context" | "source">,
  pending: { base: CallContext; context: CallContext } | null,
  saved: CallContext | null,
  now: number,
): CallContext {
  const held = contextOf(t, saved);
  return pending ? mergeDraft(pending.base, pending.context, held, now) : { ...held, appliedAt: held.appliedAt ?? now };
}

/** A draft of `t`'s context from another Transcript's, `theirs`, never applied: what Use the context of opens. */
export const copiedDraft = (t: Pick<Transcript, "context" | "source">, theirs: CallContext, saved: CallContext | null): { base: CallContext; context: CallContext } => ({
  base: contextOf(t, saved),
  context: { ...theirs, appliedAt: null },
});

/**
 * The first Start asks before it starts (the owner's word: it costs a click and spares a call told nothing, or
 * another's): a Transcript never started, its context never applied, none being edited; the default too.
 */
export const asksFirst = (t: Pick<Transcript, "context" | "source">, draft: CallContext | null): boolean =>
  t.source === null && (t.context?.appliedAt ?? null) === null && draft === null;

export const isStrings = (value: unknown): value is string[] => Array.isArray(value) && value.every((item) => typeof item === "string");

const lower = (terms: string[]) => new Set(terms.map((term) => term.toLocaleLowerCase()));

/**
 * A draft applied over what the Transcript holds now (Q128): only what the draft changed since `base`, the context
 * when Edit opened it, so a suggestion accepted or a fix's term added meanwhile is kept. A field edited in the draft
 * is the draft's; the terms the draft added are added and those it removed are removed.
 */
export function mergeDraft(base: CallContext, draft: CallContext, current: CallContext, now: number): CallContext {
  const sameLanguages = base.languages.mode === draft.languages.mode && base.languages.codes.join() === draft.languages.codes.join();
  const was = lower(base.terms);
  const kept = lower(draft.terms);
  const removed = new Set([...was].filter((term) => !kept.has(term)));
  const added = draft.terms.filter((term) => !was.has(term.toLocaleLowerCase()));
  return {
    languages: sameLanguages ? current.languages : draft.languages,
    topic: draft.topic === base.topic ? current.topic : draft.topic,
    terms: addTerms(current.terms.filter((term) => !removed.has(term.toLocaleLowerCase())), added.join(",")),
    other: draft.other === base.other ? current.other : draft.other,
    appliedAt: now,
  };
}

/** A stored Transcript's context, or `undefined` when it is not one this page wrote. */
export function readContext(value: unknown): CallContext | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const c = value as Record<string, unknown>;
  const languages = c.languages as Record<string, unknown> | undefined;
  const ok =
    typeof languages === "object" &&
    languages !== null &&
    (languages.mode === "free" || languages.mode === "force") &&
    isStrings(languages.codes) &&
    typeof c.topic === "string" &&
    isStrings(c.terms) &&
    typeof c.other === "string" &&
    (c.appliedAt === null || (typeof c.appliedAt === "number" && Number.isFinite(c.appliedAt)));
  return ok ? (c as CallContext) : undefined;
}
