import { fixedWordsOf, type Transcript } from "./transcript.ts";

// Search in the Transcribe view (spec speech/07, Q136): in the Transcript
// open, every occurrence of the query in its utterances, in order, whatever
// the case and the accents ("perche" finds "perché"); on the list, the
// Transcripts that hold it, in their title or their utterances. A part of an
// utterance's text is a hit, a fixed word (Q133), both, or plain.

/** A character as a search compares it: lower case, no accent; one for one, so offsets hold. */
function foldChar(c: string): string {
  const base = c.normalize("NFD").replace(/\p{M}/gu, "").toLocaleLowerCase();
  if (base.length === 1) return base;
  const lower = c.toLocaleLowerCase();
  return lower.length === 1 ? lower : c;
}

/** `text` as a search compares it, of the same length. */
export const fold = (text: string): string => Array.from(text.split(""), foldChar).join("");

/** An occurrence: the utterance, and where in its text. */
export type Hit = { utterance: number; start: number; end: number };

/** Every occurrence of `query` in `t`'s utterances, in order, none overlapping; none for a blank query. */
export function hitsIn(t: Transcript, query: string): Hit[] {
  const q = fold(query.trim());
  if (q === "") return [];
  const hits: Hit[] = [];
  for (const u of t.utterances) {
    const text = fold(u.text);
    for (let at = text.indexOf(q); at !== -1; at = text.indexOf(q, at + q.length)) hits.push({ utterance: u.id, start: at, end: at + q.length });
  }
  return hits;
}

/** How a Transcript holds `query`, for its row on the list: its utterances' hits, and whether its title does. */
export function matchesOf(t: Transcript, title: string, query: string): { hits: number; title: boolean } {
  const q = fold(query.trim());
  return { hits: hitsIn(t, query).length, title: q !== "" && fold(title).includes(q) };
}

/** A hit within one utterance's text, and whether the search stands on it. */
export type HitSpan = { start: number; end: number; current: boolean };

/** A part of an utterance's text as the sheet draws it. */
export type Part = { text: string; fixed: boolean; hit: "current" | "other" | null };

/**
 * `text` in parts: the words fixes put in (`fixed`, as words of their own) and the search's `hits`, `current` the
 * one the search stands on.
 */
export function partsOf(text: string, fixed: string[], hits: HitSpan[]): Part[] {
  const fixes: [number, number][] = [];
  if (fixed.length > 0) for (const m of text.matchAll(fixedWordsOf(fixed))) fixes.push([m.index, m.index + m[0].length]);
  const cuts = new Set([0, text.length, ...fixes.flat(), ...hits.flatMap((h) => [h.start, h.end])]);
  const points = [...cuts].filter((p) => p >= 0 && p <= text.length).sort((a, b) => a - b);
  const parts: Part[] = [];
  for (let i = 0; i + 1 < points.length; i++) {
    const [from, to] = [points[i], points[i + 1]];
    const hit = hits.find((h) => h.start <= from && to <= h.end);
    const part: Part = { text: text.slice(from, to), fixed: fixes.some(([a, b]) => a <= from && to <= b), hit: hit ? (hit.current ? "current" : "other") : null };
    const last = parts.at(-1);
    if (last && last.fixed === part.fixed && last.hit === part.hit) last.text += part.text;
    else parts.push(part);
  }
  return parts;
}
