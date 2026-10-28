import { clock } from "./export.ts";
import { heardUntil, type Utterance } from "./transcript.ts";

// Where "What did I miss?" reads from (spec speech/07, the owner's pick): the
// last five minutes, unless the user was away — the page hidden or its window
// left for a minute while listening, so from what they had seen — or picked a
// moment on the time spine. Asking it, or Escape, goes back to the five
// minutes. The sheet tints the spine over what it reads.

/** How far back "What did I miss?" reads by default: the utterances that ended in the last five minutes. */
export const MISSED_MS = 5 * 60_000;

/** How long the page must be left before a return counts as an absence. */
export const AWAY_MS = 60_000;

/** Away: from the first utterance that ended after what the user had seen. Picked: from the utterance at `startMs`. */
export type MissedFrom = { kind: "away"; seenUntilMs: number } | { kind: "picked"; startMs: number };

/** What "What did I miss?" reads: from utterance `index` on, named by its start, and what its button says of it. */
export type MissedSince = { index: number; startMs: number; detail: string };

export function missedSince(utterances: Utterance[], from: MissedFrom | null): MissedSince {
  const end = heardUntil(utterances);
  const index =
    from?.kind === "picked"
      ? utterances.findIndex((u) => u.startMs >= from.startMs)
      : utterances.findIndex((u) => u.endMs > (from ? from.seenUntilMs : end - MISSED_MS));
  const at = index === -1 ? utterances.length : index;
  const startMs = utterances[at]?.startMs ?? (from === null ? 0 : from.kind === "picked" ? from.startMs : from.seenUntilMs);
  if (from === null) return { index: at, startMs, detail: "last 5 min" };
  const minutes = Math.round((end - startMs) / 60_000);
  return { index: at, startMs, detail: minutes >= 1 ? `since ${clock(startMs)} · ${minutes} min` : `since ${clock(startMs)}` };
}

/**
 * Escape puts back the last five minutes, but not from a field, whose Escape is its own, nor while a drawer is out,
 * which it closes first.
 */
export const escapeClears = (key: string, target: { closest?: (selector: string) => unknown } | null, drawerOut: boolean): boolean =>
  key === "Escape" && !drawerOut && !target?.closest?.("input, textarea, select, [contenteditable]");

/** The point a time on the spine picks: that one, or none again when it is the one picked already. */
export function pickFrom(current: MissedFrom | null, startMs: number): MissedFrom | null {
  return current?.kind === "picked" && current.startMs === startMs ? null : { kind: "picked", startMs };
}

/**
 * The absences while listening: a leave noted once with what the Transcript listened to held then, a return that
 * comes after `AWAY_MS` or more an absence from there; a shorter one, none.
 */
export class AwayWatch {
  private left: { at: number; transcript: number; seenUntilMs: number } | null = null;

  leave(at: number, transcript: number, seenUntilMs: number) {
    this.left ??= { at, transcript, seenUntilMs };
  }

  back(at: number): { transcript: number; from: MissedFrom } | null {
    const left = this.left;
    this.left = null;
    if (!left || at - left.at < AWAY_MS) return null;
    return { transcript: left.transcript, from: { kind: "away", seenUntilMs: left.seenUntilMs } };
  }
}
