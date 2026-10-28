// A moment named in the notes or an answer, [mm:ss], as a link to it (spec
// speech/07, Q141): the Markdown gets links of its own (`#t=mm:ss`), a click
// on one is a jump, and the sheet brings the utterance of that moment into
// view and lights it. The page has one sheet, so a jump is page-wide.

const listeners = new Set<(ms: number) => void>();

/** The sheet's: called with each jump's time, ms on the Transcript's clock. */
export function onJump(listener: (ms: number) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function jumpTo(ms: number) {
  for (const listener of listeners) listener(ms);
}

/** `mm:ss` or `h:mm:ss` (a `~` before it allowed) in ms; `null` for anything else. */
export function clockMs(clock: string): number | null {
  const m = /^~?(?:(\d+):)?(\d{1,2}):(\d{2})$/.exec(clock.trim());
  if (!m) return null;
  return ((Number(m[1] ?? 0) * 60 + Number(m[2])) * 60 + Number(m[3])) * 1_000;
}

/** `markdown` with each `[mm:ss]` that is not already a link's text made a link to its moment. */
export const linkTimes = (markdown: string): string => markdown.replace(/\[(~?(?:\d+:)?\d{1,2}:\d{2})\](?!\()/g, "[$1](#t=$1)");

/** A click inside rendered notes or answers: a jump when it was on a time's link, the link's own navigation stopped. */
export function jumpOnClick(e: { target: EventTarget | null; preventDefault: () => void }) {
  const link = (e.target as Element | null)?.closest?.('a[href^="#t="]');
  if (!link) return;
  e.preventDefault();
  const ms = clockMs(decodeURIComponent(link.getAttribute("href")!.slice(3)));
  if (ms !== null) jumpTo(ms);
}

/** The utterance a jump to `ms` lands on: the last that starts within that second or before it. */
export function landing(utterances: { id: number; startMs: number }[], ms: number): number | null {
  let found: number | null = null;
  for (const u of utterances) if (u.startMs < ms + 1_000) found = u.id;
  return found ?? utterances[0]?.id ?? null;
}
