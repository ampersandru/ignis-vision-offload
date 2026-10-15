/**
 * The loop guard's rewinds in a reasoning text (ADR 0052, the server's
 * `loop_guard.rs`). Where ignis found the reasoning looping it took the
 * generation back and wrote a tag, `[LOOP DETECTED, REWINDING n TOKENS / m
 * CHARS · h]`: the `m` characters (code points) right before the blank line
 * that opens it are no longer the model's context, and the page strikes them
 * through. `h` is a short hash of them, which the server checks when the
 * reasoning comes back to it; the page does not need to.
 *
 * The tag before the hash, `[… m CHARS]`, counted the characters before
 * every trailing blank, not just the tag's own: it is still read that way.
 */

/** One run of reasoning text: kept, taken back, or a rewind's tag. */
export type ReasoningPiece =
  | { kind: "text"; text: string }
  | { kind: "struck"; text: string }
  | { kind: "tag"; text: string; tokens: number; chars: number; hash?: string };

const TAG = /\[LOOP DETECTED, REWINDING (\d+) TOKENS \/ (\d+) CHARS(?: · ([0-9a-f]{6}))?\]/g;

/** The blank line the server opens a tag with. */
const TAG_OPENING = "\n\n";

/** `reasoning` cut into its pieces, in order. Text with no tag is one piece. */
export function splitRewinds(reasoning: string): ReasoningPiece[] {
  const pieces: ReasoningPiece[] = [];
  const push = (kind: "text" | "struck", text: string) => {
    if (text) pieces.push({ kind, text });
  };
  let from = 0;
  for (const match of reasoning.matchAll(TAG)) {
    const before = reasoning.slice(from, match.index);
    const hash = match[3];
    // The hashed tag counts exactly the characters before its own blank
    // line, a line break of the struck text included; the old one stopped
    // short of every trailing blank.
    const body = hash === undefined
      ? before.replace(/\s+$/, "")
      : before.endsWith(TAG_OPENING) ? before.slice(0, -TAG_OPENING.length) : before;
    const chars = Array.from(body);
    const struck = Math.min(Number(match[2]), chars.length);
    push("text", chars.slice(0, chars.length - struck).join(""));
    push("struck", chars.slice(chars.length - struck).join(""));
    push("text", before.slice(body.length));
    const tag: ReasoningPiece = { kind: "tag", text: match[0], tokens: Number(match[1]), chars: Number(match[2]) };
    pieces.push(hash === undefined ? tag : { ...tag, hash });
    from = match.index + match[0].length;
  }
  push("text", reasoning.slice(from));
  return pieces;
}
