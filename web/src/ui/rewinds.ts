/**
 * The loop guard's rewinds in a reasoning text (exploration, the server's
 * `loop_guard.rs`). Where ignis found the reasoning looping it took the
 * generation back and wrote a tag, `[LOOP DETECTED, REWINDING n TOKENS / m
 * CHARS]`: the `m` characters (code points) before it are no longer the
 * model's context, and the page strikes them through.
 */

/** One run of reasoning text: kept, taken back, or a rewind's tag. */
export type ReasoningPiece =
  | { kind: "text"; text: string }
  | { kind: "struck"; text: string }
  | { kind: "tag"; text: string; tokens: number; chars: number };

const TAG = /\[LOOP DETECTED, REWINDING (\d+) TOKENS \/ (\d+) CHARS\]/g;

/** `reasoning` cut into its pieces, in order. Text with no tag is one piece. */
export function splitRewinds(reasoning: string): ReasoningPiece[] {
  const pieces: ReasoningPiece[] = [];
  const push = (kind: "text" | "struck", text: string) => {
    if (text) pieces.push({ kind, text });
  };
  let from = 0;
  for (const match of reasoning.matchAll(TAG)) {
    const before = reasoning.slice(from, match.index);
    // The tag opens on a blank line the server wrote; the count stops short
    // of it.
    const body = before.replace(/\s+$/, "");
    const chars = Array.from(body);
    const struck = Math.min(Number(match[2]), chars.length);
    push("text", chars.slice(0, chars.length - struck).join(""));
    push("struck", chars.slice(chars.length - struck).join(""));
    push("text", before.slice(body.length));
    pieces.push({ kind: "tag", text: match[0], tokens: Number(match[1]), chars: Number(match[2]) });
    from = match.index + match[0].length;
  }
  push("text", reasoning.slice(from));
  return pieces;
}
