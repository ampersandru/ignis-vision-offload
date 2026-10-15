import { describe, expect, it } from "vitest";
import { splitRewinds } from "./rewinds.ts";

/** The tag as the server wrote it before ADR 0052's hash. */
const tag = (tokens: number, chars: number) => `[LOOP DETECTED, REWINDING ${tokens} TOKENS / ${chars} CHARS]`;

/** The tag as the server writes it now: `m` and a hash of the struck text. */
const hashed = (tokens: number, chars: number, hash = "0a1b2c") =>
  `[LOOP DETECTED, REWINDING ${tokens} TOKENS / ${chars} CHARS · ${hash}]`;

describe("splitRewinds", () => {
  it("leaves reasoning without a rewind as one piece", () => {
    expect(splitRewinds("plain thinking")).toEqual([{ kind: "text", text: "plain thinking" }]);
  });

  it("strikes the characters the tag counts, stopping at the blank line before it", () => {
    const reasoning = `Kept.\nthe 301? the 301?\n\n${tag(9, 17)}\n\nWait, I'm repeating. Moving on.`;
    expect(splitRewinds(reasoning)).toEqual([
      { kind: "text", text: "Kept.\n" },
      { kind: "struck", text: "the 301? the 301?" },
      { kind: "text", text: "\n\n" },
      { kind: "tag", text: tag(9, 17), tokens: 9, chars: 17 },
      { kind: "text", text: "\n\nWait, I'm repeating. Moving on." },
    ]);
  });

  it("counts code points, as the server does, not UTF-16 units", () => {
    const pieces = splitRewinds(`ok 🙂🙂\n\n${tag(2, 2)}`);
    expect(pieces[0]).toEqual({ kind: "text", text: "ok " });
    expect(pieces[1]).toEqual({ kind: "struck", text: "🙂🙂" });
  });

  it("handles every rewind of a reply, each against the text since the last", () => {
    const reasoning = `a b c\n\n${tag(1, 1)}\n\nnudge one. x y\n\n${tag(1, 3)}\n\nclosed`;
    const struck = splitRewinds(reasoning).filter((p) => p.kind === "struck").map((p) => p.text);
    expect(struck).toEqual(["c", "x y"]);
  });

  it("never strikes past the start of the text it belongs to", () => {
    const pieces = splitRewinds(`short\n\n${tag(50, 500)}`);
    expect(pieces[0]).toEqual({ kind: "struck", text: "short" });
  });
});

describe("splitRewinds, the hashed tag", () => {
  it("strikes exactly the characters before the tag's own blank line, and carries the hash", () => {
    const reasoning = `Kept.\nthe 301? the 301?\n\n${hashed(9, 17)}\n\nWait, I'm repeating. Moving on.`;
    expect(splitRewinds(reasoning)).toEqual([
      { kind: "text", text: "Kept.\n" },
      { kind: "struck", text: "the 301? the 301?" },
      { kind: "text", text: "\n\n" },
      { kind: "tag", text: hashed(9, 17), tokens: 9, chars: 17, hash: "0a1b2c" },
      { kind: "text", text: "\n\nWait, I'm repeating. Moving on." },
    ]);
  });

  it("counts a line break the struck text ended with, as the server does", () => {
    // The struck text ends in a line break: three before the tag, the
    // server's count takes the first of them.
    const pieces = splitRewinds(`Kept. the loop\n\n\n${hashed(4, 9)}\n\nnudge`);
    expect(pieces.slice(0, 3)).toEqual([
      { kind: "text", text: "Kept. " },
      { kind: "struck", text: "the loop\n" },
      { kind: "text", text: "\n\n" },
    ]);
    // A trailing space is a character like any other.
    expect(splitRewinds(`a loop \n\n${hashed(2, 5)}`)[1]).toEqual({ kind: "struck", text: "loop " });
  });

  it("counts code points, and never strikes past the start of its text", () => {
    expect(splitRewinds(`ok 🙂🙂\n\n${hashed(2, 2)}`).slice(0, 2)).toEqual([
      { kind: "text", text: "ok " },
      { kind: "struck", text: "🙂🙂" },
    ]);
    expect(splitRewinds(`short\n\n${hashed(50, 500)}`)[0]).toEqual({ kind: "struck", text: "short" });
  });

  it("reads both forms in one reply", () => {
    const reasoning = `a b c\n\n${tag(1, 1)}\n\nnudge one. x y\n\n${hashed(1, 3)}\n\nclosed`;
    const struck = splitRewinds(reasoning).filter((p) => p.kind === "struck").map((p) => p.text);
    expect(struck).toEqual(["c", "x y"]);
  });

  it("takes a hash that is not six lowercase hex digits for no tag at all", () => {
    const text = `x\n\n[LOOP DETECTED, REWINDING 1 TOKENS / 1 CHARS · ZZZZZZ]`;
    expect(splitRewinds(text)).toEqual([{ kind: "text", text }]);
  });
});
