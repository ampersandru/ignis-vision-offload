import { describe, expect, it } from "vitest";
import { splitRewinds } from "./rewinds.ts";

const tag = (tokens: number, chars: number) => `[LOOP DETECTED, REWINDING ${tokens} TOKENS / ${chars} CHARS]`;

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
