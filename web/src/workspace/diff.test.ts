import { describe, expect, it } from "vitest";
import { changeAnchors, type DiffLine, folded, lineDiff, sideBySide } from "./diff.ts";

// The line diff between two versions of a text file, as the panel's history
// shows it: each line kept, removed or added, with its numbers in the version
// before and the version after, and long runs of kept lines folded away.

/** A diff written as the history reads it: "  " kept, "- " removed, "+ " added. */
const signs = (lines: DiffLine[]) => lines.map((l) => `${{ same: " ", removed: "-", added: "+" }[l.kind]} ${l.text}`);

describe("the line diff", () => {
  it("keeps every line of two equal texts, numbered in both", () => {
    expect(lineDiff("a\nb\n", "a\nb\n")).toEqual([
      { kind: "same", text: "a", before: 1, after: 1 },
      { kind: "same", text: "b", before: 2, after: 2 },
    ]);
  });

  it("shows a changed line as the old one removed, then the new one added", () => {
    expect(lineDiff("a\nb\nc\n", "a\nB\nc\n")).toEqual([
      { kind: "same", text: "a", before: 1, after: 1 },
      { kind: "removed", text: "b", before: 2 },
      { kind: "added", text: "B", after: 2 },
      { kind: "same", text: "c", before: 3, after: 3 },
    ]);
  });

  it("finds the lines kept between lines removed and added anywhere", () => {
    expect(signs(lineDiff("# Trip\n- Rome\n- Pisa\n- Naples\n- Bari\n", "# Trip\n- Pisa\n- Naples\n- Lecce\n- Bari\n"))).toEqual([
      "  # Trip",
      "- - Rome",
      "  - Pisa",
      "  - Naples",
      "+ - Lecce",
      "  - Bari",
    ]);
    expect(lineDiff("one\ntwo\nthree", "zero\none\nthree\nfour").map((l) => [l.kind, "before" in l ? l.before : undefined, "after" in l ? l.after : undefined])).toEqual([
      ["added", undefined, 1],
      ["same", 1, 2],
      ["removed", 2, undefined],
      ["same", 3, 3],
      ["added", undefined, 4],
    ]);
  });

  it("adds every line of a file that was empty, and removes every line of one emptied", () => {
    expect(signs(lineDiff("", "a\nb"))).toEqual(["+ a", "+ b"]);
    expect(signs(lineDiff("a\nb\n", ""))).toEqual(["- a", "- b"]);
  });

  it("compares lines without their line endings", () => {
    expect(signs(lineDiff("a\r\nb\r\n", "a\nb"))).toEqual(["  a", "  b"]);
  });

  it("finds two changes far apart in a long file", () => {
    const long = Array.from({ length: 3000 }, (_, i) => `line ${i + 1}`);
    const after = long.map((l, i) => (i === 9 ? "ten" : i === 2989 ? "two thousand nine hundred ninety" : l));
    const changed = lineDiff(long.join("\n"), after.join("\n")).filter((l) => l.kind !== "same");
    expect(changed).toEqual([
      { kind: "removed", text: "line 10", before: 10 },
      { kind: "added", text: "ten", after: 10 },
      { kind: "removed", text: "line 2990", before: 2990 },
      { kind: "added", text: "two thousand nine hundred ninety", after: 2990 },
    ]);
  });

  it("gives back both versions, keeping as many lines as they have in common", () => {
    // Random texts of a few distinct lines, against the longest common subsequence counted another way.
    let seed = 7;
    const random = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
    const text = () => Array.from({ length: Math.floor(random() * 12) }, () => "abcd"[Math.floor(random() * 4)]);
    const common = (x: string[], y: string[]): number => {
      let prev = Array<number>(y.length + 1).fill(0);
      for (let i = 1; i <= x.length; i++) {
        const cur = Array<number>(y.length + 1).fill(0);
        for (let j = 1; j <= y.length; j++) cur[j] = x[i - 1] === y[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
        prev = cur;
      }
      return prev[y.length];
    };
    for (let round = 0; round < 300; round++) {
      const x = text();
      const y = text();
      const lines = lineDiff(x.join("\n"), y.join("\n"));
      expect(lines.filter((l) => l.kind !== "added").map((l) => l.text)).toEqual(x);
      expect(lines.filter((l) => l.kind !== "removed").map((l) => l.text)).toEqual(y);
      expect(lines.filter((l) => l.kind === "same")).toHaveLength(common(x, y));
    }
  });

  it("shows two texts too different to compare line by line as the old lines removed and the new ones added, between the lines they share at each end", () => {
    // A limit of 3 lines removed and added: the four lines between the shared ends take 4 (two removed, two added).
    expect(signs(lineDiff("top\n1\n2\n3\n4\nend", "top\n2\n1\n4\n3\nend", 3))).toEqual([
      "  top",
      "- 1",
      "- 2",
      "- 3",
      "- 4",
      "+ 2",
      "+ 1",
      "+ 4",
      "+ 3",
      "  end",
    ]);
  });
});

describe("the folded diff", () => {
  const ten = Array.from({ length: 10 }, (_, i) => `line ${i + 1}`);

  it("keeps the lines near a change and folds the runs further away, a lone line shown rather than folded", () => {
    const after = ten.map((l, i) => (i === 4 ? "line five" : l));
    const rows = folded(lineDiff(ten.join("\n"), after.join("\n")), 3);
    expect(rows.map((r) => (r.kind === "skipped" ? `… ${r.count}` : `${r.kind} ${r.text}`))).toEqual([
      "same line 1",
      "same line 2",
      "same line 3",
      "same line 4",
      "removed line 5",
      "added line five",
      "same line 6",
      "same line 7",
      "same line 8",
      "… 2",
    ]);
  });

  it("shows a line it would fold alone between two changes", () => {
    // Line 6 is the only one further than three lines from both changes.
    const after = ten.map((l, i) => (i === 1 ? "two" : i === 9 ? "ten" : l));
    const rows = folded(lineDiff(ten.join("\n"), after.join("\n")), 3);
    expect(rows.some((r) => r.kind === "skipped")).toBe(false);
    expect(rows).toHaveLength(12);
  });

  it("folds two equal texts whole", () => {
    expect(folded(lineDiff(ten.join("\n"), ten.join("\n")), 3)).toEqual([{ kind: "skipped", count: 10 }]);
  });
});

// The two-column layout and the blocks previous and next visit.

/** Side-by-side rows written as the sheet reads them: "before | after", with the numbers, or "(n unchanged)". */
const cells = (rows: ReturnType<typeof sideBySide>) =>
  rows.map((r) => (r.kind === "skipped" ? `(${r.count} unchanged)` : `${r.left ? `${r.left.n} ${r.left.text}` : "-"} | ${r.right ? `${r.right.n} ${r.right.text}` : "-"}`));

describe("the side-by-side diff", () => {
  it("faces a kept line with itself, numbered in each version", () => {
    expect(cells(sideBySide(lineDiff("a\nb\n", "a\nb\n"), true))).toEqual(["1 a | 1 a", "2 b | 2 b"]);
  });

  it("pairs a run of removed lines with the run of added lines row by row", () => {
    const rows = sideBySide(lineDiff("k\nx\ny\nk2\n", "k\nX\nY\nk2\n"));
    expect(cells(rows)).toEqual(["1 k | 1 k", "2 x | 2 X", "3 y | 3 Y", "4 k2 | 4 k2"]);
    expect(rows.map((r) => r.kind === "pair" && r.changed)).toEqual([false, true, true, false]);
  });

  it("pads the shorter side with empty cells", () => {
    expect(cells(sideBySide(lineDiff("k\nx\ny\nz\nk2\n", "k\nX\nk2\n")))).toEqual(["1 k | 1 k", "2 x | 2 X", "3 y | -", "4 z | -", "5 k2 | 3 k2"]);
    expect(cells(sideBySide(lineDiff("k\nk2\n", "k\nA\nB\nk2\n")))).toEqual(["1 k | 1 k", "- | 2 A", "- | 3 B", "2 k2 | 4 k2"]);
  });

  const long = Array.from({ length: 20 }, (_, i) => `line ${i + 1}`);
  const lines = lineDiff(long.join("\n"), [...long.slice(0, 19), "the end"].join("\n"));

  it("folds the kept lines far from a change across both columns", () => {
    const rows = cells(sideBySide(lines));
    expect(rows[0]).toBe("(16 unchanged)");
    expect(rows).toContain("17 line 17 | 17 line 17");
    expect(rows.at(-1)).toBe("20 line 20 | 20 the end");
  });

  it("shows every line when asked to show all", () => {
    const rows = sideBySide(lines, true);
    expect(rows).toHaveLength(20);
    expect(rows.some((r) => r.kind === "skipped")).toBe(false);
  });
});

describe("the change anchors", () => {
  const before = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\n";
  const after = "a\nB\nc\nd\ne\nf\ng\nh\ni\nj\nK\nL\n";

  it("names the first row of each block of changed rows, in the unified rows", () => {
    const rows = folded(lineDiff(before, after));
    expect(rows.map((r) => r.kind)).toEqual(["same", "removed", "added", "same", "same", "same", "skipped", "same", "same", "same", "removed", "added", "added"]);
    expect(changeAnchors(rows)).toEqual([1, 10]);
  });

  it("names the same blocks in the side-by-side rows", () => {
    expect(changeAnchors(sideBySide(lineDiff(before, after)))).toEqual([1, 9]);
  });

  it("has none for two equal texts", () => {
    expect(changeAnchors(sideBySide(lineDiff("a\n", "a\n")))).toEqual([]);
  });
});
