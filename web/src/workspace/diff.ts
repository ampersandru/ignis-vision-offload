import { linesOf } from "./files.ts";

// The line diff between two versions of a text file, as the workspace panel's
// history shows it (spec #170). Lines are compared without their line
// endings. The lines both versions share at their start and their end are
// kept as they are; the lines between are compared by Myers' algorithm, the
// fewest lines removed and added, a removal shown before the addition that
// replaces it. Its time and memory grow with the number of changes, not the
// file's length; two texts that differ by more than `limit` lines show the
// lines between removed, then the new ones added, rather than hold the page.

/** A line of the diff, numbered from 1 in the version it is in: both for a kept line. */
export type DiffLine =
  | { kind: "same"; text: string; before: number; after: number }
  | { kind: "removed"; text: string; before: number }
  | { kind: "added"; text: string; after: number };

/** A row of the folded diff: a line, or a run of kept lines folded away. */
export type DiffRow = DiffLine | { kind: "skipped"; count: number };

/** The most lines removed and added the diff looks for, past which it shows the lines between the shared ends replaced whole. */
export const DIFF_LIMIT = 1_000;

export function lineDiff(before: string, after: string, limit = DIFF_LIMIT): DiffLine[] {
  const a = linesOf(before);
  const b = linesOf(after);
  let start = 0;
  while (start < a.length && start < b.length && a[start] === b[start]) start++;
  let end = 0;
  while (end < a.length - start && end < b.length - start && a[a.length - 1 - end] === b[b.length - 1 - end]) end++;
  const head: DiffLine[] = [];
  for (let i = 0; i < start; i++) head.push({ kind: "same", text: a[i], before: i + 1, after: i + 1 });
  const tail: DiffLine[] = [];
  for (let k = end; k > 0; k--) tail.push({ kind: "same", text: a[a.length - k], before: a.length - k + 1, after: b.length - k + 1 });
  const x = a.slice(start, a.length - end);
  const y = b.slice(start, b.length - end);
  const middle = shortest(x, y, limit) ?? [...x.map((_, i) => ["removed", i] as Step), ...y.map((_, j) => ["added", j] as Step)];
  const lines = middle.map(([kind, i, j]): DiffLine =>
    kind === "same"
      ? { kind, text: x[i], before: start + i + 1, after: start + (j ?? 0) + 1 }
      : kind === "removed"
        ? { kind, text: x[i], before: start + i + 1 }
        : { kind, text: y[i], after: start + i + 1 },
  );
  return [...head, ...lines, ...tail];
}

/** A step of the edit: a line kept (its index in both), removed (its index in x) or added (its index in y). */
type Step = ["same", number, number] | ["removed", number] | ["added", number];

/**
 * The shortest edit from x to y (Myers, 1986), or null when it takes more
 * than `limit` lines removed and added. `v[k]` holds how far along x the
 * furthest path on diagonal k (x - y) has reached; each round's values are
 * kept, for diagonals -d..d, to walk the path back.
 */
function shortest(x: string[], y: string[], limit: number): Step[] | null {
  const n = x.length;
  const m = y.length;
  const max = Math.min(n + m, limit);
  const offset = max + 1;
  const v = new Int32Array(2 * max + 3);
  const trace: Int32Array[] = [];
  let found = -1;
  for (let d = 0; d <= max && found < 0; d++) {
    trace.push(v.slice(offset - d, offset + d + 1));
    for (let k = -d; k <= d; k += 2) {
      let i = k === -d || (k !== d && v[offset + k - 1] < v[offset + k + 1]) ? v[offset + k + 1] : v[offset + k - 1] + 1;
      let j = i - k;
      while (i < n && j < m && x[i] === y[j]) i++, j++;
      v[offset + k] = i;
      if (i >= n && j >= m) {
        found = d;
        break;
      }
    }
  }
  if (found < 0) return null;
  const steps: Step[] = [];
  let i = n;
  let j = m;
  for (let d = found; d > 0; d--) {
    const at = (k: number) => trace[d][k + d];
    const k = i - j;
    const down = k === -d || (k !== d && at(k - 1) < at(k + 1));
    const pk = down ? k + 1 : k - 1;
    const pi = at(pk);
    const pj = pi - pk;
    while (i > pi && j > pj) steps.push(["same", --i, --j]);
    if (down) steps.push(["added", --j]);
    else steps.push(["removed", --i]);
  }
  while (i > 0 && j > 0) steps.push(["same", --i, --j]);
  return steps.reverse();
}

/** The diff with each run of kept lines further than `context` lines from any change folded into one row; a lone line is shown rather than folded. */
export function folded(lines: DiffLine[], context = 3): DiffRow[] {
  // Each line's distance to the nearest change, from both sides.
  const far = lines.map(() => Infinity);
  for (let i = 0, last = -Infinity; i < lines.length; i++) {
    if (lines[i].kind !== "same") last = i;
    far[i] = i - last;
  }
  for (let i = lines.length - 1, next = Infinity; i >= 0; i--) {
    if (lines[i].kind !== "same") next = i;
    far[i] = Math.min(far[i], next - i);
  }
  const rows: DiffRow[] = [];
  let hidden: DiffLine[] = [];
  const flush = () => {
    if (hidden.length === 1) rows.push(hidden[0]);
    else if (hidden.length > 1) rows.push({ kind: "skipped", count: hidden.length });
    hidden = [];
  };
  lines.forEach((line, i) => {
    if (line.kind === "same" && far[i] > context) hidden.push(line);
    else {
      flush();
      rows.push(line);
    }
  });
  flush();
  return rows;
}

/** One side of a side-by-side row: the line and its number in that version, or nothing where the other side has a line the first lacks. */
export type Cell = { kind: DiffLine["kind"]; text: string; n: number } | null;

/** A row of the side-by-side diff: the before line and the after line it faces (`changed` unless both are the same line), or a run of kept lines folded across both columns. */
export type SideRow = { kind: "pair"; left: Cell; right: Cell; changed: boolean } | { kind: "skipped"; count: number };

/**
 * The diff in two columns, the before on the left and the after on the right.
 * A run of removed lines next to a run of added ones pairs up row by row, the
 * shorter side padded with empty cells; a kept line faces itself. Kept lines
 * far from a change fold across both columns unless `showAll`.
 */
export function sideBySide(lines: DiffLine[], showAll = false): SideRow[] {
  const rows: SideRow[] = [];
  let removed: Cell[] = [];
  let added: Cell[] = [];
  const flush = () => {
    for (let i = 0; i < Math.max(removed.length, added.length); i++) rows.push({ kind: "pair", left: removed[i] ?? null, right: added[i] ?? null, changed: true });
    removed = [];
    added = [];
  };
  for (const row of showAll ? lines : folded(lines)) {
    if (row.kind === "removed") removed.push({ kind: "removed", text: row.text, n: row.before });
    else if (row.kind === "added") added.push({ kind: "added", text: row.text, n: row.after });
    else {
      flush();
      if (row.kind === "skipped") rows.push(row);
      else rows.push({ kind: "pair", left: { kind: "same", text: row.text, n: row.before }, right: { kind: "same", text: row.text, n: row.after }, changed: false });
    }
  }
  flush();
  return rows;
}

/** The indexes of the rows that start a change block, a run of changed rows: what the previous and next buttons visit. Works on the unified rows and the side-by-side ones alike. */
export function changeAnchors(rows: (DiffRow | SideRow)[]): number[] {
  const changed = (r: DiffRow | SideRow) => (r.kind === "pair" ? r.changed : r.kind === "removed" || r.kind === "added");
  const anchors: number[] = [];
  rows.forEach((row, i) => {
    if (changed(row) && (i === 0 || !changed(rows[i - 1]))) anchors.push(i);
  });
  return anchors;
}

/** The lines the diff removes and adds. */
export function changeCounts(lines: DiffLine[]): { removed: number; added: number } {
  const removed = lines.filter((l) => l.kind === "removed").length;
  return { removed, added: lines.filter((l) => l.kind === "added").length };
}
