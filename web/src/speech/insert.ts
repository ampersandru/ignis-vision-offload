/**
 * `text` put into `value` at `at` (spec speech/04, Q17), with a space added on
 * a side where it would otherwise touch a word, and the range it now holds,
 * for the field to select. The model's text comes trimmed of the spaces its
 * tokens carry.
 */
export function insertAt(value: string, at: number, text: string): { value: string; start: number; end: number } {
  const words = text.trim();
  const cut = Math.min(Math.max(at, 0), value.length);
  const before = value.slice(0, cut);
  const after = value.slice(cut);
  const lead = before !== "" && !/\s$/.test(before) ? " " : "";
  const trail = after !== "" && !/^\s/.test(after) ? " " : "";
  const start = before.length + lead.length;
  return { value: before + lead + words + trail + after, start, end: start + words.length };
}

/** The words in `text`, as "Inserted N words" counts them. */
export function wordCount(text: string): number {
  return text.split(/\s+/).filter((word) => word !== "").length;
}
