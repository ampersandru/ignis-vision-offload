import { useState } from "react";
import { caption } from "../ui/classes.ts";
import { folded, lineDiff } from "./diff.ts";
import { type Author, latest, timeLabel, type Version, type WorkspaceFile } from "./files.ts";
import { type Draft, editing, sizeLabel } from "./panel.ts";

// A file of the workspace panel by hand (spec #170): a text file's editor,
// whose Save makes one version authored by the user and whose typing makes
// none; and any file's history, every version with its author and time, the
// diff between two versions of a text file, and a restore of any version but
// the newest, which makes it the newest again.

/** What a Save did: the version it made, or why it made none. */
export type SaveOutcome = { ok: true; version: number } | { ok: false; error: string };

const BUTTON = "cut shrink-0 bg-surface px-2.5 py-1.5 font-display text-xs font-semibold text-ink [--cut-size:6px] hover:bg-line disabled:cursor-not-allowed disabled:opacity-50";

const SELECT = "min-w-0 rounded-[2px] border border-line bg-surface px-2 py-1 text-xs text-ink focus:border-ember focus:outline-none";

const SAVE = "cut shrink-0 bg-ember px-3 py-1.5 font-display text-xs font-semibold text-[#1c2026] [--cut-size:6px] hover:bg-[#ff7a45] disabled:cursor-not-allowed disabled:opacity-50";

/** A text file's text in a box: Save, or Ctrl+S, keeps it as a version of the user's; Discard changes takes the newest version back. */
export function FileEditor({ file, whom, onSave }: { file: WorkspaceFile; whom: (author: Author) => string; onSave: (file: WorkspaceFile, text: string) => Promise<SaveOutcome> }) {
  const [draft, setDraft] = useState<Draft>(() => ({ base: latest(file).n, text: latest(file).text ?? "" }));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const shown = editing(file, draft);
  const newest = latest(file);

  async function save() {
    if (!shown.dirty || saving) return;
    setSaving(true);
    setError(null);
    const outcome = await onSave(file, shown.text);
    setSaving(false);
    if (outcome.ok) setDraft({ base: outcome.version, text: shown.text });
    else setError(outcome.error);
  }

  return (
    <div className="flex flex-col gap-2">
      <textarea
        aria-label={`The text of ${file.name}`}
        // With LF line endings, as the box reads its value back: a value with CR in it would be set again at each render, the caret moved.
        value={shown.text.replace(/\r\n/g, "\n")}
        spellCheck={false}
        className="min-h-[40vh] w-full resize-y rounded-[2px] border border-line bg-surface px-2.5 py-2 font-mono text-xs leading-relaxed text-ink focus:border-ember focus:outline-none"
        onChange={(e) => setDraft({ base: shown.base, text: e.currentTarget.value })}
        onKeyDown={(e) => {
          if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
            e.preventDefault();
            void save();
          }
        }}
      />
      {shown.since && (
        <p className="text-xs leading-snug text-ash">
          Since you started, {whom(shown.since.author)} made version {shown.since.n}. Save makes your text version {newest.n + 1}; version {shown.since.n} stays in the
          history.
        </p>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" className={SAVE} disabled={!shown.dirty || saving} onClick={() => void save()}>
          Save
        </button>
        <button type="button" className={BUTTON} disabled={!shown.dirty || saving} onClick={() => setDraft({ base: newest.n, text: newest.text ?? "" })}>
          Discard changes
        </button>
        <span className="text-xs text-ash">{saving ? "Saving…" : shown.dirty ? "Not saved yet. Ctrl+S saves." : "Ctrl+S saves."}</span>
      </div>
      {error && <p className="text-xs leading-snug text-fault">{error}</p>}
    </div>
  );
}

/** Every version of a file, the newest first, with a restore of each but the newest, and for a text file the diff between two of them. */
export function FileHistory({ file, whom, onRestore }: { file: WorkspaceFile; whom: (author: Author) => string; onRestore: (n: number) => void }) {
  const texts = file.kind === "text" ? file.versions.filter((v) => v.text !== undefined) : [];
  const [pair, setPair] = useState<[number, number] | null>(null);
  const newest = latest(file);
  const [from, to] = pair ?? (texts.length >= 2 ? [texts[texts.length - 2].n, texts[texts.length - 1].n] : [0, 0]);
  const textOf = (n: number) => texts.find((v) => v.n === n)?.text;
  const choose = (side: 0 | 1, n: number) => setPair(side === 0 ? [n, to] : [from, n]);
  const option = (v: Version) => (
    <option key={v.n} value={v.n}>
      {`Version ${v.n}, ${whom(v.author)}`}
    </option>
  );

  return (
    <div className="flex flex-col gap-3">
      {texts.length >= 2 && (
        <div className="flex flex-col gap-2">
          <p className={caption}>Compare two versions</p>
          <div className="flex flex-wrap items-center gap-1.5 text-xs text-ash">
            <select aria-label="Compare from version" value={from} className={SELECT} onChange={(e) => choose(0, Number(e.currentTarget.value))}>
              {texts.map(option)}
            </select>
            to
            <select aria-label="Compare to version" value={to} className={SELECT} onChange={(e) => choose(1, Number(e.currentTarget.value))}>
              {texts.map(option)}
            </select>
          </div>
          <DiffView before={textOf(from) ?? ""} after={textOf(to) ?? ""} />
        </div>
      )}
      <div className="flex flex-col gap-1">
        <p className={caption}>Versions</p>
        <ol className="flex flex-col">
          {[...file.versions].reverse().map((v) => (
            <li key={v.n} className="flex min-w-0 items-center gap-2 border-b border-line py-1.5 last:border-b-0">
              <div className="min-w-0 flex-1">
                <p className="font-display text-xs font-semibold text-ink">{`Version ${v.n}`}</p>
                <p className="text-xs leading-snug text-ash">
                  {timeLabel(v.at)}, {whom(v.author)}; {v.deleted ? "deleted the file" : sizeLabel(v)}
                </p>
              </div>
              {!v.deleted && v !== newest && (
                <button type="button" aria-label={`Restore version ${v.n} of ${file.name}`} className={BUTTON} onClick={() => onRestore(v.n)}>
                  Restore
                </button>
              )}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

const SIGN = { same: "", removed: "-", added: "+" };

const ROW = { same: "text-ink", removed: "bg-fault/10 text-fault", added: "bg-good/10 text-good" };

/** The lines `after` removes from `before` and adds to it, numbered in both, the unchanged lines far from a change folded. */
export function DiffView({ before, after }: { before: string; after: string }) {
  const lines = lineDiff(before, after);
  const removed = lines.filter((l) => l.kind === "removed").length;
  const added = lines.filter((l) => l.kind === "added").length;
  if (removed === 0 && added === 0) {
    return (
      <p className="text-xs leading-snug text-ash">
        {before === after ? "The two versions hold the same text." : "The two versions differ only in their line endings, or the newline at the end."}
      </p>
    );
  }
  return (
    <div className="flex flex-col gap-1">
      <p className="text-xs text-ash">{`${removed} ${removed === 1 ? "line" : "lines"} removed, ${added} added`}</p>
      <div className="max-h-[50vh] overflow-auto bg-surface">
        <table className="w-full border-collapse font-mono text-xs leading-relaxed">
          <tbody>
            {folded(lines).map((row, i) =>
              row.kind === "skipped" ? (
                <tr key={i} data-diff="skipped" className="text-ash">
                  <td colSpan={4} className="px-2 py-0.5 text-center font-display">
                    {`${row.count} unchanged lines`}
                  </td>
                </tr>
              ) : (
                <tr key={i} data-diff={row.kind} className={ROW[row.kind]}>
                  <td className="w-px px-1.5 text-right text-ash tabular-nums select-none">{row.kind === "added" ? "" : row.before}</td>
                  <td className="w-px px-1.5 text-right text-ash tabular-nums select-none">{row.kind === "removed" ? "" : row.after}</td>
                  <td className="w-px px-1 select-none">{SIGN[row.kind]}</td>
                  <td className="pr-2 whitespace-pre-wrap break-all">{row.text}</td>
                </tr>
              ),
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
