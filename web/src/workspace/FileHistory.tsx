import { type ReactNode, type RefObject, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { caption } from "../ui/classes.ts";
import { IconClose } from "../ui/icons.tsx";
import { type Cell, changeAnchors, changeCounts, type DiffRow, folded, lineDiff, type SideRow, sideBySide } from "./diff.ts";
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
  const [expanded, setExpanded] = useState(false);
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
          <DiffView before={textOf(from) ?? ""} after={textOf(to) ?? ""} onExpand={() => setExpanded(true)} />
          {expanded && <DiffFullscreen name={file.name} versions={texts} whom={whom} from={from} to={to} onChoose={choose} onClose={() => setExpanded(false)} />}
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

// The full-screen sheet is the kiln's dark in both schemes, so it carries the
// dark scheme's ink, ash, fault and good by value, as the preview's sheet does.
const SHEET_ROW = { same: "text-[#eae8e4]", removed: "bg-[#ff7b6b]/10 text-[#ff7b6b]", added: "bg-[#3ecf8e]/10 text-[#3ecf8e]" };

const SHEET_SELECT = "min-w-0 rounded-[2px] border border-kiln-line bg-[#12151a] px-2 py-1 text-xs text-[#eae8e4] focus:border-[#ff5a1f] focus:outline-none";

/** What two versions without a changed line say: the same text, or only their line endings differ. */
function SameText({ before, after }: { before: string; after: string }) {
  return (
    <p className="text-xs leading-snug text-ash">
      {before === after ? "The two versions hold the same text." : "The two versions differ only in their line endings, or the newline at the end."}
    </p>
  );
}

const countLabel = (removed: number, added: number) => `${removed} ${removed === 1 ? "line" : "lines"} removed, ${added} added`;

/** The lines `after` removes from `before` and adds to it, numbered in both, the unchanged lines far from a change folded; `onExpand` adds the button that opens it full screen. */
export function DiffView({ before, after, onExpand }: { before: string; after: string; onExpand?: () => void }) {
  const lines = lineDiff(before, after);
  const { removed, added } = changeCounts(lines);
  if (removed === 0 && added === 0) return <SameText before={before} after={after} />;
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs text-ash">{countLabel(removed, added)}</p>
        {onExpand && (
          <button type="button" className={BUTTON} onClick={onExpand}>
            Expand
          </button>
        )}
      </div>
      <div className="max-h-[50vh] overflow-auto bg-surface">
        <UnifiedTable rows={folded(lines)} sheet={false} />
      </div>
    </div>
  );
}

/** The unified rows as a table; in the sheet each row carries its index and the rows that start a change block are marked, for previous and next to scroll to. */
function UnifiedTable({ rows, sheet }: { rows: DiffRow[]; sheet: boolean }) {
  const palette = sheet ? SHEET_ROW : ROW;
  const ash = sheet ? "text-[#939ba4]" : "text-ash";
  const anchors = sheet ? changeAnchors(rows) : [];
  return (
    <table className="w-full border-collapse font-mono text-xs leading-relaxed">
      <tbody>
        {rows.map((row, i) =>
          row.kind === "skipped" ? (
            <tr key={i} data-diff="skipped" {...(sheet ? { "data-row": i } : {})} className={ash}>
              <td colSpan={4} className="px-2 py-0.5 text-center font-display">
                {`${row.count} unchanged lines`}
              </td>
            </tr>
          ) : (
            <tr key={i} data-diff={row.kind} {...(sheet ? { "data-row": i } : {})} {...(anchors.includes(i) ? { "data-anchor": "" } : {})} className={palette[row.kind]}>
              <td className={`w-px px-1.5 text-right ${ash} tabular-nums select-none`}>{row.kind === "added" ? "" : row.before}</td>
              <td className={`w-px px-1.5 text-right ${ash} tabular-nums select-none`}>{row.kind === "removed" ? "" : row.after}</td>
              <td className="w-px px-1 select-none">{SIGN[row.kind]}</td>
              <td className="pr-2 whitespace-pre-wrap break-all">{row.text}</td>
            </tr>
          ),
        )}
      </tbody>
    </table>
  );
}

/** The two columns of a side-by-side diff: before on the left, after on the right, each with its line numbers, a folded run across both. */
function SideTable({ rows }: { rows: SideRow[] }) {
  const anchors = changeAnchors(rows);
  const side = (cell: Cell, edge: boolean) => {
    const tone = cell ? SHEET_ROW[cell.kind] : "";
    const fill = cell?.kind === "removed" ? "bg-[#ff7b6b]/10" : cell?.kind === "added" ? "bg-[#3ecf8e]/10" : "";
    return [
      <td key="n" className={`w-12 px-1.5 text-right align-top text-[#939ba4] tabular-nums select-none ${fill}${edge ? " border-l border-kiln-line" : ""}`}>
        {cell?.n ?? ""}
      </td>,
      <td key="t" className={`w-1/2 pr-2 align-top whitespace-pre-wrap break-all ${tone}`}>
        {cell?.text ?? ""}
      </td>,
    ];
  };
  return (
    <table className="w-full table-fixed border-collapse font-mono text-xs leading-relaxed">
      <tbody>
        {rows.map((row, i) =>
          row.kind === "skipped" ? (
            <tr key={i} data-diff="skipped" data-row={i} className="text-[#939ba4]">
              <td colSpan={4} className="px-2 py-0.5 text-center font-display">
                {`${row.count} unchanged lines`}
              </td>
            </tr>
          ) : (
            <tr key={i} data-diff={row.changed ? "changed" : "same"} data-row={i} {...(anchors.includes(i) ? { "data-anchor": "" } : {})}>
              {side(row.left, false)}
              {side(row.right, true)}
            </tr>
          ),
        )}
      </tbody>
    </table>
  );
}

/** How the full-screen diff lays the two versions out. */
export type DiffLayout = "unified" | "split";

const LAYOUTS: { id: DiffLayout; label: string }[] = [
  { id: "split", label: "Side by side" },
  { id: "unified", label: "Unified" },
];

const SHEET_BUTTON = "font-display text-[13px] font-medium disabled:cursor-not-allowed disabled:opacity-40";

const underlined = (on: boolean) => `border-b-2 pb-0.5 ${on ? "border-[#ff5a1f] text-[#eae8e4]" : "border-transparent text-[#939ba4] hover:text-[#eae8e4]"}`;

/** The rows the full-screen diff shows: the lines in two columns or one, folded unless `showAll`. */
function sheetRows(lines: ReturnType<typeof lineDiff>, layout: DiffLayout, showAll: boolean): DiffRow[] | SideRow[] {
  return layout === "split" ? sideBySide(lines, showAll) : showAll ? lines : folded(lines);
}

type SheetProps = {
  name: string;
  versions: Version[];
  whom: (author: Author) => string;
  from: number;
  to: number;
  onChoose: (side: 0 | 1, n: number) => void;
  onClose: () => void;
};

/** The full-screen diff's sheet, header and body, as props: the state and the key handling are `DiffFullscreen`'s. */
export function DiffSheet({
  name,
  versions,
  whom,
  from,
  to,
  onChoose,
  layout,
  onLayout,
  showAll,
  onShowAll,
  at,
  onGo,
  onClose,
  bodyRef,
}: SheetProps & {
  layout: DiffLayout;
  onLayout: (layout: DiffLayout) => void;
  showAll: boolean;
  onShowAll: (on: boolean) => void;
  /** The change block last visited, from 0. */
  at: number;
  onGo: (block: number) => void;
  bodyRef?: RefObject<HTMLDivElement | null>;
}) {
  const textOf = (n: number) => versions.find((v) => v.n === n)?.text ?? "";
  const before = textOf(from);
  const after = textOf(to);
  const lines = lineDiff(before, after);
  const { removed, added } = changeCounts(lines);
  const same = removed === 0 && added === 0;
  const rows = sheetRows(lines, layout, showAll);
  const blocks = changeAnchors(rows).length;
  const select = (label: string, side: 0 | 1, value: number): ReactNode => (
    <select aria-label={label} value={value} className={SHEET_SELECT} onChange={(e) => onChoose(side, Number(e.currentTarget.value))}>
      {versions.map((v) => (
        <option key={v.n} value={v.n}>
          {`Version ${v.n}, ${whom(v.author)}`}
        </option>
      ))}
    </select>
  );
  return (
    <div role="dialog" aria-modal="true" aria-label={`Differences in ${name}`} className="memory-sheet fixed inset-0 z-[60] flex flex-col bg-kiln text-[#eae8e4]">
      <header className="flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-kiln-line px-4 py-2.5 md:px-6">
        <h2 className="min-w-0 truncate font-display text-[15px] font-semibold">{name}</h2>
        <div className="flex flex-wrap items-center gap-1.5 text-xs text-[#939ba4]">
          {select("Compare from version", 0, from)}
          to
          {select("Compare to version", 1, to)}
        </div>
        {!same && <p className="text-xs text-[#939ba4]">{countLabel(removed, added)}</p>}
        {!same && (
          <>
            <div role="radiogroup" aria-label="Layout" className="flex gap-4 font-display text-[13px] font-medium">
              {LAYOUTS.map((l) => (
                <button key={l.id} type="button" role="radio" aria-checked={layout === l.id} onClick={() => onLayout(l.id)} className={underlined(layout === l.id)}>
                  {l.label}
                </button>
              ))}
            </div>
            <button type="button" aria-pressed={showAll} onClick={() => onShowAll(!showAll)} className={`${SHEET_BUTTON} ${underlined(showAll)}`}>
              Show all lines
            </button>
            <div className="flex gap-4">
              <button type="button" title="Previous change (p)" disabled={at <= 0} onClick={() => onGo(at - 1)} className={`${SHEET_BUTTON} text-[#939ba4] hover:text-[#eae8e4]`}>
                Previous change
              </button>
              <button
                type="button"
                title="Next change (n)"
                disabled={at >= blocks - 1}
                onClick={() => onGo(at + 1)}
                className={`${SHEET_BUTTON} text-[#939ba4] hover:text-[#eae8e4]`}
              >
                Next change
              </button>
            </div>
          </>
        )}
        <button
          type="button"
          aria-label="Close the diff"
          title="Close (Esc)"
          onClick={onClose}
          className="-mr-2 ml-auto grid size-9 place-items-center text-[#939ba4] hover:bg-kiln-line hover:text-white"
        >
          <IconClose />
        </button>
      </header>
      <div ref={bodyRef} className="min-h-0 flex-1 overflow-auto bg-[#12151a] p-3 md:p-5">
        {same ? (
          <SameText before={before} after={after} />
        ) : layout === "split" ? (
          <SideTable rows={rows as SideRow[]} />
        ) : (
          <UnifiedTable rows={rows as DiffRow[]} sheet />
        )}
      </div>
    </div>
  );
}

/**
 * The diff over the whole window, in a portal as the preview is: Side by side
 * at first, the first change in view; p and n visit the changes, Esc closes.
 */
function DiffFullscreen(props: SheetProps) {
  const { versions, from, to, onClose } = props;
  const [layout, setLayout] = useState<DiffLayout>("split");
  const [showAll, setShowAll] = useState(false);
  const [at, setAt] = useState(0);
  const body = useRef<HTMLDivElement>(null);
  const textOf = (n: number) => versions.find((v) => v.n === n)?.text ?? "";
  const anchors = changeAnchors(sheetRows(lineDiff(textOf(from), textOf(to)), layout, showAll));

  const go = (block: number) => {
    const k = Math.max(0, Math.min(block, anchors.length - 1));
    setAt(k);
    body.current?.querySelector(`[data-row="${anchors[k]}"]`)?.scrollIntoView({ block: "center" });
  };

  // The first change is in view when the sheet opens, and again whenever what it shows changes.
  useEffect(() => go(0), [layout, showAll, from, to]);

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") return onClose();
      const typing = e.target instanceof HTMLElement && /^(SELECT|INPUT|TEXTAREA)$/.test(e.target.tagName);
      if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "n" && at < anchors.length - 1) go(at + 1);
      else if (e.key === "p" && at > 0) go(at - 1);
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  });

  return createPortal(<DiffSheet {...props} layout={layout} onLayout={setLayout} showAll={showAll} onShowAll={setShowAll} at={at} onGo={go} bodyRef={body} />, document.body);
}
