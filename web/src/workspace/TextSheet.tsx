import { type ReactNode, useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { isHtmlName } from "../ui/HtmlPreview.tsx";
import { IconClose } from "../ui/icons.tsx";
import { Markdown } from "../ui/Markdown.tsx";
import { SHEET_BUTTON, underlined } from "./FileHistory.tsx";
import { BUTTON } from "./looks.tsx";
import { CsvTable } from "./previews.tsx";
import { isCsvName, isMarkdownName, isSvgName, svgDataUrl } from "./site.ts";

// A text file over the whole window, as the HTML preview and the diff are:
// its newest version on a page of its own, read-only (the Edit tab is the
// place to change it).

export type TextKind = "markdown" | "csv" | "svg" | "text";

/** How the full screen draws a text file of this name; an HTML file has a full screen of its own. */
export function textKind(name: string): TextKind | null {
  if (isHtmlName(name)) return null;
  if (isMarkdownName(name)) return "markdown";
  if (isCsvName(name)) return "csv";
  if (isSvgName(name)) return "svg";
  return "text";
}

export type TextView = "drawn" | "source";

/** The ways to read a kind: the first is the one the sheet opens on; other text has only its source. */
const VIEWS: Record<TextKind, { id: TextView; label: string }[]> = {
  markdown: [
    { id: "drawn", label: "Rendered" },
    { id: "source", label: "Source" },
  ],
  csv: [
    { id: "drawn", label: "Table" },
    { id: "source", label: "Source" },
  ],
  svg: [
    { id: "drawn", label: "Picture" },
    { id: "source", label: "Source" },
  ],
  text: [],
};

/** The button that opens the file full screen. */
export function ExpandButton({ onClick }: { onClick: () => void }) {
  return (
    <button type="button" className={BUTTON} onClick={onClick}>
      Expand
    </button>
  );
}

/** A text's lines numbered from 1, in the mono face, wrapped; the newline that ends the text starts no line. */
function NumberedSource({ text }: { text: string }) {
  const lines = text.replace(/\r\n?/g, "\n").replace(/\n$/, "").split("\n");
  return (
    <table className="w-full border-collapse font-mono text-xs leading-relaxed">
      <tbody>
        {lines.map((line, i) => (
          <tr key={i} className="align-top">
            <td className="w-px pr-4 text-right whitespace-nowrap text-ash tabular-nums select-none">{i + 1}</td>
            <td className="break-words whitespace-pre-wrap">{line}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** The full-screen text's sheet, header and body, as props: the state and the Esc key are `TextFullscreen`'s. */
export function TextSheet({
  name,
  version,
  text,
  kind,
  view,
  onView,
  full,
  onFull,
  onClose,
}: {
  name: string;
  version: number;
  text: string;
  kind: TextKind;
  view: TextView;
  onView: (view: TextView) => void;
  full: boolean;
  onFull: (on: boolean) => void;
  onClose: () => void;
}) {
  const views = VIEWS[kind];
  let page: ReactNode;
  if (kind === "text" || view === "source") page = <NumberedSource text={text} />;
  else if (kind === "markdown") page = <Markdown text={text} />;
  else if (kind === "csv") page = <CsvTable text={text} sheet />;
  else page = <img src={svgDataUrl(text)} alt={name} className="stage-checker mx-auto block max-w-full object-contain" />;
  return (
    <div role="dialog" aria-modal="true" aria-label={name} className="memory-sheet fixed inset-0 z-[60] flex flex-col bg-kiln text-[#eae8e4]">
      <header className="flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-kiln-line px-4 py-2.5 md:px-6">
        <h2 className="min-w-0 truncate font-display text-[15px] font-semibold">{name}</h2>
        <p className="text-xs text-[#939ba4]">{`version ${version}`}</p>
        {views.length > 0 && (
          <div role="radiogroup" aria-label="Show as" className="flex gap-4 font-display text-[13px] font-medium">
            {views.map((w) => (
              <button key={w.id} type="button" role="radio" aria-checked={view === w.id} onClick={() => onView(w.id)} className={underlined(view === w.id)}>
                {w.label}
              </button>
            ))}
          </div>
        )}
        <button type="button" aria-pressed={full} onClick={() => onFull(!full)} className={`${SHEET_BUTTON} ${underlined(full)}`}>
          Full width
        </button>
        <button
          type="button"
          aria-label="Close the full screen"
          title="Close (Esc)"
          onClick={onClose}
          className="-mr-2 ml-auto grid size-9 place-items-center text-[#939ba4] hover:bg-kiln-line hover:text-white"
        >
          <IconClose />
        </button>
      </header>
      <div className="min-h-0 flex-1 overflow-auto bg-[#12151a] p-3 md:p-5">
        <div data-page className={`mx-auto min-h-full bg-surface px-6 py-8 text-ink shadow-[0_24px_60px_rgb(0_0_0/0.45)] md:px-10 ${full ? "w-full" : "w-full max-w-[75ch]"}`}>
          {page}
        </div>
      </div>
    </div>
  );
}

/** The text over the whole window, in a portal as the HTML preview is: drawn (or the source) at first, at the page's width; Esc closes. */
export function TextFullscreen({ name, version, text, kind, onClose }: { name: string; version: number; text: string; kind: TextKind; onClose: () => void }) {
  const [view, setView] = useState<TextView>("drawn");
  const [full, setFull] = useState(false);
  useEffect(() => {
    const close = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose]);
  return createPortal(<TextSheet name={name} version={version} text={text} kind={kind} view={view} onView={setView} full={full} onFull={setFull} onClose={onClose} />, document.body);
}
