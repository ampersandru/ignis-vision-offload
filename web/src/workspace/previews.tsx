import { useEffect, useMemo, useState } from "react";
import { HtmlPreview } from "../ui/HtmlPreview.tsx";
import { Markdown } from "../ui/Markdown.tsx";
import { latest, present, type Version, type WorkspaceFile } from "./files.ts";
import { dataUrl, inlineSite, parseCsv, siteRefs, type SiteFiles, svgDataUrl } from "./site.ts";

// The panel's previews: an HTML file with the files it names put into it, and
// Markdown, SVG and CSV files drawn rather than shown as text. Each is one
// the editor tab beside them is its source.

/** What the workspace's text files put into a page at once: stylesheets, scripts and SVGs; the pictures to read are `pictures`. */
export function readySiteFiles(html: string, from: string, files: WorkspaceFile[]): { ready: Map<string, string>; pictures: { file: string; version: Version }[] } {
  const ready = new Map<string, string>();
  const pictures: { file: string; version: Version }[] = [];
  const named = new Map(files.filter(present).map((f) => [f.name, f]));
  for (const { tag, file } of siteRefs(html, from)) {
    const f = named.get(file);
    if (!f || ready.has(file) || pictures.some((p) => p.file === file)) continue;
    const v = latest(f);
    if (tag === "image") {
      if (v.text !== undefined && f.kind === "text" && /\.svg$/i.test(f.name)) ready.set(file, svgDataUrl(v.text));
      else if (f.kind === "picture" && v.bytes) pictures.push({ file, version: v });
    } else if (v.text !== undefined) ready.set(file, v.text);
  }
  return { ready, pictures };
}

/** An HTML file with the stylesheets, scripts and pictures of its workspace that it names put into the page it runs; the pictures arrive as they are read. */
export function SitePreview({ file, text, files, content }: { file: WorkspaceFile; text: string; files: WorkspaceFile[]; content: (version: Version) => Promise<Blob | null> }) {
  const { ready, pictures } = useMemo(() => readySiteFiles(text, file.name, files), [text, file.name, files]);
  const key = pictures.map((p) => `${p.file}@${p.version.n}`).join("\n");
  const [read, setRead] = useState<{ key: string; urls: Map<string, string> }>({ key: "", urls: new Map() });
  useEffect(() => {
    if (pictures.length === 0) return;
    let live = true;
    void Promise.all(
      pictures.map(async ({ file: name, version }) => {
        const blob = await content(version).catch(() => null);
        return blob ? ([name, dataUrl(blob.type || version.bytes?.type || "", new Uint8Array(await blob.arrayBuffer()))] as const) : null;
      }),
    ).then((got) => live && setRead({ key, urls: new Map(got.filter((g) => g !== null)) }));
    return () => {
      live = false;
    };
  }, [key]);
  const all: SiteFiles = read.key === key ? new Map([...ready, ...read.urls]) : ready;
  return <HtmlPreview html={text} name={file.name} shown={all.size > 0 ? inlineSite(text, file.name, all) : undefined} />;
}

export function MarkdownPreview({ text }: { text: string }) {
  return (
    <div className="max-h-[60vh] overflow-auto">
      <Markdown text={text} />
    </div>
  );
}

/** An SVG drawn as an image, which a browser does without running the scripts in it. */
export function SvgPreview({ text, name }: { text: string; name: string }) {
  return <img src={svgDataUrl(text)} alt={name} className="stage-checker block max-h-[60vh] max-w-full object-contain" />;
}

/** The most rows of a CSV file the table draws. */
export const CSV_ROWS = 500;

/** A CSV file as a table, its first row the header. */
export function CsvTable({ text }: { text: string }) {
  const rows = parseCsv(text);
  if (rows.length === 0) return <p className="text-xs leading-snug text-ash">The file is empty.</p>;
  const [head, ...body] = rows;
  const width = Math.max(...rows.map((r) => r.length));
  const cells = (row: string[]) => Array.from({ length: width }, (_, i) => row[i] ?? "");
  const shown = body.slice(0, CSV_ROWS);
  return (
    <div className="flex flex-col gap-1.5">
      <div className="max-h-[60vh] overflow-auto bg-surface">
        <table className="w-full border-collapse text-left text-xs text-ink">
          <thead className="sticky top-0 bg-line">
            <tr>
              {cells(head).map((c, i) => (
                <th key={i} className="px-2 py-1 font-display font-semibold whitespace-nowrap">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((row, r) => (
              <tr key={r} className="border-t border-line">
                {cells(row).map((c, i) => (
                  <td key={i} className="px-2 py-1 align-top whitespace-pre-wrap">
                    {c}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {body.length > CSV_ROWS && (
        <p className="text-xs text-ash tabular-nums">
          The first {CSV_ROWS} of {body.length} rows.
        </p>
      )}
    </div>
  );
}
