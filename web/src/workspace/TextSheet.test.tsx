import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { USER, type Version, type WorkspaceFile } from "./files.ts";
import type { PanelActions } from "./WorkspacePanel.tsx";
import { FileViewer } from "./WorkspacePanel.tsx";
import { sheetAfter, TextSheet, type TextKind, type TextView, textKind } from "./TextSheet.tsx";

// The full-screen text as the page renders it: the sheet's markup per kind
// and the Expand button on the viewer. Clicks and Esc are not exercised: there
// is no DOM here.

const at = 0;
const text = (t: string): Version => ({ n: 3, author: USER, at, text: t });
const file = (name: string, kind: WorkspaceFile["kind"], v: Version): WorkspaceFile => ({ format: 1, id: `f-${name}`, workspace: "w", name, kind, versions: [v] });
const actions: PanelActions = {
  create: async () => null,
  upload: () => {},
  rename: async () => null,
  remove: () => {},
  download: () => {},
  content: async () => null,
  derive: async () => null,
  save: async () => ({ ok: true, version: 2 }),
  restore: () => {},
  exportZip: () => {},
  importZip: () => {},
};
const viewer = (f: WorkspaceFile) => renderToStaticMarkup(<FileViewer file={f} whom={() => "you"} onBack={() => {}} actions={actions} files={[f]} />);

const sheet = (kind: TextKind, body: string, over: { view?: TextView; full?: boolean } = {}) =>
  renderToStaticMarkup(
    <TextSheet name={`a.${kind}`} version={3} text={body} kind={kind} view={over.view ?? "drawn"} onView={() => {}} full={over.full ?? false} onFull={() => {}} onClose={() => {}} />,
  );

describe("the full-screen text", () => {
  it("heads the sheet with the file, its version, the kind's views, Full width and Close", () => {
    const html = sheet("markdown", "# Trip\n");
    expect(html).toMatch(/role="dialog"[^>]*aria-modal="true"/);
    expect(html).toContain(">a.markdown</h2>");
    expect(html).toContain("version 3");
    expect(html).toMatch(/<button[^>]*role="radio"[^>]*aria-checked="true"[^>]*>Rendered<\/button>/);
    expect(html).toMatch(/<button[^>]*role="radio"[^>]*aria-checked="false"[^>]*>Source<\/button>/);
    expect(html).toMatch(/<button[^>]*aria-pressed="false"[^>]*>Full width<\/button>/);
    expect(html).toContain('title="Close (Esc)"');
  });

  it("draws Markdown on a page in the app's surface and ink, and its text as numbered lines", () => {
    const html = sheet("markdown", "# Trip\n\n- **bread**\n");
    expect(html).toContain("<h1>Trip</h1>");
    expect(html).toContain("<strong>bread</strong>");
    expect(html).toMatch(/data-page[^>]*class="[^"]*bg-surface[^"]*text-ink[^"]*max-w-\[75ch\]/);
    const source = sheet("markdown", "# Trip\n\n- **bread**\n", { view: "source" });
    expect(source).not.toContain("<h1>");
    expect(source).toMatch(/aria-checked="true"[^>]*>Source<\/button>/);
    expect(source).toMatch(/<td[^>]*>1<\/td><td[^>]*># Trip<\/td>/);
    expect(source).toMatch(/<td[^>]*>3<\/td><td[^>]*>- \*\*bread\*\*<\/td>/);
    expect(source).not.toMatch(/<td[^>]*>4<\/td>/);
    expect(source).toContain("font-mono");
  });

  it("draws a CSV as a table, and an SVG as a picture, with the source one choice away", () => {
    const csv = sheet("csv", "name,note\nBo,hi\n");
    expect(csv).toMatch(/<th[^>]*>name<\/th>/);
    expect(csv).toMatch(/<td[^>]*>hi<\/td>/);
    expect(csv).toMatch(/aria-checked="true"[^>]*>Table<\/button>/);
    const svg = sheet("svg", '<svg xmlns="http://www.w3.org/2000/svg"><script>x</script></svg>');
    expect(svg).toMatch(/<img[^>]*src="data:image\/svg\+xml;charset=utf-8,%3Csvg/);
    expect(svg).not.toContain("<script>");
    expect(svg).toMatch(/aria-checked="true"[^>]*>Picture<\/button>/);
    expect(sheet("svg", "<svg/>", { view: "source" })).not.toContain("<img");
  });

  it("shows other text as numbered lines and offers no views", () => {
    const html = sheet("text", "one\ntwo\n");
    expect(html).not.toContain("radiogroup");
    expect(html).toMatch(/<td[^>]*>2<\/td><td[^>]*>two<\/td>/);
  });

  it("fills the sheet's width with Full width on, and is about 75 characters wide with it off", () => {
    const off = sheet("text", "x");
    const on = sheet("text", "x", { full: true });
    expect(off).toContain("max-w-[75ch]");
    expect(off).toMatch(/aria-pressed="false"[^>]*>Full width/);
    expect(on).not.toContain("max-w-[75ch]");
    expect(on).toMatch(/aria-pressed="true"[^>]*>Full width/);
  });

  it("names the kinds: HTML has a full screen of its own", () => {
    expect(textKind("a.html")).toBeNull();
    expect(textKind("a.md")).toBe("markdown");
    expect(textKind("a.csv")).toBe("csv");
    expect(textKind("a.svg")).toBe("svg");
    expect(textKind("a.txt")).toBe("text");
  });
});

describe("the Expand button on the viewer", () => {
  it("is there for Markdown, SVG, CSV and other text, and not for HTML, a picture or audio", () => {
    for (const name of ["a.md", "a.svg", "a.csv", "a.txt"]) expect(viewer(file(name, "text", text("x")))).toMatch(/<button[^>]*>Expand<\/button>/);
    expect(viewer(file("a.html", "text", text("<p>x</p>")))).not.toContain(">Expand<");
    const bytes = { n: 1, author: USER, at, bytes: { store: "image" as const, id: "sha", type: "image/png", size: 3 } };
    expect(viewer(file("a.png", "picture", bytes))).not.toContain(">Expand<");
    expect(viewer(file("a.wav", "audio", { n: 1, author: USER, at, bytes: { store: "blob" as never, id: "x", type: "audio/wav", size: 3 } as never }))).not.toContain(">Expand<");
  });
});

describe("the full screen once its file changes", () => {
  const notes = file("notes.md", "text", text("# Trip\n"));
  const later = (f: WorkspaceFile, change: Partial<Version>): WorkspaceFile => ({ ...f, versions: [...f.versions, { n: f.versions.length + 3, author: USER, at, ...change }] });
  /** The sheet's state after each of the file's states in turn, from open on `notes.md`. */
  const after = (states: WorkspaceFile[]) => states.reduce<string | null>((open, f) => sheetAfter(open, f), "notes.md");

  it("stays open on a new version, which it then shows", () => {
    expect(after([later(notes, { text: "# Trip\n- Rome\n" })])).toBe("notes.md");
  });

  it("closes when the file is deleted, and a restore then does not open it again", () => {
    const deleted = later(notes, { deleted: true });
    expect(after([deleted])).toBeNull();
    expect(after([deleted, later(deleted, { text: "# Trip\n" })])).toBeNull();
  });

  it("closes on a rename, and renaming to HTML and back does not open it again", () => {
    expect(after([{ ...notes, name: "trip.md" }])).toBeNull();
    expect(after([{ ...notes, name: "notes.html" }, notes])).toBeNull();
  });

  it("stays closed until Expand opens it", () => {
    expect(sheetAfter(null, notes)).toBeNull();
  });
});
