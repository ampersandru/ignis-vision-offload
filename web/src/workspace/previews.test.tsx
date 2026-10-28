import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { USER, type Version, type WorkspaceFile } from "./files.ts";
import { CSV_ROWS, CsvTable, readySiteFiles } from "./previews.tsx";
import { idleActions as actions } from "./fixture.ts";
import { FileViewer } from "./WorkspacePanel.tsx";

// The previews as the panel renders them. Clicks (the Source tab) and the
// reading of a picture are not exercised: there is no DOM here.

const at = 0;
const text = (t: string): Version => ({ n: 1, author: USER, at, text: t });
const file = (name: string, kind: WorkspaceFile["kind"], v: Version): WorkspaceFile => ({ format: 1, id: `f-${name}`, workspace: "w", name, kind, versions: [v] });

const whom = () => "you";
const view = (f: WorkspaceFile, files: WorkspaceFile[] = [f]) => renderToStaticMarkup(<FileViewer file={f} whom={whom} onBack={() => {}} actions={actions} files={files} />);

describe("an HTML file with its site", () => {
  const index = file("site/index.html", "text", text('<link rel="stylesheet" href="main.css"><script src="app.js"></script><img src="logo.png"><img src="https://x.test/y.png">'));
  const logo = file("site/logo.png", "picture", { n: 1, author: USER, at, bytes: { store: "image", id: "sha", type: "image/png", size: 3 } });
  const files = [index, file("site/main.css", "text", text("h1{color:red}")), file("site/app.js", "text", text("go()")), logo];

  it("shows the page with its stylesheet and script put in, in the same sandbox", () => {
    const html = view(index, files);
    expect(html).toMatch(/<iframe[^>]*sandbox="allow-scripts allow-forms"/);
    expect(html).toContain("&lt;style&gt;h1{color:red}&lt;/style&gt;");
    expect(html).toContain("&lt;script&gt;go()&lt;/script&gt;");
    expect(html).not.toContain("allow-same-origin");
  });

  it("leaves an absolute reference alone and a picture for the read that follows", () => {
    const html = view(index, files);
    expect(html).toContain("https://x.test/y.png");
    expect(html).toContain("src=&quot;logo.png&quot;");
    expect(readySiteFiles(index.versions[0].text ?? "", index.name, files).pictures.map((p) => p.file)).toEqual(["site/logo.png"]);
  });

  it("ignores a deleted file", () => {
    const gone = { ...files[1], versions: [text("x"), { n: 2, author: USER, at, deleted: true as const }] };
    expect(view(index, [index, gone]).includes("&lt;style&gt;")).toBe(false);
  });
});

describe("Markdown, SVG and CSV files", () => {
  it("opens Markdown rendered as a reply is, with its text one tab away, in the editor", () => {
    const html = view(file("notes.md", "text", text("# Trip\n\n- **bread**")));
    expect(html).toContain("<h1>Trip</h1>");
    expect(html).toContain("<strong>bread</strong>");
    expect(html).toMatch(/role="tab"[^>]*>Edit</);
  });

  it("opens an SVG as an image, which runs none of its scripts", () => {
    const html = view(file("logo.svg", "text", text('<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>')));
    expect(html).toMatch(/<img[^>]*src="data:image\/svg\+xml;charset=utf-8,%3Csvg/);
    expect(html).not.toContain("<script>");
    expect(html).toMatch(/role="tab"[^>]*>Edit</);
  });

  it("opens a CSV as a table with a header row", () => {
    const html = view(file("data.csv", "text", text('name,note\n"Rossi, Anna","a ""b"""\nBo')));
    expect(html).toMatch(/<th[^>]*>name<\/th>/);
    expect(html).toMatch(/<td[^>]*>Rossi, Anna<\/td>/);
    expect(html).toContain("a &quot;b&quot;");
    expect(html).toMatch(/role="tab"[^>]*>Edit</);
  });

  it("keeps other text files in the editor alone", () => {
    expect(view(file("a.txt", "text", text("plain")))).not.toContain("<iframe");
  });

  it("caps a long table", () => {
    const rows = Array.from({ length: CSV_ROWS + 5 }, (_, i) => `r${i}`).join("\n");
    const html = renderToStaticMarkup(<CsvTable text={`h\n${rows}`} />);
    expect(html).toContain(`The first ${CSV_ROWS} of ${CSV_ROWS + 5} rows.`);
    expect(html).not.toContain(`>r${CSV_ROWS}<`);
  });
});
