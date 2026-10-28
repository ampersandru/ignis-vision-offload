import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { UserTurn } from "../conversation/UserTurn.tsx";
import { LocalStrip } from "../tools/local/LocalStrip.tsx";
import type { LocalRun } from "../tools/local/local.ts";
import type { Version, WorkspaceFile } from "./files.ts";
import { StaticWorkspaceFiles } from "./WorkspaceFiles.tsx";

// A reply's card for a file it wrote opens the version that reply wrote,
// read from the workspace; once a newer version exists the card says so and
// offers the latest. An older chat's create_file card renders as it always did.

const version = (n: number, text: string): Version => ({ n, author: { by: "chat", chat: "k" }, at: 0, text });

const page = (versions: Version[]): WorkspaceFile => ({ format: 1, id: "f1", workspace: "w1", name: "site/index.html", kind: "text", versions });

const wrote = (n: number): LocalRun => ({
  callId: `c${n}`,
  tool: "write_file",
  status: "done",
  written: { workspace: "w1", file: "f1", name: "site/index.html", version: n },
  result: "Wrote it",
});

const render = (runs: LocalRun[], files: WorkspaceFile[]) =>
  renderToStaticMarkup(
    <StaticWorkspaceFiles files={files}>
      <LocalStrip runs={runs} />
    </StaticWorkspaceFiles>,
  );

describe("a reply's file card", () => {
  it("opens the version its reply wrote, and is not superseded while that is the newest", () => {
    const html = render([wrote(1)], [page([version(1, "<p>One</p>")])]);
    expect(html).toContain("site/index.html");
    expect(html).toContain("version 1");
    expect(html).toContain("&lt;p&gt;One&lt;/p&gt;");
    expect(html).not.toContain("Superseded");
  });

  it("is marked superseded once a newer version exists, still opens its own, and offers the latest", () => {
    const html = render([wrote(1)], [page([version(1, "<p>One</p>"), version(2, "<p>Two</p>")])]);
    expect(html).toContain("Superseded by version 2");
    expect(html).toContain("&lt;p&gt;One&lt;/p&gt;");
    expect(html).not.toContain("&lt;p&gt;Two&lt;/p&gt;");
    expect(html).toMatch(/<button[^>]*>Open the latest<\/button>/);
  });

  it("is marked deleted once a newer version deletes the file, and still opens its own", () => {
    const html = render([wrote(1)], [page([version(1, "<p>One</p>"), version(2, "<p>Two</p>"), { n: 3, author: { by: "user" }, at: 0, deleted: true }])]);
    expect(html).toContain("Deleted in version 3; the workspace panel restores it from its history.");
    expect(html).toContain("&lt;p&gt;One&lt;/p&gt;");
    expect(html).not.toContain("Superseded");
    expect(html).not.toContain("Open the latest");
  });

  it("says so when the workspace no longer has the file", () => {
    expect(render([wrote(2)], [])).toContain("site/index.html, version 2: no longer in the workspace.");
  });

  it("renders an older chat's create_file card as before", () => {
    const html = render([{ callId: "c", tool: "create_file", status: "done", file: { name: "notes.md", content: "# Hi" } }], []);
    expect(html).toContain("notes.md");
    expect(html).toContain("4 chars");
    expect(html).toContain(">Download<");
  });

  it("opens the version an edit made, as a write's", () => {
    const edit: LocalRun = { ...wrote(2), tool: "edit_file" };
    const html = render([edit], [page([version(1, "<p>One</p>"), version(2, "<p>Two</p>")])]);
    expect(html).toContain("version 2");
    expect(html).toContain("&lt;p&gt;Two&lt;/p&gt;");
  });

  it("names the file a deletion took out, a read that showed no lines, and a refusal", () => {
    const html = render(
      [
        { callId: "d", tool: "delete_file", status: "done", written: { workspace: "w1", file: "f1", name: "draft.md", version: 3 } },
        { callId: "r", tool: "read_file", status: "done", seen: { workspace: "w1", file: "f2", name: "data.zip", version: 1 } },
        { callId: "e", tool: "edit_file", status: "failed", error: "notes.md changed since you last read it" },
      ],
      [],
    );
    expect(html).toMatch(/Delete<\/span><span[^>]*>draft\.md<\/span>/);
    expect(html).toMatch(/Read<\/span><span[^>]*>data\.zip, version 1<\/span>/);
    expect(html).toMatch(/Edit<\/span>/);
    expect(html).toContain("notes.md changed since you last read it");
  });

  it("names what list_files and read_file did, and an older chat's read in characters", () => {
    const html = render(
      [
        { callId: "a", tool: "list_files", status: "done", listed: 3 },
        { callId: "b", tool: "read_file", status: "done", read: { name: "notes.md", from: 1, to: 40, total: 120, lines: true } },
        { callId: "c", tool: "read_file", status: "done", read: { name: "old.txt", from: 0, to: 500, total: 900 } },
      ],
      [],
    );
    expect(html).toContain("3 files");
    expect(html).toContain("notes.md, lines 1–40 of 120");
    expect(html).toContain("old.txt, characters 0–500 of 900");
  });
});

describe("a prompt that attached files", () => {
  it("shows the names they took in the workspace under its text", () => {
    const message = { id: 1, role: "user" as const, content: "Summarise", reasoning: "", streaming: false, attached: ["brief.pdf", "notes (2).txt"] };
    const html = renderToStaticMarkup(<UserTurn message={message} actions={{ canRerun: true, onSave: () => {}, onFork: () => {} }} onResend={() => {}} />);
    expect(html).toContain('aria-label="Attached to the workspace"');
    expect(html.indexOf("brief.pdf")).toBeGreaterThan(html.indexOf("Summarise"));
    expect(html).toContain("notes (2).txt");
  });
});
