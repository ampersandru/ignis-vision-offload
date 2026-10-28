import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { USER, type Version, type WorkspaceFile } from "./files.ts";
import { DiffView, FileEditor, FileHistory } from "./FileHistory.tsx";
import { authorName } from "./panel.ts";

// A text file's editor, and any file's history with the diff between two
// versions and a restore, as they render. What a Save and a Restore do is the
// module's (workspace.test.ts, manage.test.ts) and the panel's (panel.test.ts):
// clicks are not exercised here.

const at = new Date(2026, 9, 28, 14, 5).getTime();
const chats = { here: (key: string) => key === "chat-a", title: (key: string) => ({ "chat-b": "Budget" })[key] };
const whom = (author: Version["author"]) => authorName(author, chats);

const v = (n: number, author: Version["author"], text: string | null, minute = n): Version => ({
  n,
  author,
  at: at + minute * 60_000,
  ...(text === null ? { deleted: true as const } : { text }),
});

const THIS = { by: "chat" as const, chat: "chat-a" };
const AGENT = { by: "chat" as const, chat: "chat-a", agent: { call: "c1", name: "researcher" } };
const OTHER = { by: "chat" as const, chat: "chat-b" };

const notes: WorkspaceFile = {
  format: 1,
  id: "f1",
  workspace: "w1",
  name: "notes.md",
  kind: "text",
  versions: [v(1, USER, "# Trip\n- Rome\n"), v(2, THIS, "# Trip\n- Rome\n- Pisa\n"), v(3, AGENT, "# Trip\n- Rome\n- Pisa\n- Lucca\n"), v(4, OTHER, "# Trip\n- Naples\n- Pisa\n- Lucca\n")],
};

const save = async () => ({ ok: true as const, version: 5 });

describe("a text file's editor", () => {
  it("holds the newest version's text, with Save and Discard off until something is typed", () => {
    const html = renderToStaticMarkup(<FileEditor file={notes} whom={whom} onSave={save} />);
    expect(html).toMatch(/<textarea[^>]*aria-label="The text of notes.md"[^>]*>/);
    expect(html).toContain("# Trip\n- Naples\n- Pisa\n- Lucca\n</textarea>");
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Save<\/button>/);
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Discard changes<\/button>/);
    expect(html).toContain("Ctrl+S saves");
  });
});

describe("a file's history", () => {
  const render = (file: WorkspaceFile) => renderToStaticMarkup(<FileHistory file={file} whom={whom} onRestore={() => {}} />);

  it("lists every version, newest first, with its time and its author", () => {
    const html = render(notes);
    const rows = ["Version 4", "Version 3", "Version 2", "Version 1"].map((s) => html.indexOf(`>${s}<`));
    expect(rows.every((i) => i > -1)).toBe(true);
    expect([...rows].sort((a, b) => a - b)).toEqual(rows);
    expect(html).toContain("2026-10-28 14:09");
    expect(html).toContain("2026-10-28 14:06");
    for (const who of ["the chat “Budget”", "this chat&#x27;s sub-agent researcher", "this chat", "you"]) expect(html).toContain(who);
  });

  it("offers to restore every version but the newest", () => {
    const html = render(notes);
    for (const n of [1, 2, 3]) expect(html).toContain(`aria-label="Restore version ${n} of notes.md"`);
    expect(html).not.toContain('aria-label="Restore version 4 of notes.md"');
  });

  it("names a deletion, and offers a deleted file's versions before it to restore", () => {
    const gone: WorkspaceFile = { ...notes, versions: [v(1, USER, "x"), v(2, THIS, null)] };
    const html = render(gone);
    expect(html).toMatch(/Version 2<.*deleted the file/s);
    expect(html).toContain('aria-label="Restore version 1 of notes.md"');
    expect(html).not.toContain('aria-label="Restore version 2 of notes.md"');
  });

  it("compares the two newest versions, and lets any two of a text file's versions be chosen", () => {
    const html = render(notes);
    expect(html).toMatch(/<select[^>]*aria-label="Compare from version"/);
    expect(html).toMatch(/<select[^>]*aria-label="Compare to version"/);
    // Every version holding text is offered on both sides; the newest two are chosen.
    expect(html.match(/<option value="\d">/g)?.length ?? 0).toBeGreaterThanOrEqual(6);
    expect(html).toMatch(/<option value="3" selected="">/);
    expect(html).toMatch(/<option value="4" selected="">/);
    expect(html).toContain("- Rome");
    expect(html).toContain("- Naples");
  });

  it("compares nothing in a file with one version, or one that holds no text", () => {
    const one: WorkspaceFile = { ...notes, versions: [v(1, USER, "x")] };
    expect(render(one)).not.toContain("<select");
    const zip: WorkspaceFile = {
      ...notes,
      name: "data.zip",
      kind: "other",
      versions: [
        { n: 1, author: USER, at, bytes: { store: "blob", id: "b1", type: "application/zip", size: 10 } },
        { n: 2, author: USER, at, bytes: { store: "blob", id: "b2", type: "application/zip", size: 2048 } },
      ],
    };
    const html = render(zip);
    expect(html).not.toContain("<select");
    expect(html).toContain("2 KB");
    expect(html).toContain('aria-label="Restore version 1 of data.zip"');
  });
});

describe("the diff between two versions", () => {
  const render = (before: string, after: string) => renderToStaticMarkup(<DiffView before={before} after={after} />);

  it("shows each line removed and added with its sign and its numbers, and how many", () => {
    const html = render("# Trip\n- Rome\n- Pisa\n", "# Trip\n- Naples\n- Pisa\n");
    expect(html).toContain("1 line removed, 1 added");
    expect(html).toMatch(/<tr[^>]*data-diff="removed"[^>]*><td[^>]*>2<\/td><td[^>]*><\/td><td[^>]*>-<\/td><td[^>]*>- Rome<\/td><\/tr>/);
    expect(html).toMatch(/<tr[^>]*data-diff="added"[^>]*><td[^>]*><\/td><td[^>]*>2<\/td><td[^>]*>\+<\/td><td[^>]*>- Naples<\/td><\/tr>/);
    expect(html).toMatch(/<tr[^>]*data-diff="same"[^>]*><td[^>]*>3<\/td><td[^>]*>3<\/td><td[^>]*><\/td><td[^>]*>- Pisa<\/td><\/tr>/);
  });

  it("folds the unchanged lines far from a change", () => {
    const long = Array.from({ length: 20 }, (_, i) => `line ${i + 1}`);
    const html = render(long.join("\n"), [...long.slice(0, 19), "the end"].join("\n"));
    // Lines 17 to 19 are the three before the change.
    expect(html).toContain("16 unchanged lines");
    expect(html).not.toContain(">line 16<");
    expect(html).toContain(">line 17<");
  });

  it("says when two versions hold the same text, or differ only in their line endings", () => {
    expect(render("a\nb\n", "a\nb\n")).toContain("The two versions hold the same text.");
    expect(render("a\r\nb\r\n", "a\nb")).toContain("The two versions differ only in their line endings, or the newline at the end.");
  });
});
