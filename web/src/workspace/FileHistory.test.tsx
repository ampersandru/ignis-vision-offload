import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { USER, type Version, type WorkspaceFile } from "./files.ts";
import { DiffSheet, DiffView, FileEditor, FileHistory } from "./FileHistory.tsx";
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

  it("holds a file with CRLF line endings as the browser's box holds it, with LF ones", () => {
    // A box given a value with CR in it reads back without, and would be given it again at each render, its caret moved.
    const crlf: WorkspaceFile = { ...notes, versions: [v(1, USER, "a\r\nb\r\n")] };
    const html = renderToStaticMarkup(<FileEditor file={crlf} whom={whom} onSave={save} />);
    expect(html).toContain(">a\nb\n</textarea>");
    expect(html).not.toContain("\r");
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

describe("the Expand button of the panel's diff", () => {
  it("opens the diff full screen from beside the count, and the panel keeps its unified table", () => {
    const html = renderToStaticMarkup(<FileHistory file={notes} whom={whom} onRestore={() => {}} />);
    expect(html).toMatch(/<p[^>]*>1 line removed, 1 added<\/p><button[^>]*>Expand<\/button>/);
    expect(html).toContain('data-diff="removed"');
    expect(html).not.toContain('role="dialog"');
  });

  it("is not in a diff of two equal texts, nor without a handler", () => {
    expect(renderToStaticMarkup(<DiffView before="a\n" after="a\n" onExpand={() => {}} />)).not.toContain("Expand");
    expect(renderToStaticMarkup(<DiffView before="a\n" after="b\n" />)).not.toContain("Expand");
  });
});

describe("the full-screen diff", () => {
  const versions = [v(1, USER, "# Trip\n- Rome\n- Pisa\n"), v(2, THIS, "# Trip\n- Naples\n- Bari\n- Pisa\n")];
  const sheet = (over: Partial<Parameters<typeof DiffSheet>[0]> = {}) =>
    renderToStaticMarkup(
      <DiffSheet
        name="notes.md"
        versions={versions}
        whom={whom}
        from={1}
        to={2}
        onChoose={() => {}}
        layout="split"
        onLayout={() => {}}
        showAll={false}
        onShowAll={() => {}}
        at={0}
        onGo={() => {}}
        onClose={() => {}}
        {...over}
      />,
    );

  it("heads the sheet with the file, both version selects, the count, the layouts, Show all lines, the changes and Close", () => {
    const html = sheet();
    expect(html).toMatch(/role="dialog"[^>]*aria-modal="true"/);
    expect(html).toContain(">notes.md</h2>");
    expect(html).toMatch(/<select[^>]*aria-label="Compare from version"[^>]*>/);
    expect(html).toMatch(/<select[^>]*aria-label="Compare to version"[^>]*>/);
    expect(html).toContain("1 line removed, 2 added");
    expect(html).toMatch(/role="radiogroup"[^>]*aria-label="Layout"/);
    expect(html).toMatch(/<button[^>]*role="radio"[^>]*aria-checked="true"[^>]*>Side by side<\/button>/);
    expect(html).toMatch(/<button[^>]*role="radio"[^>]*aria-checked="false"[^>]*>Unified<\/button>/);
    expect(html).toMatch(/<button[^>]*aria-pressed="false"[^>]*>Show all lines<\/button>/);
    expect(html).toMatch(/<button[^>]*title="Previous change \(p\)"[^>]*disabled=""[^>]*>Previous change<\/button>/);
    expect(html).toMatch(/<button[^>]*title="Next change \(n\)"[^>]*disabled=""[^>]*>Next change<\/button>/);
    expect(html).toContain('title="Close (Esc)"');
  });

  it("lays the versions side by side: before left, after right, a removed line facing the added one, the shorter side padded", () => {
    const html = sheet();
    expect(html).toMatch(/<tr[^>]*data-diff="changed"[^>]*><td[^>]*>2<\/td><td[^>]*>- Rome<\/td><td[^>]*>2<\/td><td[^>]*>- Naples<\/td><\/tr>/);
    expect(html).toMatch(/<tr[^>]*data-diff="changed"[^>]*><td[^>]*><\/td><td[^>]*><\/td><td[^>]*>3<\/td><td[^>]*>- Bari<\/td><\/tr>/);
    expect(html).toMatch(/<tr[^>]*data-diff="same"[^>]*><td[^>]*>3<\/td><td[^>]*>- Pisa<\/td><td[^>]*>4<\/td><td[^>]*>- Pisa<\/td><\/tr>/);
    expect(html).toContain("whitespace-pre-wrap break-all");
    expect((html.match(/data-anchor/g) ?? []).length).toBe(1);
  });

  it("lays them in one table when Unified is chosen, with the sign column", () => {
    const html = sheet({ layout: "unified" });
    expect(html).toMatch(/<button[^>]*aria-checked="true"[^>]*>Unified<\/button>/);
    expect(html).toMatch(/<tr[^>]*data-diff="removed"[^>]*><td[^>]*>2<\/td><td[^>]*><\/td><td[^>]*>-<\/td><td[^>]*>- Rome<\/td><\/tr>/);
    expect(html).toMatch(/<tr[^>]*data-diff="added"[^>]*><td[^>]*><\/td><td[^>]*>3<\/td><td[^>]*>\+<\/td><td[^>]*>- Bari<\/td><\/tr>/);
  });

  it("folds the unchanged lines across both columns, unless Show all lines is on", () => {
    const long = Array.from({ length: 20 }, (_, i) => `line ${i + 1}`);
    const two = [v(1, USER, long.join("\n")), v(2, THIS, [...long.slice(0, 19), "the end"].join("\n"))];
    expect(sheet({ versions: two })).toMatch(/<td colSpan="4"[^>]*>16 unchanged lines<\/td>/);
    const all = sheet({ versions: two, showAll: true });
    expect(all).not.toContain("unchanged lines");
    expect(all).toContain(">line 1<");
    expect(all).toMatch(/<button[^>]*aria-pressed="true"[^>]*>Show all lines<\/button>/);
  });

  it("enables Next change while more changes follow, and Previous change after the first", () => {
    const long = Array.from({ length: 30 }, (_, i) => `line ${i + 1}`);
    const edited = long.map((l, i) => (i === 2 || i === 25 ? `${l}!` : l));
    const two = [v(1, USER, long.join("\n")), v(2, THIS, edited.join("\n"))];
    const first = sheet({ versions: two });
    expect(first).not.toMatch(/title="Next change \(n\)"[^>]*disabled=""/);
    expect(first).toMatch(/title="Previous change \(p\)"[^>]*disabled=""/);
    expect((first.match(/data-anchor/g) ?? []).length).toBe(2);
    expect(sheet({ versions: two, at: 1 })).toMatch(/title="Next change \(n\)"[^>]*disabled=""/);
  });

  it("says when two versions hold the same text, or differ only in their line endings", () => {
    const same = [v(1, USER, "a\nb\n"), v(2, THIS, "a\nb\n")];
    expect(sheet({ versions: same })).toContain("The two versions hold the same text.");
    expect(sheet({ versions: same })).not.toContain("Show all lines");
    const eol = [v(1, USER, "a\r\nb\r\n"), v(2, THIS, "a\nb")];
    expect(sheet({ versions: eol })).toContain("The two versions differ only in their line endings, or the newline at the end.");
  });
});
