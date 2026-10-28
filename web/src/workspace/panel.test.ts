import { describe, expect, it } from "vitest";
import { createMemoryStore } from "../store/memory.ts";
import { latest, USER, type Version, type WorkspaceFile } from "./files.ts";
import { authorName, createFile, downloadName, editing, fileTree, refusal, saveText, sizeLabel, uploadFiles } from "./panel.ts";
import { createWorkspaces } from "./workspaces.ts";

// What the workspace panel does, apart from how it looks: its tree of the
// files' names, the sizes it shows, and its actions on the workspace module,
// a workspace born for the chat when it has none.

const text = (t: string): Version => ({ n: 1, author: USER, at: 0, text: t });
const file = (name: string, versions: Version[] = [text("x")]): WorkspaceFile => ({ format: 1, id: name, workspace: "w", name, kind: "text", versions });

describe("the panel's tree", () => {
  it("splits the names on /, folders before files, each by name, and leaves deleted files out", () => {
    const tree = fileTree([
      file("site/index.html"),
      file("notes.md"),
      file("site/css/main.css"),
      file("README"),
      file("gone.txt", [text("x"), { n: 2, author: USER, at: 1, deleted: true }]),
      file("site/app.js"),
    ]);
    const shape = (nodes: ReturnType<typeof fileTree>): unknown[] =>
      nodes.map((n) => (n.kind === "folder" ? { [n.path]: shape(n.children) } : n.file.name));
    expect(shape(tree)).toEqual([{ site: [{ "site/css": ["site/css/main.css"] }, "site/app.js", "site/index.html"] }, "notes.md", "README"]);
    const site = tree[0];
    expect(site.kind === "folder" && site.children.map((n) => n.name)).toEqual(["css", "app.js", "index.html"]);
  });

  it("is empty for a workspace with no files", () => {
    expect(fileTree([])).toEqual([]);
  });

  it("holds the deleted files too when they are shown, in their places", () => {
    const tree = fileTree([file("notes.md"), file("site/old.html", [text("x"), { n: 2, author: USER, at: 1, deleted: true }]), file("gone.txt", [text("x"), { n: 2, author: USER, at: 1, deleted: true }])], true);
    const shape = (nodes: ReturnType<typeof fileTree>): unknown[] =>
      nodes.map((n) => (n.kind === "folder" ? { [n.path]: shape(n.children) } : n.file.name));
    expect(shape(tree)).toEqual([{ site: ["site/old.html"] }, "gone.txt", "notes.md"]);
  });
});

describe("who the panel says made a version", () => {
  const chats = { here: (key: string) => key === "chat-a", title: (key: string) => ({ "chat-b": "Budget" })[key] };

  it("is you, this chat or one of its sub-agents, or another chat by its title", () => {
    expect(authorName(USER, chats)).toBe("you");
    expect(authorName({ by: "chat", chat: "chat-a" }, chats)).toBe("this chat");
    expect(authorName({ by: "chat", chat: "chat-a", agent: { call: "c1", name: "researcher" } }, chats)).toBe("this chat's sub-agent researcher");
    expect(authorName({ by: "chat", chat: "chat-b" }, chats)).toBe("the chat “Budget”");
    expect(authorName({ by: "chat", chat: "chat-b", agent: { call: "c2", name: "writer" } }, chats)).toBe("the sub-agent writer of the chat “Budget”");
    // A chat the page no longer has, deleted since.
    expect(authorName({ by: "chat", chat: "chat-z" }, chats)).toBe("another chat");
  });
});

describe("the editor over the file as it is now", () => {
  const by = { by: "chat" as const, chat: "chat-a" };
  const v = (n: number, t: string, author: Version["author"] = USER): Version => ({ n, author, at: n, text: t });
  const notes = (...versions: Version[]) => file("notes.md", versions);

  it("is unchanged while its text is the version it opened", () => {
    expect(editing(notes(v(1, "a")), { base: 1, text: "a" })).toEqual({ base: 1, text: "a", dirty: false, since: null });
    expect(editing(notes(v(1, "a")), { base: 1, text: "ab" })).toEqual({ base: 1, text: "ab", dirty: true, since: null });
  });

  it("follows a newer version while nothing is typed", () => {
    expect(editing(notes(v(1, "a"), v(2, "b", by)), { base: 1, text: "a" })).toEqual({ base: 2, text: "b", dirty: false, since: null });
  });

  it("keeps what is typed when a newer version lands, and names it", () => {
    const file = notes(v(1, "a"), v(2, "b", by));
    expect(editing(file, { base: 1, text: "ax" })).toEqual({ base: 1, text: "ax", dirty: true, since: file.versions[1] });
  });

  it("is unchanged once the version it saved is read back, and while it is not yet", () => {
    expect(editing(notes(v(1, "a"), v(2, "ax")), { base: 1, text: "ax" })).toEqual({ base: 2, text: "ax", dirty: false, since: null });
    expect(editing(notes(v(1, "a")), { base: 2, text: "ax" })).toEqual({ base: 2, text: "ax", dirty: false, since: null });
  });

  it("counts no change in the line endings the text box takes away", () => {
    expect(editing(notes(v(1, "a\r\nb\r\n")), { base: 1, text: "a\nb\n" })).toMatchObject({ dirty: false });
  });
});

describe("what the panel shows of a file", () => {
  it("gives its size in bytes, text counted as it would be saved", () => {
    expect(sizeLabel(text("héllo"))).toBe("6 B");
    expect(sizeLabel({ n: 1, author: USER, at: 0, bytes: { store: "blob", id: "b", type: "application/zip", size: 1536 } })).toBe("1.5 KB");
    // An older chat's PDF came with its text alone.
    expect(sizeLabel({ n: 1, author: USER, at: 0, derived: "[Page 1]" })).toBe("8 characters of text");
  });

  it("downloads a file under the last part of its name", () => {
    expect(downloadName("site/css/main.css")).toBe("main.css");
    expect(downloadName("notes.md")).toBe("notes.md");
  });

  it("says what the browser refused", () => {
    expect(refusal(new Error("QuotaExceededError"))).toBe("Not saved in this browser: QuotaExceededError");
  });
});

function binding(title = "Trip plan") {
  let n = 0;
  const workspaces = createWorkspaces(createMemoryStore(), { newId: () => `id${++n}` });
  let bound: string | undefined;
  const chat = workspaces.chat("chat-a", { bound: () => bound, name: () => title, onBorn: (id) => (bound = id) });
  return { workspaces, chat, bound: () => bound };
}

describe("the panel's actions", () => {
  it("creates an empty text file as the user's, the workspace born with it", async () => {
    const { workspaces, chat, bound } = binding();
    expect(await createFile(workspaces, chat, "notes/todo.md")).toMatchObject({ ok: true, written: { name: "notes/todo.md", version: 1 } });
    expect(await workspaces.get(bound()!)).toMatchObject({ name: "Trip plan" });
    const [made] = await workspaces.files(bound()!);
    expect(made).toMatchObject({ name: "notes/todo.md", kind: "text" });
    expect(latest(made)).toMatchObject({ text: "", author: USER });
    // A name the workspace holds takes a number, as an attached file's does.
    expect(await createFile(workspaces, chat, "notes/todo.md")).toMatchObject({ ok: true, written: { name: "notes/todo (2).md" } });
  });

  it("refuses an empty file under a name that is no name, or that names a kind other than text, and births nothing", async () => {
    const { workspaces, chat } = binding();
    expect(await createFile(workspaces, chat, "  ")).toEqual({ ok: false, error: "The file needs a name, like notes.md or site/index.html." });
    expect(await createFile(workspaces, chat, "../x.md")).toEqual({ ok: false, error: '"../x.md" is not a file name.' });
    expect(await createFile(workspaces, chat, "logo.png")).toEqual({ ok: false, error: "A new file is a text file, and logo.png names a picture file." });
    expect(await workspaces.list()).toEqual([]);
  });

  it("uploads files of any kind as the user's, the workspace born with them", async () => {
    const { workspaces, chat, bound } = binding();
    const names = await uploadFiles(
      workspaces,
      chat,
      [new File(["# A"], "a.md", { type: "text/markdown" }), new File([new Uint8Array([80, 75, 0])], "data.zip", { type: "application/zip" })],
      async () => "",
    );
    expect(names).toEqual(["a.md", "data.zip"]);
    const files = await workspaces.files(bound()!);
    expect(files.map((f) => [f.name, f.kind, latest(f).author])).toEqual([
      ["a.md", "text", USER],
      ["data.zip", "other", USER],
    ]);
    expect(await uploadFiles(workspaces, chat, [], async () => "")).toEqual([]);
  });
});

describe("the editor's Save", () => {
  it("makes one version of the user's per save", async () => {
    const { workspaces, chat, bound } = binding();
    await createFile(workspaces, chat, "notes.md");
    const at = async () => (await workspaces.files(bound()!))[0];
    expect(await saveText(workspaces, await at(), "one")).toMatchObject({ ok: true, written: { version: 2 } });
    expect(await saveText(workspaces, await at(), "two")).toMatchObject({ ok: true, written: { version: 3 } });
    expect((await at()).versions.map((v) => [v.n, v.author, v.text])).toEqual([
      [1, USER, ""],
      [2, USER, "one"],
      [3, USER, "two"],
    ]);
  });

  it("keeps a file's CRLF line endings, which the browser's text box turns into LF", async () => {
    const { workspaces, chat, bound } = binding();
    const id = await chat.workspace();
    await workspaces.add(id, [{ name: "list.txt", kind: "text", text: "a\r\nb\r\n" }], USER);
    const [file] = await workspaces.files(bound()!);
    await saveText(workspaces, file, "a\nb\nc\n");
    expect(latest((await workspaces.files(id))[0]).text).toBe("a\r\nb\r\nc\r\n");
  });
});
