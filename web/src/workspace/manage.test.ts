import { describe, expect, it } from "vitest";
import { bindWorkspace, createSession, openSession, removeSession } from "../sessions/sessions.ts";
import { createMemoryStore } from "../store/memory.ts";
import { freedBy, latest, present, USER, type Version, type WorkspaceFile } from "./files.ts";
import { commitAttached } from "./attach.ts";
import { uploadFiles } from "./panel.ts";
import { madeIn } from "./pictures.ts";
import { sendFile } from "./send.ts";
import { runWorkspaceTool } from "./tools.ts";
import { createWorkspaces, filesCollection } from "./workspaces.ts";

// The user's side of the workspace module, the panel's and the picker's
// actions, over the in-memory store: deleting, renaming, a name unique in its
// workspace, and deleting a workspace with what it frees.

/** A clock that moves one minute per reading. */
function clock() {
  let minute = 0;
  return () => new Date(2026, 9, 28, 14, minute++).getTime();
}

function module() {
  let n = 0;
  const store = createMemoryStore();
  return { store, workspaces: createWorkspaces(store, { now: clock(), newId: () => `id${++n}` }) };
}

const CHAT = { by: "chat" as const, chat: "chat-a" };

describe("deleting a file", () => {
  it("is a new version marking it deleted, by its author, the versions before it kept", async () => {
    const { workspaces } = module();
    const w = await workspaces.create("Trip");
    const [notes, todo] = await workspaces.add(w.id, [{ name: "notes.md", kind: "text", text: "# Trip" }, { name: "todo.txt", kind: "text", text: "pack" }], USER);
    const byUser = await workspaces.deleteFile(w.id, notes.file, USER);
    expect(byUser).toEqual({ ok: true, written: { workspace: w.id, file: notes.file, name: "notes.md", version: 2 } });
    // The model deletes as the chat's reader, which has read the file (the guard, workspace.test.ts).
    const chat = workspaces.chat(CHAT.chat, { bound: () => w.id, name: () => "Trip", onBorn: () => {} }).reader;
    workspaces.read(chat, (await workspaces.file(w.id, todo.file))!);
    const byChat = await workspaces.deleteFile(w.id, todo.file, chat);
    expect(byChat).toMatchObject({ ok: true, written: { name: "todo.txt", version: 2 } });
    const [a, b] = await workspaces.files(w.id);
    expect(a.versions.map((v) => [v.n, v.author, v.text, v.deleted])).toEqual([
      [1, USER, "# Trip", undefined],
      [2, USER, undefined, true],
    ]);
    expect(latest(b)).toMatchObject({ n: 2, author: CHAT, deleted: true });
    expect([a, b].some(present)).toBe(false);
  });

  it("refuses a file that is not there, or already deleted", async () => {
    const { workspaces } = module();
    const w = await workspaces.create("Trip");
    const [notes] = await workspaces.add(w.id, [{ name: "notes.md", kind: "text", text: "x" }], USER);
    await workspaces.deleteFile(w.id, notes.file, USER);
    expect(await workspaces.deleteFile(w.id, notes.file, USER)).toEqual({ ok: false, error: "notes.md is already deleted." });
    expect(await workspaces.deleteFile(w.id, "nope", USER)).toEqual({ ok: false, error: "The workspace has no such file." });
  });

  it("announces the change", async () => {
    const { workspaces } = module();
    const w = await workspaces.create("Trip");
    const [notes] = await workspaces.add(w.id, [{ name: "notes.md", kind: "text", text: "x" }], USER);
    const heard: unknown[] = [];
    workspaces.subscribe((change) => heard.push(change));
    await workspaces.deleteFile(w.id, notes.file, USER);
    expect(heard).toEqual([{ workspace: w.id, file: notes.file }]);
  });
});

describe("restoring a version", () => {
  it("makes it the newest: a new version of the user's with its content, the versions between kept", async () => {
    const { workspaces } = module();
    const w = await workspaces.create("Trip");
    const [notes] = await workspaces.add(w.id, [{ name: "notes.md", kind: "text", text: "- Rome\n" }], USER);
    const chat = workspaces.chat(CHAT.chat, { bound: () => w.id, name: () => "Trip", onBorn: () => {} }).reader;
    workspaces.read(chat, (await workspaces.file(w.id, notes.file))!);
    await workspaces.writeText(w.id, "notes.md", "- Naples\n", chat);
    const heard: unknown[] = [];
    workspaces.subscribe((change) => heard.push(change));
    expect(await workspaces.restore(w.id, notes.file, 1)).toEqual({ ok: true, written: { workspace: w.id, file: notes.file, name: "notes.md", version: 3 } });
    const file = (await workspaces.file(w.id, notes.file))!;
    expect(file.versions.map((v) => [v.n, v.author, v.text])).toEqual([
      [1, USER, "- Rome\n"],
      [2, CHAT, "- Naples\n"],
      [3, USER, "- Rome\n"],
    ]);
    expect(heard).toEqual([{ workspace: w.id, file: notes.file }]);
  });

  it("brings a deleted file back, of any kind, with the bytes and the text its version held", async () => {
    const { workspaces } = module();
    const w = await workspaces.create("Trip");
    const [brief] = await workspaces.add(w.id, [{ name: "brief.pdf", kind: "pdf", blob: new Blob(["%PDF"]), derived: "[Page 1]\nThe brief" }], USER);
    await workspaces.deleteFile(w.id, brief.file, USER);
    expect(await workspaces.restore(w.id, brief.file, 1)).toMatchObject({ ok: true, written: { version: 3 } });
    const file = (await workspaces.file(w.id, brief.file))!;
    expect(present(file)).toBe(true);
    const [first, , restored] = file.versions;
    expect(restored).toEqual({ n: 3, author: USER, at: restored.at, bytes: first.bytes, derived: "[Page 1]\nThe brief" });
    expect(await (await workspaces.content(restored))!.text()).toBe("%PDF");
  });

  it("refuses the newest version, a deletion, a version the file never had and a file that is not there", async () => {
    const { workspaces } = module();
    const w = await workspaces.create("Trip");
    const [notes] = await workspaces.add(w.id, [{ name: "notes.md", kind: "text", text: "x" }], USER);
    expect(await workspaces.restore(w.id, notes.file, 1)).toEqual({ ok: false, error: "Version 1 is already the newest of notes.md." });
    await workspaces.deleteFile(w.id, notes.file, USER);
    expect(await workspaces.restore(w.id, notes.file, 2)).toEqual({ ok: false, error: "Version 2 of notes.md is its deletion, which has no content to restore." });
    expect(await workspaces.restore(w.id, notes.file, 7)).toEqual({ ok: false, error: "notes.md has no version 7." });
    expect(await workspaces.restore(w.id, "nope", 1)).toEqual({ ok: false, error: "The workspace has no such file." });
    expect((await workspaces.file(w.id, notes.file))!.versions).toHaveLength(2);
  });
});

describe("a file's content", () => {
  it("is its text, or its bytes from the store, for a download", async () => {
    const { workspaces } = module();
    const w = await workspaces.create("A");
    const zip = new Blob([new Uint8Array([80, 75, 3, 4])], { type: "application/zip" });
    await workspaces.add(w.id, [{ name: "a.md", kind: "text", text: "# A" }, { name: "b.zip", kind: "other", blob: zip }, { name: "old.pdf", kind: "pdf", derived: "[Page 1]" }], USER);
    const [a, b, old] = await workspaces.files(w.id);
    const text = await workspaces.content(latest(a));
    expect(text?.type).toBe("text/plain;charset=utf-8");
    expect(await text?.text()).toBe("# A");
    const got = await workspaces.content(latest(b));
    expect(got?.type).toBe("application/zip");
    expect([...new Uint8Array(await got!.arrayBuffer())]).toEqual([80, 75, 3, 4]);
    // An older chat's PDF came with its text alone: there are no bytes to give.
    expect(await workspaces.content(latest(old))).toBeNull();
  });
});

describe("deleting a workspace", () => {
  const bytes = (fill: number) => new Blob([new Uint8Array(64).fill(fill)], { type: "application/zip" });

  it("deletes it with its files, and frees the bytes no other workspace holds", async () => {
    const { store, workspaces } = module();
    const a = await workspaces.create("A");
    const b = await workspaces.create("B");
    // The same bytes in both workspaces are one blob; the other bytes are A's alone.
    const [shared] = await workspaces.add(a.id, [{ name: "shared.zip", kind: "other", blob: bytes(1) }, { name: "own.zip", kind: "other", blob: bytes(2) }], USER);
    await workspaces.add(b.id, [{ name: "copy.zip", kind: "other", blob: bytes(1) }], USER);
    const blobOf = async (name: string) => latest((await workspaces.files(a.id)).find((f) => f.name === name)!).bytes!.id;
    const one = await blobOf("shared.zip");
    const two = await blobOf("own.zip");
    const heard: unknown[] = [];
    workspaces.subscribe((change) => heard.push(change));
    expect(await workspaces.remove(a.id, [])).toEqual({ files: 2 });
    expect(await workspaces.get(a.id)).toBeNull();
    expect(await workspaces.files(a.id)).toEqual([]);
    expect((await workspaces.list()).map((w) => w.name)).toEqual(["B"]);
    expect(await store.getBlob(two)).toBeNull();
    expect(await store.getBlob(one)).not.toBeNull();
    expect(heard).toContainEqual({ workspace: a.id, file: shared.file });
    expect(heard).toContainEqual({ workspace: a.id });
    await workspaces.remove(b.id, []);
    expect(await store.getBlob(one)).toBeNull();
  });

  it("frees the bytes of every version, a deleted file's included", async () => {
    const { store, workspaces } = module();
    const a = await workspaces.create("A");
    const [file] = await workspaces.add(a.id, [{ name: "old.zip", kind: "other", blob: bytes(3) }], USER);
    const [kept] = await workspaces.files(a.id);
    await workspaces.deleteFile(a.id, file.file, USER);
    await workspaces.remove(a.id, []);
    expect(await store.getBlob(kept.versions[0].bytes!.id)).toBeNull();
  });

  it("leaves the chats bound to it naming a workspace that is gone: the next file of one starts a new workspace, which it binds in place of the old", async () => {
    const { workspaces } = module();
    let bound: string | undefined;
    const born: [string, string | undefined][] = [];
    const chat = workspaces.chat("chat-a", { bound: () => bound, name: () => "Trip", onBorn: (id, replaced) => (born.push([id, replaced]), (bound = id)) });
    await uploadFiles(workspaces, chat, [new File(["# Trip"], "notes.md")]);
    const [first] = await workspaces.list();
    await workspaces.remove(first.id, []);
    // The chat still names it; its next file goes to a new workspace, named after the chat.
    expect(await uploadFiles(workspaces, chat, [new File(["- Rome"], "plan.md")])).toEqual(["plan.md"]);
    const [second] = await workspaces.list();
    expect(second).toMatchObject({ name: "Trip" });
    expect(second.id).not.toBe(first.id);
    expect(born).toEqual([
      [first.id, undefined],
      [second.id, first.id],
    ]);
  });

  it("deleted in another tab is the same: the chat's next write starts a new workspace, and leaves no file behind in the one that is gone", async () => {
    const store = createMemoryStore();
    let n = 0;
    const here = createWorkspaces(store, { now: clock(), newId: () => `here${++n}` });
    const there = createWorkspaces(store, { now: clock(), newId: () => `there${++n}` });
    let bound: string | undefined;
    const chat = here.chat("chat-a", { bound: () => bound, name: () => "Trip", onBorn: (id) => (bound = id) });
    expect((await runWorkspaceTool("write_file", { name: "a.md", content: "a" }, { workspaces: here, chat })).ok).toBe(true);
    const gone = bound!;
    await there.remove(gone, []);
    const deleted = "This chat's workspace was deleted: a new one is created with the next file you write or the user attaches.";
    expect(await runWorkspaceTool("list_files", {}, { workspaces: here, chat })).toEqual({ ok: true, run: { listed: 0, result: deleted } });
    expect(await runWorkspaceTool("read_file", { name: "a.md" }, { workspaces: here, chat })).toEqual({ ok: false, error: deleted });
    const wrote = await runWorkspaceTool("write_file", { name: "b.md", content: "b" }, { workspaces: here, chat });
    expect(wrote.ok && wrote.run.result).toBe('Created b.md, version 1 (1 line, 1 character), in the workspace "Trip". The user sees it in your reply.');
    expect(bound).not.toBe(gone);
    expect(await store.listDocs(filesCollection(gone))).toEqual([]);
    expect((await here.list()).map((w) => w.id)).toEqual([bound]);
  });
});

describe("a workspace born with its first file", () => {
  /** The workspace module over a store that refuses what `refuses` names while `refuse.now` holds, and a chat of it with no workspace yet. */
  function refusing(refuses: "putBlob" | "putImage" | "files") {
    const store = createMemoryStore();
    const refuse = { now: true };
    const no = <T>(then: () => Promise<T>) => (refuse.now ? Promise.reject(new Error("The quota is exceeded.")) : then());
    const refused = {
      ...store,
      ...(refuses === "files"
        ? { putDoc: (c: string, id: string, v: unknown) => (c.startsWith("workspace-files/") ? no(() => store.putDoc(c, id, v)) : store.putDoc(c, id, v)) }
        : refuses === "putBlob"
          ? { putBlob: (blob: Blob) => no(() => store.putBlob(blob)) }
          : { putImage: (blob: Blob, info: { width: number; height: number }) => no(() => store.putImage(blob, info)) }),
    };
    let n = 0;
    const workspaces = createWorkspaces(refused, { now: clock(), newId: () => `id${++n}`, measure: async () => ({ width: 8, height: 6 }) });
    const born: string[] = [];
    const chat = workspaces.chat("chat-a", { bound: () => born.at(-1), name: () => "Trip", onBorn: (id) => born.push(id) });
    return { workspaces, chat, born, refuse };
  }

  it("leaves no workspace when the store refuses the files a prompt attaches, and the chat unbound", async () => {
    const { workspaces, chat, born } = refusing("putBlob");
    await expect(commitAttached(workspaces, chat, [{ name: "memo.wav", kind: "audio", blob: new Blob([new Uint8Array([1])]) }])).rejects.toThrow("The quota is exceeded.");
    expect(await workspaces.list()).toEqual([]);
    expect(born).toEqual([]);
  });

  it("leaves no workspace when the store refuses the picture make_image made", async () => {
    const { workspaces, chat, born } = refusing("putImage");
    await expect(madeIn(workspaces, chat).land(new Blob([new Uint8Array([2])], { type: "image/png" }), "made-1.png")).rejects.toThrow("The quota is exceeded.");
    expect(await workspaces.list()).toEqual([]);
    expect(born).toEqual([]);
  });

  it("leaves no workspace when the store refuses a file sent to a new workspace", async () => {
    const { workspaces } = refusing("putBlob");
    await expect(sendFile(workspaces, { kind: "new", name: "Fox" }, { name: "fox.wav", kind: "audio", blob: new Blob([new Uint8Array([3])]) })).rejects.toThrow("The quota is exceeded.");
    expect(await workspaces.list()).toEqual([]);
  });

  it("leaves no workspace when the store refuses the model's first write, and is born with the next one the store takes", async () => {
    const { workspaces, chat, born, refuse } = refusing("files");
    await expect(runWorkspaceTool("write_file", { name: "notes.md", content: "x" }, { workspaces, chat })).rejects.toThrow("The quota is exceeded.");
    expect(await workspaces.list()).toEqual([]);
    expect(born).toEqual([]);
    refuse.now = false;
    expect((await runWorkspaceTool("write_file", { name: "notes.md", content: "x" }, { workspaces, chat })).ok).toBe(true);
    expect((await workspaces.list()).map((w) => [w.id, w.name])).toEqual([[born[0], "Trip"]]);
    expect(born).toHaveLength(1);
    expect((await workspaces.files(born[0])).map((f) => f.name)).toEqual(["notes.md"]);
  });
});

describe("the pictures deleting a workspace frees", () => {
  const picture = (id: string): Version => ({ n: 1, author: USER, at: 0, bytes: { store: "image", id, type: "image/png", size: 10 } });
  const file = (workspace: string, name: string, versions: Version[]): WorkspaceFile => ({ format: 1, id: name, workspace, name, kind: "picture", versions });

  it("are those no other workspace and nothing the page names elsewhere holds, and none while that is not known", () => {
    const gone = [file("a", "one.png", [picture("p1"), { ...picture("p2"), n: 2 }]), file("a", "two.png", [picture("p3")])];
    const others = [file("b", "copy.png", [picture("p1")])];
    expect(freedBy(gone, others, ["p3"])).toEqual({ blobs: [], images: ["p2"] });
    expect(freedBy(gone, others, null)).toEqual({ blobs: [], images: [] });
  });
});

describe("a workspace and its chats", () => {
  it("outlives the chat that made it: a new chat picks it before its first file, and works on its files", async () => {
    const { workspaces } = module();
    // Chat 1 is born into a workspace with its first file, then deleted.
    let list = { sessions: [createSession(1)], activeId: 1 };
    const first = workspaces.chat("key-1", { bound: () => list.sessions.find((s) => s.id === 1)?.workspaceId, name: () => "Trip plan", onBorn: (id) => (list = { ...list, sessions: bindWorkspace(list.sessions, 1, id) }) });
    await workspaces.writeText(await first.workspace(), "notes.md", "# Trip", first.reader);
    list = openSession(removeSession(list, 1, 2), 3);
    const [kept] = await workspaces.list();
    expect(kept).toMatchObject({ name: "Trip plan" });
    // Chat 2 binds to it from the picker's list, and the model finds the file there.
    list = { ...list, sessions: bindWorkspace(list.sessions, list.activeId, kept.id) };
    const second = workspaces.chat("key-2", { bound: () => list.sessions.find((s) => s.id === list.activeId)?.workspaceId, name: () => "", onBorn: () => {} });
    expect(await second.workspace()).toBe(kept.id);
    const listed = await runWorkspaceTool("list_files", {}, { workspaces, chat: second });
    expect(listed.ok && listed.run.result).toContain("- notes.md: text");
    expect(await workspaces.list()).toHaveLength(1);
  });

  it("says a file uploaded of a kind the model cannot read is not readable", async () => {
    const { workspaces } = module();
    let bound: string | undefined;
    const chat = workspaces.chat("key-1", { bound: () => bound, name: () => "Data", onBorn: (id) => (bound = id) });
    await uploadFiles(workspaces, chat, [new File([new Uint8Array([80, 75, 3, 4, 0])], "sheet.xlsx")]);
    const listed = await runWorkspaceTool("list_files", {}, { workspaces, chat });
    expect(listed.ok && listed.run.result).toContain("- sheet.xlsx: other, 5 bytes;");
    const read = await runWorkspaceTool("read_file", { name: "sheet.xlsx" }, { workspaces, chat });
    expect(read.ok && read.run.result).toBe("[sheet.xlsx, version 1: a file of a kind read_file cannot read; 5 bytes.]");
  });
});

describe("renaming a workspace", () => {
  it("changes its name, and refuses a blank one", async () => {
    const { workspaces } = module();
    const w = await workspaces.create("Trip plan");
    const heard: unknown[] = [];
    workspaces.subscribe((change) => heard.push(change));
    expect(await workspaces.rename(w.id, "  Rome in May ")).toMatchObject({ ok: true, workspace: { id: w.id, name: "Rome in May" } });
    expect(await workspaces.get(w.id)).toMatchObject({ name: "Rome in May", created: w.created });
    expect(await workspaces.rename(w.id, "   ")).toEqual({ ok: false, error: "A workspace needs a name." });
    expect(await workspaces.rename("nope", "X")).toEqual({ ok: false, error: "That workspace was deleted." });
    expect(heard).toEqual([{ workspace: w.id }]);
  });
});

describe("renaming a file", () => {
  it("changes its name, never its id, kind or versions, and may move it into a folder", async () => {
    const { workspaces } = module();
    const w = await workspaces.create("Site");
    const [page] = await workspaces.add(w.id, [{ name: "index.html", kind: "text", text: "<p>Hi</p>" }], USER);
    const heard: unknown[] = [];
    workspaces.subscribe((change) => heard.push(change));
    expect(await workspaces.renameFile(w.id, page.file, " site\\index.html ")).toEqual({
      ok: true,
      written: { workspace: w.id, file: page.file, name: "site/index.html", version: 1 },
    });
    const [file] = await workspaces.files(w.id);
    expect(file).toMatchObject({ id: page.file, name: "site/index.html", kind: "text" });
    expect(file.versions).toHaveLength(1);
    expect(heard).toEqual([{ workspace: w.id, file: page.file }]);
  });

  it("refuses a name another file holds, a deleted one included, and a name that is no name", async () => {
    const { workspaces } = module();
    const w = await workspaces.create("Site");
    const [a, b, gone] = await workspaces.add(
      w.id,
      [{ name: "a.md", kind: "text", text: "a" }, { name: "b.md", kind: "text", text: "b" }, { name: "old.md", kind: "text", text: "o" }],
      USER,
    );
    await workspaces.deleteFile(w.id, gone.file, USER);
    expect(await workspaces.renameFile(w.id, a.file, "b.md")).toEqual({ ok: false, error: "A file named b.md is already in the workspace." });
    expect(await workspaces.renameFile(w.id, a.file, "old.md")).toEqual({ ok: false, error: "A deleted file named old.md keeps that name in its history." });
    expect(await workspaces.renameFile(w.id, a.file, "../x.md")).toEqual({ ok: false, error: '"../x.md" is not a file name.' });
    expect(await workspaces.renameFile(w.id, gone.file, "new.md")).toEqual({ ok: false, error: "old.md is deleted." });
    // The same name again changes nothing, and is no error.
    expect(await workspaces.renameFile(w.id, b.file, "b.md")).toMatchObject({ ok: true, written: { name: "b.md" } });
    expect((await workspaces.files(w.id)).map((f) => f.name)).toEqual(["a.md", "b.md", "old.md"]);
  });
});
