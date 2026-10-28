import { describe, expect, it } from "vitest";
import type { Settings } from "../api/request.ts";
import type { StreamOptions, StreamResult } from "../api/stream.ts";
import { bornInto, createSession, type Message, moveWorkspace, type Session } from "../sessions/sessions.ts";
import { createMemoryStore } from "../store/memory.ts";
import { runAgents } from "../tools/agents/agents.ts";
import { agentExtras, NO_TOOLS } from "../tools/index.ts";
import { type LocalRun, runLocalCalls } from "../tools/local/local.ts";
import { latest, USER } from "./files.ts";
import { saveText } from "./panel.ts";
import { READ_LIMIT, seenIn, type WorkspaceAccess } from "./tools.ts";
import { createWorkspaces, type Workspaces } from "./workspaces.ts";

// The workspace module driven through its two entrances, over the in-memory
// store: the model's tool calls, through the local tools' router as a turn
// runs them, and the user's actions.

const settings: Settings = {
  systemPrompt: "",
  temperature: 1,
  topP: 0.95,
  maxTokens: 100,
  reasoningEffort: "none",
  thinkingBudget: null,
  laneTag: "interactive",
};

/** A clock that moves one minute per reading, from 28 October 2026 at 14:00 local time. */
function clock() {
  let minute = 0;
  return () => new Date(2026, 9, 28, 14, minute++).getTime();
}

function module(): Workspaces {
  let n = 0;
  return createWorkspaces(createMemoryStore(), { now: clock(), newId: () => `id${++n}` });
}

/** A chat as the conversation loop binds it: its key, its title, and the workspace it holds once one is born. */
function chat(workspaces: Workspaces, key = "chat-a", title = "Trip plan") {
  let bound: string | undefined;
  const born: string[] = [];
  const binding = workspaces.chat(key, {
    bound: () => bound,
    name: () => title,
    onBorn: (id) => {
      bound = id;
      born.push(id);
    },
  });
  const access: WorkspaceAccess = { workspaces, chat: binding };
  return { access, born, bound: () => bound, bind: (id: string) => (bound = id) };
}

let callN = 0;
const call = (name: string, args: object) => ({ id: `c${++callN}`, name, arguments: JSON.stringify(args) });

async function run(access: WorkspaceAccess, name: string, args: object): Promise<LocalRun> {
  const [ran] = await runLocalCalls([call(name, args)], { settings, signal: new AbortController().signal, jsSafetyCheck: false, workspace: access }, () => {});
  return ran;
}

describe("a chat's workspace", () => {
  it("is born with the chat's first file, named after the chat, and the chat binds to it", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    expect(await workspaces.list()).toEqual([]);
    const wrote = await run(a.access, "write_file", { name: "notes.md", content: "# Trip\n\n- Rome\n" });
    expect(wrote.status).toBe("done");
    expect(a.born).toHaveLength(1);
    const [born] = await workspaces.list();
    expect(born).toMatchObject({ id: a.bound(), name: "Trip plan" });
    expect(wrote.written).toEqual({ workspace: a.bound(), file: expect.any(String), name: "notes.md", version: 1 });
    expect(wrote.result).toBe('Created notes.md, version 1 (3 lines, 15 characters), in the workspace "Trip plan". The user sees it in your reply.');
  });

  it("is born once when one reply writes two files at once", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    await runLocalCalls(
      [call("write_file", { name: "a.txt", content: "a" }), call("write_file", { name: "b.txt", content: "b" })],
      { settings, signal: new AbortController().signal, jsSafetyCheck: false, workspace: a.access },
      () => {},
    );
    expect(a.born).toHaveLength(1);
    expect(await workspaces.list()).toHaveLength(1);
    expect((await workspaces.files(a.bound()!)).map((f) => f.name).sort()).toEqual(["a.txt", "b.txt"]);
  });

  it("keeps the binding it was born with: a later turn reads the file back and rewrites it as a new version", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    await run(a.access, "write_file", { name: "notes.md", content: "# Trip\n\n- Rome\n" });
    const id = a.bound();
    // A later turn: the chat comes back with the workspace it recorded.
    const later = chat(workspaces);
    later.bind(id!);
    const read = await run(later.access, "read_file", { name: "notes.md" });
    expect(read.result).toBe("1\t# Trip\n2\t\n3\t- Rome\n\n[notes.md, version 1: lines 1–3 of 3.]");
    expect(read.read).toEqual({ name: "notes.md", from: 1, to: 3, total: 3, lines: true });
    const rewrote = await run(later.access, "write_file", { name: "notes.md", content: "# Trip\n\n- Rome\n- Naples\n" });
    expect(rewrote.result).toBe('Wrote notes.md, version 2 (4 lines, 24 characters), in the workspace "Trip plan". The user sees it in your reply.');
    expect(later.born).toEqual([]);
    const [file] = await workspaces.files(id!);
    expect(file.versions.map((v) => [v.n, v.author, v.text])).toEqual([
      [1, { by: "chat", chat: "chat-a" }, "# Trip\n\n- Rome\n"],
      [2, { by: "chat", chat: "chat-a" }, "# Trip\n\n- Rome\n- Naples\n"],
    ]);
    expect(rewrote.written).toEqual({ workspace: id, file: file.id, name: "notes.md", version: 2 });
  });

  it("lists each file's name, kind, size, last change and author", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    await run(a.access, "write_file", { name: "site/index.html", content: "<p>Hi</p>" });
    const id = a.bound()!;
    await workspaces.add(id, [{ name: "brief.pdf", kind: "pdf", blob: new Blob([new Uint8Array(1500)], { type: "application/pdf" }), derived: "[Page 1]\nBrief" }], USER);
    const b = chat(workspaces, "chat-b", "Other");
    b.bind(id);
    await run(b.access, "write_file", { name: "todo.txt", content: "one\n" });
    const listed = await run(a.access, "list_files", {});
    expect(listed.listed).toBe(3);
    expect(listed.result).toBe(
      [
        'The workspace "Trip plan" holds 3 files:',
        "- brief.pdf: PDF, 1,500 bytes, its text 14 characters; version 1, changed 2026-10-28 14:02 by the user",
        "- site/index.html: text, 1 line, 9 characters; version 1, changed 2026-10-28 14:01 by you",
        "- todo.txt: text, 1 line, 4 characters; version 1, changed 2026-10-28 14:04 by another chat",
      ].join("\n"),
    );
  });

  it("keeps a name with folders, unique in the workspace", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    const wrote = await run(a.access, "write_file", { name: " ./site\\css//main.css ", content: "p{}" });
    expect(wrote.written?.name).toBe("site/css/main.css");
    expect((await run(a.access, "write_file", { name: "../up.txt", content: "x" })).error).toBe(
      'The call needs a "name": the file name with its extension, like notes.md or site/index.html.',
    );
    const id = a.bound()!;
    const added = await workspaces.add(id, [{ name: "site/css/main.css", kind: "text", text: "q{}" }, { name: "site/css/main.css", kind: "text", text: "r{}" }], USER);
    expect(added.map((w) => w.name)).toEqual(["site/css/main (2).css", "site/css/main (3).css"]);
    expect((await workspaces.files(id)).map((f) => f.name)).toEqual(["site/css/main (2).css", "site/css/main (3).css", "site/css/main.css"]);
  });

  it("does not exist while the chat has no file", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    expect((await run(a.access, "list_files", {})).result).toBe(
      "This chat has no workspace yet: it is created with the first file you write or the user attaches.",
    );
    expect((await run(a.access, "read_file", { name: "notes.md" })).error).toBe(
      "This chat has no workspace yet: it is created with the first file you write or the user attaches.",
    );
    expect(a.born).toEqual([]);
    expect(await workspaces.list()).toEqual([]);
  });
});

describe("read_file", () => {
  async function withFile(text: string, name = "long.txt") {
    const workspaces = module();
    const a = chat(workspaces);
    await run(a.access, "write_file", { name, content: text });
    return a;
  }

  it("reads the range asked for, and says where the rest starts", async () => {
    const a = await withFile(Array.from({ length: 12 }, (_, i) => `line ${i + 1}`).join("\n"));
    const read = await run(a.access, "read_file", { name: "long.txt", offset: 9, limit: 2 });
    expect(read.result).toBe(" 9\tline 9\n10\tline 10\n\n[long.txt, version 1: lines 9–10 of 12. Call read_file with offset 11 for the rest.]");
    expect((await run(a.access, "read_file", { name: "long.txt", offset: 11 })).result).toBe(
      "11\tline 11\n12\tline 12\n\n[long.txt, version 1: lines 11–12 of 12.]",
    );
    expect((await run(a.access, "read_file", { name: "long.txt", offset: 13 })).error).toBe("long.txt has 12 lines: line 13 is past its end.");
  });

  it("returns at most the read limit per call", async () => {
    // 1,000 lines of 30 characters, each numbered in 35 ("1000", a tab, the line) and a newline: 555 fit in 20,000.
    const a = await withFile(Array.from({ length: 1000 }, () => "x".repeat(30)).join("\n"));
    const read = await run(a.access, "read_file", { name: "long.txt" });
    const body = read.result!.slice(0, read.result!.lastIndexOf("\n\n["));
    expect(body.length).toBe(555 * 35 + 554);
    expect(body.length).toBeLessThanOrEqual(READ_LIMIT);
    expect(read.read).toEqual({ name: "long.txt", from: 1, to: 555, total: 1000, lines: true });
    expect(read.result!.endsWith("[long.txt, version 1: lines 1–555 of 1000. Call read_file with offset 556 for the rest.]")).toBe(true);
  });

  it("returns the start of a line too long for one read", async () => {
    const a = await withFile("y".repeat(30_000));
    const read = await run(a.access, "read_file", { name: "long.txt" });
    expect(read.result).toBe(`1\t${"y".repeat(READ_LIMIT - 2)}\n\n[long.txt, version 1: lines 1–1 of 1. Line 1 has 30,000 characters: only its start fits.]`);
  });

  it("reads a PDF as its extracted text, the same way", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    const id = await a.access.chat.workspace();
    await workspaces.add(id, [{ name: "brief.pdf", kind: "pdf", blob: new Blob([new Uint8Array([1, 2])], { type: "application/pdf" }), derived: "[Page 1]\nHello\n\n[Page 2]\nBye" }], USER);
    expect((await run(a.access, "read_file", { name: "brief.pdf" })).result).toBe(
      "1\t[Page 1]\n2\tHello\n3\t\n4\t[Page 2]\n5\tBye\n\n[brief.pdf, version 1: lines 1–5 of 5.]",
    );
  });

  it("names the files when the name is wrong, and says what it cannot read", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    // A workspace with no file yet, the chat bound to it with the picker.
    const id = a.bind((await workspaces.create("Trip plan")).id);
    expect((await run(a.access, "read_file", { name: "x.txt" })).error).toBe('No file is named "x.txt": the workspace holds no files.');
    await workspaces.add(id, [{ name: "a.txt", kind: "text", text: "" }, { name: "data.zip", kind: "other", blob: new Blob([new Uint8Array([80, 75])]) }], USER);
    expect((await run(a.access, "read_file", { name: "x.txt" })).error).toBe('No file is named "x.txt". The files are: "a.txt", "data.zip".');
    expect((await run(a.access, "read_file", { name: "a.txt" })).result).toBe("[a.txt, version 1: the file is empty.]");
    // Said as the call's result, with the version: what the model has seen of a file it may still delete.
    expect((await run(a.access, "read_file", { name: "data.zip" })).result).toBe("[data.zip, version 1: a file of a kind read_file cannot read; 2 bytes.]");
  });
});

describe("edit_file", () => {
  it("replaces the one occurrence of old, as a new version of the file", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    await run(a.access, "write_file", { name: "notes.md", content: "# Trip\n\n- Rome\n- Pisa\n" });
    const edited = await run(a.access, "edit_file", { name: "notes.md", old: "- Pisa", new: "- Florence\n- Siena" });
    expect(edited.status).toBe("done");
    expect(edited.result).toBe("Edited notes.md, version 2 (5 lines, 34 characters). The user sees it in your reply.");
    const [file] = await workspaces.files(a.bound()!);
    expect(file.versions.map((v) => [v.n, v.author, v.text])).toEqual([
      [1, { by: "chat", chat: "chat-a" }, "# Trip\n\n- Rome\n- Pisa\n"],
      [2, { by: "chat", chat: "chat-a" }, "# Trip\n\n- Rome\n- Florence\n- Siena\n"],
    ]);
    expect(edited.written).toEqual({ workspace: a.bound(), file: file.id, name: "notes.md", version: 2 });
  });

  it("refuses a passage that occurs nowhere or more than once, and says which", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    await run(a.access, "write_file", { name: "list.txt", content: "a = 1\nb = 1\nc = 2\nd = 1\n" });
    expect((await run(a.access, "edit_file", { name: "list.txt", old: "e = 1", new: "e = 2" })).error).toBe(
      'The passage to replace ("old") occurs nowhere in list.txt. Copy it exactly as read_file shows it, without the line numbers.',
    );
    expect((await run(a.access, "edit_file", { name: "list.txt", old: " = 1", new: " = 3" })).error).toBe(
      'The passage to replace ("old") occurs 3 times in list.txt, on lines 1, 2 and 4: give more of the text around the one to change, so that it occurs once.',
    );
    expect((await run(a.access, "edit_file", { name: "list.txt", old: "=", new: ":" })).error).toBe(
      'The passage to replace ("old") occurs 4 times in list.txt, on lines 1, 2, 3 and 4: give more of the text around the one to change, so that it occurs once.',
    );
    await run(a.access, "write_file", { name: "row.txt", content: "x = 1; y = 1\n" });
    expect((await run(a.access, "edit_file", { name: "row.txt", old: " = 1", new: " = 2" })).error).toBe(
      'The passage to replace ("old") occurs 2 times in row.txt, on line 1: give more of the text around the one to change, so that it occurs once.',
    );
    expect((await run(a.access, "edit_file", { name: "list.txt", old: "", new: "x" })).error).toBe('The passage to replace ("old") is empty.');
    expect((await run(a.access, "edit_file", { name: "list.txt", old: "c = 2" })).error).toBe(
      'The call needs "old", the passage to replace, and "new", the text that takes its place, both as text.',
    );
    expect((await workspaces.files(a.bound()!))[0].versions).toHaveLength(1);
  });

  it("matches a passage over several lines of a file with CRLF line endings, as read_file shows it", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    const id = await a.access.chat.workspace();
    await workspaces.add(id, [{ name: "todo.txt", kind: "text", text: "one\r\ntwo\r\nthree\r\n" }], USER);
    await run(a.access, "read_file", { name: "todo.txt" });
    expect((await run(a.access, "edit_file", { name: "todo.txt", old: "one\ntwo", new: "one\n1.5\ntwo" })).status).toBe("done");
    expect(latest((await workspaces.files(id))[0]).text).toBe("one\r\n1.5\r\ntwo\r\nthree\r\n");
  });

  it("edits text files only", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    const id = await a.access.chat.workspace();
    await workspaces.add(id, [{ name: "brief.pdf", kind: "pdf", blob: new Blob([new Uint8Array([1])]), derived: "[Page 1]\nHello" }], USER);
    await run(a.access, "read_file", { name: "brief.pdf" });
    expect((await run(a.access, "edit_file", { name: "brief.pdf", old: "Hello", new: "Bye" })).error).toBe(
      "brief.pdf is a PDF file, and only a text file can be edited.",
    );
  });
});

describe("delete_file", () => {
  it("marks the file deleted as a new version: it leaves the list, and its history keeps it", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    await run(a.access, "write_file", { name: "draft.md", content: "old idea" });
    await run(a.access, "write_file", { name: "plan.md", content: "the plan" });
    const deleted = await run(a.access, "delete_file", { name: "draft.md" });
    expect(deleted.status).toBe("done");
    expect(deleted.result).toBe("Deleted draft.md: version 2 marks it deleted, and its history keeps the versions before it.");
    expect((await run(a.access, "list_files", {})).result).toBe(
      'The workspace "Trip plan" holds 1 file:\n- plan.md: text, 1 line, 8 characters; version 1, changed 2026-10-28 14:02 by you',
    );
    const draft = (await workspaces.files(a.bound()!)).find((f) => f.name === "draft.md")!;
    expect(draft.versions.map((v) => [v.n, v.author, v.text, v.deleted])).toEqual([
      [1, { by: "chat", chat: "chat-a" }, "old idea", undefined],
      [2, { by: "chat", chat: "chat-a" }, undefined, true],
    ]);
    expect(deleted.written).toEqual({ workspace: a.bound(), file: draft.id, name: "draft.md", version: 2 });
    expect((await run(a.access, "read_file", { name: "draft.md" })).result).toBe(
      "[draft.md, version 2: deleted by you, 2026-10-28 14:03. Its history keeps the versions before it; write_file creates the file again.]",
    );
    expect((await run(a.access, "delete_file", { name: "draft.md" })).error).toBe("draft.md was deleted by you (version 2).");
    expect((await run(a.access, "edit_file", { name: "draft.md", old: "old", new: "new" })).error).toBe("draft.md was deleted by you (version 2).");
  });

  it("writes a deleted file again as its next version", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    await run(a.access, "write_file", { name: "draft.md", content: "one" });
    await run(a.access, "delete_file", { name: "draft.md" });
    const again = await run(a.access, "write_file", { name: "draft.md", content: "two" });
    expect(again.result).toBe('Created draft.md, version 3 (1 line, 3 characters), in the workspace "Trip plan". The user sees it in your reply.');
    const [draft] = await workspaces.files(a.bound()!);
    expect(draft.versions.map((v) => v.text ?? "deleted")).toEqual(["one", "deleted", "two"]);
  });

  it("is the user's deletion too, which the guard never refuses", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    await run(a.access, "write_file", { name: "draft.md", content: "one" });
    const [draft] = await workspaces.files(a.bound()!);
    const outcome = await workspaces.deleteFile(a.bound()!, draft.id, USER);
    expect(outcome).toEqual({ ok: true, written: { workspace: a.bound(), file: draft.id, name: "draft.md", version: 2 } });
    expect(latest((await workspaces.files(a.bound()!))[0])).toMatchObject({ n: 2, author: USER, deleted: true });
    expect(await workspaces.deleteFile(a.bound()!, draft.id, USER)).toEqual({ ok: false, error: "draft.md is already deleted." });
  });

  it("deletes a file of a kind the model cannot read, once it has read what it is", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    const id = await a.access.chat.workspace();
    await workspaces.add(id, [{ name: "data.zip", kind: "other", blob: new Blob([new Uint8Array([80, 75])]) }], USER);
    expect((await run(a.access, "delete_file", { name: "data.zip" })).error).toBe(
      "data.zip is a file you have not read: it is at version 1, by the user. Read it with read_file before you change it.",
    );
    await run(a.access, "read_file", { name: "data.zip" });
    expect((await run(a.access, "delete_file", { name: "data.zip" })).status).toBe("done");
  });
});

describe("the guard", () => {
  /** A workspace with notes.md written by chat a, and chat a's access. */
  async function written() {
    const workspaces = module();
    const a = chat(workspaces);
    await run(a.access, "write_file", { name: "notes.md", content: "- Rome\n" });
    const id = a.bound()!;
    const [file] = await workspaces.files(id);
    return { workspaces, a, id, file };
  }

  it("counts a reader's own writes as its reads: it edits a file it just wrote without reading it back", async () => {
    const { workspaces, a, id } = await written();
    expect((await run(a.access, "edit_file", { name: "notes.md", old: "- Rome", new: "- Rome\n- Naples" })).status).toBe("done");
    expect((await run(a.access, "write_file", { name: "notes.md", content: "- Turin\n" })).status).toBe("done");
    expect((await run(a.access, "edit_file", { name: "notes.md", old: "Turin", new: "Milan" })).status).toBe("done");
    expect((await run(a.access, "delete_file", { name: "notes.md" })).status).toBe("done");
    expect((await workspaces.files(id))[0].versions.map((v) => v.n)).toEqual([1, 2, 3, 4, 5]);
  });

  it("refuses a change over the user's, naming the file and asking for a new read, then takes it once the file is read", async () => {
    const { workspaces, a, id } = await written();
    expect(await workspaces.writeText(id, "notes.md", "- Rome\n- Venice\n", USER)).toMatchObject({ ok: true, written: { version: 2 } });
    for (const [tool, args] of [
      ["write_file", { name: "notes.md", content: "- Bari\n" }],
      ["edit_file", { name: "notes.md", old: "- Rome", new: "- Bari" }],
      ["delete_file", { name: "notes.md" }],
    ] as const) {
      const refused = await run(a.access, tool, args);
      expect(refused.status).toBe("failed");
      expect(refused.error).toBe(
        "notes.md changed since you last read it: it is at version 2, by the user, and you last read version 1. Read it again with read_file before you change it.",
      );
    }
    expect((await workspaces.files(id))[0].versions).toHaveLength(2);
    const read = await run(a.access, "read_file", { name: "notes.md" });
    expect(read.seen).toMatchObject({ name: "notes.md", version: 2 });
    expect((await run(a.access, "edit_file", { name: "notes.md", old: "- Venice", new: "- Venice\n- Bari" })).status).toBe("done");
    expect(latest((await workspaces.files(id))[0]).text).toBe("- Rome\n- Venice\n- Bari\n");
  });

  it("refuses a change to a file the reader never read, and guards no new file", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    const id = await a.access.chat.workspace();
    await workspaces.add(id, [{ name: "brief.md", kind: "text", text: "Keep it short." }], USER);
    expect((await run(a.access, "write_file", { name: "brief.md", content: "Make it long." })).error).toBe(
      "brief.md is a file you have not read: it is at version 1, by the user. Read it with read_file before you change it.",
    );
    expect((await run(a.access, "write_file", { name: "summary.md", content: "New." })).status).toBe("done");
    await run(a.access, "read_file", { name: "brief.md", offset: 1, limit: 1 });
    expect((await run(a.access, "write_file", { name: "brief.md", content: "Make it long." })).status).toBe("done");
  });

  it("refuses to write again a file deleted since the reader read it, until it reads the deletion; a deleted file it never knew is new to it", async () => {
    const { workspaces, a, id, file } = await written();
    await workspaces.deleteFile(id, file.id, USER);
    expect((await run(a.access, "write_file", { name: "notes.md", content: "- Bari\n" })).error).toBe(
      "notes.md was deleted by the user since you last read it (version 2). Read it again with read_file before you write it.",
    );
    expect((await run(a.access, "read_file", { name: "notes.md" })).result).toContain("deleted by the user");
    expect((await run(a.access, "write_file", { name: "notes.md", content: "- Bari\n" })).written).toMatchObject({ version: 3 });
    await workspaces.deleteFile(id, file.id, USER);
    const b = chat(workspaces, "chat-b", "Other");
    b.bind(id);
    expect((await run(b.access, "write_file", { name: "notes.md", content: "- Lecce\n" })).written).toMatchObject({ version: 5 });
  });

  it("keeps two sub-agents on one file apart: neither overwrites the other's change without reading it", async () => {
    const { workspaces, a, id } = await written();
    const one: WorkspaceAccess = { ...a.access, reader: a.access.chat.agent({ call: "call_1", name: "one" }) };
    const two: WorkspaceAccess = { ...a.access, reader: a.access.chat.agent({ call: "call_2", name: "two" }) };
    // Neither has read what the conversation wrote.
    expect((await run(one, "edit_file", { name: "notes.md", old: "Rome", new: "Roma" })).error).toBe(
      "notes.md is a file you have not read: it is at version 1, by the conversation that started you. Read it with read_file before you change it.",
    );
    await run(one, "read_file", { name: "notes.md" });
    await run(two, "read_file", { name: "notes.md" });
    expect((await run(one, "edit_file", { name: "notes.md", old: "- Rome", new: "- Rome\n- Pisa" })).status).toBe("done");
    expect((await run(two, "edit_file", { name: "notes.md", old: "- Rome", new: "- Rome\n- Lucca" })).error).toBe(
      "notes.md changed since you last read it: it is at version 2, by the sub-agent one, and you last read version 1. Read it again with read_file before you change it.",
    );
    // Nor does the conversation, which read version 1 too.
    expect((await run(a.access, "write_file", { name: "notes.md", content: "- Bari\n" })).error).toContain("by the sub-agent one, and you last read version 1");
    await run(two, "read_file", { name: "notes.md" });
    expect((await run(two, "edit_file", { name: "notes.md", old: "- Pisa", new: "- Pisa\n- Lucca" })).status).toBe("done");
    expect(latest((await workspaces.files(id))[0]).text).toBe("- Rome\n- Pisa\n- Lucca\n");
  });

  it("lets one of two sub-agents' edits at once win, and refuses the other", async () => {
    const { workspaces, a, id } = await written();
    const agents = ["call_1", "call_2"].map((call): WorkspaceAccess => ({ ...a.access, reader: a.access.chat.agent({ call, name: call }) }));
    for (const agent of agents) await run(agent, "read_file", { name: "notes.md" });
    const runs = await Promise.all(agents.map((agent, i) => run(agent, "write_file", { name: "notes.md", content: `- take ${i + 1}\n` })));
    expect(runs.map((r) => r.status).sort()).toEqual(["done", "failed"]);
    expect((await workspaces.files(id))[0].versions).toHaveLength(2);
  });

  it("makes each agent of a reply a reader of its own, and an agent of a later reply with the same call id one that has read nothing", async () => {
    const { workspaces, a, id } = await written();
    // Each agent reads notes.md, then writes it: the request after a read is the write, and the agents that
    // read write only once both have read.
    let read = 0;
    let bothRead = () => {};
    const reads = new Promise<void>((resolve) => (bothRead = resolve));
    const stream = async (o: StreamOptions): Promise<StreamResult> => {
      const asked = o.request.turns.length;
      const task = o.request.turns[0].content;
      if (asked === 1 && task.startsWith("read")) o.onEvent({ kind: "tool_call", call: { id: "r", name: "read_file", arguments: '{"name":"notes.md"}' } });
      else if (asked <= 3 && !o.request.turns.some((t) => t.toolCallId === "w")) {
        if (task.startsWith("read")) {
          if (++read === 2) bothRead();
          await reads;
        }
        o.onEvent({ kind: "tool_call", call: { id: "w", name: "write_file", arguments: JSON.stringify({ name: "notes.md", content: `${task}\n` }) } });
      } else o.onEvent({ kind: "content", text: "Done." });
      return { ok: true, timeline: { sentAt: 0, firstTokenAt: 1, lastTokenAt: 2, endedAt: 3, finishReason: "stop", stopped: false } };
    };
    const options = {
      settings,
      tools: agentExtras({ ...NO_TOOLS, workspace: true }),
      local: { jsSafetyCheck: false, workspace: a.access },
      signal: new AbortController().signal,
      onUpdate: () => {},
      stream,
    };
    const writes = (ran: Awaited<ReturnType<typeof runAgents>>) => ran.map((r) => r.local?.find((l) => l.tool === "write_file"));
    // One reply: two agents read version 1; the first to write wins, the other is refused.
    const first = await runAgents(
      [
        { callId: "call_0", name: "one", prompt: "read, then write one" },
        { callId: "call_1", name: "two", prompt: "read, then write two" },
      ],
      options,
    );
    expect(writes(first).map((w) => w?.status).sort()).toEqual(["done", "failed"]);
    expect(writes(first).find((w) => w?.status === "failed")?.error).toMatch(/^notes\.md changed since you last read it: it is at version 2, by the sub-agent (one|two)/);
    // A later reply's agent takes call_0 again, and writes without a read: it has read nothing.
    const [later] = await runAgents([{ callId: "call_0", name: "three", prompt: "write three" }], options);
    expect(later.local?.[0]).toMatchObject({ tool: "write_file", status: "failed" });
    expect(later.local?.[0].error).toMatch(/^notes\.md is a file you have not read: it is at version 2/);
    expect((await workspaces.files(id))[0].versions).toHaveLength(2);
  });

  it("refuses two chats on one workspace each other's unseen changes, the same way", async () => {
    const { workspaces, a, id } = await written();
    const b = chat(workspaces, "chat-b", "Other");
    b.bind(id);
    await run(b.access, "read_file", { name: "notes.md" });
    expect((await run(b.access, "edit_file", { name: "notes.md", old: "Rome", new: "Roma" })).status).toBe("done");
    expect((await run(a.access, "edit_file", { name: "notes.md", old: "Roma", new: "Rome" })).error).toBe(
      "notes.md changed since you last read it: it is at version 2, by another chat, and you last read version 1. Read it again with read_file before you change it.",
    );
    await run(a.access, "read_file", { name: "notes.md" });
    expect((await run(a.access, "edit_file", { name: "notes.md", old: "Roma", new: "Rome" })).status).toBe("done");
    expect((await run(b.access, "delete_file", { name: "notes.md" })).error).toContain("it is at version 3, by another chat, and you last read version 2");
  });
});

describe("the user's hand edits", () => {
  it("are a version of the user's per save, and the model's write, edit or deletion is refused until it reads the file again", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    await run(a.access, "write_file", { name: "notes.md", content: "- Rome\n" });
    const id = a.bound()!;
    const before = (await workspaces.files(id))[0];
    // The editor's Save, as the panel calls it.
    expect(await saveText(workspaces, before, "- Rome\n- Venice\n")).toMatchObject({ ok: true, written: { name: "notes.md", version: 2 } });
    const saved = (await workspaces.files(id))[0];
    expect(saved.versions.map((v) => [v.n, v.author.by, v.text])).toEqual([
      [1, "chat", "- Rome\n"],
      [2, "user", "- Rome\n- Venice\n"],
    ]);
    for (const [tool, args] of [
      ["write_file", { name: "notes.md", content: "- Bari\n" }],
      ["edit_file", { name: "notes.md", old: "- Rome", new: "- Bari" }],
      ["delete_file", { name: "notes.md" }],
    ] as const) {
      expect((await run(a.access, tool, args)).error).toBe(
        "notes.md changed since you last read it: it is at version 2, by the user, and you last read version 1. Read it again with read_file before you change it.",
      );
    }
    expect((await workspaces.files(id))[0].versions).toHaveLength(2);
    expect((await run(a.access, "read_file", { name: "notes.md" })).result).toContain("- Venice");
    expect((await run(a.access, "edit_file", { name: "notes.md", old: "- Venice", new: "- Venice\n- Bari" })).status).toBe("done");
    expect(latest((await workspaces.files(id))[0])).toMatchObject({ n: 3, text: "- Rome\n- Venice\n- Bari\n" });
  });

  it("include a restore, which the model reads before it changes the file", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    await run(a.access, "write_file", { name: "notes.md", content: "- Rome\n" });
    await run(a.access, "write_file", { name: "notes.md", content: "- Turin\n" });
    const id = a.bound()!;
    const [file] = await workspaces.files(id);
    await workspaces.restore(id, file.id, 1);
    expect((await run(a.access, "edit_file", { name: "notes.md", old: "Rome", new: "Roma" })).error).toBe(
      "notes.md changed since you last read it: it is at version 3, by the user, and you last read version 2. Read it again with read_file before you change it.",
    );
    await run(a.access, "read_file", { name: "notes.md" });
    expect((await run(a.access, "edit_file", { name: "notes.md", old: "Rome", new: "Roma" })).status).toBe("done");
  });
});

describe("a chat's reads across turns", () => {
  let messageN = 0;
  /** A reply as the chat keeps it: the calls it made, and their local runs. */
  const reply = (local: LocalRun[], more: Partial<Message> = {}): Message => ({
    id: ++messageN,
    role: "assistant",
    content: "",
    reasoning: "",
    streaming: false,
    toolCalls: local.map((r) => ({ id: r.callId, name: r.tool, arguments: "{}" })),
    local,
    ...more,
  });
  const prompt = (content: string): Message => ({ id: ++messageN, role: "user", content, reasoning: "", streaming: false });

  it("hold after a reload: a resumed chat changes a file its history read, and not one the user changed since", async () => {
    const store = createMemoryStore();
    let n = 0;
    const before = createWorkspaces(store, { now: clock(), newId: () => `id${++n}` });
    const a = chat(before);
    const id = await a.access.chat.workspace();
    await before.add(id, [{ name: "plan.md", kind: "text", text: "- one\n" }, { name: "notes.md", kind: "text", text: "- a\n" }], USER);
    const history = [reply([await run(a.access, "read_file", { name: "plan.md" }), await run(a.access, "read_file", { name: "notes.md" })])];
    await before.writeText(id, "notes.md", "- a\n- b\n", USER);
    // The page reloads: a new module on the same store, and the chat's reader starts from its history.
    const after = createWorkspaces(store, { now: clock(), newId: () => `id${++n}` });
    const binding = after.chat("chat-a", { bound: () => id, name: () => "Trip plan", onBorn: () => {}, seen: seenIn(history) });
    const resumed: WorkspaceAccess = { workspaces: after, chat: binding };
    expect((await run(resumed, "edit_file", { name: "plan.md", old: "one", new: "two" })).status).toBe("done");
    expect((await run(resumed, "edit_file", { name: "notes.md", old: "- b", new: "- c" })).error).toBe(
      "notes.md changed since you last read it: it is at version 2, by the user, and you last read version 1. Read it again with read_file before you change it.",
    );
  });

  it("go with the reply that made them: a turn run again without that reply has not read the file", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    const id = await a.access.chat.workspace();
    await workspaces.add(id, [{ name: "brief.md", kind: "text", text: "Keep it short." }], USER);
    const dropped = reply([await run(a.access, "read_file", { name: "brief.md" })]);
    const asked = prompt("Read the brief.");
    expect(seenIn([asked, dropped])).toEqual([{ file: expect.any(String), version: 1 }]);
    // The reply is regenerated: the turn starts again from the prompt alone.
    const again = workspaces.chat("chat-a", { bound: () => id, name: () => "Trip plan", onBorn: () => {}, seen: seenIn([asked]) });
    expect((await run({ workspaces, chat: again }, "write_file", { name: "brief.md", content: "Long." })).error).toMatch(/^brief\.md is a file you have not read/);
  });

  it("are the chat's own: what its sub-agents read or wrote does not count", () => {
    const agentWrite: LocalRun = { callId: "w", tool: "write_file", status: "done", written: { workspace: "w1", file: "f2", name: "x.md", version: 4 } };
    const deleted: LocalRun = { callId: "d", tool: "delete_file", status: "done", written: { workspace: "w1", file: "f1", name: "a.md", version: 3 } };
    const messages = [
      reply([
        { callId: "r", tool: "read_file", status: "done", seen: { workspace: "w1", file: "f1", name: "a.md", version: 2 } },
        { callId: "e", tool: "edit_file", status: "failed", error: "refused" },
      ]),
      reply([deleted], { agents: [{ callId: "a1", name: "helper", prompt: "", status: "done", reasoning: "", content: "", local: [agentWrite] }] }),
    ];
    expect(seenIn(messages)).toEqual([
      { file: "f1", version: 2 },
      { file: "f1", version: 3 },
    ]);
  });

  it("are those of the history the model reads: a failed reply, which a request leaves out, has read nothing", () => {
    const read = (file: string): LocalRun => ({ callId: `r-${file}`, tool: "read_file", status: "done", seen: { workspace: "w1", file, name: `${file}.md`, version: 1 } });
    const messages = [prompt("Read both."), reply([read("f1")]), reply([read("f2")], { error: "503: engine full" })];
    expect(seenIn(messages)).toEqual([{ file: "f1", version: 1 }]);
  });
});

describe("a chat moved to another workspace (#404)", () => {
  let messageN = 100;
  const prompt = (content: string): Message => ({ id: ++messageN, role: "user", content, reasoning: "", streaming: false });
  const reply = (local: LocalRun[], more: Partial<Message> = {}): Message => ({
    id: ++messageN,
    role: "assistant",
    content: "",
    reasoning: "",
    streaming: false,
    toolCalls: local.map((r) => ({ id: r.callId, name: r.tool, arguments: "{}" })),
    local,
    ...more,
  });

  /**
   * Chat 1 in a list of sessions, as the conversation loop holds it: each turn binds it afresh, its reader starting
   * from what its history read, and a workspace born for it binds it in place of a deleted one.
   */
  function chats(workspaces: Workspaces, sessions: Session[]) {
    let list = sessions;
    const turn = (history: Message[]): WorkspaceAccess => ({
      workspaces,
      chat: workspaces.chat("chat-a", {
        bound: () => list.find((s) => s.id === 1)?.workspaceId,
        name: () => "Trip plan",
        onBorn: (id, replaced) => (list = bornInto(list, 1, id, replaced)),
        seen: seenIn(history),
      }),
    });
    return { turn, list: () => list, move: (to: string | undefined) => (list = moveWorkspace(list, 1, to, { at: 0, name: "Trip" })) };
  }

  /** The workspaces Trip and Site, each holding a notes.md, a plan.md and a todo.md of the user's. */
  async function twoWorkspaces(workspaces: Workspaces) {
    const files = (place: string) => ["notes.md", "plan.md", "todo.md"].map((name) => ({ name, kind: "text" as const, text: `${place} ${name}` }));
    const trip = await workspaces.create("Trip");
    await workspaces.add(trip.id, files("Trip"), USER);
    const site = await workspaces.create("Site");
    await workspaces.add(site.id, files("Site"), USER);
    return { trip: trip.id, site: site.id };
  }

  it("has read nothing in the workspace it moves to: a write, edit or delete there is refused until the model reads the file", async () => {
    const workspaces = module();
    const { trip, site } = await twoWorkspaces(workspaces);
    const c = chats(workspaces, [{ ...createSession(1), workspaceId: trip }]);
    // In Trip, the chat reads two files and rewrites one; a sub-agent of it reads and edits the third.
    const first = c.turn([]);
    const helper: WorkspaceAccess = { ...first, reader: first.chat.agent({ call: "a1", name: "helper" }) };
    const own = [
      await run(first, "read_file", { name: "notes.md" }),
      await run(first, "read_file", { name: "plan.md" }),
      await run(first, "write_file", { name: "plan.md", content: "Trip plan" }),
    ];
    const agents = [await run(helper, "read_file", { name: "todo.md" }), await run(helper, "edit_file", { name: "todo.md", old: "Trip", new: "Rome" })];
    expect([...own, ...agents].map((r) => [r.status, r.written?.workspace ?? r.seen?.workspace])).toEqual(Array(5).fill(["done", trip]));
    const history = [prompt("Plan the trip."), reply(own), reply([], { agents: [{ callId: "a1", name: "helper", prompt: "", status: "done", reasoning: "", content: "", local: agents }] })];
    c.move(site);
    // The next turn, in Site, where the same names are other files.
    const next = c.turn(history);
    const notRead = (name: string) => `${name} is a file you have not read: it is at version 1, by the user. Read it with read_file before you change it.`;
    expect((await run(next, "write_file", { name: "plan.md", content: "Site plan" })).error).toBe(notRead("plan.md"));
    expect((await run(next, "edit_file", { name: "notes.md", old: "Site", new: "Small site" })).error).toBe(notRead("notes.md"));
    expect((await run(next, "delete_file", { name: "todo.md" })).error).toBe(notRead("todo.md"));
    // Its sub-agents of this turn work in Site, and have read nothing there either.
    const agent: WorkspaceAccess = { ...next, reader: next.chat.agent({ call: "a2", name: "helper" }) };
    expect((await run(agent, "edit_file", { name: "todo.md", old: "Site", new: "Rome" })).error).toBe(notRead("todo.md"));
    expect((await run(agent, "write_file", { name: "found.md", content: "facts" })).written?.workspace).toBe(site);
    // Once read, each change is taken, in Site.
    await run(next, "read_file", { name: "notes.md" });
    const edited = await run(next, "edit_file", { name: "notes.md", old: "Site", new: "Small site" });
    expect(edited.written).toMatchObject({ workspace: site, name: "notes.md", version: 2 });
    // Trip keeps its files as the chat left them.
    expect((await workspaces.files(trip)).map((f) => [f.name, latest(f).text])).toEqual([
      ["notes.md", "Trip notes.md"],
      ["plan.md", "Trip plan"],
      ["todo.md", "Rome todo.md"],
    ]);
  });

  it("out of its own, starts a new workspace named after the chat with its next file; the one it left keeps its files and its other chats", async () => {
    const workspaces = module();
    const c = chats(workspaces, [createSession(1)]);
    // The chat's first file gives it a workspace.
    const first = c.turn([]);
    const wrote = await run(first, "write_file", { name: "notes.md", content: "# Trip" });
    const born = c.list()[0].workspaceId!;
    expect(wrote.written?.workspace).toBe(born);
    // Another chat works in it.
    const other = workspaces.chat("chat-b", { bound: () => born, name: () => "", onBorn: () => {} });
    c.move(undefined);
    expect(c.list()[0].workspaceId).toBeUndefined();
    const next = c.turn([prompt("Notes."), reply([wrote])]);
    expect((await run(next, "list_files", {})).result).toBe("This chat has no workspace yet: it is created with the first file you write or the user attaches.");
    const again = await run(next, "write_file", { name: "plan.md", content: "- Rome" });
    const [now] = (await workspaces.list()).filter((w) => w.id !== born);
    expect(again.written?.workspace).toBe(now.id);
    expect(now.name).toBe("Trip plan");
    expect(c.list()[0].workspaceId).toBe(now.id);
    expect((await workspaces.files(born)).map((f) => f.name)).toEqual(["notes.md"]);
    expect((await run({ workspaces, chat: other }, "list_files", {})).result).toMatch(/^The workspace "Trip plan" holds 1 file:\n- notes\.md: text/);
  });
});

describe("write_file", () => {
  it("writes text only", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    expect((await run(a.access, "write_file", { name: "logo.png", content: "x" })).error).toBe("write_file writes text files, and logo.png names a picture file.");
    expect((await run(a.access, "write_file", { name: "notes.md" })).error).toBe('The call needs "content" as text.');
    expect(a.born).toEqual([]);
    const id = await a.access.chat.workspace();
    await workspaces.add(id, [{ name: "brief", kind: "pdf", blob: new Blob([new Uint8Array([1])]), derived: "" }], USER);
    expect((await run(a.access, "write_file", { name: "brief", content: "x" })).error).toBe("brief is a PDF file, and only a text file can be written whole.");
  });

  it("authors a sub-agent's write as the sub-agent of its chat", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    const agent: WorkspaceAccess = { ...a.access, reader: a.access.chat.agent({ call: "call_7", name: "researcher" }) };
    await run(agent, "write_file", { name: "found.md", content: "facts" });
    const [file] = await workspaces.files(a.bound()!);
    expect(file.versions[0].author).toEqual({ by: "chat", chat: "chat-a", agent: { call: "call_7", name: "researcher" } });
    expect((await run(a.access, "list_files", {})).result).toContain("by the sub-agent researcher");
    expect((await run(agent, "list_files", {})).result).toContain("by you");
  });

  it("is run by a sub-agent through its own tool loop, as a reader of its own", async () => {
    const workspaces = module();
    const a = chat(workspaces);
    const stream = async (o: StreamOptions): Promise<StreamResult> => {
      if (o.request.turns.length === 1) o.onEvent({ kind: "tool_call", call: { id: "w1", name: "write_file", arguments: '{"name":"report.md","content":"done"}' } });
      else o.onEvent({ kind: "content", text: "Wrote report.md." });
      return { ok: true, timeline: { sentAt: 0, firstTokenAt: 1, lastTokenAt: 2, endedAt: 3, finishReason: "stop", stopped: false } };
    };
    const [ran] = await runAgents([{ callId: "call_9", name: "writer", prompt: "write the report" }], {
      settings,
      tools: agentExtras({ ...NO_TOOLS, workspace: true }),
      local: { jsSafetyCheck: false, workspace: a.access },
      signal: new AbortController().signal,
      onUpdate: () => {},
      stream,
    });
    expect(ran.local?.[0]).toMatchObject({ tool: "write_file", status: "done", written: { name: "report.md", version: 1 } });
    const [file] = await workspaces.files(a.bound()!);
    expect(file.versions[0].author).toEqual({ by: "chat", chat: "chat-a", agent: { call: "call_9", name: "writer" } });
  });
});
