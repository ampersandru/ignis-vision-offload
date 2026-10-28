import { describe, expect, it } from "vitest";
import type { Settings } from "../api/request.ts";
import type { StreamOptions, StreamResult } from "../api/stream.ts";
import { createMemoryStore } from "../store/memory.ts";
import { runAgents } from "../tools/agents/agents.ts";
import { agentExtras, NO_TOOLS } from "../tools/index.ts";
import { type LocalRun, runLocalCalls } from "../tools/local/local.ts";
import { USER } from "./files.ts";
import { READ_LIMIT, type WorkspaceAccess } from "./tools.ts";
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
    const id = await a.access.chat.workspace();
    expect((await run(a.access, "read_file", { name: "x.txt" })).error).toBe('No file is named "x.txt": the workspace holds no files.');
    await workspaces.add(id, [{ name: "a.txt", kind: "text", text: "" }, { name: "data.zip", kind: "other", blob: new Blob([new Uint8Array([80, 75])]) }], USER);
    expect((await run(a.access, "read_file", { name: "x.txt" })).error).toBe('No file is named "x.txt". The files are: "a.txt", "data.zip".');
    expect((await run(a.access, "read_file", { name: "a.txt" })).result).toBe("[a.txt, version 1: the file is empty.]");
    expect((await run(a.access, "read_file", { name: "data.zip" })).error).toBe("data.zip is a file of a kind read_file cannot read.");
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
    const agent: WorkspaceAccess = { ...a.access, agent: { call: "call_7", name: "researcher" } };
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
