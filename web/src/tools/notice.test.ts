import { describe, expect, it } from "vitest";
import { buildChatRequest, conversationTurns, type Settings, type Turn } from "../api/request.ts";
import { buildResponseCreate, responseItems } from "../api/responses.ts";
import { exchangeOf } from "../app/useConversation.ts";
import { createSession, type Message, moveWorkspace, type Session } from "../sessions/sessions.ts";
import { CHAT_VERSION, restoreChats, toStored } from "../sessions/stored.ts";
import { createMemoryStore } from "../store/memory.ts";
import { USER, WORKSPACE_FORMAT, type WorkspaceFile } from "../workspace/files.ts";
import { noticeFiles } from "../workspace/notice.ts";
import { saveText } from "../workspace/panel.ts";
import { madeIn } from "../workspace/pictures.ts";
import type { WorkspaceAccess } from "../workspace/tools.ts";
import { createWorkspaces, type Workspaces } from "../workspace/workspaces.ts";
import type { AgentRun } from "./agents/agents.ts";
import { dateTimePrompt } from "./datetime/datetime.ts";
import type { ImageRun } from "./image/image.ts";
import { ALL_TOOLS, toolExtras } from "./index.ts";
import { type LocalRun, runLocalCalls } from "./local/local.ts";
import { browserMemory, type MemoryNote } from "./local/memory.ts";
import { agentNotice, turnNotice } from "./notice.ts";

// The turn notice (ADR 0072): what the page tells the model about a turn rides
// in a developer message ahead of the turn's prompt, kept with the turn, so the
// system prompt and the tools block never change with the memory notes. These
// tests drive the request seam: the notice a new prompt is given, and the
// request bodies both transports send for the conversation that carries it.

const now = new Date("2026-10-28T09:30:00+01:00");

const settings: Settings = {
  systemPrompt: "",
  temperature: 0.7,
  topP: 0.9,
  maxTokens: null,
  reasoningEffort: "none",
  thinkingBudget: null,
  laneTag: "interactive",
};

const note = (id: string, title: string, savedAt = "2026-10-27T10:00:00.000Z"): MemoryNote => ({ id, title, body: `${title}, in full.`, savedAt });

/** A prompt as the page sends it: the notice the page built for it spread on the message. */
const prompt = (id: number, content: string, notice: Partial<Message> = {}): Message => ({ id, role: "user", content, reasoning: "", streaming: false, ...notice });

/** The turns a request carries for these messages, as the conversation loop maps them. */
const wireTurns = (messages: Message[]) => conversationTurns(messages.map(exchangeOf));

const developer = (text: string) => ({ type: "message", role: "developer", content: [{ type: "input_text", text }] });
const userItem = (text: string) => ({ type: "message", role: "user", content: [{ type: "input_text", text }] });

describe("the system prompt", () => {
  it("lists no memory title, and is the same whatever notes are saved", () => {
    const before = toolExtras(ALL_TOOLS, { now });
    browserMemory.save("Language", "Prefers answers in Italian.");
    browserMemory.save("Team", "Supports AS Roma.");
    const after = toolExtras(ALL_TOOLS, { now });
    browserMemory.clear();
    expect(after).toEqual(before);
    expect(after.ignisPrompt).toContain("memory_read");
    expect(after.ignisPrompt).not.toContain("Language");
  });
});

describe("a chat's first turn", () => {
  it("carries the titles of the memory notes in a developer message ahead of its prompt, on both transports", () => {
    const notes = [note("m1", "Language"), note("m2", "Team")];
    const turns = wireTurns([prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes }))]);
    const notice = "# Memory notes\nSaved notes, by title:\n- [m1] Language\n- [m2] Team";
    expect(buildChatRequest(settings, turns).messages).toEqual([
      { role: "developer", content: notice },
      { role: "user", content: "Hi" },
    ]);
    expect(responseItems(turns)).toEqual([developer(notice), userItem("Hi")]);
  });

  it("keeps today's cap: the newest titles that fit in 4,000 characters, in their order, and a line counting the older ones", () => {
    // 50 notes, m10 to m59, whose lines are 99 characters each: 40 of them fit with their line breaks.
    const notes = Array.from({ length: 50 }, (_, i) => note(`m${i + 10}`, `${"t".repeat(90)}${i % 10}`));
    const text = turnNotice(ALL_TOOLS, [], { now, notes }).notice ?? "";
    const lines = text.split("\n");
    expect(lines.slice(0, 2)).toEqual(["# Memory notes", "Saved notes, by title:"]);
    expect(lines.slice(2, -1).map((line) => line.slice(0, 7))).toEqual(Array.from({ length: 40 }, (_, i) => `- [m${i + 20}]`));
    expect(lines.at(-1)).toBe("(10 older notes not listed.)");
  });
});

/** A reply that called local tools, memory or workspace, as the chat ran them: the runs it shows, and the results after it. */
function calledLocal(id: number, runs: LocalRun[]): Message[] {
  const calls = runs.map((run) => ({ id: run.callId, name: run.tool, arguments: "{}" }));
  return [
    { id, role: "assistant", content: "", reasoning: "", streaming: false, toolCalls: calls, local: runs },
    ...runs.map((run, i): Message => ({ id: id + 1 + i, role: "tool", content: run.result ?? "", toolCallId: run.callId, reasoning: "", streaming: false })),
  ];
}

const saveRun = (callId: string, saved: MemoryNote): LocalRun => ({ callId, tool: "memory_save", status: "done", note: saved, result: `Saved as ${saved.id}.` });
const deleteRun = (callId: string, id: string): LocalRun => ({ callId, tool: "memory_delete", status: "done", deletedId: id, result: `Deleted ${id}.` });
const answer = (id: number, content: string): Message => ({ id, role: "assistant", content, reasoning: "", streaming: false });

describe("a later turn", () => {
  const first = [note("m1", "Language"), note("m2", "Team"), note("m3", "Editor")];
  const trip = note("m4", "Trip", "2026-10-28T08:31:00.000Z");
  /** A first turn told of `first`, whose reply saved m4 and deleted m3 itself. */
  const history = [
    prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes: first })),
    ...calledLocal(2, [saveRun("c1", trip), deleteRun("c2", "m3")]),
    answer(5, "Noted."),
  ];

  it("tells only what someone other than this chat saved, changed or deleted since the last notice", () => {
    // The user edits m1 and deletes m2 in the memory manager; another chat saves m5.
    const notes = [note("m1", "Language and tone", "2026-10-28T08:40:00.000Z"), trip, note("m5", "Holidays", "2026-10-28T08:45:00.000Z")];
    const next = prompt(6, "Next", turnNotice(ALL_TOOLS, history, { now, notes }));
    const notice = [
      "# Memory notes",
      "Changed by the user or another chat since the last notice:",
      "- deleted [m2] Team",
      "- changed [m1] Language and tone",
      "- saved [m5] Holidays",
    ].join("\n");
    expect(next.notice).toBe(notice);
    const messages = buildChatRequest(settings, wireTurns([...history, next])).messages;
    expect(messages.slice(-2)).toEqual([
      { role: "developer", content: notice },
      { role: "user", content: "Next" },
    ]);
    expect(responseItems(wireTurns([...history, next])).slice(-2)).toEqual([developer(notice), userItem("Next")]);
  });

  it("tells of a note this chat saved once the user changes it, and then not again", () => {
    const edited = { ...trip, title: "Trip to Lisbon", savedAt: "2026-10-28T08:50:00.000Z" };
    const notes = [note("m1", "Language"), note("m2", "Team"), edited];
    const next = prompt(6, "Next", turnNotice(ALL_TOOLS, history, { now, notes }));
    expect(next.notice).toBe("# Memory notes\nChanged by the user or another chat since the last notice:\n- changed [m4] Trip to Lisbon");
    expect(turnNotice(ALL_TOOLS, [...history, next, answer(7, "Fine.")], { now, notes })).toEqual({});
  });

  it("counts what the chat's own agents saved or deleted as the chat's own", () => {
    const agentRun: AgentRun = { callId: "a1", name: "agent", prompt: "remember it", status: "done", reasoning: "", content: "Done.", local: [saveRun("c9", trip), deleteRun("c10", "m3")] };
    const withAgent = [
      prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes: first })),
      { ...answer(2, ""), toolCalls: [{ id: "a1", name: "agent", arguments: "{}" }], agents: [agentRun] },
      { id: 3, role: "tool" as const, content: "Done.", toolCallId: "a1", reasoning: "", streaming: false },
      answer(4, "Noted."),
    ];
    expect(turnNotice(ALL_TOOLS, withAgent, { now, notes: [note("m1", "Language"), note("m2", "Team"), trip] })).toEqual({});
  });
});

describe("a turn with nothing to report", () => {
  it("carries no notice: the request has no developer message on either transport", () => {
    const notes = [note("m1", "Language")];
    const history = [prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes })), answer(2, "Hello.")];
    const next = turnNotice(ALL_TOOLS, history, { now, notes });
    expect(next).toEqual({});
    const turns = wireTurns([...history, prompt(3, "Again", next)]);
    expect(buildChatRequest(settings, turns).messages.slice(2)).toEqual([
      { role: "assistant", content: "Hello." },
      { role: "user", content: "Again" },
    ]);
    expect(responseItems(turns).at(-1)).toEqual(userItem("Again"));
  });

  it("carries none at a first turn with no note saved, and never repeats what the chat then saves itself", () => {
    const first = turnNotice(ALL_TOOLS, [], { now, notes: [] });
    expect(first.notice).toBeUndefined();
    const trip = note("m1", "Trip", "2026-10-28T08:31:00.000Z");
    const history = [prompt(1, "Hi", first), ...calledLocal(2, [saveRun("c1", trip)]), answer(4, "Noted.")];
    expect(turnNotice(ALL_TOOLS, history, { now, notes: [trip] }).notice).toBeUndefined();
  });

  it("carries none while the memory or every tool is off", () => {
    const notes = [note("m1", "Language")];
    expect(turnNotice({ ...ALL_TOOLS, memory: false }, [], { now, notes })).toEqual({});
    expect(turnNotice({ ...ALL_TOOLS, enabled: false }, [], { now, notes })).toEqual({});
  });
});

describe("a notice the model never read", () => {
  const notes = [note("m1", "Language")];
  const listing = "# Memory notes\nSaved notes, by title:\n- [m1] Language";

  it("does not count once its prompt is dropped with a failed reply: the next prompt hears the titles again", () => {
    const history = [prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes })), { ...answer(2, ""), error: "the engine is full" }];
    expect(wireTurns(history)).toEqual([]);
    expect(turnNotice(ALL_TOOLS, history, { now, notes }).notice).toBe(listing);
  });

  it("goes with the prompt a rerun drops: the first prompt sent again is a first turn again", () => {
    const history = [prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes })), answer(2, "Hello.")];
    // A rerun of the first prompt sends it again after what came before it: nothing.
    expect(turnNotice(ALL_TOOLS, history.slice(0, 0), { now, notes }).notice).toBe(listing);
  });
});

describe("a notice once sent", () => {
  it("is kept with its turn and re-sent byte-identical by every later request, through a reload, on both transports", () => {
    const live = { ...ALL_TOOLS, dateTimeLive: true };
    const first = prompt(1, "Hi", turnNotice(live, [], { now, notes: [note("m1", "Language")] }));
    const session: Session = { ...createSession(1, now), messages: [first, answer(2, "Hello.")] };
    const [reloaded] = restoreChats([toStored(session, { key: "k", order: 1 })], 10).sessions;
    // An hour later, with another note saved: the first turn still goes out as it was sent.
    const later = new Date(now.getTime() + 3_600_000);
    const notes = [note("m1", "Language"), note("m2", "Team", "2026-10-28T09:00:00.000Z")];
    const next = prompt(20, "Again", turnNotice(live, reloaded.messages, { now: later, notes }));
    const before = wireTurns(session.messages);
    const after = wireTurns([...reloaded.messages, next]);
    const chat = (turns: Turn[]) => buildChatRequest(settings, turns).messages;
    expect(chat(before)[0]).toEqual({ role: "developer", content: first.notice });
    expect(chat(after).slice(0, chat(before).length)).toEqual(chat(before));
    expect(responseItems(after).slice(0, responseItems(before).length)).toEqual(responseItems(before));
    expect(next.notice).toBe(`${dateTimePrompt(later)}\n\n# Memory notes\nChanged by the user or another chat since the last notice:\n- saved [m2] Team`);
  });
});

describe("the live date", () => {
  const live = { ...ALL_TOOLS, dateTimeLive: true };
  const notes = [note("m1", "Language")];

  it("rides in the same notice as the memory notes, ahead of them", () => {
    expect(turnNotice(live, [], { now, notes }).notice).toBe(`${dateTimePrompt(now)}\n\n# Memory notes\nSaved notes, by title:\n- [m1] Language`);
  });

  it("alone, is the developer message the moment always was", () => {
    const history = [prompt(1, "Hi", turnNotice(live, [], { now, notes })), answer(2, "Hello.")];
    expect(turnNotice(live, history, { now, notes }).notice).toBe(dateTimePrompt(now));
  });
});

describe("a chat saved before the turn notice", () => {
  const moment = "# Date and time\nIt is now Monday 27 October 2026 at 10:00 in the user's time zone (Europe/Rome, GMT+01:00). Use it for anything that depends on today's date or the current time.";
  /** As the page kept a chat before the turn notice: the moment on its prompt, nothing else. */
  const stored = {
    version: CHAT_VERSION,
    key: "older",
    order: 1,
    title: "Hi",
    startedAt: now.getTime(),
    log: [],
    attachments: [],
    messages: [
      { id: 1, role: "user", content: "Hi", reasoning: "", streaming: false, dateTime: moment },
      { id: 2, role: "assistant", content: "Hello.", reasoning: "", streaming: false },
    ],
  };

  it("re-sends the moment its turns carry exactly as before, on both transports", () => {
    const [older] = restoreChats([stored], 10).sessions;
    const turns = wireTurns(older.messages);
    expect(buildChatRequest(settings, turns).messages).toEqual([
      { role: "developer", content: moment },
      { role: "user", content: "Hi" },
      { role: "assistant", content: "Hello." },
    ]);
    expect(responseItems(turns)).toEqual([
      developer(moment),
      userItem("Hi"),
      { type: "message", role: "assistant", content: [{ type: "output_text", text: "Hello." }] },
    ]);
  });

  it("tells its next prompt every title, as a first turn: it was never told of them", () => {
    const [older] = restoreChats([stored], 10).sessions;
    expect(turnNotice(ALL_TOOLS, older.messages, { now, notes: [note("m1", "Language")] }).notice).toBe(
      "# Memory notes\nSaved notes, by title:\n- [m1] Language",
    );
  });
});

describe("an agent", () => {
  it("hears every title in a notice ahead of its prompt, without the moment: it keeps the session's in its system prompt", () => {
    const live = { ...ALL_TOOLS, dateTimeLive: true };
    expect(agentNotice(live, [note("m1", "Language"), note("m2", "Team")])).toBe("# Memory notes\nSaved notes, by title:\n- [m1] Language\n- [m2] Team");
    expect(agentNotice(live, [])).toBeUndefined();
    expect(agentNotice({ ...ALL_TOOLS, memory: false }, [note("m1", "Language")])).toBeUndefined();
  });
});

// The workspace half of the notice (#393): the files of the chat's workspace,
// whole at its first turn and later only what someone other than this chat
// created, changed, renamed or deleted, driven through the workspace module
// over the in-memory store and its tools as a turn runs them.

/** A clock that moves one minute per reading, from 28 October 2026 at 09:00 local time. */
function clock() {
  let minute = 0;
  return () => new Date(2026, 9, 28, 9, minute++).getTime();
}

function workspaceModule(): Workspaces {
  let n = 0;
  return createWorkspaces(createMemoryStore(), { now: clock(), newId: () => `id${++n}`, measure: async () => ({ width: 1024, height: 1024 }) });
}

/** A chat bound to `workspace`, as the picker or its first file binds it, under its store key. */
function boundChat(workspaces: Workspaces, workspace: string, key = "chat-a"): WorkspaceAccess {
  return { workspaces, chat: workspaces.chat(key, { bound: () => workspace, name: () => "Trip plan", onBorn: () => {} }) };
}

let callN = 0;

/** One workspace call as the local tools' router runs it in a turn, by the chat's reader, or by `access.reader`. */
async function call(access: WorkspaceAccess, name: string, args: object): Promise<LocalRun> {
  const calls = [{ id: `w${++callN}`, name, arguments: JSON.stringify(args) }];
  const [ran] = await runLocalCalls(calls, { settings, signal: new AbortController().signal, jsSafetyCheck: false, workspace: access }, () => {});
  return ran;
}

/** A workspace "Trip" holding the user's `files`, each a text file. */
async function tripWith(workspaces: Workspaces, files: Record<string, string>) {
  const trip = await workspaces.create("Trip");
  await workspaces.add(trip.id, Object.entries(files).map(([name, text]) => ({ name, kind: "text" as const, text })), USER);
  const named = async (name: string) => (await workspaces.files(trip.id)).find((f) => f.name === name)!;
  return { id: trip.id, files: () => workspaces.files(trip.id), named };
}

const textFile = (id: string, name: string, text: string, at: number): WorkspaceFile => ({
  format: WORKSPACE_FORMAT,
  id,
  workspace: "w1",
  name,
  kind: "text",
  versions: [{ n: 1, author: USER, at, text }],
});

describe("an agent, with a workspace", () => {
  it("hears the files of the chat's workspace in its notice, as a chat's first turn does, beside the titles, while the switch is on", async () => {
    const workspaces = workspaceModule();
    const trip = await tripWith(workspaces, { "notes.md": "# Trip\n\n- Rome\n" });
    const files = await trip.files();
    const listing = ["# Workspace files", "Files in the chat's workspace:", "- notes.md: text, 3 lines, 15 characters"].join("\n");
    expect(agentNotice(ALL_TOOLS, [note("m1", "Language")], files)).toBe(`# Memory notes\nSaved notes, by title:\n- [m1] Language\n\n${listing}`);
    expect(agentNotice({ ...ALL_TOOLS, memory: false }, [], files)).toBe(listing);
    expect(agentNotice({ ...ALL_TOOLS, workspace: false }, [], files)).toBeUndefined();
    expect(agentNotice(ALL_TOOLS, [], [])).toBeUndefined();
  });
});

describe("a chat's first turn, with a workspace", () => {
  it("lists the files of the workspace the picker bound it to, with their kind and size, on both transports", async () => {
    const workspaces = workspaceModule();
    const trip = await tripWith(workspaces, { "notes.md": "# Trip\n\n- Rome\n" });
    const other = boundChat(workspaces, trip.id, "chat-b");
    await call(other, "write_file", { name: "site/index.html", content: "<h1>Trip</h1>" });
    await call(other, "write_file", { name: "old.md", content: "x" });
    await call(other, "delete_file", { name: "old.md" });
    const first = prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes: [], files: await trip.files() }));
    const notice = ["# Workspace files", "Files in the chat's workspace:", "- notes.md: text, 3 lines, 15 characters", "- site/index.html: text, 1 line, 13 characters"].join("\n");
    expect(first.notice).toBe(notice);
    const turns = wireTurns([first]);
    expect(buildChatRequest(settings, turns).messages).toEqual([
      { role: "developer", content: notice },
      { role: "user", content: "Hi" },
    ]);
    expect(responseItems(turns)).toEqual([developer(notice), userItem("Hi")]);
  });

  it("rides behind the moment and the memory notes, in the same notice", async () => {
    const workspaces = workspaceModule();
    const trip = await tripWith(workspaces, { "notes.md": "- Rome\n" });
    const live = { ...ALL_TOOLS, dateTimeLive: true };
    expect(turnNotice(live, [], { now, notes: [note("m1", "Language")], files: await trip.files() }).notice).toBe(
      `${dateTimePrompt(now)}\n\n# Memory notes\nSaved notes, by title:\n- [m1] Language\n\n# Workspace files\nFiles in the chat's workspace:\n- notes.md: text, 1 line, 7 characters`,
    );
  });

  it("has no files section while the chat has no workspace, and never tells the chat of the file it writes itself", async () => {
    const workspaces = workspaceModule();
    let bound: string | undefined;
    const chat = workspaces.chat("chat-a", { bound: () => bound, name: () => "Trip plan", onBorn: (id) => (bound = id) });
    const first = turnNotice(ALL_TOOLS, [], { now, notes: [], files: await noticeFiles(workspaces, bound) });
    expect(first).toEqual({ knownNotes: [], knownFiles: [] });
    const wrote = await call({ workspaces, chat }, "write_file", { name: "plan.md", content: "- Rome\n" });
    const history = [prompt(1, "Hi", first), ...calledLocal(2, [wrote]), answer(4, "Written.")];
    expect(turnNotice(ALL_TOOLS, history, { now, notes: [], files: await noticeFiles(workspaces, bound) })).toEqual({});
  });

  it("keeps the memory's cap: the files changed last that fit in 4,000 characters, in the order they changed, and a line counting the others", () => {
    // 50 files, f10 to f59, changed in that order, whose lines are 99 characters each: 40 of them fit with their line breaks.
    const files = Array.from({ length: 50 }, (_, i) => textFile(`id${i}`, `f${i + 10}-${"x".repeat(63)}.md`, "x", i));
    const lines = (turnNotice(ALL_TOOLS, [], { now, notes: [], files }).notice ?? "").split("\n");
    expect(lines.slice(0, 2)).toEqual(["# Workspace files", "Files in the chat's workspace:"]);
    expect(lines[2]).toBe(`- f20-${"x".repeat(63)}.md: text, 1 line, 1 character`);
    expect(lines.slice(2, -1).map((line) => line.slice(0, 5))).toEqual(Array.from({ length: 40 }, (_, i) => `- f${i + 20}`));
    expect(lines.at(-1)).toBe("(10 older files not listed; list_files lists them all.)");
  });
});

describe("a later turn, with a workspace", () => {
  it("tells only what the user or another chat created, changed, renamed or deleted since the last notice, on both transports", async () => {
    const workspaces = workspaceModule();
    const trip = await tripWith(workspaces, { "budget.csv": "item,cost\n", "notes.md": "# Trip\n", "old.md": "x" });
    const history = [prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes: [], files: await trip.files() })), answer(2, "Hello.")];
    // In the panel the user edits notes.md, renames budget.csv and deletes old.md; another chat writes todo.md.
    await saveText(workspaces, await trip.named("notes.md"), "# Trip\n\n- Rome\n");
    await workspaces.renameFile(trip.id, (await trip.named("budget.csv")).id, "costs.csv");
    await workspaces.deleteFile(trip.id, (await trip.named("old.md")).id, USER);
    await call(boundChat(workspaces, trip.id, "chat-b"), "write_file", { name: "todo.md", content: "- Book\n" });
    const next = prompt(3, "Next", turnNotice(ALL_TOOLS, history, { now, notes: [], files: await trip.files() }));
    const notice = [
      "# Workspace files",
      "Changed by the user or another chat since the last notice:",
      "- renamed budget.csv to costs.csv",
      "- changed notes.md: text, 3 lines, 15 characters",
      "- deleted old.md",
      "- created todo.md: text, 1 line, 7 characters",
    ].join("\n");
    expect(next.notice).toBe(notice);
    const turns = wireTurns([...history, next]);
    expect(buildChatRequest(settings, turns).messages.slice(-2)).toEqual([
      { role: "developer", content: notice },
      { role: "user", content: "Next" },
    ]);
    expect(responseItems(turns).slice(-2)).toEqual([developer(notice), userItem("Next")]);
    // Told once: the prompt after it hears nothing more of them.
    expect(turnNotice(ALL_TOOLS, [...history, next, answer(4, "Noted.")], { now, notes: [], files: await trip.files() })).toEqual({});
  });

  it("says nothing of what the chat wrote, edited, deleted or made itself, nor of what its agents did", async () => {
    const workspaces = workspaceModule();
    const trip = await tripWith(workspaces, { "budget.csv": "item,cost\n", "notes.md": "# Trip\n", "old.md": "x" });
    const a = boundChat(workspaces, trip.id);
    const first = prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes: [], files: await trip.files() }));
    const own = [
      await call(a, "read_file", { name: "notes.md" }),
      await call(a, "edit_file", { name: "notes.md", old: "# Trip", new: "# Trip to Rome" }),
      await call(a, "write_file", { name: "plan.md", content: "- Day 1\n" }),
      await call(a, "read_file", { name: "old.md" }),
      await call(a, "delete_file", { name: "old.md" }),
    ];
    // One of its agents reads and deletes budget.csv and writes agenda.md; make_image lands a picture.
    const helper = { ...a, reader: a.chat.agent({ call: "a1", name: "agent" }) };
    const agentRun: AgentRun = {
      callId: "a1",
      name: "agent",
      prompt: "tidy up",
      status: "done",
      reasoning: "",
      content: "Done.",
      local: [
        await call(helper, "read_file", { name: "budget.csv" }),
        await call(helper, "delete_file", { name: "budget.csv" }),
        await call(helper, "write_file", { name: "agenda.md", content: "- 9:00\n" }),
      ],
    };
    const picture = await madeIn(workspaces, a.chat).land(new Blob(["png"], { type: "image/png" }), "map.png");
    const made: ImageRun = { callId: "i1", prompt: "a map", references: [], transparent: false, count: 1, status: "done", made: [{ name: "map.png", size: "1024 × 1024", file: picture }] };
    expect([...own, ...(agentRun.local ?? [])].map((run) => run.status)).toEqual(Array(8).fill("done"));
    const history = [
      first,
      ...calledLocal(2, own),
      { ...answer(8, ""), toolCalls: [{ id: "a1", name: "agent", arguments: "{}" }], agents: [agentRun] },
      { id: 9, role: "tool" as const, content: "Done.", toolCallId: "a1", reasoning: "", streaming: false },
      { ...answer(10, ""), toolCalls: [{ id: "i1", name: "make_image", arguments: "{}" }], imageRuns: [made] },
      { id: 11, role: "tool" as const, content: "Made map.png.", toolCallId: "i1", reasoning: "", streaming: false },
      answer(12, "Tidied."),
    ];
    const next = turnNotice(ALL_TOOLS, history, { now, notes: [], files: await trip.files() });
    expect(next.notice).toBeUndefined();
    expect(turnNotice(ALL_TOOLS, [...history, prompt(13, "Next", next), answer(14, "Fine.")], { now, notes: [], files: await trip.files() })).toEqual({});
  });

  it("tells of every file it knew once the user deletes the workspace, and the chat has none", async () => {
    const workspaces = workspaceModule();
    const trip = await tripWith(workspaces, { "notes.md": "# Trip\n", "plan.md": "- Rome\n" });
    const history = [prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes: [], files: await trip.files() })), answer(2, "Hello.")];
    await workspaces.remove(trip.id, []);
    // Deleting a workspace unbinds its chats.
    expect(turnNotice(ALL_TOOLS, history, { now, notes: [], files: await noticeFiles(workspaces, undefined) })).toEqual({
      notice: "# Workspace files\nChanged by the user or another chat since the last notice:\n- deleted notes.md\n- deleted plan.md",
      knownFiles: [],
    });
  });

  it("does not tell again a change the chat read in its turn", async () => {
    const workspaces = workspaceModule();
    const trip = await tripWith(workspaces, { "notes.md": "# Trip\n" });
    const first = prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes: [], files: await trip.files() }));
    await saveText(workspaces, await trip.named("notes.md"), "# Trip\n\n- Rome\n");
    const read = await call(boundChat(workspaces, trip.id), "read_file", { name: "notes.md" });
    const history = [first, ...calledLocal(2, [read]), answer(4, "It says Rome.")];
    expect(turnNotice(ALL_TOOLS, history, { now, notes: [], files: await trip.files() })).toEqual({});
  });

  it("does not repeat the files a prompt attaches, which it names itself, nor tell of them after", async () => {
    const workspaces = workspaceModule();
    // The first prompt attaches brief.txt, the workspace's only file.
    const trip = await tripWith(workspaces, { "brief.txt": "Rome, 3 days\n" });
    const firstNotice = turnNotice(ALL_TOOLS, [], { now, notes: [], files: await trip.files(), attached: ["brief.txt"] });
    expect(firstNotice.notice).toBeUndefined();
    const history = [prompt(1, "Plan it", { attached: ["brief.txt"], ...firstNotice }), answer(2, "Planned.")];
    // The user edits brief.txt in the panel; the next prompt attaches map.txt.
    await saveText(workspaces, await trip.named("brief.txt"), "Rome, 4 days\n");
    await workspaces.add(trip.id, [{ name: "map.txt", kind: "text", text: "Colosseum\n" }], USER);
    const nextNotice = turnNotice(ALL_TOOLS, history, { now, notes: [], files: await trip.files(), attached: ["map.txt"] });
    const next = prompt(3, "And the map", { attached: ["map.txt"], ...nextNotice });
    const notice = "# Workspace files\nChanged by the user or another chat since the last notice:\n- changed brief.txt: text, 1 line, 13 characters";
    expect(next.notice).toBe(notice);
    expect(buildChatRequest(settings, wireTurns([...history, next])).messages.slice(-2)).toEqual([
      { role: "developer", content: notice },
      { role: "user", content: "And the map\n\n[Attached to the workspace: map.txt]" },
    ]);
    // A prompt that attaches a file and has nothing else to tell carries no notice, and the chat knows the file after it.
    await workspaces.add(trip.id, [{ name: "route.txt", kind: "text", text: "Via Appia\n" }], USER);
    const later = [...history, next, answer(4, "Mapped.")];
    const third = turnNotice(ALL_TOOLS, later, { now, notes: [], files: await trip.files(), attached: ["route.txt"] });
    expect(third.notice).toBeUndefined();
    const last = [...later, prompt(5, "And the route", { attached: ["route.txt"], ...third }), answer(6, "Routed.")];
    expect(turnNotice(ALL_TOOLS, last, { now, notes: [], files: await trip.files() })).toEqual({});
  });
});

describe("a notice with the Workspace switch off", () => {
  it("names no file, and the first prompt after the switch is back on hears every one", async () => {
    const workspaces = workspaceModule();
    const trip = await tripWith(workspaces, { "notes.md": "- Rome\n" });
    const notes = [note("m1", "Language")];
    const off = turnNotice({ ...ALL_TOOLS, workspace: false }, [], { now, notes, files: await trip.files() });
    expect(off.notice).toBe("# Memory notes\nSaved notes, by title:\n- [m1] Language");
    expect(off.knownFiles).toBeUndefined();
    expect(turnNotice({ ...ALL_TOOLS, enabled: false }, [], { now, notes, files: await trip.files() })).toEqual({});
    const history = [prompt(1, "Hi", off), answer(2, "Hello.")];
    expect(turnNotice(ALL_TOOLS, history, { now, notes, files: await trip.files() }).notice).toBe(
      "# Workspace files\nFiles in the chat's workspace:\n- notes.md: text, 1 line, 7 characters",
    );
  });
});

describe("a notice of the workspace the model never read", () => {
  const listing = "# Workspace files\nFiles in the chat's workspace:\n- notes.md: text, 1 line, 7 characters";

  it("does not count once its prompt is dropped with a failed reply: the next prompt hears every file again", async () => {
    const workspaces = workspaceModule();
    const trip = await tripWith(workspaces, { "notes.md": "- Rome\n" });
    const history = [prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes: [], files: await trip.files() })), { ...answer(2, ""), error: "the engine is full" }];
    expect(history[0].notice).toBe(listing);
    expect(turnNotice(ALL_TOOLS, history, { now, notes: [], files: await trip.files() }).notice).toBe(listing);
  });

  it("is not sent when the store cannot read the workspace: the next prompt tells what it could not", async () => {
    const workspaces = workspaceModule();
    const trip = await tripWith(workspaces, { "notes.md": "- Rome\n" });
    const files = await noticeFiles({ files: () => Promise.reject(new Error("the store is closed")) }, trip.id);
    expect(files).toBeUndefined();
    const history = [prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes: [], files })), answer(2, "Hello.")];
    expect(history[0].notice).toBeUndefined();
    expect(turnNotice(ALL_TOOLS, history, { now, notes: [], files: await noticeFiles(workspaces, trip.id) }).notice).toBe(listing);
  });
});

describe("a notice of the workspace once sent", () => {
  it("is re-sent byte-identical through a reload, on both transports, and the next prompt hears only what changed since", async () => {
    const workspaces = workspaceModule();
    const trip = await tripWith(workspaces, { "notes.md": "# Trip\n" });
    const first = prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes: [], files: await trip.files() }));
    const session: Session = { ...createSession(1, now), messages: [first, answer(2, "Hello.")], workspaceId: trip.id };
    const [reloaded] = restoreChats([toStored(session, { key: "k", order: 1 })], 10).sessions;
    await saveText(workspaces, await trip.named("notes.md"), "# Trip\n\n- Rome\n");
    const next = prompt(20, "Again", turnNotice(ALL_TOOLS, reloaded.messages, { now, notes: [], files: await trip.files() }));
    expect(next.notice).toBe("# Workspace files\nChanged by the user or another chat since the last notice:\n- changed notes.md: text, 3 lines, 15 characters");
    const before = wireTurns(session.messages);
    const after = wireTurns([...reloaded.messages, next]);
    const chat = (turns: Turn[]) => buildChatRequest(settings, turns).messages;
    expect(chat(before)[0]).toEqual({ role: "developer", content: "# Workspace files\nFiles in the chat's workspace:\n- notes.md: text, 1 line, 7 characters" });
    expect(chat(after).slice(0, chat(before).length)).toEqual(chat(before));
    expect(responseItems(after).slice(0, responseItems(before).length)).toEqual(responseItems(before));
  });
});

// A chat moved to another workspace, or out of its own (#404): the next notice
// says so, names the workspace left and lists the one it works in now whole,
// as a first turn does; the turns after it hear only others' changes again.

/** A workspace "Small site" holding the user's `files`, each a text file. */
async function siteWith(workspaces: Workspaces, files: Record<string, string>) {
  const site = await workspaces.create("Small site");
  await workspaces.add(site.id, Object.entries(files).map(([name, text]) => ({ name, kind: "text" as const, text })), USER);
  return { id: site.id, files: () => workspaces.files(site.id) };
}

/** A chat bound to the workspace "Trip" that has talked: its first turn was told of Trip's files. */
async function talkedInTrip(workspaces: Workspaces) {
  const trip = await tripWith(workspaces, { "notes.md": "# Trip\n" });
  const session: Session = {
    ...createSession(1, now),
    workspaceId: trip.id,
    messages: [prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes: [], files: await trip.files() })), answer(2, "Hello.")],
  };
  return { trip, session };
}

const LEFT_TRIP = 'This chat moved out of the workspace "Trip": its files are no longer this chat\'s.';

describe("a chat moved to another workspace", () => {
  it("is told at its next prompt, naming the workspace left, then hears the new one's files whole, on both transports", async () => {
    const workspaces = workspaceModule();
    const { trip, session } = await talkedInTrip(workspaces);
    const site = await siteWith(workspaces, { "index.html": "<h1>Site</h1>", "style.css": "h1 {}" });
    const [moved] = moveWorkspace([session], 1, site.id, { at: 100, name: "Trip" });
    const next = prompt(3, "Next", turnNotice(ALL_TOOLS, moved.messages, { now, notes: [], files: await site.files(), moved: moved.moved }));
    const notice = ["# Workspace files", LEFT_TRIP, "Files in the chat's workspace now:", "- index.html: text, 1 line, 13 characters", "- style.css: text, 1 line, 5 characters"].join("\n");
    expect(next.notice).toBe(notice);
    expect(next.moved).toEqual({ from: trip.id, name: "Trip", at: 100 });
    const turns = wireTurns([...moved.messages, next]);
    expect(buildChatRequest(settings, turns).messages.slice(-2)).toEqual([
      { role: "developer", content: notice },
      { role: "user", content: "Next" },
    ]);
    expect(responseItems(turns).slice(-2)).toEqual([developer(notice), userItem("Next")]);
  });

  it("hears only others' changes from the turn after, as before the move, and nothing of the workspace it left", async () => {
    const workspaces = workspaceModule();
    const { trip, session } = await talkedInTrip(workspaces);
    const site = await siteWith(workspaces, { "index.html": "<h1>Site</h1>" });
    const [moved] = moveWorkspace([session], 1, site.id, { at: 100, name: "Trip" });
    const history = [...moved.messages, prompt(3, "Next", turnNotice(ALL_TOOLS, moved.messages, { now, notes: [], files: await site.files(), moved: moved.moved })), answer(4, "Noted.")];
    expect(turnNotice(ALL_TOOLS, history, { now, notes: [], files: await site.files(), moved: moved.moved })).toEqual({});
    // The user edits the site's page, and Trip's notes, which are no longer this chat's.
    const [page] = await site.files();
    await saveText(workspaces, page, "<h1>Small site</h1>");
    await saveText(workspaces, await trip.named("notes.md"), "# Trip\n\n- Rome\n");
    const later = prompt(5, "Again", turnNotice(ALL_TOOLS, history, { now, notes: [], files: await site.files(), moved: moved.moved }));
    const notice = "# Workspace files\nChanged by the user or another chat since the last notice:\n- changed index.html: text, 1 line, 19 characters";
    expect(later.notice).toBe(notice);
    expect(later.moved).toBeUndefined();
    expect(buildChatRequest(settings, wireTurns([...history, later])).messages.slice(-2)).toEqual([
      { role: "developer", content: notice },
      { role: "user", content: "Again" },
    ]);
    expect(responseItems(wireTurns([...history, later])).slice(-2)).toEqual([developer(notice), userItem("Again")]);
  });

  it("to a new workspace, is told the chat has none now; the workspace its next file starts is its own, and no news", async () => {
    const workspaces = workspaceModule();
    const { session } = await talkedInTrip(workspaces);
    const [left] = moveWorkspace([session], 1, undefined, { at: 100, name: "Trip" });
    const next = prompt(3, "Next", turnNotice(ALL_TOOLS, left.messages, { now, notes: [], files: await noticeFiles(workspaces, left.workspaceId), moved: left.moved }));
    const notice = ["# Workspace files", LEFT_TRIP, "The chat has no workspace now: one is created with the next file written or attached."].join("\n");
    expect(next.notice).toBe(notice);
    expect(next.knownFiles).toEqual([]);
    const turns = wireTurns([...left.messages, next]);
    expect(buildChatRequest(settings, turns).messages.slice(-2)).toEqual([
      { role: "developer", content: notice },
      { role: "user", content: "Next" },
    ]);
    expect(responseItems(turns).slice(-2)).toEqual([developer(notice), userItem("Next")]);
    // The reply writes a file: a new workspace is born, named after the chat, and the turn after hears nothing.
    let bound: string | undefined;
    const chat = workspaces.chat("chat-a", { bound: () => bound, name: () => "Trip plan", onBorn: (id) => (bound = id) });
    const wrote = await call({ workspaces, chat }, "write_file", { name: "plan.md", content: "- Rome\n" });
    const history = [...left.messages, next, ...calledLocal(4, [wrote]), answer(6, "Written.")];
    expect(turnNotice(ALL_TOOLS, history, { now, notes: [], files: await noticeFiles(workspaces, bound), moved: left.moved })).toEqual({});
  });

  it("from a workspace that was deleted, is told the same way, the workspace left unnamed", async () => {
    const workspaces = workspaceModule();
    const { trip, session } = await talkedInTrip(workspaces);
    await workspaces.remove(trip.id, []);
    const site = await siteWith(workspaces, { "index.html": "<h1>Site</h1>" });
    const [moved] = moveWorkspace([session], 1, site.id, { at: 100 });
    const next = prompt(3, "Next", turnNotice(ALL_TOOLS, moved.messages, { now, notes: [], files: await site.files(), moved: moved.moved }));
    const notice = [
      "# Workspace files",
      "This chat moved out of its workspace, which was deleted: its files are no longer this chat's.",
      "Files in the chat's workspace now:",
      "- index.html: text, 1 line, 13 characters",
    ].join("\n");
    expect(next.notice).toBe(notice);
    expect(buildChatRequest(settings, wireTurns([...moved.messages, next])).messages.slice(-2)).toEqual([
      { role: "developer", content: notice },
      { role: "user", content: "Next" },
    ]);
    expect(responseItems(wireTurns([...moved.messages, next])).slice(-2)).toEqual([developer(notice), userItem("Next")]);
  });

  it("with the Workspace switch off, is not told and hears no file; the first prompt with the switch back on tells it", async () => {
    const workspaces = workspaceModule();
    const { session } = await talkedInTrip(workspaces);
    const site = await siteWith(workspaces, { "index.html": "<h1>Site</h1>" });
    const [moved] = moveWorkspace([session], 1, site.id, { at: 100, name: "Trip" });
    const off = turnNotice({ ...ALL_TOOLS, workspace: false }, moved.messages, { now, notes: [note("m1", "Language")], files: await site.files(), moved: moved.moved });
    expect(off).toEqual({
      notice: "# Memory notes\nChanged by the user or another chat since the last notice:\n- saved [m1] Language",
      knownNotes: [{ id: "m1", title: "Language", savedAt: "2026-10-27T10:00:00.000Z" }],
    });
    expect(turnNotice({ ...ALL_TOOLS, enabled: false }, moved.messages, { now, notes: [], files: await site.files(), moved: moved.moved })).toEqual({});
    const history = [...moved.messages, prompt(3, "Next", off), answer(4, "Noted.")];
    const on = turnNotice(ALL_TOOLS, history, { now, notes: [note("m1", "Language")], files: await site.files(), moved: moved.moved });
    expect(on.notice).toBe(["# Workspace files", LEFT_TRIP, "Files in the chat's workspace now:", "- index.html: text, 1 line, 13 characters"].join("\n"));
    expect(on.moved).toEqual(moved.moved);
  });

  it("keeps the first turn's cap for the new workspace's files: the files changed last that fit, and a line counting the others", async () => {
    const workspaces = workspaceModule();
    const { session } = await talkedInTrip(workspaces);
    const [moved] = moveWorkspace([session], 1, "w1", { at: 100, name: "Trip" });
    // 50 files, f10 to f59, changed in that order, whose lines are 99 characters each: 40 of them fit with their line breaks.
    const files = Array.from({ length: 50 }, (_, i) => textFile(`id${i}`, `f${i + 10}-${"x".repeat(63)}.md`, "x", i));
    const lines = (turnNotice(ALL_TOOLS, moved.messages, { now, notes: [], files, moved: moved.moved }).notice ?? "").split("\n");
    expect(lines.slice(0, 3)).toEqual(["# Workspace files", LEFT_TRIP, "Files in the chat's workspace now:"]);
    expect(lines.slice(3, -1).map((line) => line.slice(0, 5))).toEqual(Array.from({ length: 40 }, (_, i) => `- f${i + 20}`));
    expect(lines.at(-1)).toBe("(10 older files not listed; list_files lists them all.)");
  });

  it("names neither a file the prompt attaches, which it names itself, nor an empty workspace's files", async () => {
    const workspaces = workspaceModule();
    const { session } = await talkedInTrip(workspaces);
    const site = await siteWith(workspaces, { "brief.txt": "Rome\n" });
    const [moved] = moveWorkspace([session], 1, site.id, { at: 100, name: "Trip" });
    expect(turnNotice(ALL_TOOLS, moved.messages, { now, notes: [], files: await site.files(), attached: ["brief.txt"], moved: moved.moved }).notice).toBe(
      ["# Workspace files", LEFT_TRIP].join("\n"),
    );
    const [brief] = await site.files();
    await workspaces.deleteFile(site.id, brief.id, USER);
    expect(turnNotice(ALL_TOOLS, moved.messages, { now, notes: [], files: await site.files(), moved: moved.moved }).notice).toBe(
      ["# Workspace files", LEFT_TRIP, "The chat's workspace holds no files."].join("\n"),
    );
  });

  it("is told again when the prompt that told it goes: dropped with a failed reply, or by a rerun from before it", async () => {
    const workspaces = workspaceModule();
    const { session } = await talkedInTrip(workspaces);
    const site = await siteWith(workspaces, { "index.html": "<h1>Site</h1>" });
    const [moved] = moveWorkspace([session], 1, site.id, { at: 100, name: "Trip" });
    const options = { now, notes: [], files: await site.files(), moved: moved.moved };
    const told = turnNotice(ALL_TOOLS, moved.messages, options);
    const failed = [...moved.messages, prompt(3, "Next", told), { ...answer(4, ""), error: "the engine is full" }];
    expect(turnNotice(ALL_TOOLS, failed, options)).toEqual(told);
    // A rerun of the first prompt sends it after nothing: a first turn, whose listing of the new workspace says it all.
    const first = turnNotice(ALL_TOOLS, [], options);
    expect(first.notice).toBe("# Workspace files\nFiles in the chat's workspace:\n- index.html: text, 1 line, 13 characters");
    expect(first.moved).toEqual(moved.moved);
  });

  it("is re-sent byte-identical through a reload, on both transports, and not told again after it", async () => {
    const workspaces = workspaceModule();
    const { session } = await talkedInTrip(workspaces);
    const site = await siteWith(workspaces, { "index.html": "<h1>Site</h1>" });
    const [moved] = moveWorkspace([session], 1, site.id, { at: 100, name: "Trip" });
    // Reloaded between the move and the next prompt, then after it.
    const [between] = restoreChats([toStored(moved, { key: "k", order: 1 })], 10).sessions;
    const next = prompt(20, "Next", turnNotice(ALL_TOOLS, between.messages, { now, notes: [], files: await site.files(), moved: between.moved }));
    expect(next.notice).toContain(LEFT_TRIP);
    const told: Session = { ...between, messages: [...between.messages, next, answer(21, "Noted.")] };
    const [reloaded] = restoreChats([toStored(told, { key: "k", order: 1 })], 30).sessions;
    const chat = (turns: Turn[]) => buildChatRequest(settings, turns).messages;
    const before = wireTurns(told.messages);
    const after = wireTurns(reloaded.messages);
    expect(chat(after)).toEqual(chat(before));
    expect(chat(after).at(-3)).toEqual({ role: "developer", content: next.notice });
    expect(responseItems(after)).toEqual(responseItems(before));
    expect(turnNotice(ALL_TOOLS, reloaded.messages, { now, notes: [], files: await site.files(), moved: reloaded.moved })).toEqual({});
  });
});

describe("the system prompt and the tools block, with a workspace", () => {
  it("are the same whatever the workspace holds: its files reach the notice alone, on both transports", async () => {
    const workspaces = workspaceModule();
    const trip = await tripWith(workspaces, { "notes.md": "# Trip\n", "site/index.html": "<h1>Trip</h1>" });
    const extras = toolExtras(ALL_TOOLS, { now });
    const empty = wireTurns([prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes: [], files: [] }))]);
    const full = wireTurns([prompt(1, "Hi", turnNotice(ALL_TOOLS, [], { now, notes: [], files: await trip.files() }))]);
    const [a, b] = [empty, full].map((turns) => buildChatRequest(settings, turns, extras));
    expect(JSON.stringify(b.messages[0])).toBe(JSON.stringify(a.messages[0]));
    expect(JSON.stringify(b.tools)).toBe(JSON.stringify(a.tools));
    expect(a.messages.slice(1)).toEqual([{ role: "user", content: "Hi" }]);
    expect(b.messages[1]).toEqual({ role: "developer", content: expect.stringContaining("- site/index.html: text, 1 line, 13 characters") });
    const [x, y] = [empty, full].map((turns) => buildResponseCreate({ settings, turns, extras }, "s", responseItems(turns)));
    expect(y.instructions).toBe(x.instructions);
    expect(JSON.stringify(y.tools)).toBe(JSON.stringify(x.tools));
    expect(x.instructions).not.toContain("notes.md");
  });

  it("are the same when the chat moves to another workspace or out of its own: the move reaches the notice alone, on both transports", async () => {
    const workspaces = workspaceModule();
    const { session } = await talkedInTrip(workspaces);
    const site = await siteWith(workspaces, { "index.html": "<h1>Site</h1>" });
    const extras = toolExtras(ALL_TOOLS, { now });
    const stayed = wireTurns([...session.messages, prompt(3, "Next", turnNotice(ALL_TOOLS, session.messages, { now, notes: [], files: await workspaces.files(session.workspaceId!) }))]);
    const moves = await Promise.all(
      [site.id, undefined].map(async (to) => {
        const [moved] = moveWorkspace([session], 1, to, { at: 100, name: "Trip" });
        return wireTurns([...moved.messages, prompt(3, "Next", turnNotice(ALL_TOOLS, moved.messages, { now, notes: [], files: await noticeFiles(workspaces, to), moved: moved.moved }))]);
      }),
    );
    const head = (turns: Turn[]) => {
      const body = buildChatRequest(settings, turns, extras);
      return JSON.stringify([body.messages[0], body.tools]);
    };
    const instructions = (turns: Turn[]) => {
      const body = buildResponseCreate({ settings, turns, extras }, "s", responseItems(turns));
      return JSON.stringify([body.instructions, body.tools]);
    };
    for (const turns of moves) {
      expect(head(turns)).toBe(head(stayed));
      expect(instructions(turns)).toBe(instructions(stayed));
      expect(buildChatRequest(settings, turns, extras).messages.at(-2)).toEqual({ role: "developer", content: expect.stringContaining(LEFT_TRIP) });
    }
    expect(head(stayed)).not.toContain("Trip");
  });
});
