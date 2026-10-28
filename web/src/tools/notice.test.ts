import { describe, expect, it } from "vitest";
import { buildChatRequest, conversationTurns, type Settings, type Turn } from "../api/request.ts";
import { responseItems } from "../api/responses.ts";
import { exchangeOf } from "../app/useConversation.ts";
import { createSession, type Message, type Session } from "../sessions/sessions.ts";
import { CHAT_VERSION, restoreChats, toStored } from "../sessions/stored.ts";
import type { AgentRun } from "./agents/agents.ts";
import { dateTimePrompt } from "./datetime/datetime.ts";
import { ALL_TOOLS, toolExtras } from "./index.ts";
import type { LocalRun } from "./local/local.ts";
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

/** A reply that called memory tools, as the chat ran them: the runs it shows, and the results after it. */
function calledMemory(id: number, runs: LocalRun[]): Message[] {
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
    ...calledMemory(2, [saveRun("c1", trip), deleteRun("c2", "m3")]),
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
    const history = [prompt(1, "Hi", first), ...calledMemory(2, [saveRun("c1", trip)]), answer(4, "Noted.")];
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
