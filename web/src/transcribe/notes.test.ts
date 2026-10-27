import { afterEach, describe, expect, it } from "vitest";
import type { ModelState } from "../api/model.ts";
import {
  chooseNoteKind,
  dueForUpdate,
  FIRST_LIVE_UTTERANCES,
  kindOf,
  LIVE_GAP_MS,
  NOTE_KIND_KEY,
  NOTE_KINDS,
  noNotes,
  notesBlocked,
  notesInput,
  notesPrompt,
  notesSettings,
  notesStatus,
  readNotes,
  reloadNoteKind,
  tidyNotes,
  type Notes,
} from "./notes.ts";
import { addUtterance, createTranscript, type Transcript } from "./transcript.ts";

// Notes on a Transcript (spec speech/07, Q124-Q127): the text model reads the
// transcript and writes the sections a kind of conversation wants, on a press
// or, kept up while listening, about once a minute.

function call(lines: [string, string][]): Transcript {
  let t = { ...createTranscript(1, { source: "call", at: new Date(2026, 9, 27, 15, 42) }), title: "27 Oct 15:42 · Call" };
  lines.forEach(([who, text], i) => {
    t = addUtterance(t, { stream: "s", startMs: i * 10_000, endMs: i * 10_000 + 4_000, text, voice: who === "me" ? { kind: "me" } : { kind: "others" } });
  });
  return t;
}

afterEach(() => {
  Object.defineProperty(globalThis, "localStorage", { value: undefined, configurable: true, writable: true });
  reloadNoteKind();
});

const written = (patch: Partial<Notes> = {}): Notes => ({ kind: "work", text: "## Riassunto\n- ok", writtenAs: "work", covered: 4, writtenAt: new Date(2026, 9, 27, 16, 5).getTime(), fromMs: null, ...patch });

describe("the kinds of conversation", () => {
  it("are General, Work call, Game session and Lesson, each with its sections", () => {
    expect(NOTE_KINDS.map((k) => k.label)).toEqual(["General", "Work call", "Game session", "Lesson"]);
    const sections = (kind: string) => NOTE_KINDS.find((k) => k.kind === kind)?.sections.map((s) => s.name);
    expect(sections("work")).toEqual(["Summary", "Topics", "People", "Decisions", "Action items", "Open questions"]);
    expect(sections("game")).toEqual(["Games", "Players", "Results", "Highlights"]);
  });

  it("start at the one this browser picked last, General at first; a Transcript keeps its own", () => {
    const storage = new Map<string, string>();
    const local = { getItem: (key: string) => storage.get(key) ?? null, setItem: (key: string, value: string) => void storage.set(key, value) };
    Object.defineProperty(globalThis, "localStorage", { value: local, configurable: true, writable: true });
    reloadNoteKind();
    const t = call([]);
    expect(kindOf(t)).toBe("general");
    chooseNoteKind("game");
    expect(storage.get(NOTE_KIND_KEY)).toBe("game");
    expect(kindOf(t)).toBe("game");
    expect(kindOf({ ...t, notes: noNotes("lesson") })).toBe("lesson");
    storage.set(NOTE_KIND_KEY, "party");
    reloadNoteKind();
    expect(kindOf(t)).toBe("general");
  });

  it("ask the model for their sections in order, under the headings as given, in the transcript's language, and nothing it did not hear", () => {
    const prompt = notesPrompt("work");
    const at = (heading: string) => prompt.indexOf(`## ${heading}\n`);
    expect(at("Summary")).toBeGreaterThan(0);
    expect(at("Summary")).toBeLessThan(at("Topics"));
    expect(at("Action items")).toBeLessThan(at("Open questions"));
    expect(prompt).toContain("Keep each heading exactly as given. Write the text under them in the same language as the transcript");
    expect(prompt).toContain("[mm:ss]");
    expect(prompt).not.toContain("## Games");
  });

  it("send no thinking, on the agent lane, and never at greedy with a top_p ignis would refuse", () => {
    const settings = notesSettings("game");
    expect(settings).toMatchObject({ systemPrompt: notesPrompt("game"), reasoningEffort: "none", laneTag: "agent", thinkingBudget: null });
    expect(settings.temperature === 0 && settings.topP !== 1).toBe(false);
  });
});

describe("what the model reads", () => {
  it("is the title and the transcript as its .txt, so a later request only appends", () => {
    const t = call([
      ["me", "Partiamo dal budget."],
      ["others", "Va bene."],
    ]);
    expect(notesInput(t, null)).toEqual({ content: "27 Oct 15:42 · Call\n\n[00:00] Me: Partiamo dal budget.\n[00:10] Others: Va bene.\n", fromMs: null });
    const longer = notesInput(addUtterance(t, { stream: "s", startMs: 20_000, endMs: 21_000, text: "Poi?", voice: { kind: "me" } }), null);
    expect(longer.content.startsWith(notesInput(t, null).content)).toBe(true);
  });

  it("is the transcript's tail when the whole would not fit the model's context, and says where it starts", () => {
    const t = call([
      ["me", "a".repeat(3_000)],
      ["others", "b".repeat(3_000)],
      ["me", "c".repeat(3_000)],
    ]);
    // 1,500 tokens for the notes and 1,000 for the prompt leave 2,500 tokens: about 7,500 characters.
    const { content, fromMs } = notesInput(t, 5_000);
    expect(fromMs).toBe(10_000);
    expect(content).not.toContain("aaa");
    expect(content).toContain("[00:10] Others: bbb");
    expect(notesInput(t, 100_000).fromMs).toBeNull();
  });
});

describe("keeping the notes up while listening", () => {
  it("writes the first once a few utterances are in, then about once a minute while something new was said", () => {
    expect(FIRST_LIVE_UTTERANCES).toBe(3);
    expect(dueForUpdate({ notes: undefined, utterances: 2, sinceLastMs: null })).toBe(false);
    expect(dueForUpdate({ notes: undefined, utterances: 3, sinceLastMs: null })).toBe(true);
    expect(dueForUpdate({ notes: written({ covered: 4 }), utterances: 4, sinceLastMs: LIVE_GAP_MS * 2 })).toBe(false);
    expect(dueForUpdate({ notes: written({ covered: 4 }), utterances: 5, sinceLastMs: LIVE_GAP_MS - 1 })).toBe(false);
    expect(dueForUpdate({ notes: written({ covered: 4 }), utterances: 5, sinceLastMs: LIVE_GAP_MS })).toBe(true);
  });

  it("writes them again as soon as another kind is picked", () => {
    expect(dueForUpdate({ notes: written({ covered: 4, kind: "game" }), utterances: 4, sinceLastMs: 1_000 })).toBe(true);
  });

  it("writes once more at the end when something came after the last notes", () => {
    expect(dueForUpdate({ notes: written({ covered: 4 }), utterances: 5, sinceLastMs: 1_000, final: true })).toBe(true);
    expect(dueForUpdate({ notes: written({ covered: 4 }), utterances: 4, sinceLastMs: 1_000, final: true })).toBe(false);
    expect(dueForUpdate({ notes: undefined, utterances: 1, sinceLastMs: null, final: true })).toBe(true);
  });
});

describe("what the panel says", () => {
  it("is when the notes were written, from how much, and what came since", () => {
    expect(notesStatus(written(), 4)).toBe("Written at 16:05 from 4 utterances.");
    expect(notesStatus(written(), 7)).toBe("Written at 16:05 from 4 utterances. 3 more since.");
    expect(notesStatus(written({ covered: 1 }), 2)).toBe("Written at 16:05 from 1 utterance. 1 more since.");
    expect(notesStatus(written({ fromMs: 125_000 }), 4)).toBe("Written at 16:05 from 4 utterances, from 02:05 on: the start did not fit the model's context.");
  });

  it("says which kind they were written as, when another is picked", () => {
    expect(notesStatus(written({ kind: "lesson" }), 4)).toBe("Written as Work call notes at 16:05. Write notes to redo them as Lesson notes.");
    expect(notesStatus(undefined, 4)).toBeNull();
  });

  it("says why no notes can be written while no text model serves", () => {
    const ready: ModelState = { state: "ready", id: "qwen3.8-27b", contextLimit: 262_144, kind: "text" };
    expect(notesBlocked(ready)).toBeNull();
    expect(notesBlocked({ state: "unloaded", id: "qwen3.8-27b", contextLimit: null })).toBeNull();
    expect(notesBlocked({ ...ready, kind: "image" })).toBe("The image model is loaded: notes need a text model. Switch in the Server view.");
    expect(notesBlocked({ state: "loading", to: "qwen3.8-27b", since: 0 })).toBe("A model is loading: notes can be written once it serves.");
    expect(notesBlocked({ state: "error", message: "down" })).toBe("No text model serves: notes need one. Load it in the Server view.");
  });
});

describe("the notes as kept", () => {
  it("are what the model wrote, without a fence around it", () => {
    expect(tidyNotes("```markdown\n## Summary\n- ok\n```\n")).toBe("## Summary\n- ok");
    expect(tidyNotes("  ## Summary\n- `x`\n")).toBe("## Summary\n- `x`");
  });

  it("read back from the store as written, and a malformed one as none", () => {
    expect(readNotes(written())).toEqual(written());
    expect(readNotes({ ...written(), kind: "party" })).toBeUndefined();
    expect(readNotes({ ...written(), text: 3 })).toBeUndefined();
    expect(readNotes(null)).toBeUndefined();
  });
});
