import { describe, expect, it } from "vitest";
import { IDBFactory } from "fake-indexeddb";
import { openIndexedDbStore } from "../store/indexeddb.ts";
import { createMemoryStore } from "../store/memory.ts";
import { activeOf, openTranscript, removeTranscript, startList, titleOf, UNTITLED } from "./sessions.ts";
import { restoreTranscripts, TRANSCRIBE_SESSIONS, TRANSCRIBE_VERSION, toStored, worthKeeping } from "./stored.ts";
import { addUtterance, beginListening, createTranscript, mergeSpeakers, type Transcript } from "./transcript.ts";

// The Transcripts as the view lists them and the page's store keeps them
// (spec speech/07, "The Transcript", Q86, Q101): a new kind of session, saved
// as each utterance completes, text, times, speakers and corrections, and
// never audio.

function heardTwice(id = 3): Transcript {
  let t = createTranscript(id, { source: "call", at: new Date(2026, 9, 25, 15, 42) });
  t = addUtterance(t, { stream: "mic-1", startMs: 0, endMs: 900, text: "Ci sei?", voice: { kind: "me" } });
  t = addUtterance(t, { stream: "shared-2", startMs: 1_000, endMs: 2_000, text: "Sì.", voice: { kind: "speaker", index: 0 } });
  return { ...t, clockMs: 2_400 };
}

describe("the Transcripts listed", () => {
  it("open with one untitled Transcript; a new one is reused while nothing is heard in it", () => {
    const list = startList(1);
    expect(titleOf(activeOf(list))).toBe(UNTITLED);
    expect(openTranscript(list, 2)).toBe(list);
    const heard = { transcripts: [heardTwice(1)], activeId: 1 };
    const opened = openTranscript(heard, 2);
    expect(opened.transcripts.map((t) => t.id)).toEqual([2, 1]);
    expect(opened.activeId).toBe(2);
  });

  it("removing the open one opens the next, the last leaves a new one", () => {
    const list = { transcripts: [heardTwice(2), heardTwice(1)], activeId: 2 };
    expect(removeTranscript(list, 2, 9)).toEqual({ transcripts: [list.transcripts[1]], activeId: 1 });
    expect(removeTranscript({ transcripts: [heardTwice(1)], activeId: 1 }, 1, 9).transcripts.map((t) => t.id)).toEqual([9]);
  });

  it("a listening titles an untitled Transcript by its date, time and source, and a live one keeps its wall-clock start; a Continue keeps both", () => {
    const at = new Date(2026, 9, 25, 15, 42);
    const begun = beginListening(createTranscript(1), { source: "microphone", at });
    expect(begun).toMatchObject({ title: "25 Oct 15:42 · Microphone", source: "microphone", startedAt: at.getTime() });
    const renamed = beginListening({ ...createTranscript(1), title: "Standup" }, { source: "call", at });
    expect(renamed.title).toBe("Standup");
    const later = new Date(2026, 9, 26, 9, 0);
    expect(beginListening(heardTwice(), { source: "microphone", at: later })).toMatchObject({ title: "25 Oct 15:42 · Call", startedAt: at.getTime() });
    expect(beginListening(createTranscript(1), { source: "file", at, fileName: "talk.mp3" })).toMatchObject({ title: "25 Oct 15:42 · talk.mp3", startedAt: null });
  });
});

describe("a Transcript in the store", () => {
  it("is plain data, its corrections with it, and comes back as it was", async () => {
    const original = mergeSpeakers(heardTwice(), 2, 1);
    const stored = toStored(original, { key: "key-1", order: 1 });
    expect(stored).toMatchObject({ version: TRANSCRIBE_VERSION, key: "key-1", order: 1 });
    expect(stored).not.toHaveProperty("id");
    // Held to the store's contract, on each adapter: a document of its own collection, a copy in and a copy out.
    for (const store of [createMemoryStore(), openIndexedDbStore(new IDBFactory(), "transcribe-test")]) {
      await store.putDoc(TRANSCRIBE_SESSIONS, stored.key, stored);
      const [back] = restoreTranscripts(await store.listDocs(TRANSCRIBE_SESSIONS), 1).transcripts;
      expect(back).toEqual({ ...original, id: back.id });
    }
    expect(JSON.stringify(stored)).not.toMatch(/audio|pcm/i);
  });

  it("is worth keeping once an utterance is heard in it", () => {
    expect(worthKeeping(createTranscript(1))).toBe(false);
    expect(worthKeeping(heardTwice())).toBe(true);
  });

  it("comes back newest first with new ids, and leaves out what it cannot read", () => {
    const older = toStored(heardTwice(9), { key: "a", order: 2 });
    const newer = toStored(heardTwice(2), { key: "b", order: 9 });
    const { transcripts, stamps, nextId } = restoreTranscripts([older, null, { version: TRANSCRIBE_VERSION + 1 }, newer, { version: 1 }], 5);
    expect(transcripts.map((t) => t.id)).toEqual([6, 5]);
    expect(stamps.get(6)).toEqual({ key: "b", order: 9 });
    expect(nextId).toBe(7);
  });
});
