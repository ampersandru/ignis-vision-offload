import { describe, expect, it } from "vitest";
import { addUtterance, createTranscript, editText, type Heard, mergeSpeakers, moveUtterance, renameSpeaker, splitUtterance, type Transcript } from "./transcript.ts";

// A Transcript (spec speech/07, "Utterances and speakers"): its utterances in
// time order, each with its speaker, numbered as voices first speak, and the
// corrections the page makes to them, which never go back to the server (Q109).

const heard = (stream: string, startMs: number, text: string, voice: Heard["voice"], endMs = startMs + 1_000): Heard => ({ stream, startMs, endMs, text, voice });

const names = (t: Transcript) => t.utterances.map((u) => [u.text, u.speaker === null ? null : t.speakers.find((s) => s.id === u.speaker)?.name]);

describe("an utterance heard", () => {
  it("takes Speaker 1, 2… in the order voices first speak, each stream's voices its own", () => {
    let t = createTranscript(1, { source: "microphone", at: new Date(2026, 9, 25, 15, 42) });
    t = addUtterance(t, heard("s1", 0, "Buongiorno.", { kind: "speaker", index: 3 }));
    t = addUtterance(t, heard("s1", 2_000, "Ciao.", { kind: "speaker", index: 0 }));
    t = addUtterance(t, heard("s1", 4_000, "Iniziamo.", { kind: "speaker", index: 3 }));
    // A new session (a Resume) numbers its voices afresh: its 3 is not the first session's.
    t = addUtterance(t, heard("s2", 6_000, "Eccomi.", { kind: "speaker", index: 3 }));
    expect(names(t)).toEqual([
      ["Buongiorno.", "Speaker 1"],
      ["Ciao.", "Speaker 2"],
      ["Iniziamo.", "Speaker 1"],
      ["Eccomi.", "Speaker 3"],
    ]);
  });

  it("on a call, is Me from the microphone and Others from the shared audio with Speakers off, whatever the session", () => {
    let t = createTranscript(1, { source: "call", at: new Date(2026, 9, 25, 15, 42) });
    t = addUtterance(t, heard("mic-1", 0, "Pronto?", { kind: "me" }));
    t = addUtterance(t, heard("call-1", 1_500, "Sì, ci sono.", { kind: "others" }));
    t = addUtterance(t, heard("mic-2", 3_000, "Bene.", { kind: "me" }));
    expect(names(t)).toEqual([
      ["Pronto?", "Me"],
      ["Sì, ci sono.", "Others"],
      ["Bene.", "Me"],
    ]);
    expect(t.speakers.map((s) => s.name)).toEqual(["Me", "Others"]);
  });

  it("has no speaker without the speakers, and joins the others in time order, whenever it completes", () => {
    let t = createTranscript(1, { source: "call", at: new Date(2026, 9, 25, 15, 42) });
    t = addUtterance(t, heard("mic-1", 5_000, "dopo", { kind: "me" }));
    t = addUtterance(t, heard("call-1", 1_000, "prima", null));
    t = addUtterance(t, heard("call-1", 5_000, "insieme", null));
    expect(t.utterances.map((u) => u.text)).toEqual(["prima", "dopo", "insieme"]);
    expect(t.utterances[0].speaker).toBeNull();
  });
});

describe("a new Transcript", () => {
  it("is titled by its date, its time and its source", () => {
    const at = new Date(2026, 9, 25, 15, 42);
    expect(createTranscript(1, { source: "call", at }).title).toBe("25 Oct 15:42 · Call");
    expect(createTranscript(1, { source: "microphone", at: new Date(2026, 0, 3, 9, 5) }).title).toBe("3 Jan 09:05 · Microphone");
    expect(createTranscript(1, { source: "file", at, fileName: "standup.m4a" }).title).toBe("25 Oct 15:42 · standup.m4a");
  });
});

/** A room's Transcript: Speaker 1 says two things, Speaker 2 one. */
function room(): Transcript {
  let t = createTranscript(1, { source: "microphone", at: new Date(2026, 9, 25, 15, 42) });
  t = addUtterance(t, heard("s1", 0, "Apriamo la riunione.", { kind: "speaker", index: 0 }, 2_000));
  t = addUtterance(t, heard("s1", 2_500, "Io ho finito ieri.", { kind: "speaker", index: 1 }, 4_000));
  t = addUtterance(t, heard("s1", 5_000, "Bene, avanti.", { kind: "speaker", index: 0 }, 6_000));
  return t;
}

const id = (t: Transcript, name: string) => t.speakers.find((s) => s.name === name)!.id;

describe("the corrections (Q109)", () => {
  it("rename names every utterance of the speaker, past and future", () => {
    let t = renameSpeaker(room(), id(room(), "Speaker 1"), "  Ada ");
    t = addUtterance(t, heard("s1", 7_000, "Chiudo io.", { kind: "speaker", index: 0 }));
    expect(names(t).map(([, n]) => n)).toEqual(["Ada", "Speaker 2", "Ada", "Ada"]);
    // A blank name keeps the one it had.
    expect(renameSpeaker(t, id(t, "Ada"), " ").speakers.map((s) => s.name)).toEqual(["Ada", "Speaker 2"]);
  });

  it("move takes one utterance to another speaker, or to a new one, and nothing else", () => {
    const t = room();
    const moved = moveUtterance(t, t.utterances[2].id, id(t, "Speaker 2"));
    expect(names(moved).map(([, n]) => n)).toEqual(["Speaker 1", "Speaker 2", "Speaker 2"]);
    const fresh = moveUtterance(t, t.utterances[1].id, "new");
    expect(names(fresh).map(([, n]) => n)).toEqual(["Speaker 1", "Speaker 3", "Speaker 1"]);
    // The channel still speaks as before: a move is one-off.
    const after = addUtterance(moved, heard("s1", 9_000, "Ancora io.", { kind: "speaker", index: 0 }));
    expect(names(after).at(-1)![1]).toBe("Speaker 1");
  });

  it("merge makes two speakers one label, and the future utterances of either take it", () => {
    const t = mergeSpeakers(room(), id(room(), "Speaker 2"), id(room(), "Speaker 1"));
    expect(t.speakers.map((s) => s.name)).toEqual(["Speaker 1"]);
    expect(names(t).map(([, n]) => n)).toEqual(["Speaker 1", "Speaker 1", "Speaker 1"]);
    const later = addUtterance(t, heard("s1", 9_000, "Di nuovo io.", { kind: "speaker", index: 1 }));
    expect(names(later).at(-1)![1]).toBe("Speaker 1");
    // The next new voice is not a second Speaker 2.
    const third = addUtterance(later, heard("s1", 11_000, "Arrivo ora.", { kind: "speaker", index: 2 }));
    expect(names(third).at(-1)![1]).toBe("Speaker 3");
  });

  it("split cuts an utterance's text at the cursor, the second half to a chosen or new speaker, its time proportional and approximate (Q98)", () => {
    let t = createTranscript(1, { source: "microphone", at: new Date(2026, 9, 25, 15, 42) });
    t = addUtterance(t, heard("s1", 10_000, "Ho finito. Anch'io!", { kind: "speaker", index: 0 }, 13_800));
    const [whole] = t.utterances;
    const split = splitUtterance(t, whole.id, 10, "new");
    expect(split.utterances.map(({ text, startMs, endMs, approxStart, approxEnd }) => ({ text, startMs, endMs, approxStart, approxEnd }))).toEqual([
      // 10 of 19 characters: 10 s + 3.8 s × 10 / 19 = 12 s.
      { text: "Ho finito.", startMs: 10_000, endMs: 12_000, approxStart: undefined, approxEnd: true },
      { text: "Anch'io!", startMs: 12_000, endMs: 13_800, approxStart: true, approxEnd: undefined },
    ]);
    expect(names(split).map(([, n]) => n)).toEqual(["Speaker 1", "Speaker 2"]);
    expect(splitUtterance(t, whole.id, 10, id(t, "Speaker 1")).utterances.map((u) => u.speaker)).toEqual([whole.speaker, whole.speaker]);
    // At either end there is nothing to split.
    expect(splitUtterance(t, whole.id, 0, "new")).toBe(t);
    expect(splitUtterance(t, whole.id, 19, "new")).toBe(t);
  });

  it("edits an utterance's text", () => {
    const t = room();
    expect(editText(t, t.utterances[1].id, "Io ho finito ieri sera.").utterances[1].text).toBe("Io ho finito ieri sera.");
  });
});
