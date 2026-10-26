import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { SECURE_CONTEXT, TURNED_OFF } from "../speech/availability.ts";
import { Deck, deckHint, HEADPHONES, startBlocked, Strip, stripStatus, timeLeft } from "./Deck.tsx";
import { IDLE, type ListeningState } from "./listening.ts";
import { Sheet, speakerColor } from "./Sheet.tsx";
import { startList } from "./sessions.ts";
import { fileStem, transcriptSummary } from "./TranscribeView.tsx";
import { addUtterance, createTranscript, sessionsWithSpeakers, splitUtterance, type Transcript } from "./transcript.ts";
import type { TranscribeActions, TranscribeState } from "./useTranscribe.ts";

// The Transcribe view drawn in its states (spec speech/07): the sheet, the
// deck and its strip, the list's rows.

const none = () => {};
const actions: TranscribeActions = {
  select: none,
  newTranscript: none,
  remove: none,
  rename: none,
  start: none,
  pause: none,
  resume: none,
  stop: none,
  setContext: none,
  setSpeakers: none,
  setHoldMic: none,
  setSilence: none,
  renameSpeaker: none,
  moveUtterance: none,
  mergeSpeakers: none,
  splitUtterance: none,
  editText: none,
};

const at = (state: Partial<ListeningState>): ListeningState => ({ ...IDLE, ...state });

function room(): Transcript {
  let t = createTranscript(1, { source: "microphone", at: new Date(2026, 9, 25, 15, 42) });
  t = addUtterance(t, { stream: "mic-1", startMs: 4_000, endMs: 8_000, text: "Apriamo. Io ho finito ieri.", voice: { kind: "speaker", index: 0 } });
  return t;
}

const sheet = (t: Transcript, listening: ListeningState = IDLE, here = false) =>
  renderToStaticMarkup(<Sheet transcript={t} live={here ? listening.live : []} phase={listening.phase} call={listening.source === "call"} here={here} actions={actions} />);

describe("the sheet", () => {
  it("hangs each utterance on the spine: its mm:ss, its speaker as a menu in its hue (a move, a new speaker), its text", () => {
    const html = sheet(room());
    expect(html).toContain(">00:04<");
    expect(html).toMatch(/<select aria-label="Speaker of the utterance at 00:04"[^>]*style="color:var\(--speaker-1\)"/);
    expect(html).toContain('<option value="1" selected="">Speaker 1</option>');
    expect(html).toContain('<option value="new">New speaker</option>');
    expect(html).toContain("Apriamo. Io ho finito ieri.");
    expect(html).toContain('value="25 Oct 15:42 · Microphone"');
    expect(html).toContain("Started 25 Oct 2026, 15:42");
  });

  it("marks a split's guessed start with ~ (Q98)", () => {
    const t = room();
    const html = sheet(splitUtterance(t, t.utterances[0].id, 9, "new"));
    expect(html).toContain(">~00:05<");
  });

  it("shows the utterances under way at the foot, the tail light, the spine burning while it listens (Q82)", () => {
    const listening = at({ phase: "listening", transcript: 1, live: [{ key: "mic-1:item_2", startMs: 9_000, stable: "Bene", tail: " avanti", lane: "mic" }] });
    const html = sheet(room(), listening, true);
    expect(html).toContain(">00:09<");
    expect(html).toMatch(/<span class="text-ink">Bene<\/span><span class="text-ash"> avanti<\/span>/);
    expect(html).toContain("spine-live");
    // Another Transcript's listening shows nothing here.
    expect(sheet(room(), listening, false)).not.toContain("avanti");
  });

  it("names a call's utterance under way by its side before its speaker is known", () => {
    const listening = at({ phase: "listening", source: "call", transcript: 1, live: [{ key: "mic-1:a", startMs: 0, stable: "", tail: "Pronto", lane: "mic" }] });
    expect(sheet(createTranscript(1), listening, true)).toMatch(/text-ash">Me<\/span>/);
  });

  it("invites a start while nothing is heard", () => {
    expect(sheet(createTranscript(1))).toContain("Nothing heard yet.");
  });
});

describe("the speakers' hues", () => {
  it("are Me in ink, Others in ash, and six for the voices", () => {
    expect(speakerColor({ id: 3, name: "Me", role: "me" })).toBe("var(--ink)");
    expect(speakerColor({ id: 4, name: "Others", role: "others" })).toBe("var(--ash)");
    expect(speakerColor({ id: 7, name: "Speaker 5", role: "voice" })).toBe("var(--speaker-1)");
  });
});

describe("the deck", () => {
  const state = (patch: Partial<TranscribeState> = {}): TranscribeState => {
    const list = startList();
    return { list, active: list.transcripts[0], context: "", speakers: true, speakersUnavailable: null, holdMic: true, storeError: null, ...patch };
  };
  const deck = (s: TranscribeState, listening: ListeningState = IDLE) =>
    renderToStaticMarkup(<Deck state={s} listening={listening} actions={actions} speech={{ available: true }} secure here />);

  it("offers the context, the three sources, the Speakers toggle on by default, and Start, or Continue on a Transcript heard in", () => {
    const html = deck(state());
    expect(html).toContain(">Context<");
    for (const source of ["Microphone", "Call", "File"]) expect(html).toContain(`>${source}</button>`);
    expect(html).toMatch(/role="switch" aria-checked="true"/);
    expect(html).toContain(">Start</button>");
    expect(deck(state({ active: room() }))).toContain(">Continue</button>");
  });

  it("strikes the Speakers toggle with the diarizer's reason (Q112)", () => {
    const html = deck(state({ speakers: false, speakersUnavailable: "Speakers are turned off on this server (`diarizer.model = off`)." }));
    expect(html).toContain("line-through");
    expect(html).toContain('aria-disabled="true"');
    expect(html).toContain("Speakers are turned off on this server");
  });

  it("offers to hold the mic on a call (Q121), and says when it is held", () => {
    const call = deck(state(), at({ phase: "listening", source: "call", transcript: 1, held: true, levels: { mic: [0], shared: [0.6] } }));
    expect(call).toMatch(/role="switch" aria-checked="true"[^>]*>.*Hold my mic while they speak/);
    expect(call).toMatch(/Mic<span class="ml-1 text-\[#ff8a4c\]">held<\/span>/);
    expect(deck(state())).not.toContain("Hold my mic");
  });

  it("is the kiln strip while a listening runs", () => {
    const html = deck(state(), at({ phase: "listening", transcript: 1, clockMs: 61_000, levels: { mic: [0, 0.5] } }));
    expect(html).toContain(">Pause<");
    expect(html).toContain(">Stop<");
    expect(html).toContain(">01:01<");
    expect(html).not.toContain(">Start</button>");
  });
});

describe("the hint line", () => {
  it("says what the listening said, else why Start cannot, else headphones before a call", () => {
    const notice = { text: "x", tone: "fault" as const };
    expect(deckHint({ notice, idle: true, blocked: "y", source: "call" })).toBe(notice);
    expect(deckHint({ notice: null, idle: true, blocked: TURNED_OFF, source: "call" })).toEqual({ text: TURNED_OFF, tone: "ash" });
    expect(deckHint({ notice: null, idle: true, blocked: null, source: "call" })).toEqual({ text: HEADPHONES, tone: "ash" });
    expect(deckHint({ notice: null, idle: true, blocked: null, source: "microphone" })).toBeNull();
    expect(deckHint({ notice: null, idle: false, blocked: null, source: "call" })).toBeNull();
  });

  it("blocks a microphone or a call outside a secure context, a file never for that; and each on an unavailable speech model", () => {
    expect(startBlocked("microphone", { secure: false })).toBe(SECURE_CONTEXT);
    expect(startBlocked("file", { secure: false })).toBeNull();
    expect(startBlocked("file", { secure: true, speech: { available: false, reason: "the speech model is turned off (`speech.model` is `off`)" } })).toBe(TURNED_OFF);
  });
});

describe("the strip", () => {
  const strip = (listening: ListeningState) => renderToStaticMarkup(<Strip listening={listening} here actions={actions} />);

  it("shows a file's progress and the time left in place of the levels", () => {
    const html = strip(at({ phase: "listening", file: { name: "talk.mp3", fraction: 0.42, leftMs: 95_000 } }));
    expect(html).toContain('role="progressbar"');
    expect(html).toContain('aria-valuenow="42"');
    expect(html).toContain("about 2 min left");
    expect(html).toContain("Transcribing talk.mp3");
  });

  it("offers Resume while paused, Stop now while finishing, and Cancel while a busy server is waited on", () => {
    expect(strip(at({ phase: "paused" }))).toContain(">Resume<");
    expect(strip(at({ phase: "finishing" }))).toContain(">Stop now<");
    expect(strip(at({ phase: "waiting" }))).toContain(">Cancel<");
    expect(stripStatus(at({ phase: "finishing" }))).toBe("Finishing the last words");
    expect(timeLeft(40_000)).toBe("about 40 s left");
  });
});

describe("the Transcripts list", () => {
  it("says how many utterances each holds, or that it is the one listening", () => {
    const t = room();
    expect(transcriptSummary(t, IDLE)).toBe("1 utterance");
    expect(transcriptSummary(createTranscript(2), IDLE)).toBe("Nothing heard yet");
    expect(transcriptSummary(t, at({ phase: "paused", transcript: 1 }))).toBe("Paused");
  });

  it("names a download after the title, and tells a Transcript heard over more than one session", () => {
    expect(fileStem("25 Oct 15:42 · Call")).toBe("25-Oct-15-42-Call");
    const t = room();
    expect(sessionsWithSpeakers(t)).toBe(1);
    expect(sessionsWithSpeakers(addUtterance(t, { stream: "mic-2", startMs: 9_000, endMs: 9_500, text: "Eccomi.", voice: { kind: "speaker", index: 0 } }))).toBe(2);
  });
});
