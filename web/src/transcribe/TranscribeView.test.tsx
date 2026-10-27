import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { SECURE_CONTEXT, TURNED_OFF } from "../speech/availability.ts";
import { ContextCard } from "./Context.tsx";
import { Deck, deckHint, HEADPHONES, startBlocked, Strip, stripStatus, timeLeft } from "./Deck.tsx";
import { IDLE, type ListeningState } from "./listening.ts";
import { NotesPanel, notesButton } from "./Notes.tsx";
import type { AnalystState } from "./analyst.ts";
import { SuggestionsPanel, suggestionText } from "./Suggestions.tsx";
import { suggest } from "./suggestions.ts";
import type { NotesWriterState } from "./notesWriter.ts";
import { Heading, Sheet, speakerColor } from "./Sheet.tsx";
import { startList } from "./sessions.ts";
import { fileStem, TranscribeView, transcriptSummary } from "./TranscribeView.tsx";
import { addUtterance, createTranscript, fixUtterances, sessionsWithSpeakers, splitUtterance, type Transcript } from "./transcript.ts";
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
  editContext: none,
  changeContext: none,
  applyContext: none,
  cancelContext: none,
  setSpeakers: none,
  setHoldMic: none,
  setSilence: none,
  renameSpeaker: none,
  moveUtterance: none,
  mergeSpeakers: none,
  splitUtterance: none,
  editText: none,
  setNoteKind: none,
  writeNotes: none,
  stopNotes: none,
  setNotesLive: none,
  acceptSuggestion: none,
  dismissSuggestion: none,
  dropSuggestedTerm: none,
  suggestNow: none,
  stopSuggesting: none,
  setSuggestionsOn: none,
  undoFixes: none,
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
  });

  it("is headed, above it, by the title, the start and the exports (Q129)", () => {
    const html = renderToStaticMarkup(
      <Heading transcript={room()} onRename={none}>
        <button type="button">Copy</button>
      </Heading>,
    );
    expect(html).toContain('value="25 Oct 15:42 · Microphone"');
    expect(html).toContain("Started 25 Oct 2026, 15:42");
    expect(html).toContain(">Copy</button>");
  });

  it("marks the words a fix put in, and offers to undo it (Q133)", () => {
    const html = sheet(fixUtterances(room(), "Apriamo", "Apriremo", [1]));
    expect(html).toContain("decoration-dotted");
    expect(html).toMatch(/<mark[^>]*>Apriremo<\/mark>/);
    expect(html).toContain('title="Heard as: Apriamo. Io ho finito ieri."');
    expect(html).toContain(">Undo fix<");
    expect(sheet(room())).not.toContain("Undo fix");
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

const state = (patch: Partial<TranscribeState> = {}): TranscribeState => {
  const list = startList();
  return { list, active: list.transcripts[0], draft: null, speakers: true, speakersUnavailable: null, holdMic: true, notesLive: false, notesBlocked: null, suggestionsOn: true, storeError: null, ...patch };
};

describe("the deck", () => {
  const deck = (s: TranscribeState, listening: ListeningState = IDLE) =>
    renderToStaticMarkup(<Deck state={s} listening={listening} actions={actions} speech={{ available: true }} secure here onEditContext={none} />);

  it("offers the three sources, the Speakers toggle on by default, and Start, or Continue on a Transcript heard in", () => {
    const html = deck(state());
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

describe("the context (Q128)", () => {
  const told = (t: Transcript): Transcript => ({
    ...t,
    context: { languages: { mode: "free", codes: [] }, topic: "Supporto IT", terms: ["PostgreSQL", "VPN"], other: "", appliedAt: new Date(2026, 9, 27, 15, 44).getTime() },
  });
  const card = (t: Transcript, draft: TranscribeState["draft"] = null, live = false) =>
    renderToStaticMarkup(<ContextCard transcript={t} draft={draft} live={live} actions={actions} />);

  it("reads, applied, as a record with when it was applied and what it takes of the server's 512 tokens, and offers Edit", () => {
    const html = card(told(room()));
    expect(html).toContain(">Supporto IT<");
    expect(html).toContain(">PostgreSQL<");
    expect(html).toMatch(/Applied at 15:44 · about \d+ of 512 tokens/);
    expect(html).toContain(">Edit<");
    expect(html).not.toContain(">Apply<");
  });

  it("is a draft once edited, applied by Apply, from the next utterance while listening", () => {
    const t = told(room());
    const html = card(t, t.context!, true);
    expect(html).toContain('value="Supporto IT"');
    expect(html).toContain('aria-label="Remove PostgreSQL"');
    expect(html).toContain(">Apply<");
    expect(html).toContain("Holds from the next utterance.");
    expect(card(t, t.context!, false)).toContain("A Start applies it too.");
  });

  it("detects the languages by default and forces one on request, saying what each does (Q130)", () => {
    const t = told(room());
    const detect = card(t, { ...t.context!, languages: { mode: "free", codes: ["it", "en"] } });
    expect(detect).toMatch(/type="radio"[^>]*checked=""[^>]*\/><span[^>]*>Detect</);
    expect(detect).toContain('aria-label="Remove English"');
    expect(detect).toContain("The model detects each utterance");
    const force = card(t, { ...t.context!, languages: { mode: "force", codes: ["it"] } });
    expect(force).toContain("Every utterance is transcribed as Italian");
    expect(card({ ...t, context: { ...t.context!, languages: { mode: "force", codes: ["it"] } } })).toContain("Italian (forced)");
  });

  it("offers Add when nothing is told, and the deck says so in a line with Edit", () => {
    expect(card(room())).toContain(">Add<");
    const deckLine = renderToStaticMarkup(<Deck state={state({ active: told(room()) })} listening={IDLE} actions={actions} secure here onEditContext={none} />);
    expect(deckLine).toContain("Supporto IT · 2 terms");
    expect(renderToStaticMarkup(<Deck state={state({ draft: told(room()).context! })} listening={IDLE} actions={actions} secure here onEditContext={none} />)).toContain(
      "Being edited: a Start applies it.",
    );
  });
});

describe("the suggestions (Q132)", () => {
  const idle: AnalystState = { looking: null, failed: null };
  const panel = (t: Transcript, looking: AnalystState = idle, patch: Partial<TranscribeState> = {}) =>
    renderToStaticMarkup(<SuggestionsPanel transcript={t} state={state(patch)} actions={actions} looking={looking} />);

  it("wait as cards with what they propose, why, their verb, Always and Dismiss", () => {
    let t = suggest(room(), { kind: "languages", languages: { mode: "force", codes: ["it"] }, why: "2 short utterances came out as Thai and Hindi, among 9 in Italian." }, 1);
    t = suggest(t, { kind: "terms", terms: ["PostgreSQL", "VPN"] }, 2);
    const html = panel(t);
    expect(html).toContain(">Force Italian<");
    expect(html).toContain("came out as Thai and Hindi");
    expect(html).toContain('aria-label="Not PostgreSQL"');
    expect(html).toContain(">Add them<");
    expect(html.match(/>Always</g)).toHaveLength(2);
    expect(html.match(/>Dismiss</g)).toHaveLength(2);
  });

  it("of a fix says where, and that it can be undone", () => {
    const fixed = suggest(room(), { kind: "fix", from: "Apriamo", to: "Apriremo", utterances: [1] }, 1);
    const html = panel(fixed);
    expect(html).toContain("“Apriamo” is Apriremo");
    expect(html).toContain("Misheard at 00:04.");
    expect(html).toContain(">Fix<");
  });

  it("say what each proposes", () => {
    expect(suggestionText({ id: 1, status: "open", kind: "languages", languages: { mode: "free", codes: ["it", "en"] }, why: "" })).toEqual({ headline: "Detect Italian and English", verb: "Apply" });
    expect(suggestionText({ id: 1, status: "open", kind: "title", title: "27 Oct 15:42 · Assistenza" })).toEqual({ headline: "Rename to “27 Oct 15:42 · Assistenza”", verb: "Rename" });
    expect(suggestionText({ id: 1, status: "open", kind: "topic", topic: "IT" }).verb).toBe("Use");
  });

  it("say when the text model is reading, when they are off, and why a look failed; Suggest now waits for a text model", () => {
    const t = room();
    expect(panel(t, { looking: t.id, failed: null })).toContain("The text model is reading the transcript.");
    expect(panel(t, { looking: t.id, failed: null })).toContain(">Stop<");
    expect(panel(t, idle, { suggestionsOn: false })).toContain("Suggestions while listening are off in Settings.");
    expect(panel(t, { looking: null, failed: { transcript: t.id, message: "ignis answered 500" } })).toContain("No suggestions: ignis answered 500");
    expect(panel(t, idle, { notesBlocked: "No text model serves: notes need one. Load it in the Server view." })).toMatch(/disabled=""[^>]*>Suggest now</);
  });
});

describe("the notes", () => {
  const idle: NotesWriterState = { writing: null, text: "", failed: null };
  const written = (t: Transcript): Transcript => ({
    ...t,
    notes: { kind: "work", text: "## Riassunto\n- Si apre.", writtenAs: "work", covered: 1, writtenAt: new Date(2026, 9, 27, 16, 5).getTime(), fromMs: null },
  });
  const panel = (t: Transcript, writer: NotesWriterState = idle, patch: Partial<TranscribeState> = {}) =>
    renderToStaticMarkup(<NotesPanel transcript={t} state={state(patch)} actions={actions} writer={writer} />);

  it("offers the four kinds, Write notes, and says nothing goes to the model before a press (Q124)", () => {
    const html = panel(room());
    expect(html).toMatch(/role="radio" aria-checked="true"[^>]*><span[^>]*>General<\/span>/);
    expect(html).toContain("Game session");
    expect(html).toContain(">Write notes<");
    expect(html).toContain("Nothing goes to it until you press Write notes");
    expect(html).toMatch(/role="switch" aria-checked="false"[^>]*>.*Keep updating while listening/);
  });

  it("shows the notes written, when and from how much, and offers to update them", () => {
    const html = panel(addUtterance(written(room()), { stream: "mic-1", startMs: 9_000, endMs: 9_500, text: "Poi?", voice: null }));
    expect(html).toContain("<h2>Riassunto</h2>");
    expect(html).toContain("Written at 16:05 from 1 utterance. 1 more since.");
    expect(html).toContain(">Update notes<");
    expect(html).toMatch(/role="radio" aria-checked="true"[^>]*><span[^>]*>Work call<\/span>/);
  });

  it("streams a writing under the heat, the button a Stop", () => {
    const t = written(room());
    const html = panel(t, { writing: t.id, text: "## Argomenti\n- Il budget", failed: null });
    expect(html).toContain("spine-live");
    expect(html).toContain("<h2>Argomenti</h2>");
    expect(html).not.toContain("Riassunto");
    expect(notesButton(t, { writing: t.id, text: "", failed: null }, null)).toEqual({ label: "Stop", disabled: false, stops: true });
  });

  it("cannot write with no text model, nothing heard, or another Transcript's notes under way, and says why it failed", () => {
    const t = room();
    expect(notesButton(t, idle, "No text model serves: notes need one. Load it in the Server view.").disabled).toBe(true);
    expect(notesButton(createTranscript(2), idle, null).disabled).toBe(true);
    expect(notesButton(t, { writing: 9, text: "", failed: null }, null).disabled).toBe(true);
    expect(panel(t, idle, { notesBlocked: "The image model is loaded: notes need a text model. Switch in the Server view." })).toContain("The image model is loaded");
    expect(panel(t, { writing: null, text: "", failed: { transcript: t.id, message: "The model wrote nothing." } })).toContain("The notes were not written: The model wrote nothing.");
  });
});

const idleHandle = { getState: () => IDLE, subscribe: () => () => {} };
const assistants = {
  notes: { getState: () => ({ writing: null, text: "", failed: null }), subscribe: () => () => {} },
  analyst: { getState: (): AnalystState => ({ looking: null, failed: null }), subscribe: () => () => {} },
};

describe("the view", () => {
  it("heads the transcript with its exports, and opens the right panel on the Call tab, beside Notes and Settings (Q129)", () => {
    const html = renderToStaticMarkup(
      <TranscribeView state={state({ active: room() })} actions={actions} listening={idleHandle} assistants={assistants} drawer={null} onDrawer={none} onOpenInChat={none} />,
    );
    expect(html.indexOf('aria-label="Transcript title"')).toBeLessThan(html.indexOf(">Download"));
    expect(html).toContain(">Open in chat<");
    expect(html).toMatch(/role="tab" id="transcribe-tab-call" aria-selected="true"/);
    expect(html).toMatch(/role="tab" id="transcribe-tab-notes" aria-selected="false"/);
    expect(html).toMatch(/role="tab" id="transcribe-tab-settings" aria-selected="false"/);
    expect(html).toContain(">Context<");
    expect(html).toContain('id="transcribe-speakers"');
    expect(html).not.toContain("Utterance ends after");
    expect(html).toContain(">Suggest now<");
    expect(html).not.toContain("suggestions waiting");
  });

  it("rings the bell with the open Transcript's suggestions waiting (Q132)", () => {
    const t = suggest(suggest(room(), { kind: "topic", topic: "Supporto IT" }, 1), { kind: "terms", terms: ["PostgreSQL"] }, 2);
    const html = renderToStaticMarkup(
      <TranscribeView state={state({ active: t })} actions={actions} listening={idleHandle} assistants={assistants} drawer={null} onDrawer={none} onOpenInChat={none} />,
    );
    expect(html).toContain('aria-label="2 suggestions"');
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
