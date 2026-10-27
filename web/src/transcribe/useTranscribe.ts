import { useEffect, useMemo, useRef, useState } from "react";
import { keyRequired } from "../api/auth.ts";
import { STORED_FLAGS, useStoredFlag } from "../app/storedFlag.ts";
import type { SpeechStatus } from "../api/model.ts";
import { speakersUnavailable } from "../speech/availability.ts";
import { captureStream, pickSharedAudio, startCapture } from "../speech/capture.ts";
import { openListening } from "../speech/realtime.ts";
import { mergeLoaded } from "../store/docs.ts";
import { pageStore } from "../store/index.ts";
import { useStoredSessions } from "../store/useStoredSessions.ts";
import { decodeFile } from "./file.ts";
import { filePace } from "./filePace.ts";
import { chooseUtteranceSilence, utteranceSilence } from "./silence.ts";
import { chooseNoteKind, kindOf, type NoteKind, noNotes } from "./notes.ts";
import { type CallContext, EMPTY_CONTEXT, told } from "./context.ts";
import { type NotesHandle, NotesWriter } from "./notesWriter.ts";
import { Analyst, type AnalystHandle } from "./analyst.ts";
import { accept, analysisOffers, dismiss, dropTerm, languageOffer, suggest } from "./suggestions.ts";
import { ListeningController, type ListeningHandle, type ListeningSource } from "./listening.ts";
import { activeOf, freeTranscriptId, openTranscript, removeTranscript, startList, type TranscriptList, updateTranscript } from "./sessions.ts";
import { restoreTranscripts, TRANSCRIBE_SESSIONS, toStored, worthKeeping } from "./stored.ts";
import { addUtterance, beginListening, clockOf, editText, mergeSpeakers, moveUtterance, renameSpeaker, splitUtterance, type Transcript } from "./transcript.ts";

// The Transcribe view's state (spec speech/07), held by the page and not by
// the view, as the Image view's is: the header shows a listening from every
// view, and a listening goes on while the user looks at another one. The
// Transcripts are kept in the page's store (stored.ts); the listening is one
// ListeningController, whose completed utterances land in the Transcript it
// started on, whichever one is open by then.

export type TranscribeState = {
  list: TranscriptList;
  active: Transcript;
  /**
   * The open Transcript's context as it is being edited, `null` when it is not (Q128): its own, applied, is
   * `active.context`. A Start applies it first.
   */
  draft: CallContext | null;
  /** The Speakers toggle, per listening, on by default (Q112). */
  speakers: boolean;
  /** Why the Speakers toggle is struck, when it is: the diarizer the server reports. */
  speakersUnavailable: string | null;
  /** On a call, hold the mic while the far side speaks (Q121): this browser's, on at first. */
  holdMic: boolean;
  /** Keep a listening's notes up as it goes (Q126): this browser's, off at first. */
  notesLive: boolean;
  /** Why no notes can be written now (notes.ts), `null` while a text model serves; the suggestions' model too. */
  notesBlocked: string | null;
  /** Suggest while listening (Q132): this browser's, on at first. */
  suggestionsOn: boolean;
  /** What keeping the Transcripts in this browser ran into. */
  storeError: string | null;
};

export type TranscribeActions = {
  select: (id: number) => void;
  newTranscript: () => void;
  remove: (id: number) => void;
  rename: (id: number, title: string) => void;
  /** Start, or Continue on a Transcript that holds utterances (Q97); inside the press. */
  start: (source: ListeningSource) => void;
  pause: () => void;
  resume: () => void;
  stop: () => void;
  /** Edits the open Transcript's context: a draft, from what it was told. */
  editContext: () => void;
  changeContext: (draft: CallContext) => void;
  /** Applies the draft: the Transcript keeps it, and a listening into it is told from the next utterance on. */
  applyContext: () => void;
  cancelContext: () => void;
  setSpeakers: (on: boolean) => void;
  setHoldMic: (on: boolean) => void;
  /** The silence that ends an utterance (Q122): the viewer's, and every session's at once. */
  setSilence: (silenceMs: number) => void;
  renameSpeaker: (speaker: number, name: string) => void;
  moveUtterance: (utterance: number, to: number | "new") => void;
  mergeSpeakers: (from: number, into: number) => void;
  splitUtterance: (utterance: number, at: number, to: number | "new") => void;
  editText: (utterance: number, text: string) => void;
  /** The open Transcript's kind of notes, and this browser's next default. */
  setNoteKind: (kind: NoteKind) => void;
  /** Writes the open Transcript's notes now. */
  writeNotes: () => void;
  stopNotes: () => void;
  setNotesLive: (on: boolean) => void;
  /** The open Transcript's suggestion accepted; `always`, and every later one of its kind (Q132). */
  acceptSuggestion: (id: number, always?: boolean) => void;
  dismissSuggestion: (id: number) => void;
  /** Takes a term out of a terms suggestion, never to be offered again. */
  dropSuggestedTerm: (id: number, term: string) => void;
  /** Reads the open Transcript for suggestions now. */
  suggestNow: () => void;
  stopSuggesting: () => void;
  setSuggestionsOn: (on: boolean) => void;
};

/** The page's helpers on the text model: the notes' writer and the look for suggestions. */
export type Assistants = { notes: NotesHandle; analyst: AnalystHandle };

/** The text model, as the notes need it: why none serves, and its context. */
export type TextModel = { blocked: string | null; contextLimit: number | null };

/** How often keeping the notes up looks whether it is time. */
const NOTES_LOOK_MS = 5_000;

export function useTranscribe({
  speech,
  text,
  onUnavailable,
}: {
  speech?: SpeechStatus;
  text: TextModel;
  onUnavailable: () => void;
}): [TranscribeState, TranscribeActions, ListeningHandle, Assistants] {
  const [list, setList] = useState<TranscriptList>(() => startList());
  /** The draft, and the Transcript it is of. */
  const [drafting, setDrafting] = useState<{ id: number; context: CallContext } | null>(null);
  const draftNow = useRef(drafting);
  draftNow.current = drafting;
  const [speakers, setSpeakers] = useState(true);
  const [holdMic, storeHoldMic] = useStoredFlag(STORED_FLAGS.transcribeHoldMic, true);
  const holdMicRef = useRef(holdMic);
  holdMicRef.current = holdMic;
  const listNow = useRef(list);
  listNow.current = list;
  const latest = useRef({ onUnavailable, text });
  latest.current = { onUnavailable, text };
  const [notesLive, storeNotesLive] = useStoredFlag(STORED_FLAGS.transcribeNotesLive, false);
  const [suggestionsOn, storeSuggestionsOn] = useStoredFlag(STORED_FLAGS.transcribeSuggestions, true);

  const [controller] = useState(
    () =>
      new ListeningController({
        open: (options, onEvent) => openListening(options, onEvent),
        captureMic: (onChunk) => startCapture(onChunk),
        pickShared: (onEnded) => pickSharedAudio(onEnded),
        captureShared: (stream, onChunk) => captureStream(stream, onChunk),
        decode: (file) => decodeFile(file),
        filePaceSeconds: filePace,
        onUtterance: (id, utterance) => setList((l) => updateTranscript(l, id, (t) => addUtterance(t, utterance))),
        onClock: (id, clockMs) => setList((l) => updateTranscript(l, id, (t) => ({ ...t, clockMs: Math.max(t.clockMs, clockMs) }))),
        onKeyRequired: keyRequired,
        onUnavailable: () => latest.current.onUnavailable(),
      }),
  );
  useEffect(() => () => controller.dispose(), [controller]);

  const [writer] = useState(
    () =>
      new NotesWriter({
        onWritten: (id, written) => setList((l) => updateTranscript(l, id, (t) => ({ ...t, notes: { ...(t.notes ?? noNotes(written.writtenAs ?? "general")), ...written } }))),
      }),
  );
  useEffect(() => () => writer.stop(), [writer]);

  const [analyst] = useState(
    () =>
      new Analyst({
        onFound: (id, found) => setList((l) => updateTranscript(l, id, (t) => analysisOffers(t, found).reduce((x, offer) => suggest(x, offer, Date.now()), t))),
      }),
  );
  useEffect(() => () => analyst.stop(), [analyst]);

  // Keeping up while listening, a look every few seconds: the languages the server heard, then, while a text model
  // serves, the notes and the look for suggestions, each on its own cadence.
  useEffect(() => {
    if (!notesLive && !suggestionsOn) return;
    const look = () => {
      const { text: model } = latest.current;
      const listening = controller.getState();
      const on = listening.phase === "idle" ? null : listening.transcript;
      const find = (id: number) => listNow.current.transcripts.find((t) => t.id === id);
      const t = on === null ? undefined : find(on);
      const offer = suggestionsOn && t ? languageOffer(t) : null;
      if (t && offer && suggest(t, offer, Date.now()) !== t) setList((l) => updateTranscript(l, t.id, (x) => suggest(x, offer, Date.now())));
      if (model.blocked !== null) return;
      if (notesLive) void writer.look(on, find, kindOf, model.contextLimit);
      if (suggestionsOn) void analyst.look(on, find, model.contextLimit);
    };
    const timer = setInterval(look, NOTES_LOOK_MS);
    return () => clearInterval(timer);
  }, [notesLive, suggestionsOn, controller, writer, analyst]);

  const kept = useStoredSessions({
    store: pageStore(),
    collection: TRANSCRIBE_SESSIONS,
    view: "transcribe",
    items: list.transcripts,
    activeId: list.activeId,
    toStored,
    worthKeeping,
    adopt: async (values, activeKey) => {
      const restored = restoreTranscripts(values, freeTranscriptId(listNow.current));
      const reopened = [...restored.stamps].find(([, stamp]) => stamp.key === activeKey)?.[0];
      setList((l) => {
        const merged = mergeLoaded(l.transcripts, l.activeId, restored.transcripts, reopened, worthKeeping);
        return { transcripts: merged.items, activeId: merged.activeId };
      });
      return { loaded: restored.transcripts, stamps: restored.stamps };
    },
  });

  const struck = speech ? speakersUnavailable(speech) : null;
  // A listening is told what its Transcript holds, applied or renamed since: each session sends a change alone.
  useEffect(() => {
    const listening = controller.getState();
    if (listening.phase === "idle") return;
    const t = list.transcripts.find((x) => x.id === listening.transcript);
    if (t) controller.setContext(told(t.context, t.speakers));
  }, [list, controller]);

  // Read at the press: the toggle as struck by the server's word.
  const speakersRef = useRef(speakers && struck === null);
  speakersRef.current = speakers && struck === null;
  const active = activeOf(list);
  const change = (id: number, apply: (t: Transcript) => Transcript) => setList((l) => updateTranscript(l, id, apply));
  /** A correction is the open Transcript's. */
  const correct = (apply: (t: Transcript) => Transcript) => setList((l) => updateTranscript(l, l.activeId, apply));

  const actions = useMemo<TranscribeActions>(
    () => ({
      select: (id) => setList((l) => ({ ...l, activeId: id })),
      newTranscript: () => setList((l) => openTranscript(l, freeTranscriptId(l))),
      // The Transcript a listening appends to stays until the listening ends.
      remove: (id) => controller.getState().transcript !== id && setList((l) => removeTranscript(l, id, freeTranscriptId(l))),
      rename: (id, title) => {
        const trimmed = title.trim();
        if (trimmed !== "") change(id, (t) => ({ ...t, title: trimmed }));
      },
      start: (source) => {
        if (controller.getState().phase !== "idle") return;
        const pending = draftNow.current;
        const opened = activeOf(listNow.current);
        // A context being edited is applied by the Start: what it shows is what the listening is told.
        const t = pending?.id === opened.id ? { ...opened, context: { ...pending.context, appliedAt: Date.now() } } : opened;
        if (pending?.id === opened.id) setDrafting(null);
        const start = { source: source.kind, at: new Date(), ...(source.kind === "file" ? { fileName: source.file.name } : {}) };
        setList((l) => updateTranscript(l, t.id, (current) => beginListening({ ...current, context: t.context }, start)));
        controller.start({
          transcript: t.id,
          clockMs: clockOf(t),
          source,
          speakers: speakersRef.current,
          context: told(t.context, t.speakers),
          silenceMs: utteranceSilence(),
          holdMic: holdMicRef.current,
        });
      },
      pause: () => controller.pause(),
      resume: () => controller.resume(),
      stop: () => void controller.stop(),
      editContext: () => {
        const t = activeOf(listNow.current);
        setDrafting({ id: t.id, context: t.context ?? EMPTY_CONTEXT });
      },
      changeContext: (draft) => setDrafting({ id: listNow.current.activeId, context: draft }),
      applyContext: () => {
        const pending = draftNow.current;
        if (!pending) return;
        setDrafting(null);
        change(pending.id, (t) => ({ ...t, context: { ...pending.context, appliedAt: Date.now() } }));
      },
      cancelContext: () => setDrafting(null),
      setSpeakers: (on) => setSpeakers(on),
      setHoldMic: (on) => {
        holdMicRef.current = on;
        storeHoldMic(on);
        controller.setHoldMic(on);
      },
      setSilence: (silenceMs) => {
        chooseUtteranceSilence(silenceMs);
        controller.setSilence(silenceMs);
      },
      renameSpeaker: (speaker, name) => correct((t) => renameSpeaker(t, speaker, name)),
      moveUtterance: (utterance, to) => correct((t) => moveUtterance(t, utterance, to)),
      mergeSpeakers: (from, into) => correct((t) => mergeSpeakers(t, from, into)),
      splitUtterance: (utterance, at, to) => correct((t) => splitUtterance(t, utterance, at, to)),
      editText: (utterance, text) => correct((t) => editText(t, utterance, text)),
      setNoteKind: (kind) => {
        chooseNoteKind(kind);
        correct((t) => ({ ...t, notes: { ...(t.notes ?? noNotes(kind)), kind } }));
      },
      writeNotes: () => {
        const { text: model } = latest.current;
        if (model.blocked !== null) return;
        const t = activeOf(listNow.current);
        void writer.write(t, kindOf(t), model.contextLimit);
      },
      stopNotes: () => writer.stop(),
      setNotesLive: (on) => storeNotesLive(on),
      acceptSuggestion: (id, always = false) => correct((t) => accept(t, id, Date.now(), { always })),
      dismissSuggestion: (id) => correct((t) => dismiss(t, id)),
      dropSuggestedTerm: (id, term) => correct((t) => dropTerm(t, id, term)),
      suggestNow: () => {
        const { text: model } = latest.current;
        if (model.blocked === null) void analyst.analyze(activeOf(listNow.current), model.contextLimit);
      },
      stopSuggesting: () => analyst.stop(),
      setSuggestionsOn: (on) => storeSuggestionsOn(on),
    }),
    // The stored flags' setters are stable for their keys.
    [controller, writer, analyst, storeHoldMic, storeNotesLive, storeSuggestionsOn],
  );

  return [
    {
      list,
      active,
      draft: drafting?.id === active.id ? drafting.context : null,
      speakers: speakers && struck === null,
      speakersUnavailable: struck,
      holdMic,
      notesLive,
      notesBlocked: text.blocked,
      suggestionsOn,
      storeError: kept.error,
    },
    actions,
    controller,
    { notes: writer, analyst },
  ];
}
