import { useEffect, useMemo, useRef, useState } from "react";
import { keyRequired } from "../api/auth.ts";
import type { SpeechStatus } from "../api/model.ts";
import { speakersUnavailable } from "../speech/availability.ts";
import { captureStream, pickSharedAudio, startCapture } from "../speech/capture.ts";
import { openListening } from "../speech/realtime.ts";
import { mergeLoaded } from "../store/docs.ts";
import { pageStore } from "../store/index.ts";
import { useStoredSessions } from "../store/useStoredSessions.ts";
import { decodeFile } from "./file.ts";
import { filePace } from "./filePace.ts";
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
  /** The context field: sent as the prompt at the start, and by a session.update when changed (Q87). */
  context: string;
  /** The Speakers toggle, per listening, on by default (Q112). */
  speakers: boolean;
  /** Why the Speakers toggle is struck, when it is: the diarizer the server reports. */
  speakersUnavailable: string | null;
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
  setContext: (context: string) => void;
  setSpeakers: (on: boolean) => void;
  renameSpeaker: (speaker: number, name: string) => void;
  moveUtterance: (utterance: number, to: number | "new") => void;
  mergeSpeakers: (from: number, into: number) => void;
  splitUtterance: (utterance: number, at: number, to: number | "new") => void;
  editText: (utterance: number, text: string) => void;
};

export function useTranscribe({ speech, onUnavailable }: { speech?: SpeechStatus; onUnavailable: () => void }): [TranscribeState, TranscribeActions, ListeningHandle] {
  const [list, setList] = useState<TranscriptList>(() => startList());
  const [context, setContextText] = useState("");
  const [speakers, setSpeakers] = useState(true);
  const listNow = useRef(list);
  listNow.current = list;
  const latest = useRef({ onUnavailable });
  latest.current = { onUnavailable };

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
  // Read at the press: the toggle as struck by the server's word, the context as typed.
  const contextRef = useRef(context);
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
        const t = activeOf(listNow.current);
        const start = { source: source.kind, at: new Date(), ...(source.kind === "file" ? { fileName: source.file.name } : {}) };
        setList((l) => updateTranscript(l, t.id, (current) => beginListening(current, start)));
        controller.start({ transcript: t.id, clockMs: clockOf(t), source, speakers: speakersRef.current, context: contextRef.current });
      },
      pause: () => controller.pause(),
      resume: () => controller.resume(),
      stop: () => void controller.stop(),
      setContext: (text) => {
        contextRef.current = text;
        setContextText(text);
        controller.setContext(text);
      },
      setSpeakers: (on) => setSpeakers(on),
      renameSpeaker: (speaker, name) => correct((t) => renameSpeaker(t, speaker, name)),
      moveUtterance: (utterance, to) => correct((t) => moveUtterance(t, utterance, to)),
      mergeSpeakers: (from, into) => correct((t) => mergeSpeakers(t, from, into)),
      splitUtterance: (utterance, at, to) => correct((t) => splitUtterance(t, utterance, at, to)),
      editText: (utterance, text) => correct((t) => editText(t, utterance, text)),
    }),
    [controller],
  );

  return [{ list, active, context, speakers: speakers && struck === null, speakersUnavailable: struck, storeError: kept.error }, actions, controller];
}
