import { useSyncExternalStore } from "react";
import { abortableSleep, isEngineFull } from "../api/busy.ts";
import { streamChat, type StreamOptions, type StreamResult } from "../api/stream.ts";
import { dueForUpdate, type NoteKind, type Notes, notesInput, notesSettings, tidyNotes } from "./notes.ts";
import type { Transcript } from "./transcript.ts";

// The notes' writer (spec speech/07, Q124-Q127): one writing at a time, on
// the text model loaded, with no model named, so a request never switches
// one. Its text streams into the state while it writes; the notes land on the
// Transcript the writing started on (`onWritten`), whichever is open by then,
// and a Stop or a failure leaves the notes there were. A full engine is
// waited out a second at a time, as an agent's request is: the notes are a
// side request on an agent lane, behind the chat. The page holds one, so a
// writing goes on while the view is hidden.

export type NotesWriterState = {
  /** The Transcript whose notes are being written. */
  writing: number | null;
  /** What the writing has written so far. */
  text: string;
  /** The last writing that failed, and why. */
  failed: { transcript: number; message: string } | null;
};

/** What a finished writing puts in a Transcript's notes. */
export type Written = Pick<Notes, "text" | "writtenAs" | "covered" | "writtenAt" | "fromMs">;

export type NotesWriterDeps = {
  onWritten: (transcript: number, written: Written) => void;
  stream?: (options: StreamOptions) => Promise<StreamResult>;
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>;
  now?: () => number;
};

const IDLE: NotesWriterState = { writing: null, text: "", failed: null };

/** How many times a full engine is waited out before the writing says so. */
const BUSY_TRIES = 60;

export class NotesWriter {
  private state: NotesWriterState = IDLE;
  private readonly listeners = new Set<() => void>();
  private abort: AbortController | null = null;
  /** When each Transcript's last writing began: keeping up counts from it. */
  private readonly began = new Map<number, number>();
  /** The Transcript a listening appended to at the last look, and one whose listening ended since, owed its last notes. */
  private followed: number | null = null;
  private ended: number | null = null;

  constructor(private readonly deps: NotesWriterDeps) {}

  getState = (): NotesWriterState => this.state;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  /** Writes `t`'s notes as `kind`, unless a writing is under way. */
  async write(t: Transcript, kind: NoteKind, contextLimit: number | null): Promise<void> {
    if (this.state.writing !== null) return;
    const now = this.deps.now ?? Date.now;
    const stream = this.deps.stream ?? streamChat;
    const sleep = this.deps.sleep ?? abortableSleep;
    const abort = new AbortController();
    this.abort = abort;
    this.began.set(t.id, now());
    this.set({ writing: t.id, text: "", failed: null });
    const { content, fromMs } = notesInput(t, contextLimit);
    const covered = t.utterances.length;
    const request = { settings: notesSettings(kind), turns: [{ role: "user" as const, content }] };
    for (let attempt = 1; ; attempt++) {
      let text = "";
      const result = await stream({
        request,
        signal: abort.signal,
        onEvent: (event) => {
          if (event.kind !== "content" || this.abort !== abort) return;
          text += event.text;
          this.set({ ...this.state, text });
        },
      });
      // Stopped: the Stop said so already.
      if (this.abort !== abort) return;
      if (!result.ok && isEngineFull(result.message) && attempt < BUSY_TRIES) {
        await sleep(1_000, abort.signal);
        continue;
      }
      this.abort = null;
      const notes = result.ok ? tidyNotes(text) : "";
      if (notes === "") {
        this.set({ writing: null, text: "", failed: { transcript: t.id, message: result.ok ? "The model wrote nothing." : result.message } });
        return;
      }
      this.deps.onWritten(t.id, { text: notes, writtenAs: kind, covered, writtenAt: now(), fromMs });
      this.set(IDLE);
      return;
    }
  }

  /** Stops the writing under way; the notes there were stay. */
  stop() {
    if (!this.abort) return;
    this.abort.abort();
    this.abort = null;
    this.set(IDLE);
  }

  /** Writes `t`'s notes if keeping up says it is time (notes.ts); `final` as the listening ends. */
  keepUp(t: Transcript, kind: NoteKind, contextLimit: number | null, { final = false } = {}): Promise<void> {
    const began = this.began.get(t.id);
    const now = (this.deps.now ?? Date.now)();
    const due = dueForUpdate({ notes: t.notes && { ...t.notes, kind }, utterances: t.utterances.length, sinceLastMs: began === undefined ? null : now - began, final });
    return due ? this.write(t, kind, contextLimit) : Promise.resolve();
  }

  /**
   * One look while keeping up, every few seconds: `on` is the Transcript a listening appends to, `null` with none.
   * When its listening ends, the notes are written once more as soon as no writing is under way.
   */
  look(on: number | null, find: (id: number) => Transcript | undefined, kindOf: (t: Transcript) => NoteKind, contextLimit: number | null): Promise<void> {
    if (this.followed !== null && this.followed !== on) this.ended = this.followed;
    this.followed = on;
    if (this.state.writing !== null) return Promise.resolve();
    const final = this.ended !== null;
    const id = this.ended ?? on;
    this.ended = null;
    const t = id === null ? undefined : find(id);
    return t ? this.keepUp(t, kindOf(t), contextLimit, { final }) : Promise.resolve();
  }

  private set(state: NotesWriterState) {
    this.state = state;
    for (const listener of this.listeners) listener();
  }
}

export type NotesHandle = Pick<NotesWriter, "getState" | "subscribe">;

/** The writer's state, for what shows it: the text streams token by token, and only the notes redraw. */
export function useNotesWriter(handle: NotesHandle): NotesWriterState {
  return useSyncExternalStore(handle.subscribe, handle.getState, handle.getState);
}
