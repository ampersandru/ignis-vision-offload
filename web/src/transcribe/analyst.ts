import { useSyncExternalStore } from "react";
import { type AskDeps, askModel, assistRequest, Follower, transcriptInput } from "./assist.ts";
import { ANALYSIS_GAP_MS, ANALYSIS_MAX_TOKENS, ANALYSIS_SETTINGS, ANALYSIS_TASK, type Analysis, dueForAnalysis, readAnalysis } from "./suggestions.ts";
import type { Transcript } from "./transcript.ts";

// The look for suggestions (spec speech/07, Q132): the text model reads a
// Transcript and says its topic, a title and the terms to spell right
// (suggestions.ts), on Suggest now or, while listening, about every minute and
// a half when something new was said, and once more at the end. One look at a
// time, on an agent lane with thinking off, as the notes' writing; a look
// that failed waits the gap before the next. What it found goes to `onFound`
// with the Transcript it read, whichever is open by then.

export type AnalystState = {
  /** The Transcript being read. */
  looking: number | null;
  /** The last look that failed, and why. */
  failed: { transcript: number; message: string } | null;
};

export type AnalystDeps = AskDeps & {
  onFound: (transcript: number, found: Analysis) => void;
  now?: () => number;
};

const IDLE: AnalystState = { looking: null, failed: null };

export class Analyst {
  private state: AnalystState = IDLE;
  private readonly listeners = new Set<() => void>();
  private abort: AbortController | null = null;
  /** When each Transcript's last look began, and how many of its utterances it read. */
  private readonly began = new Map<number, number>();
  private readonly read = new Map<number, number>();
  private readonly follower = new Follower();

  constructor(private readonly deps: AnalystDeps) {}

  getState = (): AnalystState => this.state;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  /** Reads `t` now, unless a look is under way. */
  async analyze(t: Transcript, contextLimit: number | null): Promise<void> {
    if (this.state.looking !== null) return;
    const abort = new AbortController();
    this.abort = abort;
    this.began.set(t.id, (this.deps.now ?? Date.now)());
    this.read.set(t.id, t.utterances.length);
    this.set({ looking: t.id, failed: null });
    const { content } = transcriptInput(t, contextLimit, ANALYSIS_MAX_TOKENS);
    const asked = await askModel(assistRequest(ANALYSIS_SETTINGS, ANALYSIS_TASK, content), abort.signal, this.deps);
    if (asked.ok === "stopped") return;
    this.abort = null;
    if (asked.ok) {
      this.deps.onFound(t.id, readAnalysis(asked.text));
      this.set(IDLE);
    } else {
      this.set({ looking: null, failed: { transcript: t.id, message: asked.message } });
    }
  }

  stop() {
    if (!this.abort) return;
    this.abort.abort();
    this.abort = null;
    this.set(IDLE);
  }

  /** Reads `t` if keeping up says it is time; `final` as the listening ends. A failed look waits the gap. */
  keepUp(t: Transcript, contextLimit: number | null, { final = false } = {}): Promise<void> {
    const began = this.began.get(t.id);
    const sinceLastMs = began === undefined ? null : (this.deps.now ?? Date.now)() - began;
    if (!final && this.state.failed?.transcript === t.id && sinceLastMs !== null && sinceLastMs < ANALYSIS_GAP_MS) return Promise.resolve();
    const due = dueForAnalysis({ utterances: t.utterances.length, analyzed: this.read.get(t.id) ?? 0, sinceLastMs, final });
    return due ? this.analyze(t, contextLimit) : Promise.resolve();
  }

  /** One look while listening (assist.ts's Follower): `on`, the Transcript a listening appends to. */
  look(on: number | null, find: (id: number) => Transcript | undefined, contextLimit: number | null): Promise<void> {
    const next = this.follower.next(on, this.state.looking !== null);
    const t = next === null ? undefined : find(next.id);
    return t ? this.keepUp(t, contextLimit, { final: next!.final }) : Promise.resolve();
  }

  private set(state: AnalystState) {
    this.state = state;
    for (const listener of this.listeners) listener();
  }
}

export type AnalystHandle = Pick<Analyst, "getState" | "subscribe">;

export function useAnalyst(handle: AnalystHandle): AnalystState {
  return useSyncExternalStore(handle.subscribe, handle.getState, handle.getState);
}
