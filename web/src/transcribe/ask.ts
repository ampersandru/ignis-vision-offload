import type { ConversationRequest, Settings } from "../api/request.ts";
import { type AskDeps, askModel, assistRequest, settle, transcriptInput, Watched } from "./assist.ts";
import type { Transcript } from "./transcript.ts";

// Ask about this conversation (spec speech/07, Q139): a question about the
// Transcript, answered by the text model from the transcript alone, streamed,
// and kept with the Transcript, latest first. A person waits on it, so the
// interactive lane, thinking off: a quick answer, its moments as [mm:ss],
// which the view turns into links to them (Q141).

export const ASK_TASK = [
  "Task: answer the user's question about the conversation, from the transcript alone, briefly, in the language of the question, in Markdown.",
  "Point at the moments you draw on by their time, as [mm:ss]. When the transcript does not say, say so.",
].join("\n");

export const ASK_MAX_TOKENS = 800;

export const ASK_SETTINGS: Omit<Settings, "systemPrompt"> = {
  temperature: 0.3,
  topP: 0.9,
  maxTokens: ASK_MAX_TOKENS,
  reasoningEffort: "none",
  thinkingBudget: null,
  laneTag: "interactive",
};

/** A question asked and its answer, as the Transcript keeps them. */
export type Answered = { id: number; question: string; answer: string; at: number };

export function askRequest(t: Transcript, question: string, contextLimit: number | null): ConversationRequest {
  const { content } = transcriptInput(t, contextLimit, ASK_MAX_TOKENS);
  return assistRequest(ASK_SETTINGS, ASK_TASK, `${content}\nQuestion: ${question.trim()}`);
}

/** `answered`'s next, latest first. */
export function withAnswer(answered: Answered[] | undefined, question: string, answer: string, at: number): Answered[] {
  const all = answered ?? [];
  return [{ id: Math.max(0, ...all.map((a) => a.id)) + 1, question, answer, at }, ...all];
}

/** A stored Transcript's answers, those this page can read. */
export function readAnswered(value: unknown): Answered[] | undefined {
  if (!Array.isArray(value)) return undefined;
  return value.filter(
    (a): a is Answered =>
      typeof a === "object" && a !== null && typeof a.id === "number" && typeof a.question === "string" && typeof a.answer === "string" && typeof a.at === "number",
  );
}

export type AskerState = {
  /** The Transcript asked about, the question, and the answer so far. */
  transcript: number | null;
  question: string;
  text: string;
  writing: boolean;
  failed: string | null;
};

const IDLE: AskerState = { transcript: null, question: "", text: "", writing: false, failed: null };

export type AskerDeps = AskDeps & { onAnswered: (transcript: number, question: string, answer: string) => void };

/** One question at a time; the answer streams, and lands on the Transcript asked about. */
export class Asker extends Watched<AskerState> {
  private abort: AbortController | null = null;

  constructor(private readonly deps: AskerDeps) {
    super(IDLE);
  }

  async ask(t: Transcript, question: string, contextLimit: number | null): Promise<void> {
    if (this.state.writing || question.trim() === "") return;
    const abort = new AbortController();
    this.abort = abort;
    this.set({ transcript: t.id, question: question.trim(), text: "", writing: true, failed: null });
    const asked = await askModel(askRequest(t, question, contextLimit), abort.signal, this.deps, (text) => this.set({ ...this.state, text }));
    const settled = settle(asked);
    if (settled === "stopped") return;
    this.abort = null;
    if ("failed" in settled) {
      this.set({ ...this.state, writing: false, text: "", failed: settled.failed });
      return;
    }
    this.deps.onAnswered(t.id, this.state.question, settled.text);
    this.set(IDLE);
  }

  stop() {
    this.abort?.abort();
    this.abort = null;
    this.set(IDLE);
  }
}

export type AskerHandle = Pick<Asker, "getState" | "subscribe">;
