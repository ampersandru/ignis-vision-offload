import type { Figures } from "../metrics/figures.ts";
import type { LaneTag, ReasoningEffort } from "../api/request.ts";
import type { ToolCall } from "../api/sse.ts";
import type { AgentRun } from "../tools/agents/agents.ts";
import type { ImageRun } from "../tools/image/image.ts";
import type { Question } from "../tools/ask/ask.ts";
import type { UnknownCall } from "../tools/errors.ts";
import type { PromptImage } from "../conversation/images.ts";
import type { Attachment } from "../tools/local/attachments.ts";
import type { LocalRun } from "../tools/local/local.ts";
import type { WebRun } from "../tools/web/web.ts";

// Playground sessions: separate conversations, each with its own log of
// per-reply figures. The page's store keeps them across a reload (stored.ts).
// Every function returns a new list and leaves the one it was given alone.

export type Message = {
  id: number;
  /** `tool` messages carry a call's result back to the model; the transcript shows them through the call. */
  role: "user" | "assistant" | "tool";
  content: string;
  /**
   * Images sent with this prompt (GitHub #174). They live on the message and
   * not on the session: a fork, a resend and a regenerate all repeat the
   * turn as it went out, and the engine only has a media prefix to reuse
   * while the bytes stay the same.
   */
  images?: PromptImage[];
  /**
   * The workspace files this prompt attached, by the names they took there
   * (spec #170). The prompt names them on the wire (`withAttached`), the same
   * bytes on every request that repeats it.
   */
  attached?: string[];
  reasoning: string;
  streaming: boolean;
  /** The engine was full when this reply was sent: it waits for a lane before its first token. */
  queued?: boolean;
  figures?: Figures;
  error?: string;
  /** Changed by hand after it was sent or generated; its figures measure the original. */
  edited?: boolean;
  /** The tools an assistant reply called. */
  toolCalls?: ToolCall[];
  /** The call a `tool` message answers. */
  toolCallId?: string;
  /** The agents an assistant reply's calls started, as they run. */
  agents?: AgentRun[];
  /** The web searches and page reads an assistant reply's calls made, as they run. */
  web?: WebRun[];
  /** Calls an assistant reply made to tools the request did not declare. */
  unknownTools?: UnknownCall[];
  /** The questions an assistant reply asked the user. */
  questions?: Question[];
  /** The local tool calls an assistant reply made: code, plan, memory, files. */
  local?: LocalRun[];
  /** The pictures an assistant reply's make_image calls asked for, as they are made. */
  imageRuns?: ImageRun[];
  /** The switch a turn made before its first request, shown on the prompt it answered. */
  switch?: TurnSwitch;
  /**
   * The moment this turn was sent, as the date and time tool writes it, while
   * the tool updates every prompt. Kept with the turn rather than written
   * fresh on each request: the history a later request repeats has to be the
   * one it sent, or the engine is left no prefix to reuse.
   */
  dateTime?: string;
};

/** A switch to another model before a turn's first request: the owner's choice of model, or the way back from the image model. */
export type TurnSwitch = { to: string; from?: string; status: "switching" | "switched" | "stopped" | "failed"; error?: string };

export type LogRow = {
  n: number;
  at: string;
  laneTag: LaneTag;
  reasoningEffort: ReasoningEffort;
  /** The `thinking_budget` the request sent; absent when it sent none. Whether the budget closed it is on the figures. */
  thinkingBudget?: number;
  figures: Figures | null;
  error?: string;
  /** The agent's name, on a row an agent's request made. */
  agent?: string;
};

export type Session = {
  id: number;
  title: string;
  messages: Message[];
  log: LogRow[];
  /**
   * Files the user attached before the workspace (#170), in an older chat:
   * its next turn makes them the first files of its workspace, and empties this.
   */
  attachments: Attachment[];
  /**
   * The chat's workspace (spec #170): set when its first file is created, or
   * when the user picks one before that, and never changed afterwards. A chat
   * with no file has none.
   */
  workspaceId?: string;
  /**
   * When this conversation began: the moment the date and time tool writes
   * into its system prompt, the same for every turn. A fork inherits it, so
   * the fork and its source still open with the same tokens.
   */
  startedAt: Date;
};

export type SessionList = { sessions: Session[]; activeId: number };

export const UNTITLED = "New session";

const TITLE_LENGTH = 48;

export function createSession(id: number, startedAt = new Date()): Session {
  return { id, title: UNTITLED, messages: [], log: [], attachments: [], startedAt };
}

/** Binds a session to its workspace: once, since a chat never changes workspace. */
export function bindWorkspace(sessions: Session[], sessionId: number, workspaceId: string): Session[] {
  return sessions.map((s) => (s.id === sessionId && s.workspaceId === undefined ? { ...s, workspaceId } : s));
}

/** An older chat's attachments, gone once they are files of its workspace. */
export function clearAttachments(sessions: Session[], sessionId: number): Session[] {
  return sessions.map((s) => (s.id === sessionId && s.attachments.length > 0 ? { ...s, attachments: [] } : s));
}

/** Drops the older chat's attached file called `name`. */
export function removeAttachment(sessions: Session[], sessionId: number, name: string): Session[] {
  return sessions.map((s) => (s.id === sessionId ? { ...s, attachments: s.attachments.filter((a) => a.name !== name) } : s));
}

/** A session's title from its first prompt: the first non-blank line, shortened. */
/** The last reply with text, the chat's dictation context before the prompt (spec speech/04); `""` before any. */
export function lastReply(messages: Message[]): string {
  return [...messages].reverse().find((m) => m.role === "assistant" && m.content.trim() !== "")?.content ?? "";
}

export function titleFrom(prompt: string): string {
  const line = prompt.split("\n").find((l) => l.trim() !== "")?.trim().replace(/\s+/g, " ") ?? "";
  if (line === "") return UNTITLED;
  return line.length > TITLE_LENGTH ? `${line.slice(0, TITLE_LENGTH - 1).trimEnd()}…` : line;
}

/** A session's title from its first prompt, falling back to the image's or the file's name when the prompt is only that. */
export function titleOf(first: Message): string {
  const title = titleFrom(first.content);
  if (title !== UNTITLED) return title;
  if (first.images?.length) return titleFrom(first.images[0].name);
  return first.attached?.length ? titleFrom(first.attached[0]) : title;
}

/**
 * A session to write in. An active session with no messages yet is reused,
 * so repeated clicks do not pile up empty sessions; otherwise `newId` opens
 * at the top of the list.
 */
export function openSession(list: SessionList, newId: number): SessionList {
  const active = list.sessions.find((s) => s.id === list.activeId);
  if (active && active.messages.length === 0) return list;
  return { sessions: [createSession(newId), ...list.sessions], activeId: newId };
}

/**
 * Drops a session. Removing the active one moves to the session that took
 * its place in the list; removing the last one leaves a fresh `newId`.
 */
export function removeSession(list: SessionList, id: number, newId: number): SessionList {
  const index = list.sessions.findIndex((s) => s.id === id);
  if (index === -1) return list;
  const sessions = list.sessions.filter((s) => s.id !== id);
  if (sessions.length === 0) return { sessions: [createSession(newId)], activeId: newId };
  if (id !== list.activeId) return { sessions, activeId: list.activeId };
  return { sessions, activeId: sessions[Math.min(index, sessions.length - 1)].id };
}

/** A prompt and the reply that will stream into it; the first prompt names the session. */
export function addExchange(sessions: Session[], sessionId: number, user: Message, reply: Message): Session[] {
  return sessions.map((s) =>
    s.id === sessionId
      ? { ...s, title: s.messages.length === 0 ? titleOf(user) : s.title, messages: [...s.messages, user, reply] }
      : s,
  );
}

export function updateMessage(
  sessions: Session[],
  sessionId: number,
  messageId: number,
  change: (m: Message) => Message,
): Session[] {
  return sessions.map((s) =>
    s.id === sessionId ? { ...s, messages: s.messages.map((m) => (m.id === messageId ? change(m) : m)) } : s,
  );
}

/** Messages appended in order; a user message among them names a session that was empty. */
export function addMessages(sessions: Session[], sessionId: number, added: Message[]): Session[] {
  return sessions.map((s) => {
    if (s.id !== sessionId) return s;
    const first = s.messages.length === 0 ? added.find((m) => m.role === "user") : undefined;
    return { ...s, title: first ? titleOf(first) : s.title, messages: [...s.messages, ...added] };
  });
}

/** A reply streaming into the session as it stands (regenerate: no new prompt). */
export function addReply(sessions: Session[], sessionId: number, reply: Message): Session[] {
  return sessions.map((s) => (s.id === sessionId ? { ...s, messages: [...s.messages, reply] } : s));
}

/**
 * A new session holding the conversation up to and including `messageId`,
 * opened at the top of the list. Its log starts empty: the copied replies
 * keep their own figures, but no request has run in the fork yet.
 */
export function forkSession(list: SessionList, sessionId: number, messageId: number, newId: number): SessionList {
  const source = list.sessions.find((s) => s.id === sessionId);
  const index = source ? source.messages.findIndex((m) => m.id === messageId) : -1;
  if (!source || index === -1) return list;
  const fork: Session = {
    id: newId,
    title: `${source.title} (fork)`,
    messages: source.messages.slice(0, index + 1).map((m) => ({ ...m, streaming: false })),
    log: [],
    attachments: source.attachments,
    ...(source.workspaceId !== undefined ? { workspaceId: source.workspaceId } : {}),
    startedAt: source.startedAt,
  };
  return { sessions: [fork, ...list.sessions], activeId: newId };
}

/** Replaces a message's text, marking it edited; the next request sends the new text. */
export function editMessage(sessions: Session[], sessionId: number, messageId: number, content: string): Session[] {
  return updateMessage(sessions, sessionId, messageId, (m) =>
    m.content === content ? m : { ...m, content, edited: true },
  );
}

/**
 * Drops a message and everything after it, ahead of sending an edited
 * prompt again. The log keeps its rows: they measured requests that ran.
 */
export function truncateFrom(sessions: Session[], sessionId: number, messageId: number): Session[] {
  return sessions.map((s) => {
    if (s.id !== sessionId) return s;
    const index = s.messages.findIndex((m) => m.id === messageId);
    return index === -1 ? s : { ...s, messages: s.messages.slice(0, index) };
  });
}

/** Appends a reply's figures to the session's log, numbered from 1. */
export function addLogRow(sessions: Session[], sessionId: number, row: Omit<LogRow, "n">): Session[] {
  return sessions.map((s) => (s.id === sessionId ? { ...s, log: [...s.log, { ...row, n: s.log.length + 1 }] } : s));
}
