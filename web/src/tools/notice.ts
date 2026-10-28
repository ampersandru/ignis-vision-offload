import { keptEntries } from "../api/request.ts";
import type { Message } from "../sessions/sessions.ts";
import { type ToolsState, turnDateTime } from "./index.ts";
import { MEMORY_DELETE, MEMORY_SAVE } from "./local/local.ts";
import { markOf, type MemoryNote, memoryNotice, type NoteMark } from "./local/memory.ts";

// The turn notice (ADR 0072): the developer message the page puts ahead of a
// new prompt, after the history, with what the model is told about that turn.
// The system prompt and the tools block stay fixed, the same in every chat;
// what changes rides here. A chat's first turn hears the titles of the memory
// notes; a later one only what someone other than this chat saved, changed or
// deleted since; the moment rides along while the date and time tool updates
// every prompt. A turn with nothing to tell carries no notice. Once sent, the
// notice is kept with its prompt and every later request repeats its bytes.

/** A notice's sections, each under its own heading, in the order they are written. */
export type NoticeSections = { moment?: string; memory?: string };

const SECTIONS: (keyof NoticeSections)[] = ["moment", "memory"];

/** A notice's text: its sections in their order, or undefined when it has none — the turn then carries no notice. */
export function noticeText(sections: NoticeSections): string | undefined {
  const parts = SECTIONS.flatMap((name) => sections[name] ?? []);
  return parts.length > 0 ? parts.join("\n\n") : undefined;
}

/**
 * The memory notes as the chat knows them after `history`, undefined while it
 * was never told of them: the notes its last notice told it of, then what its
 * own calls saved and deleted since, its agents' included. Only what the
 * model reads counts: a prompt dropped with its failed reply told it nothing.
 */
export function knownNotesOf(history: Message[]): NoteMark[] | undefined {
  let known: Map<string, NoteMark> | undefined;
  for (const m of keptEntries(history.map((m) => ({ ...m, failed: m.error !== undefined })))) {
    if (m.knownNotes) known = new Map(m.knownNotes.map((n) => [n.id, n]));
    for (const run of [...(m.local ?? []), ...(m.agents ?? []).flatMap((a) => a.local ?? [])]) {
      if (run.status !== "done") continue;
      if (run.tool === MEMORY_SAVE && run.note) known?.set(run.note.id, markOf(run.note));
      if (run.tool === MEMORY_DELETE && run.deletedId) known?.delete(run.deletedId);
    }
  }
  return known && [...known.values()];
}

/**
 * What a new prompt after `history` goes out with, to spread on its message:
 * its notice, when it has anything to tell, and the memory notes the chat
 * knows from it on, when that changes. `notes` are the memory notes now.
 */
export function turnNotice(tools: ToolsState, history: Message[], { now, notes }: { now: Date; notes: MemoryNote[] }): Pick<Message, "notice" | "knownNotes"> {
  const memory = tools.enabled && tools.memory ? memoryNotice(knownNotesOf(history), notes) : {};
  const notice = noticeText({ moment: turnDateTime(tools, now), memory: memory.text });
  return { ...(notice ? { notice } : {}), ...(memory.known ? { knownNotes: memory.known } : {}) };
}

/**
 * The notice an agent's one prompt goes out behind: every title, as a chat's
 * first turn hears them, or undefined when there is none to tell. No moment:
 * an agent keeps the session's in its system prompt (`agentExtras`).
 */
export function agentNotice(tools: ToolsState, notes: MemoryNote[]): string | undefined {
  return tools.enabled && tools.memory ? noticeText({ memory: memoryNotice(undefined, notes).text }) : undefined;
}
