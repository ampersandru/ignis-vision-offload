import { keptMessages, type Message } from "../sessions/sessions.ts";
import type { WorkspaceFile } from "../workspace/files.ts";
import { type FileMark, workspaceNotice } from "../workspace/notice.ts";
import type { Written } from "../workspace/workspaces.ts";
import { type ToolsState, turnDateTime } from "./index.ts";
import { MEMORY_DELETE, MEMORY_SAVE } from "./local/local.ts";
import { markOf, type MemoryNote, memoryNotice, type NoteMark } from "./local/memory.ts";

// The turn notice (ADR 0072): the developer message the page puts ahead of a
// new prompt, after the history, with what the model is told about that turn.
// The system prompt and the tools block stay fixed, the same in every chat;
// what changes rides here. A chat's first turn hears the titles of the memory
// notes; a later one only what someone other than this chat saved, changed or
// deleted since; the moment rides along while the date and time tool updates
// every prompt. The workspace's files ride the same way (#393): every one at
// the first turn, later only what the user or another chat changed. A turn
// with nothing to tell carries no notice. Once sent, the notice is kept with
// its prompt and every later request repeats its bytes.

/** A notice's sections, each under its own heading, in the order they are written. */
export type NoticeSections = { moment?: string; memory?: string; files?: string };

const SECTIONS: (keyof NoticeSections)[] = ["moment", "memory", "files"];

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
  for (const m of keptMessages(history)) {
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
 * The workspace's files as the chat knows them after `history`, undefined
 * while it was never told of them: the files its last notice told it of,
 * then the versions its own calls wrote, edited, deleted or read since, and
 * the pictures its make_image calls landed; its agents' calls count as its
 * own, their answers telling their work. As with the memory notes, a prompt
 * dropped with its failed reply told it nothing.
 */
export function knownFilesOf(history: Message[]): FileMark[] | undefined {
  let known: Map<string, FileMark> | undefined;
  const learn = (v: Written | undefined) => {
    const was = v && known?.get(v.file);
    if (v && (!was || was.version <= v.version)) known?.set(v.file, { id: v.file, name: v.name, version: v.version });
  };
  for (const m of keptMessages(history)) {
    if (m.knownFiles) known = new Map(m.knownFiles.map((f) => [f.id, f]));
    for (const run of [...(m.local ?? []), ...(m.agents ?? []).flatMap((a) => a.local ?? [])]) learn(run.written ?? run.seen);
    for (const made of (m.imageRuns ?? []).flatMap((r) => r.made ?? [])) learn(made.file);
  }
  return known && [...known.values()];
}

/**
 * What a new prompt after `history` goes out with, to spread on its message:
 * its notice, when it has anything to tell, and the memory notes and the
 * workspace's files the chat knows from it on, when that changes. `notes`
 * are the memory notes now; `files` the chat's workspace's now
 * (workspace/notice.ts, noticeFiles), undefined when they could not be read;
 * `attached` the names of the files the prompt attaches, which it names itself.
 */
export function turnNotice(
  tools: ToolsState,
  history: Message[],
  { now, notes, files, attached }: { now: Date; notes: MemoryNote[]; files?: WorkspaceFile[]; attached?: readonly string[] },
): Pick<Message, "notice" | "knownNotes" | "knownFiles"> {
  const memory = tools.enabled && tools.memory ? memoryNotice(knownNotesOf(history), notes) : {};
  const workspace = tools.enabled && tools.workspace && files ? workspaceNotice(knownFilesOf(history), files, attached) : {};
  const notice = noticeText({ moment: turnDateTime(tools, now), memory: memory.text, files: workspace.text });
  return {
    ...(notice ? { notice } : {}),
    ...(memory.known ? { knownNotes: memory.known } : {}),
    ...(workspace.known ? { knownFiles: workspace.known } : {}),
  };
}

/**
 * The notice an agent's one prompt goes out behind: every title, as a chat's
 * first turn hears them, or undefined when there is none to tell. No moment:
 * an agent keeps the session's in its system prompt (`agentExtras`).
 */
export function agentNotice(tools: ToolsState, notes: MemoryNote[]): string | undefined {
  return tools.enabled && tools.memory ? noticeText({ memory: memoryNotice(undefined, notes).text }) : undefined;
}
