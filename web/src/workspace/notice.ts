import type { WorkspaceMove } from "../sessions/sessions.ts";
import { CHANGED_SINCE, newestFitting, noticeSection } from "../tools/local/memory.ts";
import { KIND_LABEL, latest, present, type WorkspaceFile, type WorkspaceId } from "./files.ts";
import { sizeOf } from "./tools.ts";
import type { Workspaces } from "./workspaces.ts";

// What the turn notice (ADR 0072, tools/notice.ts) tells the model of the
// chat's workspace. A chat's first turn hears every file, with its kind and
// size; a later one only the files someone other than this chat created,
// changed, renamed or deleted since: the user, from the panel, or another
// chat. What the chat knows is a mark per file, its name and the version it
// knows (knownFilesOf): a file at the version and under the name the chat
// knows is no news, whoever made that version. A file the turn's prompt
// attaches is named by the prompt (`withAttached`) and not told again here.
// A chat the user moved to another workspace, or out of its own (#404),
// hears at its next turn which one it left, then the one it works in now
// whole, as a first turn does.

/** A workspace file as a chat knows it: what a later turn's notice compares the files with. */
export type FileMark = { id: string; name: string; version: number };

export const fileMarkOf = (file: WorkspaceFile): FileMark => ({ id: file.id, name: file.name, version: latest(file).n });

const facts = (file: WorkspaceFile) => `${KIND_LABEL[file.kind]}, ${sizeOf(latest(file))}`;

/** The file lines a turn notice lists, cut to this many characters in all, as the memory titles are to theirs. */
export const NOTICE_FILES_LIMIT = 4000;

/** The lines that fit, as the memory titles do: in the order the files last changed, the newest kept first. */
function fitting(lines: { at: number; line: string }[]) {
  return newestFitting([...lines].sort((a, b) => a.at - b.at).map((l) => l.line), NOTICE_FILES_LIMIT);
}

/** The workspace section's text: its heading, what the lines are, the lines, and what was left out, if anything. */
const section = (what: string, lines: string[], leftOut: string) => noticeSection("# Workspace files", what, lines, leftOut);

const sameMarks = (a: FileMark[], b: FileMark[]) => {
  const byId = new Map(a.map((m) => [m.id, m]));
  return a.length === b.length && b.every((m) => byId.get(m.id)?.name === m.name && byId.get(m.id)?.version === m.version);
};

/**
 * What a notice says of the chat's move out of the workspace its model knew
 * (#404): the workspace left, by the name it had, or as deleted.
 */
const leftLine = (move: WorkspaceMove) =>
  move.name !== undefined
    ? `This chat moved out of the workspace "${move.name}": its files are no longer this chat's.`
    : "This chat moved out of its workspace, which was deleted: its files are no longer this chat's.";

/**
 * The workspace section of a turn notice, and the files the chat knows once
 * it has read it. `files` are the workspace's now, deleted ones included
 * (none while the chat has no workspace); `attached`, the names the turn's
 * prompt gives the files it attaches. A chat told nothing yet (`known`
 * undefined) hears every file, as many as fit; with none to tell it now
 * knows that. A chat told before hears only the changes to what it knows,
 * and nothing at all when there are none. A chat that `moved` out of the
 * workspace it was told of hears that, then every file of the one it works
 * in now, as a chat told nothing yet does, or that it has none.
 */
export function workspaceNotice(known: FileMark[] | undefined, files: WorkspaceFile[], attached: readonly string[] = [], moved?: WorkspaceMove): { text?: string; known?: FileMark[] } {
  const marks = files.filter(present).map(fileMarkOf);
  const told = (file: WorkspaceFile) => !attached.includes(file.name);
  /** Every file, as many as fit, and the line counting the others. */
  const whole = () => {
    const { listed, left } = fitting(files.filter((f) => present(f) && told(f)).map((f) => ({ at: latest(f).at, line: `- ${f.name}: ${facts(f)}` })));
    return { listed, leftOut: left > 0 ? `(${left} older files not listed; list_files lists them all.)` : "" };
  };
  // A chat never told of a workspace has none it moved out of: the files it works in now say it all.
  if (known === undefined) {
    const { listed, leftOut } = whole();
    return listed.length === 0 ? { known: marks } : { text: section("Files in the chat's workspace:", listed, leftOut), known: marks };
  }
  if (moved) {
    const { listed, leftOut } = whole();
    if (listed.length > 0) return { text: section(`${leftLine(moved)}\nFiles in the chat's workspace now:`, listed, leftOut), known: marks };
    // Files the prompt attaches are in its workspace, and the prompt names them.
    const none = marks.length > 0 ? [] : files.length > 0 ? ["The chat's workspace holds no files."] : ["The chat has no workspace now: one is created with the next file written or attached."];
    return { text: section(leftLine(moved), none, ""), known: marks };
  }
  const before = new Map(known.map((m) => [m.id, m]));
  const changes: { at: number; line: string }[] = [];
  for (const file of files) {
    const was = before.get(file.id);
    before.delete(file.id);
    const { n, at, deleted } = latest(file);
    if (deleted) {
      // A deletion the chat made, or read, is at the version it knows.
      if (was && was.version !== n) changes.push({ at, line: `- deleted ${was.name}` });
    } else if (!told(file)) {
      continue;
    } else if (!was) {
      changes.push({ at, line: `- created ${file.name}: ${facts(file)}` });
    } else if (was.version !== n) {
      changes.push({ at, line: was.name === file.name ? `- changed ${file.name}: ${facts(file)}` : `- renamed ${was.name} to ${file.name} and changed: ${facts(file)}` });
    } else if (was.name !== file.name) {
      // A rename makes no version: only the name says it.
      changes.push({ at, line: `- renamed ${was.name} to ${file.name}` });
    }
  }
  // What the chat knows and the workspace no longer holds: the workspace was deleted, or the chat works in another.
  for (const was of before.values()) changes.push({ at: 0, line: `- deleted ${was.name}` });
  if (changes.length === 0) return sameMarks(known, marks) ? {} : { known: marks };
  const { listed, left } = fitting(changes);
  return { text: section(CHANGED_SINCE, listed, left > 0 ? `(${left} older changes not listed.)` : ""), known: marks };
}

/**
 * The files a turn's notice is told of: those of the chat's workspace, deleted
 * ones included, none while it has no workspace, or undefined when the
 * store cannot read them. The notice then names no file, and the next one
 * tells what this one could not.
 */
export async function noticeFiles(workspaces: Pick<Workspaces, "files">, workspace: WorkspaceId | undefined): Promise<WorkspaceFile[] | undefined> {
  if (workspace === undefined) return [];
  return workspaces.files(workspace).catch(() => undefined);
}
