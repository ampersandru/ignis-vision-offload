import type { ToolDefinition } from "../api/request.ts";
import type { LocalRun } from "../tools/local/local.ts";
import { type Author, cleanName, KIND_LABEL, kindOf, latest, linesOf, present, readableText, type Version, type WorkspaceFile } from "./files.ts";
import type { ChatBinding, Workspaces } from "./workspaces.ts";

// The model's entrance to the workspace module: list_files, read_file and
// write_file, declared whenever the Workspace switch is on, with or without a
// workspace yet (ADR 0072: the tools block never changes with what the chat
// holds). They run in the browser like the other local tools, through the
// local tools' router; a sub-agent of the chat runs them too, as a reader of
// its own.

export const LIST_FILES = "list_files";
export const READ_FILE = "read_file";
export const WRITE_FILE = "write_file";

export const WORKSPACE_TOOL_NAMES: readonly string[] = [LIST_FILES, READ_FILE, WRITE_FILE];

/** The most read_file returns at once, in characters. */
export const READ_LIMIT = 20_000;

/** Who calls a workspace tool: the chat, and the sub-agent of the chat when one does. */
export type WorkspaceAccess = {
  workspaces: Workspaces;
  chat: ChatBinding;
  agent?: { call: string; name: string };
};

const tool = (name: string, description: string, properties: object, required: string[]): ToolDefinition => ({
  type: "function",
  function: { name, description, parameters: { type: "object", properties, required } },
});

export const LIST_FILES_TOOL = tool(LIST_FILES, "List the files of the chat's workspace: each one's name, kind, size, last change and author.", {}, []);

export const READ_FILE_TOOL = tool(
  READ_FILE,
  `Read a file of the workspace as numbered lines: a text file, or a PDF's text. Returns up to ${READ_LIMIT.toLocaleString("en-US")} characters at a time, and where to go on.`,
  {
    name: { type: "string", description: "The file's name, as list_files gives it." },
    offset: { type: "integer", description: "The first line to read, from 1; 1 by default." },
    limit: { type: "integer", description: "How many lines to read at most; as many as fit by default." },
  },
  ["name"],
);

export const WRITE_FILE_TOOL = tool(
  WRITE_FILE,
  "Create a text file in the workspace, or replace one whole. Each write is a new version of the file.",
  {
    name: { type: "string", description: "The file name with its extension; `/` makes folders, like notes.md or site/index.html." },
    content: { type: "string", description: "The whole content of the file." },
  },
  ["name", "content"],
);

export const WORKSPACE_PROMPT = `# Workspace
The chat can have a workspace: files you and the user work on together, which stay across turns and chats. It is created with its first file, which you write or the user attaches; a file the user attaches is named in their message.
- See what it holds with \`list_files\`: each file's name, kind, size, last change and author.
- Read a file with \`read_file\` as numbered lines, a range at a time; a PDF reads as its text. Read a file before you rely on it.
- Write a text file with \`write_file\`: code, a document, data — anything worth keeping. It creates the file or replaces it whole. Mention the file in your reply instead of repeating its content.
- A name may hold \`/\` for folders, like \`site/index.html\`.`;

/** What a workspace call does: the change to its run, or why it failed. */
export type WorkspaceOutcome = { ok: true; run: Partial<LocalRun> & { result: string } } | { ok: false; error: string };

const NO_WORKSPACE = "This chat has no workspace yet: it is created with the first file you write or the user attaches.";

const count = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;

const pad = (n: number) => String(n).padStart(2, "0");

/** A moment as a list line gives it, in the page's own time. */
function moment(at: number): string {
  const d = new Date(at);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** The author as the reader calling sees it. */
function authorLabel(author: Author, access: WorkspaceAccess): string {
  if (author.by === "user") return "the user";
  if (author.chat !== access.chat.key) return "another chat";
  if ((author.agent?.call ?? null) === (access.agent?.call ?? null)) return "you";
  return author.agent ? `the sub-agent ${author.agent.name}` : "the conversation that started you";
}

/** The version's size: lines and characters for text, bytes for the rest, and the text derived from them. */
function sizeOf(version: Version): string {
  if (version.text !== undefined) return `${count(linesOf(version.text).length, "line")}, ${count(version.text.length, "character")}`;
  // An older chat's PDF comes with its text alone: that chat never kept the bytes.
  const parts = [...(version.bytes ? [count(version.bytes.size, "byte")] : []), ...(version.derived !== undefined ? [`its text ${count(version.derived.length, "character")}`] : [])];
  return parts.length > 0 ? parts.join(", ") : count(0, "byte");
}

const fileLine = (file: WorkspaceFile, access: WorkspaceAccess) => {
  const v = latest(file);
  return `- ${file.name}: ${KIND_LABEL[file.kind]}, ${sizeOf(v)}; version ${v.n}, changed ${moment(v.at)} by ${authorLabel(v.author, access)}`;
};

/** The workspace's file named `name`, or why there is none. */
async function find(access: WorkspaceAccess, raw: string): Promise<{ file: WorkspaceFile } | { error: string }> {
  const id = access.chat.bound();
  if (!id) return { error: NO_WORKSPACE };
  const listed = (await access.workspaces.files(id)).filter(present);
  const name = cleanName(raw);
  const file = listed.find((f) => f.name === raw) ?? listed.find((f) => f.name === name);
  if (file) return { file };
  return {
    error:
      listed.length > 0
        ? `No file is named "${raw}". The files are: ${listed.map((f) => `"${f.name}"`).join(", ")}.`
        : `No file is named "${raw}": the workspace holds no files.`,
  };
}

const integer = (value: unknown): number | undefined => (typeof value === "number" && Number.isFinite(value) ? Math.floor(value) : undefined);

/** Lines `offset` on of `text`, numbered, as many as `limit` asks and READ_LIMIT allows. */
function readLines(name: string, version: Version, text: string, offset?: number, limit?: number): WorkspaceOutcome {
  const lines = linesOf(text);
  const total = lines.length;
  const tag = `${name}, version ${version.n}`;
  if (total === 0) return { ok: true, run: { read: { name, from: 0, to: 0, total: 0, lines: true }, result: `[${tag}: the file is empty.]` } };
  const from = Math.max(1, offset ?? 1);
  if (from > total) return { ok: false, error: `${name} has ${count(total, "line")}: line ${from} is past its end.` };
  const most = Math.max(1, limit ?? total);
  const width = String(Math.min(total, from + most - 1)).length;
  const out: string[] = [];
  let used = 0;
  let cut = "";
  for (let n = from; n <= total && out.length < most; n++) {
    const numbered = `${String(n).padStart(width)}\t${lines[n - 1]}`;
    if (used + numbered.length > READ_LIMIT) {
      if (out.length === 0) {
        out.push(numbered.slice(0, READ_LIMIT));
        cut = ` Line ${n} has ${count(lines[n - 1].length, "character")}: only its start fits.`;
      }
      break;
    }
    out.push(numbered);
    used += numbered.length + 1;
  }
  const to = from + out.length - 1;
  const more = to < total ? ` Call read_file with offset ${to + 1} for the rest.` : "";
  return {
    ok: true,
    run: { read: { name, from, to, total, lines: true }, result: `${out.join("\n")}\n\n[${tag}: lines ${from}–${to} of ${total}.${cut}${more}]` },
  };
}

/** Runs one workspace call: `name` is the tool, `args` its parsed arguments. */
export async function runWorkspaceTool(name: string, args: Record<string, unknown>, access: WorkspaceAccess): Promise<WorkspaceOutcome> {
  const author: Author = { by: "chat", chat: access.chat.key, ...(access.agent ? { agent: access.agent } : {}) };
  switch (name) {
    case LIST_FILES: {
      const id = access.chat.bound();
      if (!id) return { ok: true, run: { listed: 0, result: NO_WORKSPACE } };
      const [workspace, all] = await Promise.all([access.workspaces.get(id), access.workspaces.files(id)]);
      const listed = all.filter(present);
      const title = `The workspace "${workspace?.name ?? ""}"`;
      return {
        ok: true,
        run: {
          listed: listed.length,
          result: listed.length > 0 ? `${title} holds ${count(listed.length, "file")}:\n${listed.map((f) => fileLine(f, access)).join("\n")}` : `${title} holds no files.`,
        },
      };
    }
    case READ_FILE: {
      const raw = typeof args.name === "string" ? args.name.trim() : "";
      if (!raw) return { ok: false, error: 'The call needs the "name" of a file, as list_files gives it.' };
      const found = await find(access, raw);
      if ("error" in found) return { ok: false, error: found.error };
      const { file } = found;
      const version = latest(file);
      const text = readableText(version);
      if ((file.kind !== "text" && file.kind !== "pdf") || text === undefined) {
        return { ok: false, error: `${file.name} is ${file.kind === "other" ? "a file of a kind" : `a ${KIND_LABEL[file.kind]} file, which`} read_file cannot read.` };
      }
      return readLines(file.name, version, text, integer(args.offset), integer(args.limit));
    }
    case WRITE_FILE: {
      const raw = typeof args.name === "string" ? args.name : "";
      const clean = cleanName(raw);
      if (!clean) return { ok: false, error: 'The call needs a "name": the file name with its extension, like notes.md or site/index.html.' };
      if (typeof args.content !== "string") return { ok: false, error: 'The call needs "content" as text.' };
      const kind = kindOf(clean);
      if (kind !== "text") return { ok: false, error: `write_file writes text files, and ${clean} names a ${KIND_LABEL[kind]} file.` };
      const content = args.content;
      const outcome = await access.workspaces.writeText(await access.chat.workspace(), clean, content, author);
      if (!outcome.ok) return { ok: false, error: outcome.error };
      const { written } = outcome;
      const size = `${count(linesOf(content).length, "line")}, ${count(content.length, "character")}`;
      return {
        ok: true,
        run: {
          written,
          result: `${outcome.created ? "Created" : "Wrote"} ${written.name}, version ${written.version} (${size}), in the workspace "${outcome.workspace.name}". The user sees it in your reply.`,
        },
      };
    }
    default:
      return { ok: false, error: `"${name}" is not a workspace tool.` };
  }
}
