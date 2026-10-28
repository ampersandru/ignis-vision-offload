import { formatBytes } from "../decide/evidenceFile.ts";
import { readUpload } from "./attach.ts";
import { cleanName, KIND_LABEL, kindOf, present, USER, type Version, type WorkspaceFile } from "./files.ts";
import type { ChatBinding, FileOutcome, Workspaces } from "./workspaces.ts";

// What the workspace panel does, apart from how it looks (WorkspacePanel.tsx):
// the tree it shows of the files' names, the size it gives each one, and the
// user's actions that make a file, through the workspace module. A chat with
// no workspace yet gets one born with the first file the user makes or
// brings here, named after the chat, as with an attached file.

/** A folder of the tree, named by the part of the names before a `/`, or a file under the last part of its name. */
export type TreeNode = { kind: "folder"; name: string; path: string; children: TreeNode[] } | { kind: "file"; name: string; file: WorkspaceFile };

const byName = (a: TreeNode, b: TreeNode) => {
  if (a.kind !== b.kind) return a.kind === "folder" ? -1 : 1;
  const x = a.name.toLowerCase();
  const y = b.name.toLowerCase();
  return x < y ? -1 : x > y ? 1 : 0;
};

/** The files in the workspace as a tree of their names split on `/`: folders first, then files, each by name. */
export function fileTree(files: WorkspaceFile[]): TreeNode[] {
  const root: TreeNode[] = [];
  for (const file of files.filter(present)) {
    const parts = file.name.split("/");
    let level = root;
    for (let i = 0; i < parts.length - 1; i++) {
      const path = parts.slice(0, i + 1).join("/");
      let folder = level.find((n): n is Extract<TreeNode, { kind: "folder" }> => n.kind === "folder" && n.path === path);
      if (!folder) {
        folder = { kind: "folder", name: parts[i], path, children: [] };
        level.push(folder);
      }
      level = folder.children;
    }
    level.push({ kind: "file", name: parts[parts.length - 1], file });
  }
  const sort = (nodes: TreeNode[]): TreeNode[] => {
    nodes.sort(byName);
    for (const n of nodes) if (n.kind === "folder") sort(n.children);
    return nodes;
  };
  return sort(root);
}

/** A version's size as the panel shows it: its bytes, text counted as UTF-8 as a download saves it. */
export function sizeLabel(version: Version): string {
  if (version.text !== undefined) return formatBytes(new TextEncoder().encode(version.text).length);
  if (version.bytes) return formatBytes(version.bytes.size);
  return `${(version.derived ?? "").length.toLocaleString("en-US")} characters of text`;
}

/** The name a download saves a file under: the last part of its name. */
export const downloadName = (name: string) => name.slice(name.lastIndexOf("/") + 1);

/** What the panel says when the browser's store refused a change. */
export const refusal = (err: unknown) => `Not saved in this browser: ${err instanceof Error ? err.message : String(err)}`;

/** An empty text file of the user's, named `raw`; a name another file holds takes a number, as an attached file's does. */
export async function createFile(workspaces: Workspaces, chat: ChatBinding, raw: string): Promise<FileOutcome> {
  if (!raw.trim()) return { ok: false, error: "The file needs a name, like notes.md or site/index.html." };
  const name = cleanName(raw);
  if (!name) return { ok: false, error: `"${raw.trim()}" is not a file name.` };
  const kind = kindOf(name);
  if (kind !== "text") return { ok: false, error: `A new file is a text file, and ${name} names a ${KIND_LABEL[kind]} file.` };
  const [written] = await workspaces.add(await chat.workspace(), [{ name, kind: "text", text: "" }], USER);
  return { ok: true, written };
}

/** Files the user uploads or drops on the panel, of any kind, as the user's; the names they took in the workspace. */
export async function uploadFiles(workspaces: Workspaces, chat: ChatBinding, files: File[], readPdf?: (data: ArrayBuffer) => Promise<string>): Promise<string[]> {
  if (files.length === 0) return [];
  const incoming = await Promise.all(files.map((f) => readUpload(f, readPdf)));
  const written = await workspaces.add(await chat.workspace(), incoming, USER);
  return written.map((w) => w.name);
}
