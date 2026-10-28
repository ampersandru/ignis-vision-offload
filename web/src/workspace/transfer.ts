import { readUpload } from "./attach.ts";
import { latest, present, USER, type Workspace, type WorkspaceFile } from "./files.ts";
import type { Incoming, Workspaces } from "./workspaces.ts";
import { isDirectory, readZip, writeZip } from "./zip.ts";

// A workspace leaves the browser as a zip and comes back as a new workspace
// (spec #170, #397). The zip holds each file's current version under its
// name (`/` makes the folders); deleted files and the history are not in it.
// Importing makes a workspace named after the zip whose files start at
// version 1, the user's, each kind recognised as an upload's is.

/** The name a download of the workspace is saved under: the workspace's name, with what a file system refuses made `_`. */
export function zipName(workspace: Workspace): string {
  const stem = workspace.name.replace(/[\u0000-\u001f\u007f\\/:*?"<>|]+/g, "_").trim().replace(/^\.+/, "").trim();
  return `${stem || "workspace"}.zip`;
}

/** The workspace's present files as a zip, and the names of those it could not include (an older chat kept only the text of a PDF, so there are no bytes). */
export async function exportWorkspace(workspaces: Workspaces, workspace: Workspace, files: WorkspaceFile[]): Promise<{ zip: Blob; name: string; left: string[] }> {
  const entries = [];
  const left: string[] = [];
  for (const file of files.filter(present)) {
    const version = latest(file);
    const content = await workspaces.content(version);
    if (!content) {
      left.push(file.name);
      continue;
    }
    entries.push({ name: file.name, data: new Uint8Array(await content.arrayBuffer()), modified: version.at });
  }
  const zip = await writeZip(entries);
  return { zip: new Blob([zip as BlobPart], { type: "application/zip" }), name: zipName(workspace), left };
}

/** Whether a zip's entry is the file manager's litter rather than a file. */
const litter = (name: string) => name.startsWith("__MACOSX/");

/** The workspace a zip becomes: named after it, without `.zip`. */
export const workspaceNamed = (zipFileName: string) => zipFileName.replace(/\.zip$/i, "").trim() || "Workspace";

/**
 * A new workspace made of the zip's files; the workspace and the number of its
 * files, or why the zip could not be read (no workspace is made then).
 */
export async function importWorkspace(
  workspaces: Workspaces,
  zip: File,
  readPdf?: (data: ArrayBuffer) => Promise<string>,
): Promise<{ ok: true; workspace: Workspace; files: number } | { ok: false; error: string }> {
  let entries;
  try {
    entries = await readZip(new Uint8Array(await zip.arrayBuffer()));
  } catch (err) {
    return { ok: false, error: `${zip.name}: ${err instanceof Error ? err.message : String(err)}` };
  }
  const kept = entries.filter((e) => !isDirectory(e.name) && !litter(e.name));
  if (kept.length === 0) return { ok: false, error: `${zip.name} holds no files.` };
  const incoming: Incoming[] = [];
  for (const entry of kept) incoming.push(await readUpload(new File([entry.data as BlobPart], entry.name), readPdf));
  const workspace = await workspaces.create(workspaceNamed(zip.name));
  await workspaces.add(workspace.id, incoming, USER);
  return { ok: true, workspace, files: incoming.length };
}
