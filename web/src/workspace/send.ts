import { USER, type Workspace, type WorkspaceId } from "./files.ts";
import type { Incoming, Workspaces, Written } from "./workspaces.ts";

// A file the user sends to a workspace from elsewhere in the page (a
// transcript, a picture): it asks which workspace, a new one or an existing
// one, and the file is added through the workspace module's entrance for the
// user's own files, authored by the user.

/** Where a file goes: a workspace born for it, or one that exists. */
export type Destination = { kind: "new"; name: string } | { kind: "existing"; id: WorkspaceId };

/** What the send did: the file as the workspace named it (a taken name gets `(2)`), and the workspace. */
export type Sent = { workspace: Workspace; written: Written };

/** The file in the destination, added through the user's entrance. */
export async function sendFile(workspaces: Workspaces, destination: Destination, file: Incoming): Promise<Sent> {
  const workspace = destination.kind === "new" ? await workspaces.create(destination.name) : await workspaces.get(destination.id);
  if (!workspace) throw new Error("That workspace is gone.");
  const [written] = await workspaces.add(workspace.id, [file], USER);
  if (!written) throw new Error("The workspace did not take the file.");
  return { workspace: (await workspaces.get(workspace.id)) ?? workspace, written };
}

/** The text as a file of the destination. */
export const sendText = (workspaces: Workspaces, destination: Destination, file: { name: string; text: string }): Promise<Sent> =>
  sendFile(workspaces, destination, { name: file.name, kind: "text", text: file.text });
