import { USER, type Workspace, type WorkspaceId } from "./files.ts";
import type { Workspaces, Written } from "./workspaces.ts";

// A text the user sends to a workspace from elsewhere in the page (a
// transcript, later a picture's description): it asks which workspace, a new
// one or an existing one, and the text becomes one file the user authored.

/** Where a text goes: a workspace born for it, or one that exists. */
export type Destination = { kind: "new"; name: string } | { kind: "existing"; id: WorkspaceId };

/** What the send did: the file as the workspace named it (a taken name gets `(2)`), and the workspace. */
export type Sent = { workspace: Workspace; written: Written };

/** The text as a file of the destination, added through the user's entrance. */
export async function sendText(workspaces: Workspaces, destination: Destination, file: { name: string; text: string }): Promise<Sent> {
  const workspace = destination.kind === "new" ? await workspaces.create(destination.name) : await workspaces.get(destination.id);
  if (!workspace) throw new Error("That workspace is gone.");
  const [written] = await workspaces.add(workspace.id, [{ name: file.name, kind: "text", text: file.text }], USER);
  if (!written) throw new Error("The workspace did not take the file.");
  return { workspace: (await workspaces.get(workspace.id)) ?? workspace, written };
}
