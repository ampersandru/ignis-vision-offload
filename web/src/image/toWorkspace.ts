import type { ImageStore } from "../store/imageStore.ts";
import { sendFile, type Destination, type Sent } from "../workspace/send.ts";
import type { Workspaces } from "../workspace/workspaces.ts";
import { type ImageSession, pictureName, type PoolImage } from "./sessions.ts";

// Send to workspace: a pool picture as a picture file of a workspace (spec
// #170). The workspace module keeps pictures in this same image store, by the
// SHA-256 of their bytes, so the picture is one copy however many places name
// it; the session keeps its own record.

/** What a new workspace for a pool picture is called: the session's title. */
export const workspaceNameFor = (s: ImageSession): string => s.title;

/** The picture as a file of the destination, named as its download is; rejects when the picture is no longer in the store. */
export async function sendPicture(workspaces: Workspaces, store: ImageStore, picture: PoolImage, destination: Destination): Promise<Sent> {
  const blob = await store.getImage(picture.id);
  if (!blob) throw new Error("The picture is no longer in this browser's store.");
  return sendFile(workspaces, destination, { name: pictureName(picture), kind: "picture", blob });
}
