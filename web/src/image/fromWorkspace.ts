import type { ImageId } from "../store/imageStore.ts";
import { latest, present, type WorkspaceFile } from "../workspace/files.ts";
import type { PoolImage } from "./sessions.ts";

// Pictures from a workspace into an image session's pool (spec #170): the
// workspace's picture files that sit in the image store, each as an input of
// the next step under the pool origin `workspace`. The pool names the same
// SHA-256 id the workspace's version holds; no bytes are copied.

/** A picture file of a workspace, as its newest version holds it. */
export type WorkspacePicture = { file: string; name: string; id: ImageId; width: number; height: number };

/** The workspace's pictures the pool can take: present picture files whose newest version is an image of known size. */
export function takeablePictures(files: WorkspaceFile[]): WorkspacePicture[] {
  return files.flatMap((f): WorkspacePicture[] => {
    if (f.kind !== "picture" || !present(f)) return [];
    const bytes = latest(f).bytes;
    if (bytes?.store !== "image" || !bytes.width || !bytes.height) return [];
    return [{ file: f.id, name: f.name, id: bytes.id, width: bytes.width, height: bytes.height }];
  });
}

/** Which of `pictures` fit among the `max` a request takes when `have` are already inputs, and what to say of the rest. */
export function fitInputs(pictures: WorkspacePicture[], have: number, max: number): { taken: WorkspacePicture[]; notice: string | null } {
  const room = Math.max(0, max - have);
  return { taken: pictures.slice(0, room), notice: pictures.length > room ? `A request takes at most ${max} references.` : null };
}

/** The pool entries for the pictures: the workspace's file name is the pool picture's name. */
export const poolEntries = (pictures: WorkspacePicture[]): Omit<PoolImage, "n">[] =>
  pictures.map((p) => ({ id: p.id, width: p.width, height: p.height, origin: "workspace", name: p.name }));
