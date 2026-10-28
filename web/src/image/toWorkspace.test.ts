import { describe, expect, it } from "vitest";
import { sha256 } from "../store/imageStore.ts";
import { createMemoryStore } from "../store/memory.ts";
import { latest, USER } from "../workspace/files.ts";
import { createWorkspaces } from "../workspace/workspaces.ts";
import { fitInputs, poolEntries, takeablePictures } from "./fromWorkspace.ts";
import { addToPool, freedWithSession, imagesOf, newSession, pictureName } from "./sessions.ts";
import { sendPicture, workspaceNameFor } from "./toWorkspace.ts";

// The two moves of a picture between an image session's pool and a workspace
// (#398), over the in-memory store: the picture is one copy of its bytes,
// named by both sides; each side keeps its own records.

const png = (fill: number) => new Blob([new Uint8Array(48).fill(fill)], { type: "image/png" });
const measure = async () => ({ width: 800, height: 600 });

function setup() {
  let n = 0;
  const store = createMemoryStore();
  const workspaces = createWorkspaces(store, { now: () => 1_000 + n, newId: () => `id${++n}`, measure });
  return { store, workspaces };
}

/** A session whose pool holds the picture `fill` made, kept in the store as image sessions keep it. */
async function sessionWith(store: ReturnType<typeof createMemoryStore>, fill: number, key = "s1") {
  const image = await store.putImage(png(fill), { width: 800, height: 600 });
  const [session] = addToPool(newSession(key, 0), [{ id: image.id, width: 800, height: 600, origin: "upload", name: "fox.png" }], 1, true);
  return { session, picture: session.pool[0], id: image.id };
}

describe("send to workspace", () => {
  it("adds the pool picture to a new workspace as a picture file by the user, sharing its bytes", async () => {
    const { store, workspaces } = setup();
    const { session, picture, id } = await sessionWith(store, 1);
    const sent = await sendPicture(workspaces, store, picture, { kind: "new", name: workspaceNameFor(session) });
    expect(sent.workspace.name).toBe(session.title);
    expect(sent.written.name).toBe(pictureName(picture));
    const [file] = await workspaces.files(sent.workspace.id);
    expect(file.kind).toBe("picture");
    expect(latest(file).author).toEqual(USER);
    expect(latest(file).bytes).toMatchObject({ store: "image", id });
    // One copy of the bytes, named by both sides.
    expect((await store.listImages()).map((i) => i.id)).toEqual([id]);
    expect(await sha256((await workspaces.content(latest(file)))!)).toBe(id);
    // The session keeps its own record.
    expect(session.pool[0].id).toBe(id);
  });

  it("adds to an existing workspace, a taken name numbered by the module's rule", async () => {
    const { store, workspaces } = setup();
    const { picture } = await sessionWith(store, 2);
    const w = await workspaces.create("Fox");
    const first = await sendPicture(workspaces, store, picture, { kind: "existing", id: w.id });
    const second = await sendPicture(workspaces, store, picture, { kind: "existing", id: w.id });
    expect(first.written.name).toBe(pictureName(picture));
    expect(second.written.name).not.toBe(first.written.name);
    expect(second.written.name).toContain("(2)");
    expect((await store.listImages()).length).toBe(1);
  });

  it("refuses a picture the store no longer holds, and a workspace that is gone", async () => {
    const { store, workspaces } = setup();
    const { picture } = await sessionWith(store, 3);
    await expect(sendPicture(workspaces, store, picture, { kind: "existing", id: "nope" })).rejects.toThrow("gone");
    await store.deleteImage(picture.id);
    await expect(sendPicture(workspaces, store, picture, { kind: "new", name: "x" })).rejects.toThrow("no longer");
    expect(await workspaces.list()).toEqual([]);
  });
});

describe("pictures from a workspace", () => {
  it("lists the present picture files that sit in the image store", async () => {
    const { store, workspaces } = setup();
    const w = await workspaces.create("W");
    await workspaces.add(w.id, [{ name: "a.png", kind: "picture", blob: png(4) }, { name: "b.txt", kind: "text", text: "hi" }, { name: "c.png", kind: "picture", blob: png(5) }], USER);
    const [c] = (await workspaces.files(w.id)).filter((f) => f.name === "c.png");
    await workspaces.deleteFile(w.id, c.id, USER);
    const found = takeablePictures(await workspaces.files(w.id));
    expect(found).toEqual([{ file: expect.any(String), name: "a.png", id: await sha256(png(4)), width: 800, height: 600 }]);
    expect((await store.listImages()).length).toBe(2);
  });

  it("adds them to the pool as inputs under the workspace origin, naming the id the workspace holds", async () => {
    const { store, workspaces } = setup();
    const w = await workspaces.create("W");
    await workspaces.add(w.id, [{ name: "dir/a.png", kind: "picture", blob: png(6) }], USER);
    const found = takeablePictures(await workspaces.files(w.id));
    const [s, numbers] = addToPool(newSession("s", 0), poolEntries(found), 1, true);
    expect(s.pool).toEqual([{ n: 1, id: await sha256(png(6)), width: 800, height: 600, origin: "workspace", name: "dir/a.png" }]);
    expect(s.draft.inputs).toEqual(numbers);
    // No bytes were copied: the store still holds the one image.
    expect((await store.listImages()).length).toBe(1);
  });

  it("holds the request's reference limit", () => {
    const pictures = [1, 2, 3].map((i) => ({ file: `f${i}`, name: `${i}.png`, id: `i${i}`, width: 1, height: 1 }));
    expect(fitInputs(pictures, 0, 10)).toEqual({ taken: pictures, notice: null });
    const fit = fitInputs(pictures, 9, 10);
    expect(fit.taken).toEqual([pictures[0]]);
    expect(fit.notice).toBe("A request takes at most 10 references.");
    expect(fitInputs(pictures, 12, 10).taken).toEqual([]);
  });
});

describe("deleting either side keeps the bytes the other names", () => {
  it("deleting the session keeps a picture a workspace holds, and frees one nothing else names", async () => {
    const { store, workspaces } = setup();
    const kept = await sessionWith(store, 7);
    const lone = await sessionWith(store, 8, "s2");
    const both = addToPool(kept.session, [{ id: lone.id, width: 800, height: 600, origin: "upload" }], 2)[0];
    await sendPicture(workspaces, store, kept.picture, { kind: "new", name: "W" });
    const held = await workspaces.pictures();
    expect(freedWithSession(both, [], [], held)).toEqual([lone.id]);
    // Unknown holders free nothing.
    expect(freedWithSession(both, [], [], null)).toEqual([]);
    expect(freedWithSession(both, [], null, held)).toEqual([]);
  });

  it("deleting the workspace keeps a picture a session names", async () => {
    const { store, workspaces } = setup();
    const { session, picture, id } = await sessionWith(store, 9);
    const sent = await sendPicture(workspaces, store, picture, { kind: "new", name: "W" });
    await workspaces.remove(sent.workspace.id, imagesOf(session));
    expect(await store.getImage(id)).not.toBeNull();
    expect(await workspaces.list()).toEqual([]);
  });

  it("deleting the workspace frees a picture only it names", async () => {
    const { store, workspaces } = setup();
    const { picture, id } = await sessionWith(store, 10);
    const sent = await sendPicture(workspaces, store, picture, { kind: "new", name: "W" });
    await workspaces.remove(sent.workspace.id, []);
    expect(await store.getImage(id)).toBeNull();
  });
});
