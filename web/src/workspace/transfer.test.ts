import { describe, expect, it } from "vitest";
import { createMemoryStore } from "../store/memory.ts";
import { latest, present, USER } from "./files.ts";
import { exportWorkspace, importWorkspace, workspaceNamed, zipName } from "./transfer.ts";
import { createWorkspaces, type Workspaces } from "./workspaces.ts";
import { readZip, writeZip } from "./zip.ts";

// A workspace out as a zip and back in, over the in-memory store.

const bytes = (...n: number[]) => new Uint8Array(n);
const blobOf = (b: Uint8Array, type = "") => new Blob([b as BlobPart], { type });

function module(store = createMemoryStore()): Workspaces {
  let n = 0;
  return createWorkspaces(store, { newId: () => `id${++n}`, measure: async () => ({ width: 2, height: 3 }) });
}

const PNG = bytes(137, 80, 78, 71, 13, 10, 26, 10, 1, 2, 3);
const WAV = bytes(82, 73, 70, 70, 0, 0, 0, 0, 87, 65, 86, 69, 9, 9);
const PDF = bytes(37, 80, 68, 70, 45, 49, 46, 52, 10, 0, 255);
const BIN = bytes(0, 1, 2, 3, 250, 251);
const readPdf = async () => "the pdf text";

async function source() {
  const workspaces = module();
  const w = await workspaces.create("Trip plan");
  await workspaces.add(
    w.id,
    [
      { name: "notes.md", kind: "text", text: "# Trip\ncaffè\n" },
      { name: "site/css/main.css", kind: "text", text: "h1{}" },
      { name: "site/pic.png", kind: "picture", blob: blobOf(PNG, "image/png") },
      { name: "voice/memo.wav", kind: "audio", blob: blobOf(WAV, "audio/wav") },
      { name: "docs/ticket.pdf", kind: "pdf", blob: blobOf(PDF, "application/pdf"), derived: "x" },
      { name: "data/blob.bin", kind: "other", blob: blobOf(BIN) },
      { name: "gone.txt", kind: "text", text: "deleted" },
    ],
    USER,
  );
  const gone = (await workspaces.files(w.id)).find((f) => f.name === "gone.txt")!;
  await workspaces.deleteFile(w.id, gone.id, USER);
  const notes = (await workspaces.files(w.id)).find((f) => f.name === "notes.md")!;
  await workspaces.editText(w.id, notes.id, "Trip", "Journey", USER);
  return { workspaces, w };
}

describe("a workspace as a zip", () => {
  it("holds each present file's current version under its name, with no deleted file and no history", async () => {
    const { workspaces, w } = await source();
    const out = await exportWorkspace(workspaces, w, await workspaces.files(w.id));
    expect(out.name).toBe("Trip plan.zip");
    expect(out.left).toEqual([]);
    const entries = await readZip(new Uint8Array(await out.zip.arrayBuffer()));
    expect(entries.map((e) => e.name).sort()).toEqual(["data/blob.bin", "docs/ticket.pdf", "notes.md", "site/css/main.css", "site/pic.png", "voice/memo.wav"]);
    expect(new TextDecoder().decode(entries.find((e) => e.name === "notes.md")!.data)).toBe("# Journey\ncaffè\n");
  });

  it("round-trips text, picture, audio, PDF and other kinds with names holding /, as new version-1 files of the user's", async () => {
    const { workspaces, w } = await source();
    const out = await exportWorkspace(workspaces, w, await workspaces.files(w.id));
    const target = module();
    const back = await importWorkspace(target, new File([out.zip], "Trip plan.zip"), readPdf);
    if (!back.ok) throw new Error(back.error);
    expect(back.workspace.name).toBe("Trip plan");
    const files = await target.files(back.workspace.id);
    expect(files.map((f) => [f.name, f.kind])).toEqual([
      ["data/blob.bin", "other"],
      ["docs/ticket.pdf", "pdf"],
      ["notes.md", "text"],
      ["site/css/main.css", "text"],
      ["site/pic.png", "picture"],
      ["voice/memo.wav", "audio"],
    ]);
    expect(back.files).toBe(6);
    for (const f of files) {
      expect(f.versions).toHaveLength(1);
      expect(latest(f)).toMatchObject({ n: 1, author: USER });
      expect(present(f)).toBe(true);
    }
    const by = (name: string) => files.find((f) => f.name === name)!;
    const bytesOfFile = async (name: string) => new Uint8Array(await (await target.content(latest(by(name))))!.arrayBuffer());
    expect(latest(by("notes.md")).text).toBe("# Journey\ncaffè\n");
    expect([...(await bytesOfFile("site/pic.png"))]).toEqual([...PNG]);
    expect(latest(by("site/pic.png")).bytes).toMatchObject({ store: "image", width: 2, height: 3 });
    expect([...(await bytesOfFile("voice/memo.wav"))]).toEqual([...WAV]);
    expect([...(await bytesOfFile("docs/ticket.pdf"))]).toEqual([...PDF]);
    expect(latest(by("docs/ticket.pdf")).derived).toBe("the pdf text");
    expect([...(await bytesOfFile("data/blob.bin"))]).toEqual([...BIN]);
  });

  it("does not store a picture twice when the image store already holds it", async () => {
    const store = createMemoryStore();
    const workspaces = module(store);
    const w = await workspaces.create("A");
    await workspaces.add(w.id, [{ name: "p.png", kind: "picture", blob: blobOf(PNG, "image/png") }], USER);
    expect(await store.listImages()).toHaveLength(1);
    const out = await exportWorkspace(workspaces, w, await workspaces.files(w.id));
    const back = await importWorkspace(workspaces, new File([out.zip], "A.zip"));
    expect(back.ok).toBe(true);
    expect(await store.listImages()).toHaveLength(1);
    const ids = (await workspaces.list()).flatMap((x) => [x.id]);
    expect(ids).toHaveLength(2);
  });

  it("cleans names as the module does, skips directory entries, and numbers a duplicate", async () => {
    const text = (s: string) => new TextEncoder().encode(s);
    const zip = await writeZip([
      { name: "dir/", data: new Uint8Array(0) },
      { name: "/abs/top.txt", data: text("1") },
      { name: "a/../../escape.txt", data: text("2") },
      { name: "same.txt", data: text("3") },
      { name: "same.txt", data: text("4") },
    ]);
    const workspaces = module();
    const back = await importWorkspace(workspaces, new File([zip as BlobPart], "odd.ZIP"));
    if (!back.ok) throw new Error(back.error);
    expect(back.workspace.name).toBe("odd");
    const names = (await workspaces.files(back.workspace.id)).map((f) => f.name);
    expect(names).toEqual(["abs/top.txt", "file", "same (2).txt", "same.txt"]);
  });

  it("says a zip that cannot be read cannot, and makes no workspace", async () => {
    const workspaces = module();
    const back = await importWorkspace(workspaces, new File(["not a zip"], "bad.zip"));
    expect(back).toMatchObject({ ok: false });
    if (!back.ok) expect(back.error).toMatch(/^bad\.zip: This is not a zip ignis can read/);
    const empty = await importWorkspace(workspaces, new File([(await writeZip([{ name: "d/", data: new Uint8Array(0) }])) as BlobPart], "e.zip"));
    expect(empty).toEqual({ ok: false, error: "e.zip holds no files." });
    expect(await workspaces.list()).toEqual([]);
  });

  it("names the download after the workspace and the workspace after the zip", () => {
    const w = { format: 1 as const, id: "w", name: 'Trip: "Rome"/2026', created: 0, updated: 0 };
    expect(zipName(w)).toBe("Trip_ _Rome_2026.zip");
    expect(workspaceNamed("my files.zip")).toBe("my files");
    expect(workspaceNamed(".zip")).toBe("Workspace");
  });
});
