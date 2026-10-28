import { describe, expect, it } from "vitest";
import { createMemoryStore } from "../store/memory.ts";
import type { Attachment } from "../tools/local/attachments.ts";
import { type AttachedFile, attachedChip, commitAttached, fromLegacy, readAttached, readUpload, takeTurnFiles, turnAdds, withAttached } from "./attach.ts";
import { latest } from "./files.ts";
import { createWorkspaces } from "./workspaces.ts";

// The user's entrance to the workspace from the prompt box: a file attached
// becomes a file of the chat's workspace when its prompt is sent, born if the
// chat has none, and the prompt names it. An older chat's attachments become
// the first files of its workspace when it is resumed.

function binding(title = "Reading list", refuses?: "putBlob" | "putImage") {
  let n = 0;
  const memory = createMemoryStore();
  const no = () => Promise.reject(new Error("The quota is exceeded."));
  const store = refuses ? { ...memory, [refuses]: no } : memory;
  const workspaces = createWorkspaces(store, { newId: () => `id${++n}`, measure: async () => ({ width: 8, height: 6 }) });
  let bound: string | undefined;
  const chat = workspaces.chat("chat-a", { bound: () => bound, name: () => title, onBorn: (id) => (bound = id) });
  return { store, workspaces, chat, bound: () => bound };
}

const pdfReader = async () => "[Page 1]\nThe brief";

describe("attaching files", () => {
  it("reads a text file and a PDF for the prompt, and refuses what is neither", async () => {
    const text = await readAttached(new File(["hello\n"], "notes.txt", { type: "text/plain" }), [], pdfReader);
    expect(text).toMatchObject({ ok: true, file: { name: "notes.txt", kind: "text", text: "hello\n", size: 6 } });
    const pdf = await readAttached(new File([new Uint8Array([37, 80, 68, 70])], "brief.pdf", { type: "application/pdf" }), [], pdfReader);
    expect(pdf).toMatchObject({ ok: true, file: { name: "brief.pdf", kind: "pdf", derived: "[Page 1]\nThe brief", size: 4 } });
    expect(pdf.ok && pdf.file.blob?.size).toBe(4);
    // A `.ts` file a browser calls a video is still text when its bytes are.
    expect(await readAttached(new File(["let x = 1;"], "app.ts", { type: "video/mp2t" }), [], pdfReader)).toMatchObject({ ok: true, file: { kind: "text" } });
    expect(await readAttached(new File([new Uint8Array([0, 1, 2])], "blob.bin"), [], pdfReader)).toEqual({ ok: false, error: "blob.bin is not a text, PDF or audio file." });
    expect(await readAttached(new File(["x"], "notes.txt"), ["notes.txt"], pdfReader)).toMatchObject({ ok: true, file: { name: "notes (2).txt" } });
  });

  it("adds the attached files to the chat's workspace, born with them and named after the chat, and the prompt names them", async () => {
    const { workspaces, chat, bound } = binding();
    const read = await readAttached(new File(["hello\n"], "notes.txt", { type: "text/plain" }), [], pdfReader);
    const pdf = await readAttached(new File([new Uint8Array([37, 80, 68, 70])], "brief.pdf", { type: "application/pdf" }), [], pdfReader);
    if (!read.ok || !pdf.ok) throw new Error("unread");
    const names = await commitAttached(workspaces, chat, [read.file, pdf.file]);
    expect(names).toEqual(["notes.txt", "brief.pdf"]);
    const [workspace] = await workspaces.list();
    expect(workspace).toMatchObject({ id: bound(), name: "Reading list" });
    const files = await workspaces.files(bound()!);
    expect(files.map((f) => [f.name, f.kind, latest(f).author])).toEqual([
      ["brief.pdf", "pdf", { by: "user" }],
      ["notes.txt", "text", { by: "user" }],
    ]);
    expect(latest(files[0])).toMatchObject({ derived: "[Page 1]\nThe brief", bytes: { store: "blob", type: "application/pdf", size: 4 } });
    expect(withAttached("Summarise them", names)).toBe("Summarise them\n\n[Attached to the workspace: notes.txt, brief.pdf]");
    expect(withAttached("", ["notes.txt"])).toBe("[Attached to the workspace: notes.txt]");
    expect(withAttached("Hi", undefined)).toBe("Hi");
  });

  it("attaches audio, which becomes an audio file of the workspace with its bytes, and the prompt names it", async () => {
    const { workspaces, chat, bound } = binding();
    const read = await readAttached(new File([new Uint8Array(2048)], "standup.m4a", { type: "audio/mp4" }), [], pdfReader);
    expect(read).toMatchObject({ ok: true, file: { name: "standup.m4a", kind: "audio", size: 2048 } });
    if (!read.ok) throw new Error("unread");
    expect(read.file.blob?.size).toBe(2048);
    expect(read.file.text).toBeUndefined();
    expect(attachedChip(read.file)).toEqual({ name: "standup.m4a", kind: "audio", characters: 0, size: 2048 });
    // Its transcript is not made here: the model's first read of it asks the server.
    expect(read.file.derived).toBeUndefined();
    const names = await commitAttached(workspaces, chat, [read.file]);
    expect(withAttached("What did we decide?", names)).toBe("What did we decide?\n\n[Attached to the workspace: standup.m4a]");
    const [file] = await workspaces.files(bound()!);
    expect([file.name, file.kind, latest(file).author]).toEqual(["standup.m4a", "audio", { by: "user" }]);
    expect(latest(file).bytes).toMatchObject({ store: "blob", type: "audio/mp4", size: 2048 });
  });

  it("gives an attached file a name of its own when the workspace already has one", async () => {
    const { workspaces, chat } = binding();
    const first = await commitAttached(workspaces, chat, [{ name: "notes.txt", kind: "text", text: "a" }]);
    const again = await commitAttached(workspaces, chat, [{ name: "notes.txt", kind: "text", text: "b" }]);
    expect([...first, ...again]).toEqual(["notes.txt", "notes (2).txt"]);
    expect(await workspaces.list()).toHaveLength(1);
  });

  it("turns an older chat's attachments into the first files of its workspace", async () => {
    const { workspaces, chat, bound } = binding("Old chat");
    const older: Attachment[] = [
      { name: "notes.txt", size: 5, kind: "text", text: "hello" },
      { name: "brief.pdf", size: 900, kind: "pdf", text: "[Page 1]\nThe brief" },
    ];
    const names = await commitAttached(workspaces, chat, older.map(fromLegacy));
    expect(names).toEqual(["notes.txt", "brief.pdf"]);
    const files = await workspaces.files(bound()!);
    expect(files.map((f) => [f.name, f.kind, latest(f).text, latest(f).derived])).toEqual([
      ["brief.pdf", "pdf", undefined, "[Page 1]\nThe brief"],
      ["notes.txt", "text", "hello", undefined],
    ]);
    expect(older.map(fromLegacy).map(attachedChip)).toEqual([
      { name: "notes.txt", kind: "text", characters: 5, size: 5 },
      { name: "brief.pdf", kind: "pdf", characters: 18, size: 900 },
    ]);
  });

  it("keeps a file of any kind the panel takes in: text and a PDF read as here, any other bytes as they are", async () => {
    expect(await readUpload(new File(["hello\n"], "notes.txt", { type: "text/plain" }), pdfReader)).toEqual({ name: "notes.txt", kind: "text", text: "hello\n" });
    const pdf = await readUpload(new File([new Uint8Array([37, 80, 68, 70])], "brief.pdf", { type: "application/pdf" }), pdfReader);
    expect(pdf).toMatchObject({ name: "brief.pdf", kind: "pdf", derived: "[Page 1]\nThe brief" });
    expect(pdf.blob?.size).toBe(4);
    expect(await readUpload(new File(["let x = 1;"], "app.ts", { type: "video/mp2t" }), pdfReader)).toEqual({ name: "app.ts", kind: "text", text: "let x = 1;" });
    for (const [name, type, kind] of [["data.bin", "", "other"], ["core", "", "other"], ["photo.png", "image/png", "picture"], ["memo.wav", "audio/wav", "audio"], ["sheet.xlsx", "", "other"]] as const) {
      const read = await readUpload(new File([new Uint8Array([0, 1, 2])], name, { type }), pdfReader);
      expect(read).toMatchObject({ name, kind });
      expect(read.text).toBeUndefined();
      expect(read.blob?.size).toBe(3);
    }
    // A PDF that cannot be read is still kept, with no text.
    const broken = await readUpload(new File([new Uint8Array([1])], "bad.pdf"), async () => Promise.reject(new Error("bad XRef")));
    expect(broken).toMatchObject({ name: "bad.pdf", kind: "pdf" });
    expect(broken.derived).toBeUndefined();
  });

  it("leaves an older chat's attachments with it while the Workspace switch is off, and migrates them, first, at the first turn with it on", () => {
    const older: Attachment[] = [{ name: "notes.txt", size: 5, kind: "text", text: "hello" }];
    const box: AttachedFile[] = [{ name: "plan.md", kind: "text", text: "- go", size: 4 }];
    const photo: AttachedFile = { name: "photo.png", kind: "picture", blob: new Blob([new Uint8Array([1])]), size: 1 };
    expect(turnAdds(false, older, { files: box, pictures: [photo] })).toEqual({ files: [], pictures: [], clearsLegacy: false });
    expect(turnAdds(true, older, { files: box, pictures: [photo] })).toEqual({ files: [...older.map(fromLegacy), ...box], pictures: [photo], clearsLegacy: true });
    expect(turnAdds(true, [], { files: box, pictures: [] })).toEqual({ files: box, pictures: [], clearsLegacy: false });
  });

  it("takes a turn's files and pictures into the workspace, and names them for its prompt", async () => {
    const { workspaces, chat, bound } = binding();
    const taken = await takeTurnFiles(workspaces, chat, {
      files: [{ name: "plan.md", kind: "text", text: "- go", size: 4 }],
      pictures: [{ name: "photo.png", kind: "picture", blob: new Blob([new Uint8Array([1])], { type: "image/png" }), size: 1 }],
      clearsLegacy: false,
    });
    expect(taken).toEqual({ sent: true, named: ["plan.md", "photo.png"] });
    expect((await workspaces.files(bound()!)).map((f) => f.name)).toEqual(["photo.png", "plan.md"]);
  });

  it("sends no turn when the store refuses its files, which go back to the box, and says they were not saved in the workspace", async () => {
    const { workspaces, chat } = binding("Reading list", "putBlob");
    const memo: AttachedFile = { name: "memo.wav", kind: "audio", blob: new Blob([new Uint8Array([1])]), size: 1 };
    expect(await takeTurnFiles(workspaces, chat, { files: [memo], pictures: [], clearsLegacy: false })).toEqual({
      sent: false,
      said: "memo.wav was not saved in the workspace: nothing was sent, and nothing is lost. Not saved in this browser: The quota is exceeded.",
    });
    expect(await workspaces.list()).toEqual([]);
  });

  it("sends a turn whose pictures the store refuses, the pictures inline as before, and says they were not saved in the workspace", async () => {
    const { workspaces, chat } = binding("Reading list", "putImage");
    const picture = (name: string): AttachedFile => ({ name, kind: "picture", blob: new Blob([new Uint8Array([2])], { type: "image/png" }), size: 1 });
    expect(await takeTurnFiles(workspaces, chat, { files: [{ name: "plan.md", kind: "text", text: "- go", size: 4 }], pictures: [picture("a.png"), picture("b.png")], clearsLegacy: false })).toEqual({
      sent: true,
      named: ["plan.md"],
      said: "a.png, b.png were not saved in the workspace: the model sees them in this prompt only. Not saved in this browser: The quota is exceeded.",
    });
  });

  it("classifies a file the same way in the prompt box and in the panel: what one keeps as text the other does, and what the box refuses the panel keeps as bytes", async () => {
    const bytes = new Uint8Array([0, 1, 2]);
    const cases: [File, string][] = [
      [new File(["hello\n"], "notes.txt", { type: "text/plain" }), "text"],
      // A browser calls a `.ts` file a video: its name says text.
      [new File(["let x = 1;"], "app.ts", { type: "video/mp2t" }), "text"],
      [new File(["a,b"], "sheet.xlsx"), "other"],
      [new File([bytes], "recording", { type: "audio/webm" }), "audio"],
      [new File([bytes], "core"), "other"],
      [new File([bytes], "photo.png", { type: "image/png" }), "picture"],
      [new File([new Uint8Array([37, 80, 68, 70])], "brief.pdf", { type: "application/pdf" }), "pdf"],
    ];
    for (const [file, kind] of cases) {
      const upload = await readUpload(file, pdfReader);
      const attached = await readAttached(file, [], pdfReader);
      expect([file.name, upload.kind]).toEqual([file.name, kind]);
      expect([file.name, attached.ok ? attached.file.kind : "refused"]).toEqual([file.name, ["text", "pdf", "audio"].includes(kind) ? kind : "refused"]);
    }
  });

  it("adds nothing, and births nothing, when nothing is attached", async () => {
    const { workspaces, chat } = binding();
    expect(await commitAttached(workspaces, chat, [])).toEqual([]);
    expect(await workspaces.list()).toEqual([]);
  });
});
