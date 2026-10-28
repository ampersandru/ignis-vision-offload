import { describe, expect, it } from "vitest";
import type { Settings, Turn } from "../api/request.ts";
import type { StreamOptions, StreamResult } from "../api/stream.ts";
import { sha256 } from "../store/imageStore.ts";
import { createMemoryStore } from "../store/memory.ts";
import { runAgents } from "../tools/agents/agents.ts";
import { agentExtras, NO_TOOLS } from "../tools/index.ts";
import { type LocalRun, localToolImages, localToolResult, runLocalCalls } from "../tools/local/local.ts";
import { commitAttached, withAttached } from "./attach.ts";
import { latest, USER } from "./files.ts";
import { boxPictures } from "./pictures.ts";
import type { WorkspaceAccess } from "./tools.ts";
import { createWorkspaces, type Workspaces } from "./workspaces.ts";

// Pictures as workspace files (#394), over the in-memory store: their bytes
// kept once in the image store image sessions use, what the model sees of one
// it reads, the pictures make_image makes landing as files, and what deleting
// a workspace frees.

/** A clock that moves one minute per reading. */
function clock() {
  let minute = 0;
  return () => new Date(2026, 9, 28, 14, minute++).getTime();
}

/** A picture's bytes: what a decoder would read as `width` by `height`. */
const png = (fill: number) => new Blob([new Uint8Array(48).fill(fill)], { type: "image/png" });

/** Measures a picture as the browser would, without a canvas: every picture is 3024 by 4032. */
const measure = async () => ({ width: 3024, height: 4032 });

function module(options: { measure?: (blob: Blob) => Promise<{ width: number; height: number; thumb?: Blob }> } = {}) {
  let n = 0;
  const store = createMemoryStore();
  const workspaces = createWorkspaces(store, { now: clock(), newId: () => `id${++n}`, measure: options.measure ?? measure });
  return { store, workspaces };
}

describe("a picture file", () => {
  it("keeps its bytes once, by their SHA-256, in the image store, with its size in pixels", async () => {
    const { store, workspaces } = module();
    const a = await workspaces.create("A");
    const b = await workspaces.create("B");
    await workspaces.add(a.id, [{ name: "photo.png", kind: "picture", blob: png(1) }], USER);
    await workspaces.add(b.id, [{ name: "same.png", kind: "picture", blob: png(1) }], USER);
    const id = await sha256(png(1));
    const [file] = await workspaces.files(a.id);
    expect(file.kind).toBe("picture");
    expect(latest(file).bytes).toEqual({ store: "image", id, type: "image/png", size: 48, width: 3024, height: 4032 });
    expect((await store.listImages()).map((i) => [i.id, i.width, i.height])).toEqual([[id, 3024, 4032]]);
    expect(await store.getBlob(id)).toBeNull();
    expect(latest((await workspaces.files(b.id))[0]).bytes?.id).toBe(id);
  });

  it("is kept as bytes of a kind the model cannot read when the browser cannot decode it", async () => {
    const { store, workspaces } = module({ measure: () => Promise.reject(new Error("not decodable")) });
    const w = await workspaces.create("A");
    await workspaces.add(w.id, [{ name: "photo.heic", kind: "picture", blob: png(2) }], USER);
    const [file] = await workspaces.files(w.id);
    expect(file.kind).toBe("other");
    expect(latest(file).bytes).toMatchObject({ store: "blob", id: await sha256(png(2)) });
    expect(await store.listImages()).toEqual([]);
  });
});

describe("the pictures of the workspaces", () => {
  it("are every version's, in every workspace: what deleting a chat or an image session keeps", async () => {
    const { workspaces } = module();
    const a = await workspaces.create("A");
    const b = await workspaces.create("B");
    const [photo] = await workspaces.add(a.id, [{ name: "photo.png", kind: "picture", blob: png(1) }, { name: "notes.md", kind: "text", text: "x" }], USER);
    await workspaces.add(b.id, [{ name: "other.png", kind: "picture", blob: png(2) }, { name: "data.zip", kind: "other", blob: png(3) }], USER);
    await workspaces.deleteFile(a.id, photo.file, USER);
    expect(await workspaces.pictures()).toEqual(new Set([await sha256(png(1)), await sha256(png(2))]));
  });

  it("deleting a workspace frees only those nothing else names: another workspace, a chat or an image session", async () => {
    const { store, workspaces } = module();
    const a = await workspaces.create("A");
    const b = await workspaces.create("B");
    await workspaces.add(a.id, [1, 2, 3, 4].map((n) => ({ name: `p${n}.png`, kind: "picture" as const, blob: png(n) })), USER);
    await workspaces.add(b.id, [{ name: "copy.png", kind: "picture", blob: png(2) }], USER);
    const [one, two, three, four] = await Promise.all([1, 2, 3, 4].map((n) => sha256(png(n))));
    // An image session's pool names the third, a chat's prompt the fourth.
    await workspaces.remove(a.id, [three, four]);
    const left = new Set((await store.listImages()).map((i) => i.id));
    expect([one, two, three, four].map((id) => left.has(id))).toEqual([false, true, true, true]);
    // While the image sessions or the chats are not read yet, no picture goes.
    const c = await workspaces.create("C");
    await workspaces.add(c.id, [{ name: "p5.png", kind: "picture", blob: png(5) }], USER);
    await workspaces.remove(c.id, null);
    expect(await store.getImage(await sha256(png(5)))).not.toBeNull();
  });
});

const settings: Settings = { systemPrompt: "", temperature: 1, topP: 0.95, maxTokens: 100, reasoningEffort: "none", thinkingBudget: null, laneTag: "interactive" };

/** A chat bound to `workspace`, reading with or without vision; its pictures redrawn to 1568 on the long edge, as the canvas would. */
function reader(workspaces: Workspaces, workspace: string, vision: boolean | (() => boolean)): WorkspaceAccess & { scaled: Blob[] } {
  const scaled: Blob[] = [];
  return {
    workspaces,
    chat: workspaces.chat("chat-a", { bound: () => workspace, name: () => "Trip", onBorn: () => {} }),
    vision: typeof vision === "function" ? vision : () => vision,
    scale: async (blob) => {
      scaled.push(blob);
      return { url: "data:image/jpeg;base64,U0NBTEVE", width: 1176, height: 1568 };
    },
    scaled,
  };
}

let callN = 0;

async function read(access: WorkspaceAccess, name: string): Promise<LocalRun> {
  const call = { id: `c${++callN}`, name: "read_file", arguments: JSON.stringify({ name }) };
  const [ran] = await runLocalCalls([call], { settings, signal: new AbortController().signal, jsSafetyCheck: false, workspace: access }, () => {});
  return ran;
}

describe("read_file on a picture", () => {
  async function withPhoto() {
    const { store, workspaces } = module();
    const w = await workspaces.create("Trip");
    await workspaces.add(w.id, [{ name: "photo.png", kind: "picture", blob: png(1) }], USER);
    return { store, workspaces, w };
  }

  it("returns the picture itself, redrawn to a 1568-pixel long edge, when the loaded model has vision", async () => {
    const { workspaces, w } = await withPhoto();
    const access = reader(workspaces, w.id, true);
    const ran = await read(access, "photo.png");
    expect(ran.status).toBe("done");
    expect(await Promise.all(access.scaled.map((b) => sha256(b)))).toEqual([await sha256(png(1))]);
    expect(localToolImages(ran)).toEqual([{ name: "photo.png", url: "data:image/jpeg;base64,U0NBTEVE", width: 1176, height: 1568 }]);
    expect(localToolResult(ran)).toBe("[photo.png, version 1: a picture of 3024 × 4032 pixels, 48 bytes, shown here at 1176 × 1568.]");
  });

  it("returns the picture's name, size in pixels and bytes as text when the loaded model has no vision, and no picture", async () => {
    const { workspaces, w } = await withPhoto();
    const access = reader(workspaces, w.id, false);
    const ran = await read(access, "photo.png");
    expect(ran.status).toBe("done");
    expect(localToolImages(ran)).toEqual([]);
    expect(access.scaled).toEqual([]);
    expect(localToolResult(ran)).toBe(
      "[photo.png, version 1: a picture of 3024 × 4032 pixels, 48 bytes. The loaded model has no vision, so the picture itself is not shown to you.]",
    );
  });

  it("asks at each call whether the model has vision, so a switch between two reads changes what the second returns", async () => {
    const { workspaces, w } = await withPhoto();
    let vision = true;
    const access = reader(workspaces, w.id, () => vision);
    expect(localToolImages(await read(access, "photo.png"))).toHaveLength(1);
    vision = false;
    expect(localToolImages(await read(access, "photo.png"))).toEqual([]);
  });

  it("sends no picture when nothing says whether the model has vision", async () => {
    const { workspaces, w } = await withPhoto();
    const { vision: _vision, ...unknown } = reader(workspaces, w.id, true);
    const ran = await read(unknown, "photo.png");
    expect(localToolImages(ran)).toEqual([]);
    expect(localToolResult(ran)).toContain("no vision");
  });
});

describe("a sub-agent reading a picture", () => {
  it("sees it as the chat does, on its own next request, and keeps no copy of it in its run", async () => {
    const { workspaces } = module();
    const w = await workspaces.create("Trip");
    await workspaces.add(w.id, [{ name: "photo.png", kind: "picture", blob: png(1) }], USER);
    const sent: Turn[][] = [];
    const stream = async (o: StreamOptions): Promise<StreamResult> => {
      sent.push(o.request.turns);
      if (o.request.turns.length === 1) o.onEvent({ kind: "tool_call", call: { id: "r1", name: "read_file", arguments: '{"name":"photo.png"}' } });
      else o.onEvent({ kind: "content", text: "A dog on a beach." });
      return { ok: true, timeline: { sentAt: 0, firstTokenAt: 1, lastTokenAt: 2, endedAt: 3, finishReason: "stop", stopped: false } };
    };
    const [ran] = await runAgents([{ callId: "call_9", name: "looker", prompt: "describe photo.png" }], {
      settings,
      tools: agentExtras({ ...NO_TOOLS, workspace: true }),
      local: { jsSafetyCheck: false, workspace: reader(workspaces, w.id, true) },
      signal: new AbortController().signal,
      onUpdate: () => {},
      stream,
    });
    expect(sent[1].at(-1)).toMatchObject({ role: "tool", toolCallId: "r1", images: [{ name: "photo.png", url: "data:image/jpeg;base64,U0NBTEVE" }] });
    expect(ran.local?.[0]).toMatchObject({ tool: "read_file", status: "done", picture: { name: "photo.png", seen: true } });
    expect(ran.local?.[0].images).toBeUndefined();
  });
});

describe("a picture attached or pasted in the prompt box", () => {
  it("becomes a file of the chat's workspace, born for it, as it was picked; the prompt names it, and carries it inline as before", async () => {
    const { store, workspaces } = module();
    let bound: string | undefined;
    const chat = workspaces.chat("chat-a", { bound: () => bound, name: () => "What is this?", onBorn: (id) => (bound = id) });
    const picked = new File([png(7)], "image.png", { type: "image/png" });
    // The box holds the redraw for the wire beside the picture as it was picked.
    const box = [{ name: "image.png", url: "data:image/jpeg;base64,UkVEUkFX", width: 1176, height: 1568, full: picked }];
    const names = await commitAttached(workspaces, chat, boxPictures(box));
    expect(names).toEqual(["image.png"]);
    expect(withAttached("What is this?", names)).toBe("What is this?\n\n[Attached to the workspace: image.png]");
    const [file] = await workspaces.files(bound!);
    expect(file).toMatchObject({ name: "image.png", kind: "picture" });
    expect(latest(file).author).toEqual(USER);
    expect(latest(file).bytes).toMatchObject({ store: "image", id: await sha256(picked) });
    // The same picture pasted again, in another prompt, is another file and the same bytes.
    expect(await commitAttached(workspaces, chat, boxPictures(box))).toEqual(["image (2).png"]);
    expect(await store.listImages()).toHaveLength(1);
  });
});
