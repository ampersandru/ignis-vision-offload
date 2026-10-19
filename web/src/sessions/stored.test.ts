import { describe, expect, it } from "vitest";
import { createMemoryStore } from "../store/memory.ts";
import type { Mark } from "../markup/markup.ts";
import type { Figures } from "../metrics/figures.ts";
import type { Message, Session } from "./sessions.ts";
import { sha256 } from "../store/imageStore.ts";
import { CHAT_VERSION, freedBy, keepPictures, picturesOf, restoreChats, settle, toStored, withPictures, worthKeeping } from "./stored.ts";

// A chat session as the page's store keeps it (spec playground-enhancements/03):
// plain data, Blobs left to the image store by id; read back newest first with
// new ids, and settled — a page that went away mid-turn left nothing running.

const figures: Figures = {
  ttftMs: 120,
  decodeTokensPerSec: 80,
  durationMs: 900,
  promptTokens: 40,
  completionTokens: 60,
  finishReason: "tool_calls",
  partial: false,
};

const msg = (id: number, role: Message["role"], content: string, extra: Partial<Message> = {}): Message => ({
  id,
  role,
  content,
  reasoning: "",
  streaming: false,
  ...extra,
});

/** A session that has been through a turn with a picture, tools and their results. */
function finished(id = 4): Session {
  return {
    id,
    title: "Draw a fox",
    startedAt: new Date("2026-10-19T08:00:00Z"),
    attachments: [{ name: "notes.txt", size: 5, kind: "text", text: "hello" }],
    log: [{ n: 1, at: "08:00:01", laneTag: "interactive", reasoningEffort: "none", figures }],
    messages: [
      msg(10, "user", "Draw a fox like this", {
        dateTime: "Sunday 19 October 2026, 10:00",
        switch: { to: "qwen", status: "switched" },
        images: [
          {
            name: "fox.jpg",
            url: "data:image/jpeg;base64,AAAA",
            width: 4,
            height: 3,
            full: new Blob([new Uint8Array([1, 2])], { type: "image/png" }),
            markup: { source: new Blob([new Uint8Array([3])]), marks: [], keepOutside: true, region: new Blob([new Uint8Array([4])]) },
            kept: { full: "f".repeat(64), markup: { source: "s".repeat(64), region: "r".repeat(64), marks: [], keepOutside: true } },
          },
        ],
      }),
      msg(11, "assistant", "", {
        reasoning: "thinking",
        figures,
        toolCalls: [
          { id: "c1", name: "web_search", arguments: "{}" },
          { id: "c2", name: "ask_user", arguments: "{}" },
        ],
        web: [{ callId: "c1", tool: "web_search", target: "fox", status: "done", results: [] }],
        questions: [{ callId: "c2", question: "Red?", options: [], status: "answered", answer: "yes" }],
      }),
      msg(12, "tool", "results", { toolCallId: "c1" }),
      msg(13, "tool", "yes", { toolCallId: "c2" }),
      msg(14, "assistant", "Here it is.", { figures }),
    ],
  };
}

const counter = (from: number) => {
  let n = from;
  return () => n++;
};

describe("a chat session in the store", () => {
  it("is plain data: no Blob and no Date reaches the store", () => {
    const stored = toStored(finished(), "key-1");
    expect(JSON.parse(JSON.stringify(stored))).toEqual(stored);
    expect(stored).toMatchObject({ version: CHAT_VERSION, key: "key-1", id: 4, startedAt: Date.parse("2026-10-19T08:00:00Z") });
    const image = stored.messages[0].images![0];
    expect(image).not.toHaveProperty("full");
    expect(image).not.toHaveProperty("markup");
    expect(image.kept?.full).toBe("f".repeat(64));
  });

  it("comes back as it was, the pictures beside the redraw still to read", () => {
    const original = finished();
    const [back] = restoreChats([toStored(original, "key-1")], 1).sessions;
    const { full: _full, markup: _markup, ...redraw } = original.messages[0].images![0];
    expect(back).toEqual({
      ...original,
      id: back.id,
      messages: [{ ...original.messages[0], images: [redraw] }, ...original.messages.slice(1)],
    });
    expect(back.startedAt).toBeInstanceOf(Date);
  });

  it("names the store's pictures it keeps", () => {
    expect([...picturesOf(finished())].sort()).toEqual(["f".repeat(64), "r".repeat(64), "s".repeat(64)]);
    expect(picturesOf({ ...finished(), messages: [] }).size).toBe(0);
  });

  it("is worth keeping once it holds a message or a file", () => {
    const empty = { ...finished(), messages: [], attachments: [] };
    expect(worthKeeping(empty)).toBe(false);
    expect(worthKeeping({ ...empty, attachments: finished().attachments })).toBe(true);
    expect(worthKeeping({ ...empty, messages: [msg(1, "user", "hi")] })).toBe(true);
  });
});

describe("reading the chat sessions back", () => {
  it("lists them newest first, with new ids above every message id kept", () => {
    const older = toStored({ ...finished(3), title: "older" }, "a");
    const newer = toStored({ ...finished(7), title: "newer", messages: [msg(50, "user", "hi")] }, "b");
    const { sessions, keys, nextId } = restoreChats([older, newer], 2);
    expect(sessions.map((s) => s.title)).toEqual(["newer", "older"]);
    expect(sessions.map((s) => s.id)).toEqual([52, 51]);
    expect(nextId).toBe(53);
    expect(keys.get(52)).toBe("b");
    expect(keys.get(51)).toBe("a");
  });

  it("starts above the page's own ids when those are higher", () => {
    const { sessions, nextId } = restoreChats([toStored(finished(), "a")], 100);
    expect(sessions[0].id).toBe(100);
    expect(nextId).toBe(101);
  });

  it("leaves out what it cannot read: a newer page's schema, or something that is not a session", () => {
    const newer = { ...toStored(finished(), "a"), version: CHAT_VERSION + 1 };
    const { sessions } = restoreChats([newer, null, "x", { version: 1 }, toStored(finished(), "b")], 1);
    expect(sessions).toHaveLength(1);
  });

  it("reads a session without a log or files, as an older page could have written it", () => {
    const { log: _log, attachments: _files, ...bare } = toStored(finished(), "a");
    const [back] = restoreChats([bare], 1).sessions;
    expect(back.log).toEqual([]);
    expect(back.attachments).toEqual([]);
  });
});

describe("settling a session a page left mid-turn", () => {
  it("brings a streaming reply back stopped, with the text it had", () => {
    const s = { ...finished(), messages: [msg(1, "user", "hi"), msg(2, "assistant", "Half a sent", { streaming: true, queued: true })] };
    const [, reply] = settle(s, counter(100)).messages;
    expect(reply).toMatchObject({ content: "Half a sent", streaming: false });
    expect(reply.queued).toBeUndefined();
  });

  it("stops every tool run, skips a waiting question and stops a switch under way", () => {
    const s: Session = {
      ...finished(),
      messages: [
        msg(1, "user", "go", { switch: { to: "qwen", status: "switching" } }),
        msg(2, "assistant", "", {
          toolCalls: [],
          agents: [{ callId: "a", name: "scout", prompt: "p", status: "running", reasoning: "", content: "", web: [{ callId: "aw", tool: "web_fetch", target: "u", status: "running" }], local: [{ callId: "al", tool: "run_js", status: "checking" }] }],
          web: [{ callId: "w", tool: "web_search", target: "q", status: "running" }],
          local: [{ callId: "l", tool: "run_js", status: "running" }],
          questions: [{ callId: "q", question: "?", options: [], status: "waiting" }],
          imageRuns: [{ callId: "i", prompt: "fox", references: [], transparent: false, count: 1, status: "making" }],
        }),
      ],
    };
    const [user, reply] = settle(s, counter(100)).messages;
    expect(user.switch?.status).toBe("stopped");
    expect(reply.agents?.[0].status).toBe("stopped");
    expect(reply.agents?.[0].web?.[0].status).toBe("stopped");
    expect(reply.agents?.[0].local?.[0].status).toBe("stopped");
    expect(reply.web?.[0].status).toBe("stopped");
    expect(reply.local?.[0].status).toBe("stopped");
    expect(reply.questions?.[0].status).toBe("skipped");
    expect(reply.imageRuns?.[0].status).toBe("stopped");
  });

  it("answers every call left without a result, right after the ones it has, so the history stays well-formed", () => {
    const s: Session = {
      ...finished(),
      messages: [
        msg(1, "user", "go"),
        msg(2, "assistant", "", {
          toolCalls: [
            { id: "c1", name: "web_search", arguments: "{}" },
            { id: "c2", name: "web_fetch", arguments: "{}" },
          ],
        }),
        msg(3, "tool", "done", { toolCallId: "c1" }),
      ],
    };
    const settled = settle(s, counter(100)).messages;
    expect(settled.map((m) => [m.id, m.role, m.toolCallId])).toEqual([
      [1, "user", undefined],
      [2, "assistant", undefined],
      [3, "tool", "c1"],
      [100, "tool", "c2"],
    ]);
    expect(settled[3].content).toMatch(/stopped/i);
  });

  it("leaves a failed reply's calls unanswered, as the turn loop does: the history drops the reply", () => {
    const s: Session = {
      ...finished(),
      messages: [
        msg(1, "user", "go"),
        msg(2, "assistant", "", { toolCalls: [{ id: "c1", name: "web_search", arguments: "{}" }], error: "The stream broke." }),
      ],
    };
    expect(settle(s, counter(100))).toBe(s);
  });

  it("leaves a finished session as it was", () => {
    const s = finished();
    expect(settle(s, counter(100))).toBe(s);
  });
});

describe("the pictures beside a chat image", () => {
  it("are read back from the store, and left out where the store no longer has them", async () => {
    const store = createMemoryStore();
    const full = await store.putImage(new Blob([new Uint8Array([1, 2])], { type: "image/png" }), { width: 4, height: 3 });
    const source = await store.putImage(new Blob([new Uint8Array([3])], { type: "image/png" }), { width: 4, height: 3 });
    const region = await store.putImage(new Blob([new Uint8Array([4])], { type: "image/png" }), { width: 4, height: 3 });
    const image = { name: "fox.jpg", url: "data:,", width: 4, height: 3 };
    const s: Session = {
      ...finished(),
      messages: [
        msg(1, "user", "a", { images: [{ ...image, kept: { full: full.id, markup: { source: source.id, region: region.id, marks: [], keepOutside: true } } }] }),
        msg(2, "user", "b", { images: [{ ...image, kept: { full: "gone" } }] }),
      ],
    };
    const back = await withPictures(s, (id) => store.getImage(id));
    const [read, missing] = back.messages.map((m) => m.images![0]);
    expect([...new Uint8Array(await read.full!.arrayBuffer())]).toEqual([1, 2]);
    expect(read.markup).toMatchObject({ marks: [], keepOutside: true });
    expect([...new Uint8Array(await read.markup!.region.arrayBuffer())]).toEqual([4]);
    expect(missing.full).toBeUndefined();
    expect(missing.kept).toEqual({ full: "gone" });
  });
});

describe("keeping a sent prompt's pictures", () => {
  const bytes = (...b: number[]) => new Blob([new Uint8Array(b)], { type: "image/png" });
  const image = { name: "fox.jpg", url: "data:,", width: 4, height: 3 };

  it("puts the picture at its own size and the markup's two in the store, and names them on the image", async () => {
    const store = createMemoryStore();
    const keep = (blob: Blob) => store.putImage(blob, { width: 1, height: 1 }).then((p) => p.id);
    const marks: Mark[] = [{ kind: "ellipse", color: "#f00", width: 2, from: [0, 0], to: [1, 1] }];
    const kept = await keepPictures(
      [
        { ...image, full: bytes(1), markup: { source: bytes(2), marks, keepOutside: true, region: bytes(3) } },
        image,
      ],
      keep,
    );
    expect(kept?.[0].kept).toEqual({
      full: await sha256(bytes(1)),
      markup: { source: await sha256(bytes(2)), region: await sha256(bytes(3)), marks, keepOutside: true },
    });
    expect(kept?.[0].full).toBeDefined();
    expect(kept?.[1]).toBe(image);
    expect(await store.listImages()).toHaveLength(3);
  });

  it("has nothing to do for images already kept, or without pictures beside them", async () => {
    const keep = async () => "never";
    expect(await keepPictures([image, { ...image, full: bytes(1), kept: { full: "x" } }], keep)).toBeNull();
  });

  it("keeps nothing when the store refuses: the session is kept with the redraw alone", async () => {
    const keep = async (): Promise<string> => {
      throw new Error("QuotaExceededError");
    };
    expect(await keepPictures([{ ...image, full: bytes(1) }], keep)).toBeNull();
  });
});

describe("the pictures deleting a session frees", () => {
  const naming = (id: number, ...full: string[]): Session => ({
    ...finished(id),
    messages: full.map((f, i) => msg(i + 1, "user", "x", { images: [{ name: "p", url: "data:,", width: 1, height: 1, kept: { full: f } }] })),
  });

  it("are the ones no session left names, nor the Image view's sessions", () => {
    expect(freedBy(naming(1, "a", "b", "c", "d"), [naming(2, "b")], ["c"])).toEqual(["a", "d"]);
  });

  it("are none while the Image view has not read its sessions", () => {
    expect(freedBy(naming(1, "a"), [], null)).toEqual([]);
  });
});
