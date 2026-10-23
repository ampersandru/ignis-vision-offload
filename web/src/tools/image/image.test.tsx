import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RGBA_TAIL } from "../../image/studio.ts";
import { ImageStrip } from "./ImageStrip.tsx";
import {
  IMAGE_TOOL_NAME,
  type ImageRun,
  type ImageTask,
  imageSummary,
  imageToolResult,
  madeSoFar,
  parseImageCall,
  picturesByName,
  runImages,
  sizeForAspect,
} from "./image.ts";

// The picture tool (tools/image/image.ts): its calls read and refused, the
// pictures of a session by name, the presets for a shape, and the visit to
// the image model a reply's calls make — there, the pictures, and back to the
// text model whatever happened — over a server that switches when asked.

const call = (args: object | string, name = IMAGE_TOOL_NAME) => ({ id: "c1", name, arguments: typeof args === "string" ? args : JSON.stringify(args) });

const SIZES = ["2048x2048", "2400x1792", "1792x2400", "2528x1696", "1696x2528", "2752x1536", "1536x2752", "1024x1024"];

describe("a make_image call", () => {
  it("reads the prompt and the options it was given, and defaults the rest", () => {
    expect(parseImageCall(call({ prompt: " a fox " }))).toEqual({
      ok: true,
      task: { callId: "c1", prompt: "a fox", references: [], transparent: false, count: 1 },
    });
    expect(parseImageCall(call({ prompt: "a hat", references: ["user-1", "made-2"], aspect: "16:9", transparent: true, count: 2, seed: 7 }))).toEqual({
      ok: true,
      task: { callId: "c1", prompt: "a hat", references: ["user-1", "made-2"], aspect: "16:9", transparent: true, count: 2, seed: "7" },
    });
    // The model repeats a drawn u64 seed: kept exactly, not rounded by JSON.parse.
    expect(parseImageCall({ id: "c1", name: IMAGE_TOOL_NAME, arguments: '{"prompt":"a hat","seed":5936113633719656123}' })).toMatchObject({
      ok: true,
      task: { seed: "5936113633719656123" },
    });
  });

  it("refuses arguments it cannot run, saying what was wrong", () => {
    const error = (args: object | string) => {
      const parsed = parseImageCall(call(args));
      return parsed.ok ? null : parsed.run.error;
    };
    expect(error("{")).toContain("not valid JSON");
    expect(error({ prompt: "" })).toContain('non-empty "prompt"');
    expect(error({ prompt: "x", references: "user-1" })).toContain("list of picture names");
    expect(error({ prompt: "x", aspect: "5:4" })).toContain('"aspect" is one of');
    expect(error({ prompt: "x", count: 9 })).toContain('"count"');
    expect(error({ prompt: "x", seed: -1 })).toContain('"seed"');
    expect(parseImageCall(call({ prompt: "x" }), [])).toMatchObject({ ok: false });
  });
});

describe("the pictures and the presets", () => {
  it("names the user's pictures in the order sent, and the tool's as they were made", () => {
    const run = (made: string[]): ImageRun => ({
      callId: "c",
      prompt: "p",
      references: [],
      transparent: false,
      count: 1,
      status: "done",
      made: made.map((name) => ({ name, url: `data:${name}`, size: "1024x1024" })),
    });
    const messages = [
      { role: "user", images: [{ url: "data:u1" }, { url: "data:u2" }] },
      { role: "assistant", imageRuns: [run(["made-1"])] },
      { role: "user", images: [{ url: "data:u3" }] },
      { role: "assistant", imageRuns: [run(["made-2", "made-3"])] },
    ];
    expect([...picturesByName(messages)].map(([name, p]) => [name, p.url])).toEqual([
      ["user-1", "data:u1"],
      ["user-2", "data:u2"],
      ["made-1", "data:made-1"],
      ["user-3", "data:u3"],
      ["made-2", "data:made-2"],
      ["made-3", "data:made-3"],
    ]);
    expect(madeSoFar(messages)).toBe(3);
  });

  it("serves a shape with the largest preset of its ratio, and no shape as auto", () => {
    expect(sizeForAspect("1:1", SIZES)).toBe("2048x2048");
    expect(sizeForAspect("16:9", SIZES)).toBe("2752x1536");
    expect(sizeForAspect("4:3", ["1024x1024"])).toBe("auto");
    expect(sizeForAspect(undefined, SIZES)).toBe("auto");
  });

  it("tells the model the names, sizes and seeds, and that the user sees them", () => {
    const run: ImageRun = {
      callId: "c",
      prompt: "p",
      references: [],
      transparent: false,
      count: 1,
      status: "done",
      made: [{ name: "made-4", url: "data:x", size: "2048x2048", seed: "9" }],
    };
    expect(imageToolResult(run)).toBe("Made made-4: 2048x2048, seed 9.\nThe user sees the pictures in the conversation.");
    expect(imageToolResult({ ...run, status: "failed", error: "the queue is full" })).toContain("failed: the queue is full\nBefore it failed:\nMade made-4");
    expect(imageToolResult({ ...run, status: "stopped" })).toContain("stopped");
    expect(imageSummary([run, { ...run, status: "making", made: [] }])).toBe("1 picture, under way");
  });
});

/** A server with a text model loaded that switches when asked, makes pictures, and records what it was sent. */
function server(options: { catalog?: object[]; failImage?: boolean } = {}) {
  let loaded = { id: "qwen3.8-27b", kind: "text" };
  const log: string[] = [];
  const bodies: (string | FormData)[] = [];
  const json = (body: object, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  const fetcher = (async (path: string, init: RequestInit = {}) => {
    const method = init.method ?? "GET";
    log.push(`${method} ${path}`);
    if (path === "/v1/models") return json({ data: [{ id: loaded.id, kind: loaded.kind }], status: "ready" });
    if (path === "/v1/models/catalog") {
      return json({ data: options.catalog ?? [{ id: "ignis-image-2.1-turbo", family: "qwenimage21", switchable: true, source: "built-in", on_disk: "complete", loaded: false }] });
    }
    if (path === "/v1/config") return json({ image: { sizes: SIZES, max_n: 10 } });
    if (path === "/v1/models/switch") {
      const to = (JSON.parse(String(init.body)) as { model: string }).model;
      loaded = { id: to, kind: to.startsWith("ignis-image") ? "image" : "text" };
      return json({ from: "x", to });
    }
    if (path === "/v1/images/generations" || path === "/v1/images/edits") {
      bodies.push(init.body as string | FormData);
      if (options.failImage) return json({ error: { message: "the queue is full", code: "engine_full" } }, 503);
      const events = [
        { type: "image_generation.progress", phase: "denoising", image: 0, step: 4, total: 8 },
        { type: "image_generation.completed", image: 0, b64_json: "AAA", seed: 41, size: "2752x1536" },
      ];
      return new Response(events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join(""), { status: 200 });
    }
    throw new Error(`unexpected ${path}`);
  }) as unknown as typeof fetch;
  return { fetcher, log, bodies, loaded: () => loaded.id };
}

const task = (change: Partial<ImageTask> = {}): ImageTask => ({ callId: "c1", prompt: "a fox", references: [], transparent: false, count: 1, ...change });

async function run(tasks: ImageTask[], s: ReturnType<typeof server>, messages: Parameters<typeof picturesByName>[0] = [], signal = new AbortController().signal) {
  const updates: ImageRun[] = [];
  const runs = await runImages(tasks, { messages, signal, onUpdate: (r) => updates.push(r), fetch: s.fetcher, wait: async () => {} });
  return { runs, updates };
}

describe("a reply's visit to the image model", () => {
  it("switches there, makes the picture, and switches back to the text model before the turn goes on", async () => {
    const s = server();
    const { runs, updates } = await run([task({ aspect: "16:9", transparent: true })], s, [{ role: "assistant", imageRuns: [{ ...task(), status: "done", made: [{ name: "made-1", url: "data:x", size: "1x1" }] }] }]);
    expect(runs[0].status).toBe("done");
    expect(runs[0].made).toEqual([{ name: "made-2", url: "data:image/png;base64,AAA", size: "2752x1536", seed: "41" }]);
    expect(s.loaded()).toBe("qwen3.8-27b");
    const switches = s.log.filter((l) => l === "POST /v1/models/switch");
    expect(switches).toHaveLength(2);
    expect(s.log.indexOf("POST /v1/images/generations")).toBeGreaterThan(s.log.indexOf("POST /v1/models/switch"));
    expect(s.log.lastIndexOf("POST /v1/models/switch")).toBeGreaterThan(s.log.indexOf("POST /v1/images/generations"));
    const body = JSON.parse(s.bodies[0] as string) as Record<string, unknown>;
    expect(body).toMatchObject({ size: "2752x1536", background: "transparent", prompt_enhancement: false, stream: true });
    expect(body).not.toHaveProperty("model");
    expect(String(body.prompt)).toContain(RGBA_TAIL);
    expect(updates.map((u) => u.status)).toEqual(expect.arrayContaining(["switching", "making", "returning", "done"]));
    expect(updates.some((u) => u.line === "Denoising · step 4/8")).toBe(true);
  });

  it("sends the named pictures as an edit, in the order the call listed them", async () => {
    const s = server();
    const messages = [{ role: "user", images: [{ url: "data:image/png;base64,AQ==" }, { url: "data:image/png;base64,Ag==" }] }];
    await run([task({ references: ["user-2", "user-1"] })], s, messages);
    const form = s.bodies[0] as FormData;
    const files = form.getAll("image[]") as File[];
    expect(files.map((f) => f.name)).toEqual(["user-2-1.png", "user-1-2.png"]);
    expect([...new Uint8Array(await files[0].arrayBuffer())]).toEqual([2]);
  });

  it("sends the user's picture at its own size, not the redraw the vision tower got", async () => {
    const s = server();
    const full = new Blob([new Uint8Array([9, 9])], { type: "image/png" });
    await run([task({ references: ["user-1"] })], s, [{ role: "user", images: [{ url: "data:image/png;base64,AQ==", full }] }]);
    const sent = (s.bodies[0] as FormData).getAll("image[]") as File[];
    expect([...new Uint8Array(await sent[0].arrayBuffer())]).toEqual([9, 9]);
  });

  it("lays each picture over a marked reference's original when it asked to change only its marked area", async () => {
    const s = server();
    const source = new Blob([new Uint8Array([1])], { type: "image/png" });
    const region = new Blob([new Uint8Array([2])], { type: "image/png" });
    const marked = { url: "data:image/png;base64,AQ==", markup: { source, keepOutside: true, region } };
    const laid: { source: Blob; region: Blob }[] = [];
    const runs = await runImages([task({ references: ["user-1"] })], {
      messages: [{ role: "user", images: [marked] }],
      signal: new AbortController().signal,
      onUpdate: () => {},
      fetch: s.fetcher,
      wait: async () => {},
      keepAround: async (_url, keep) => {
        laid.push(keep);
        return "data:image/png;base64,S0VQVA==";
      },
    });
    expect(laid).toEqual([{ source, region }]);
    expect(runs[0].made?.[0]).toMatchObject({ url: "data:image/png;base64,S0VQVA==", kept: true });
    // Not asked: the picture as the model made it.
    const plain = await run([task({ references: ["user-1"] })], server(), [{ role: "user", images: [{ ...marked, markup: { ...marked.markup, keepOutside: false } }] }]);
    expect(plain.runs[0].made?.[0].kept).toBeUndefined();
  });

  it("does not send, for an edit, the seed that made a picture it edits, and says so", async () => {
    const s = server();
    const made = { ...task(), status: "done" as const, made: [{ name: "made-1", url: "data:image/png;base64,AQ==", size: "2048x2048", seed: "41" }] };
    const { runs } = await run([task({ references: ["made-1"], seed: "41" })], s, [{ role: "assistant", imageRuns: [made] }]);
    expect((s.bodies[0] as FormData).has("seed")).toBe(false);
    expect(runs[0].seedDropped).toBe(true);
    expect(imageToolResult(runs[0])).toContain("The seed was not reused");
    const other = server();
    await run([task({ references: ["made-1"], seed: "99" })], other, [{ role: "assistant", imageRuns: [made] }]);
    expect((other.bodies[0] as FormData).get("seed")).toBe("99");
  });

  it("fails a call that names a picture nobody sent or made, and still goes back", async () => {
    const s = server();
    const { runs } = await run([task({ references: ["user-1"] })], s);
    expect(runs[0].status).toBe("failed");
    expect(runs[0].error).toContain("No picture is named user-1");
    expect(s.loaded()).toBe("qwen3.8-27b");
  });

  it("says the server's refusal, and still goes back", async () => {
    const s = server({ failImage: true });
    const { runs } = await run([task()], s);
    expect(runs[0]).toMatchObject({ status: "failed" });
    expect(runs[0].error).toContain("The queue is full");
    expect(s.loaded()).toBe("qwen3.8-27b");
  });

  it("makes nothing and switches nothing when the server lists no image model", async () => {
    const s = server({ catalog: [] });
    const { runs } = await run([task(), task({ callId: "c2" })], s);
    expect(runs.map((r) => r.status)).toEqual(["failed", "failed"]);
    expect(runs[0].error).toContain("lists no image model");
    expect(s.log).not.toContain("POST /v1/models/switch");
  });

  // ADR 0064: the server would refuse a switch to an image model above the card's class, so none is begun.
  it("switches nothing to an image model above the card's class, and says why in the server's words", async () => {
    const reason = "ignis-image-2.1-turbo needs a 32 GB card or larger (its catalog entry's card_class), and this card is of the 16 GB class";
    const above = { id: "ignis-image-2.1-turbo", family: "qwenimage21", switchable: false, available: false, reason, source: "built-in", on_disk: "complete", loaded: false };
    const s = server({ catalog: [above] });
    const { runs } = await run([task(), task({ callId: "c2" })], s);
    expect(runs.map((r) => r.status)).toEqual(["failed", "failed"]);
    expect(runs[0].error).toBe(`The image model is not for this card: ${reason}`);
    expect(s.log).not.toContain("POST /v1/models/switch");
    expect(s.loaded()).toBe("qwen3.8-27b");
  });

  it("switches nowhere for a turn stopped before its calls ran", async () => {
    const s = server();
    const stop = new AbortController();
    stop.abort();
    const { runs } = await run([task()], s, [], stop.signal);
    expect(runs[0].status).toBe("stopped");
    expect(s.log).not.toContain("POST /v1/models/switch");
  });

  it("goes back to the text model when the turn is stopped mid-picture", async () => {
    const s = server();
    const stop = new AbortController();
    const fetcher = (async (path: string, init?: RequestInit) => {
      if (path === "/v1/images/generations") {
        stop.abort();
        throw new DOMException("The operation was aborted.", "AbortError");
      }
      return s.fetcher(path, init);
    }) as unknown as typeof fetch;
    const runs = await runImages([task()], { messages: [], signal: stop.signal, onUpdate: () => {}, fetch: fetcher, wait: async () => {} });
    expect(runs[0].status).toBe("stopped");
    expect(s.loaded()).toBe("qwen3.8-27b");
  });

  it("runs every call of a reply on one visit", async () => {
    const s = server();
    const { runs } = await run([task(), task({ callId: "c2" })], s);
    expect(runs.map((r) => r.made?.[0].name)).toEqual(["made-1", "made-2"]);
    expect(s.log.filter((l) => l === "POST /v1/models/switch")).toHaveLength(2);
  });
});

describe("the strip", () => {
  it("shows the switch while it runs, then each picture with its name, size, seed and download", () => {
    const working = renderToStaticMarkup(<ImageStrip runs={[{ ...task(), status: "switching", line: "Switching to ignis-image-2.1-turbo…" }]} />);
    expect(working).toContain("Switching to ignis-image-2.1-turbo…");
    expect(working).toContain("a fox");
    const done = renderToStaticMarkup(
      <ImageStrip runs={[{ ...task({ references: ["user-1"] }), status: "done", made: [{ name: "made-1", url: "data:image/png;base64,AAA", size: "2752x1536", seed: "41" }] }]} />,
    );
    expect(done).toContain('src="data:image/png;base64,AAA"');
    expect(done).toContain("made-1");
    expect(done).toContain("2752 × 1536");
    expect(done).toContain("Seed 41");
    expect(done).toContain("From user-1");
    expect(done).toContain(">PNG<");
    expect(done).toContain("1 picture");
  });
});
