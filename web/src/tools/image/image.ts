import { generateStreaming, type ImageParams, type ImageProgress } from "../../api/images.ts";
import { readModel } from "../../api/model.ts";
import type { ToolDefinition } from "../../api/request.ts";
import type { ToolCall } from "../../api/sse.ts";
import { switchAndWait } from "../../api/switching.ts";
import { fileFromDataUrl, imageModelRow, limitsOf, progressLine, refusalLine, shapeOf, transparentPrompt } from "../../image/studio.ts";
import { keepOutside } from "../../markup/markup.ts";
import { readCatalog, readConfig } from "../../server/routes.ts";
import { unknownToolError } from "../errors.ts";

// The picture tool: `make_image` lets the conversation's model make a picture,
// or change pictures of the conversation, with the image model (spec
// image-model). The model loaded and the image model never share the card —
// a switch is a full reload (ADR 0049) — so the page does what the owner
// would in the Image view: it switches to the image model, makes every
// picture the reply asked for, and switches back to the text model the turn
// was talking to before the next request, which would otherwise be refused as
// the wrong kind. The picture shows in the reply; the model is told its name,
// size and seed, not what it looks like.
//
// The pictures of a session have names the model can pass back: `user-N` is
// the Nth picture the user sent, `made-N` the Nth this tool made.

export const IMAGE_TOOL_NAME = "make_image";

/** The shapes the tool offers, each served by the preset nearest it (`image.sizes`). */
export const ASPECTS = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"] as const;

/** At most this many pictures per call: each is a full denoising run. */
export const MAX_COUNT = 4;

export const IMAGE_TOOL: ToolDefinition = {
  type: "function",
  function: {
    name: IMAGE_TOOL_NAME,
    description:
      "Make a picture, or change pictures of this conversation, with the image model. The user sees the result; you get its name, size and seed.",
    parameters: {
      type: "object",
      properties: {
        prompt: {
          type: "string",
          description:
            "The picture described: subject, setting, light, style, composition. Put any text to render in quotes. With references, what to change, naming them image 1, image 2… in the order given.",
        },
        references: {
          type: "array",
          items: { type: "string" },
          description: "Pictures to edit or to keep consistent with, by name: user-1, user-2… for the user's, made-1, made-2… for yours. At most 10.",
        },
        aspect: { type: "string", enum: [...ASPECTS], description: "The shape. Omitted: square for a new picture, the last reference's for an edit." },
        transparent: { type: "boolean", description: "A transparent background, for a sticker, an icon or a cut-out." },
        count: { type: "integer", minimum: 1, maximum: MAX_COUNT, description: "How many variants. Default 1." },
        seed: { type: "integer", minimum: 0, description: "Repeats a picture's seed, to vary the prompt and keep the rest." },
      },
      required: ["prompt"],
    },
  },
};

/** What the tool adds to the ignis system prompt. */
export const IMAGE_IGNIS_PROMPT = `# Pictures
You can make and edit pictures with \`make_image\`. The user sees each picture in the conversation; you do not.
- Call it when the user asks for a picture or for a change to one, not otherwise: a call unloads you for about half a minute, and the conversation goes on once you are loaded again.
- Describe the picture you want in the prompt, concretely: subject, setting, light, style, composition. Put any text to render in quotes.
- To change pictures, or to keep a person or a product the same across pictures, list them in \`references\` by name: the user's are user-1, user-2… in the order they sent them; yours are made-1, made-2… as the results name them. The prompt calls them image 1, image 2… in the order you list them.
- A picture the user marked — a circle, strokes in a colour — asks for a change there: pass it in \`references\` and say in the prompt what the marked part becomes, naming the colour of the marks.
- Call it once for several variants (\`count\`), not several times.
- Afterwards, say in a sentence what you made. Do not describe it as if you could see it.`;

export type ImageStatus = "queued" | "switching" | "making" | "returning" | "done" | "failed" | "stopped";

/** A picture the tool made. */
export type MadePicture = {
  name: string;
  url: string;
  size: string;
  seed?: number;
  revisedPrompt?: string;
  /** Laid over the marked picture it edits, through the marked area. */
  kept?: true;
};

/**
 * A picture of the session by name: the URL it shows by; the picture at its own size where the chat kept it (the
 * redraw for the vision tower is smaller); and, for one marked to change only its marked area, that area and the
 * original to keep around it.
 */
export type NamedPicture = { url: string; full?: Blob; keep?: { source: Blob; region: Blob } };

type SessionPicture = { url: string; full?: Blob; markup?: { source: Blob; keepOutside: boolean; region: Blob } };

/** One call as the reply shows it and the model is told about it. */
export type ImageRun = {
  callId: string;
  prompt: string;
  references: string[];
  aspect?: (typeof ASPECTS)[number];
  transparent: boolean;
  count: number;
  seed?: number;
  status: ImageStatus;
  /** What the call is waiting on, in words: the switch, a phase, a step. */
  line?: string;
  progress?: ImageProgress;
  made?: MadePicture[];
  error?: string;
};

export type ImageTask = Omit<ImageRun, "status" | "line" | "progress" | "made" | "error">;

export const isImageTool = (name: string) => name === IMAGE_TOOL_NAME;

/** A call read as a task, or as a failed run that says what was wrong with it. */
export function parseImageCall(call: ToolCall, available: string[] = [IMAGE_TOOL_NAME]): { ok: true; task: ImageTask } | { ok: false; run: ImageRun } {
  const fail = (error: string, prompt = "") => ({
    ok: false as const,
    run: { callId: call.id, prompt, references: [], transparent: false, count: 1, status: "failed" as const, error },
  });
  if (!isImageTool(call.name) || !available.includes(call.name)) return fail(unknownToolError(call.name, available));
  let args: Record<string, unknown>;
  try {
    const parsed: unknown = JSON.parse(call.arguments);
    if (typeof parsed !== "object" || parsed === null) return fail("The call's arguments are not an object.");
    args = parsed as Record<string, unknown>;
  } catch {
    return fail("The call's arguments are not valid JSON.");
  }
  const prompt = typeof args.prompt === "string" ? args.prompt.trim() : "";
  if (prompt === "") return fail('The call needs a non-empty "prompt".');
  const references = args.references === undefined ? [] : args.references;
  if (!Array.isArray(references) || references.some((r) => typeof r !== "string")) return fail('"references" is a list of picture names.', prompt);
  if (references.length > 10) return fail("At most 10 references.", prompt);
  const aspect = args.aspect;
  if (aspect !== undefined && !ASPECTS.includes(aspect as (typeof ASPECTS)[number])) return fail(`"aspect" is one of ${ASPECTS.join(", ")}.`, prompt);
  const count = args.count === undefined ? 1 : Number(args.count);
  if (!Number.isInteger(count) || count < 1 || count > MAX_COUNT) return fail(`"count" is a whole number from 1 to ${MAX_COUNT}.`, prompt);
  const seed = args.seed;
  if (seed !== undefined && !(typeof seed === "number" && Number.isSafeInteger(seed) && seed >= 0)) return fail('"seed" is a whole number.', prompt);
  return {
    ok: true,
    task: {
      callId: call.id,
      prompt,
      references: (references as string[]).map((r) => r.trim()),
      ...(aspect !== undefined ? { aspect: aspect as (typeof ASPECTS)[number] } : {}),
      transparent: args.transparent === true,
      count,
      ...(seed !== undefined ? { seed: seed as number } : {}),
    },
  };
}

/** The served size for a shape: the preset whose ratio reads as it, else `auto`. */
export function sizeForAspect(aspect: string | undefined, sizes: string[]): string {
  if (!aspect) return "auto";
  // The largest preset of that ratio: the presets are the model's own sizes; 1024x1024 is there for OpenAI clients.
  const matching = sizes.filter((s) => shapeOf(s)?.ratio === aspect);
  const area = (s: string) => (shapeOf(s)?.width ?? 0) * (shapeOf(s)?.height ?? 0);
  return matching.sort((a, b) => area(b) - area(a))[0] ?? "auto";
}

/** The pictures of a session by name: the user's in the order sent, the tool's in the order made. */
export function picturesByName(messages: { role: string; images?: SessionPicture[]; imageRuns?: ImageRun[] }[]): Map<string, NamedPicture> {
  const names = new Map<string, NamedPicture>();
  let user = 0;
  for (const m of messages) {
    if (m.role === "user") {
      for (const image of m.images ?? []) {
        const keep = image.markup?.keepOutside ? { source: image.markup.source, region: image.markup.region } : undefined;
        names.set(`user-${++user}`, { url: image.url, ...(image.full ? { full: image.full } : {}), ...(keep ? { keep } : {}) });
      }
    }
    for (const run of m.imageRuns ?? []) for (const made of run.made ?? []) names.set(made.name, { url: made.url });
  }
  return names;
}

/** A named picture as a file of the request: at its own size when the chat kept it. */
export function fileOfPicture(picture: NamedPicture, name: string): File {
  return picture.full ? new File([picture.full], name, { type: picture.full.type || "image/png" }) : fileFromDataUrl(picture.url, name);
}

/** The result laid over a marked picture's original through its marked area, as a data URL (browser only). */
async function keepAround(url: string, keep: { source: Blob; region: Blob }): Promise<string> {
  const blob = await keepOutside(fileFromDataUrl(url, "made.png"), keep);
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("the picture could not be read"));
    reader.readAsDataURL(blob);
  });
}

/** How many pictures the session's tool calls have made: the next one is `made-(n+1)`. */
export const madeSoFar = (messages: { imageRuns?: ImageRun[] }[]) => messages.reduce((n, m) => n + (m.imageRuns ?? []).reduce((k, r) => k + (r.made?.length ?? 0), 0), 0);

/** The tool result the model receives for a finished run. */
export function imageToolResult(run: ImageRun): string {
  if (run.status === "stopped") return "The make_image call was stopped before it finished.";
  const made = run.made ?? [];
  const lines = made.map(
    (p) =>
      `Made ${p.name}: ${p.size}${p.seed !== undefined ? `, seed ${p.seed}` : ""}.${p.revisedPrompt ? ` The prompt was rewritten as: ${p.revisedPrompt}` : ""}`,
  );
  if (run.status === "failed") {
    return [`The make_image call failed: ${run.error ?? "unknown error"}`, ...(made.length ? ["Before it failed:", ...lines] : [])].join("\n");
  }
  if (made.length === 0) return "The make_image call made no picture.";
  return [...lines, "The user sees the pictures in the conversation."].join("\n");
}

/** `2 pictures: 1 making, 1 done`. */
export function imageSummary(runs: ImageRun[]): string {
  const pictures = runs.reduce((n, r) => n + (r.made?.length ?? 0), 0);
  const failed = runs.filter((r) => r.status === "failed").length;
  const busy = runs.some((r) => r.status !== "done" && r.status !== "failed" && r.status !== "stopped");
  return [
    `${pictures} ${pictures === 1 ? "picture" : "pictures"}`,
    busy ? "under way" : "",
    failed ? `${failed} failed` : "",
  ]
    .filter(Boolean)
    .join(", ");
}

export type RunImagesOptions = {
  /** The session's messages as the turn stands: the pictures the calls can name. */
  messages: Parameters<typeof picturesByName>[0];
  signal: AbortSignal;
  onUpdate: (run: ImageRun) => void;
  /** Read the model again, so the page follows the switches. */
  onModelChange?: () => void;
  fetch?: typeof fetch;
  /** Waits between two reads of a switch under way. */
  wait?: (ms: number) => Promise<void>;
  /** How long a switch may take before the call gives up on it. */
  switchTimeoutMs?: number;
  /** Lays a picture over a marked reference's original through its area; the canvas one in the browser. */
  keepAround?: (url: string, keep: { source: Blob; region: Blob }) => Promise<string>;
};

/**
 * Runs a reply's picture calls one after another on one visit to the image model: it switches there, makes them, and
 * switches back to the model that was loaded, whatever happened in between — a stopped turn too, so the next message
 * finds a text model. Resolves with each run finished.
 */
export async function runImages(tasks: ImageTask[], options: RunImagesOptions): Promise<ImageRun[]> {
  if (tasks.length === 0) return [];
  const doFetch = options.fetch ?? fetch.bind(globalThis);
  const runs: ImageRun[] = tasks.map((t) => ({ ...t, status: "queued" }));
  const update = (i: number, change: Partial<ImageRun>) => {
    runs[i] = { ...runs[i], ...change };
    options.onUpdate(runs[i]);
  };
  const failAll = (error: string, from = 0) => {
    for (let i = from; i < runs.length; i++) if (runs[i].status !== "done") update(i, { status: "failed", error, line: undefined });
  };

  const [text, catalog, config] = await Promise.all([readModel(doFetch), readCatalog(doFetch), readConfig(doFetch)]);
  const imageModel = imageModelRow(catalog.ok ? catalog.value : null);
  if (!imageModel) {
    failAll("ignis lists no image model. It is fetched with `ignis-server model download ignis-image-2.1-turbo`.");
    return runs;
  }
  const back = text.state === "ready" || text.state === "unloaded" ? (text.kind === "image" ? null : text.id) : null;
  const limits = limitsOf(config.ok ? config.value : null);
  const pictures = picturesByName(options.messages);
  let made = madeSoFar(options.messages);

  if (options.signal.aborted) {
    for (let i = 0; i < runs.length; i++) update(i, { status: "stopped" });
    return runs;
  }
  try {
    for (let i = 0; i < runs.length; i++) update(i, { status: "switching", line: `Switching to ${imageModel.id}…` });
    const how = { onModelChange: options.onModelChange, fetch: doFetch, wait: options.wait, timeoutMs: options.switchTimeoutMs };
    const switched = await switchAndWait(imageModel.id, { ...how, signal: options.signal });
    if (switched === "stopped") {
      for (let i = 0; i < runs.length; i++) update(i, { status: "stopped", line: undefined });
      return runs;
    }
    if (switched) {
      failAll(switched);
      return runs;
    }
    for (let i = 0; i < runs.length; i++) {
      if (options.signal.aborted) {
        update(i, { status: "stopped", line: undefined });
        continue;
      }
      const task = runs[i];
      const missing = task.references.filter((name) => !pictures.has(name));
      if (missing.length > 0) {
        update(i, {
          status: "failed",
          line: undefined,
          error: `No picture is named ${missing.join(", ")}. The names are ${[...pictures.keys()].join(", ") || "none yet: nobody sent or made a picture"}.`,
        });
        continue;
      }
      update(i, { status: "making", line: "Sending the request…" });
      const params: ImageParams = {
        prompt: task.transparent ? transparentPrompt(task.prompt) : task.prompt,
        n: Math.min(task.count, limits.maxN),
        size: sizeForAspect(task.aspect, limits.sizes),
        seed: task.seed ?? null,
        enhance: false,
        images: task.references.map((name, k) => fileOfPicture(pictures.get(name)!, `${name}-${k + 1}.png`)),
        ...(task.transparent ? { background: "transparent" as const } : {}),
      };
      // The first reference marked to change only its area: each picture is laid over its original before it lands.
      const keep = task.references.map((name) => pictures.get(name)?.keep).find((k) => k !== undefined);
      const lay = options.keepAround ?? keepAround;
      let landing: Promise<void> = Promise.resolve();
      const pictureList: MadePicture[] = [];
      const outcome = await generateStreaming(
        params,
        {
          onProgress: (progress) => update(i, { progress, line: progressLine(progress, params.n) }),
          onImage: (image) => {
            // Named in the order they come, whatever the laying over takes.
            const name = `made-${++made}`;
            landing = landing.then(async () => {
              let url = `data:image/png;base64,${image.b64}`;
              let kept = false;
              if (keep) {
                try {
                  url = await lay(url, keep);
                  kept = true;
                } catch {
                  // The picture as the model made it, then.
                }
              }
              const picture: MadePicture = {
                name,
                url,
                size: image.size ?? params.size,
                ...(image.seed !== undefined ? { seed: image.seed + image.index } : {}),
                ...(image.revisedPrompt ? { revisedPrompt: image.revisedPrompt } : {}),
                ...(kept ? { kept: true as const } : {}),
              };
              pictures.set(name, { url });
              pictureList.push(picture);
              update(i, { made: [...pictureList] });
            });
          },
        },
        options.signal,
        doFetch,
      );
      await landing;
      if (outcome.ok) update(i, { status: "done", line: undefined, progress: undefined });
      else if (options.signal.aborted) update(i, { status: "stopped", line: undefined, progress: undefined });
      else update(i, { status: "failed", line: undefined, progress: undefined, error: refusalLine(outcome) });
    }
  } finally {
    // Back to the text model, stopped or not: the next request of the turn — or the owner's next message — needs it.
    if (back) {
      for (let i = 0; i < runs.length; i++) if (runs[i].status === "done") update(i, { status: "returning", line: `Loading ${back} again…` });
      // No signal: the way back is taken even for a stopped turn.
      const returned = await switchAndWait(back, { onModelChange: options.onModelChange, fetch: doFetch, wait: options.wait, timeoutMs: options.switchTimeoutMs });
      for (let i = 0; i < runs.length; i++) {
        if (runs[i].status === "returning") {
          update(i, returned ? { status: "failed", line: undefined, error: `The pictures were made, but ${back} did not load again: ${returned}` } : { status: "done", line: undefined });
        } else if (returned && runs[i].status === "failed") {
          update(i, { error: `${runs[i].error ?? "unknown error"} Then ${back} did not load again: ${returned}` });
        }
      }
    }
  }
  return runs;
}
