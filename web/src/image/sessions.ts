import type { ImageBackground } from "../api/images.ts";
import type { Seed } from "../api/seed.ts";
import { type ImageId, type ImageStore, type StoreUsage, unreferenced } from "../store/imageStore.ts";
import type { Mark, MarkMode } from "../markup/markup.ts";
import { AUTO_SIZE } from "./studio.ts";

// The image view's sessions: each one a line of work on one or more pictures.
// A session holds an **image pool** — every picture uploaded into it, made in
// it or marked in it, each named #N for good — and its **steps**, each a
// prompt sent with **image inputs** chosen from the pool and the pictures it
// made. The next step's inputs default to the last step's first picture, so
// an edit goes on from where the last one ended; the owner adds or drops
// inputs from the pool. A marked picture is a new pool entry beside the one it
// marks, so a step records the version it sent.
//
// Image input vs condition image (GLOSSARY.md): an input is a pool picture the
// next step uses; what reaches the server are its condition images — one per
// input, two for one marked as a mask (the picture, then its mask).
//
// Sessions are documents in the page's ImageStore (`image-sessions`), the
// pictures images in it. Every function here returns a new session.

export const SESSIONS = "image-sessions";

export type PoolOrigin = "upload" | "made" | "marked" | "chat" | "workspace";

/** The marks a `marked` pool picture carries, and what was made of them. */
export type PoolMarkup = {
  /** The pool picture marked. */
  source: number;
  marks: Mark[];
  mode: MarkMode;
  keepOutside: boolean;
  /** The marked area, white on black: sent after the source picture in mask mode. */
  mask: ImageId;
  /** The marked area as alpha, for laying a result over the source. */
  region: ImageId;
};

export type PoolImage = {
  /** Its name in the session: #1, #2… never reused. */
  n: number;
  id: ImageId;
  width: number;
  height: number;
  origin: PoolOrigin;
  /** An upload's file name. */
  name?: string;
  /** The step that made it. */
  step?: number;
  seed?: Seed;
  revisedPrompt?: string;
  background?: Exclude<ImageBackground, "auto">;
  /** Laid over its input through the marked area. */
  kept?: true;
  markup?: PoolMarkup;
};

/** What a step was sent with besides its prompt and inputs. */
export type StepSettings = { size: string; count: number; seed: Seed | null; background: ImageBackground; enhance: boolean };

export type Step = {
  n: number;
  prompt: string;
  /** Pool pictures, in the order the prompt calls them image 1, image 2… */
  inputs: number[];
  settings: StepSettings;
  /** Pool pictures it made, in the order they came. */
  outputs: number[];
  status: "running" | "done" | "failed" | "stopped";
  error?: string;
  /** What the server added: a condition image scaled, an enhancement that did not rewrite. */
  notes?: string[];
  /** The prompt as prompt enhancement rewrote it. */
  revised?: string;
  at: number;
};

/** The next step as the owner is writing it. `enhance` follows the server's setting until touched. */
export type SessionDraft = {
  prompt: string;
  inputs: number[];
  size: string;
  count: number;
  /** Blank after each step: the next step edits its picture, and must not start from the noise that made it (seedClash). */
  seedText: string;
  /**
   * The owner's choice against that: each step's seed is the last picture's, and an edit is sent the seed behind its
   * inputs. Not advised — it builds the noise up into grain — and off unless chosen; for reproducing a picture.
   */
  sameSeed: boolean;
  background: ImageBackground;
  enhance: boolean | null;
  stream: boolean;
};

export type ImageSession = {
  id: string;
  title: string;
  created: number;
  updated: number;
  pool: PoolImage[];
  steps: Step[];
  draft: SessionDraft;
};

export const UNTITLED = "New session";

const TITLE_LENGTH = 48;

export const EMPTY_SESSION_DRAFT: SessionDraft = {
  prompt: "",
  inputs: [],
  size: AUTO_SIZE,
  count: 1,
  seedText: "",
  sameSeed: false,
  background: "auto",
  enhance: null,
  stream: true,
};

export function newSession(id: string, now: number): ImageSession {
  return { id, title: UNTITLED, created: now, updated: now, pool: [], steps: [], draft: { ...EMPTY_SESSION_DRAFT } };
}

/** A session with nothing in it yet: not worth keeping in the store. */
export const isEmpty = (s: ImageSession) => s.pool.length === 0 && s.steps.length === 0;

const touched = (s: ImageSession, now: number): ImageSession => ({ ...s, updated: now });

export const pictureOf = (s: ImageSession, n: number) => s.pool.find((p) => p.n === n);

/** `ignis-41-2752x1536.png`: the seed and the size, else the pool name. */
export const pictureName = (p: PoolImage) => `ignis-${p.seed ?? `n${p.n}`}-${p.width}x${p.height}.png`;

export function setDraft(s: ImageSession, change: Partial<SessionDraft>, now: number): ImageSession {
  return touched({ ...s, draft: { ...s.draft, ...change } }, now);
}

/** Pictures into the pool, numbered on from the last; with `asInputs`, the next step's inputs too. */
export function addToPool(s: ImageSession, entries: Omit<PoolImage, "n">[], now: number, asInputs = false): [ImageSession, number[]] {
  let n = s.pool.reduce((max, p) => Math.max(max, p.n), 0);
  const added = entries.map((e): PoolImage => ({ ...e, n: ++n }));
  const numbers = added.map((p) => p.n);
  const draft = asInputs ? { ...s.draft, inputs: [...s.draft.inputs, ...numbers.filter((x) => !s.draft.inputs.includes(x))] } : s.draft;
  return [touched({ ...s, pool: [...s.pool, ...added], draft }, now), numbers];
}

/** A pool picture in or out of the next step's inputs; in, it is the last. */
export function toggleInput(s: ImageSession, n: number, now: number): ImageSession {
  if (!pictureOf(s, n)) return s;
  const inputs = s.draft.inputs.includes(n) ? s.draft.inputs.filter((x) => x !== n) : [...s.draft.inputs, n];
  return setDraft(s, { inputs }, now);
}

/**
 * A marked version of pool picture `source`, as a new pool picture that takes the place of `replacing` among the
 * inputs: the source, or an earlier marking of it being marked again.
 */
export function addMarked(
  s: ImageSession,
  source: number,
  entry: Omit<PoolImage, "n" | "origin" | "markup">,
  markup: Omit<PoolMarkup, "source">,
  now: number,
  replacing = source,
): [ImageSession, number] {
  const [withIt, [n]] = addToPool(s, [{ ...entry, origin: "marked", markup: { ...markup, source } }], now);
  const inputs = withIt.draft.inputs.includes(replacing) ? withIt.draft.inputs.map((x) => (x === replacing ? n : x)) : [...withIt.draft.inputs, n];
  return [setDraft(withIt, { inputs }, now), n];
}

/** Marks taken off: the marked picture leaves the inputs, and its source takes its place there. */
export function unmark(s: ImageSession, n: number, now: number): ImageSession {
  const source = pictureOf(s, n)?.markup?.source;
  if (source === undefined || !s.draft.inputs.includes(n)) return s;
  return setDraft(s, { inputs: s.draft.inputs.map((x) => (x === n ? source : x)) }, now);
}

/**
 * A step under way, from the draft as it was sent; the first one names the session. The prompt box empties, as the
 * chat's does: the prompt is the step's now, and comes back if the step makes nothing (endStep).
 */
export function startStep(s: ImageSession, prompt: string, inputs: number[], settings: StepSettings, now: number): [ImageSession, number] {
  const n = s.steps.length + 1;
  const step: Step = { n, prompt, inputs, settings, outputs: [], status: "running", at: now };
  const title = s.steps.length === 0 && s.title === UNTITLED ? titleOf(prompt) : s.title;
  return [touched({ ...s, title, steps: [...s.steps, step], draft: { ...s.draft, prompt: "" } }, now), n];
}

/** A picture the step made, into the pool and onto the step. */
export function landOutput(s: ImageSession, stepN: number, entry: Omit<PoolImage, "n" | "origin" | "step">, now: number): [ImageSession, number] {
  const [withIt, [n]] = addToPool(s, [{ ...entry, origin: "made", step: stepN }], now);
  return [touched({ ...withIt, steps: withIt.steps.map((st) => (st.n === stepN ? { ...st, outputs: [...st.outputs, n] } : st)) }, now), n];
}

/**
 * The step ended. One that made pictures hands the next step its first one as the only input, and clears the seed:
 * an edit that starts again from the noise that made its input builds that noise up into grain (seedClash) — unless
 * the owner chose the same seed, when the field takes that picture's. One that made none gives its prompt back to an
 * empty prompt box, to send again or change.
 */
export function endStep(
  s: ImageSession,
  stepN: number,
  end: { status: Exclude<Step["status"], "running">; error?: string; notes?: string[]; revised?: string },
  now: number,
): ImageSession {
  const steps = s.steps.map((st) =>
    st.n === stepN
      ? {
          ...st,
          status: end.status,
          ...(end.error ? { error: end.error } : {}),
          ...(end.notes?.length ? { notes: end.notes } : {}),
          ...(end.revised ? { revised: end.revised } : {}),
        }
      : st,
  );
  const step = steps.find((st) => st.n === stepN);
  const first = step?.outputs[0];
  if (first === undefined) {
    const back = step && s.draft.prompt.trim() === "" ? { draft: { ...s.draft, prompt: step.prompt } } : {};
    return touched({ ...s, steps, ...back }, now);
  }
  const seed = s.draft.sameSeed ? pictureOf(s, first)?.seed : undefined;
  const draft: SessionDraft = { ...s.draft, prompt: "", inputs: [first], seedText: seed ?? "" };
  return touched({ ...s, steps, draft }, now);
}

/**
 * A session as the store had it, made current: a step left running by a page that went away ended there, stopped;
 * a seed kept as a number (before seeds were digits, already rounded past 2^53) is read as digits; and a draft from
 * when "Keep the last seed" filled the seed field gets the field back blank (seedClash says why).
 */
export function fromStore(stored: ImageSession, now: number): ImageSession {
  const { keepSeed, ...draft } = stored.draft as SessionDraft & { keepSeed?: boolean };
  const s: ImageSession = { ...stored, draft: { ...draft, sameSeed: draft.sameSeed ?? false, ...(keepSeed ? { seedText: "" } : {}) } };
  // As the page showed it: a u64 prints without an exponent.
  const digits = (seed: unknown) => (typeof seed === "number" ? String(Math.trunc(Math.abs(seed))) : seed);
  const pool = s.pool.map((p) => (typeof p.seed === "number" ? { ...p, seed: digits(p.seed) as Seed } : p));
  const steps = s.steps.map((st) => (typeof st.settings.seed === "number" ? { ...st, settings: { ...st.settings, seed: digits(st.settings.seed) as Seed } } : st));
  return settleRunning({ ...s, pool, steps }, now);
}

/** A step left running by a page that went away: it ended there, stopped. */
export function settleRunning(s: ImageSession, now: number): ImageSession {
  if (!s.steps.some((st) => st.status === "running")) return s;
  return touched({ ...s, steps: s.steps.map((st) => (st.status === "running" ? { ...st, status: "stopped" as const } : st)) }, now);
}

/** The draft of a step's prompt, inputs and settings again, to send it once more or change it. */
export function reuseStep(s: ImageSession, stepN: number, now: number): ImageSession {
  const step = s.steps.find((st) => st.n === stepN);
  if (!step) return s;
  return setDraft(
    s,
    {
      prompt: step.prompt,
      inputs: step.inputs.filter((n) => pictureOf(s, n)),
      size: step.settings.size,
      count: step.settings.count,
      seedText: step.settings.seed ?? "",
      background: step.settings.background,
    },
    now,
  );
}

/**
 * The step that made a picture again, with the picture's own seed: its prompt, its inputs and its settings, one
 * picture. Not the picture as an input with its own seed, which would build its noise up (seedClash).
 */
export function reusePicture(s: ImageSession, n: number, now: number): ImageSession {
  const p = pictureOf(s, n);
  if (!p) return s;
  const step = p.step !== undefined ? s.steps.find((st) => st.n === p.step) : undefined;
  return setDraft(
    s,
    {
      ...(step
        ? { prompt: step.prompt, inputs: step.inputs.filter((x) => pictureOf(s, x)), size: step.settings.size, background: step.settings.background, count: 1 }
        : {}),
      ...(p.seed !== undefined ? { seedText: p.seed } : {}),
    },
    now,
  );
}

/** The seeds behind a pool picture: its own, and those of every picture it was made or marked from. */
export function seedsBehind(s: ImageSession, n: number, seen = new Set<number>()): Set<Seed> {
  const seeds = new Set<Seed>();
  if (seen.has(n)) return seeds;
  seen.add(n);
  const p = pictureOf(s, n);
  if (!p) return seeds;
  if (p.seed !== undefined) seeds.add(p.seed);
  const from = [...(p.markup ? [p.markup.source] : []), ...(p.step !== undefined ? (s.steps.find((st) => st.n === p.step)?.inputs ?? []) : [])];
  for (const m of from) for (const seed of seedsBehind(s, m, seen)) seeds.add(seed);
  return seeds;
}

/**
 * The input a seed is behind, if any. An edit that starts again from the noise that made its input — or a picture
 * that input came from — finds that noise's faint imprint in the picture and builds it up: two or three such steps
 * and the picture is grain (docs/findings/2026-10-16-image-edit-same-seed-grain.md). Such a seed is not sent.
 */
export function seedClash(s: ImageSession, inputs: number[], seed: Seed | null): number | undefined {
  if (seed === null) return undefined;
  return inputs.find((n) => seedsBehind(s, n).has(seed));
}

/** What a request sends for the inputs, in order: a mask-mode marked picture sends its source, then its mask. */
export type WireImage = { id: ImageId; name: string };

export function wireImages(s: ImageSession, inputs: number[]): WireImage[] {
  return inputs.flatMap((n): WireImage[] => {
    const p = pictureOf(s, n);
    if (!p) return [];
    if (p.markup?.mode === "mask") {
      const source = pictureOf(s, p.markup.source);
      return [
        { id: source?.id ?? p.id, name: `image-${n}.png` },
        { id: p.markup.mask, name: `image-${n}-mask.png` },
      ];
    }
    return [{ id: p.id, name: `image-${n}.png` }];
  });
}

/** The number the prompt calls each input by, and its mask's, in the order they are sent. */
export function inputNumbers(s: ImageSession, inputs: number[]): { n: number; picture: number; mask?: number }[] {
  let k = 0;
  return inputs.map((n) => {
    const picture = ++k;
    return pictureOf(s, n)?.markup?.mode === "mask" ? { n, picture, mask: ++k } : { n, picture };
  });
}

/** The first input whose marked area is all a result may change: its source picture and the area. */
export function keepFor(s: ImageSession, inputs: number[]): { source: ImageId; region: ImageId } | null {
  for (const n of inputs) {
    const p = pictureOf(s, n);
    if (p?.markup?.keepOutside) return { source: pictureOf(s, p.markup.source)?.id ?? p.id, region: p.markup.region };
  }
  return null;
}

/** Every image the session names: its pictures, and their masks and areas. */
export function imagesOf(s: ImageSession): Set<ImageId> {
  return new Set(s.pool.flatMap((p) => [p.id, ...(p.markup ? [p.markup.mask, p.markup.region] : [])]));
}

/**
 * The pictures deleting a session frees: those only it names. `others` are the other sessions, `live` the chat's
 * pictures and `held` the workspaces' (null while not known: then nothing is freed, as a picture something else
 * may name must not go).
 */
export function freedWithSession(gone: ImageSession, others: Iterable<ImageSession>, live: Iterable<ImageId> | null, held: Iterable<ImageId> | null): ImageId[] {
  if (live === null || held === null) return [];
  const keep = new Set<ImageId>([...[...others].flatMap((s) => [...imagesOf(s)]), ...live, ...held]);
  return unreferenced(imagesOf(gone), keep);
}

/** The picture that stands for the session in the list: the last one made, else the last one added. */
export function coverOf(s: ImageSession): PoolImage | undefined {
  const made = [...s.steps].reverse().flatMap((st) => st.outputs)[0];
  return (made !== undefined ? pictureOf(s, made) : undefined) ?? s.pool[s.pool.length - 1];
}

/** The last step's prompt, as prompt enhancement rewrote it: the studio's dictation context before the box's. */
export function lastPrompt(s: ImageSession): string {
  const last = s.steps[s.steps.length - 1];
  return last?.revised ?? last?.prompt ?? "";
}

export function titleOf(prompt: string): string {
  const line = prompt.trim().split("\n")[0] ?? "";
  if (line === "") return UNTITLED;
  return line.length > TITLE_LENGTH ? `${line.slice(0, TITLE_LENGTH - 1).trimEnd()}…` : line;
}

/** Newest first: the order of the list. */
export const byRecent = (a: ImageSession, b: ImageSession) => b.updated - a.updated;

/** What the store holds, and where: the foot of the sessions list. */
export function storeLine(state: { usage: StoreUsage | null; storeKind: ImageStore["kind"] }): string {
  if (state.storeKind === "memory") return "This browser keeps no store: the sessions go with the page.";
  if (!state.usage) return "Pictures are kept in this browser.";
  const mb = state.usage.bytes / 1e6;
  const size = mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${Math.round(mb)} MB`;
  return `${state.usage.images} ${state.usage.images === 1 ? "picture" : "pictures"}, ${size}, kept in this browser.`;
}
