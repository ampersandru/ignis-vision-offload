import type { PromptImage } from "../conversation/images.ts";
import { restoreOrder } from "../store/docs.ts";
import type { ImageId } from "../store/imageStore.ts";
import type { AgentRun } from "../tools/agents/agents.ts";
import type { LocalRun } from "../tools/local/local.ts";
import type { WebRun } from "../tools/web/web.ts";
import type { LogRow, Message, Session } from "./sessions.ts";

// A chat session as the page's store keeps it (spec playground-enhancements/03):
// a document of `chat-sessions`, under a key of its own (store/docs.ts). Plain
// data — the start as milliseconds, not a Date — and no Blob: a prompt image
// keeps the redraw the wire carries inline, and names the pictures beside it
// (the picked file at its own size, the markup's source and area) by their id
// in the image store, where the Image view keeps its own.
//
// Read back, a session is settled: the page that wrote it may have gone away
// mid-turn, and nothing it left running is running any more.

export const CHAT_SESSIONS = "chat-sessions";

/** The schema this page writes. A document of a later one was written by a newer page, and is left alone. */
export const CHAT_VERSION = 1;

type StoredImage = Omit<PromptImage, "full" | "markup">;

export type StoredMessage = Omit<Message, "images"> & { images?: StoredImage[] };

export type StoredChat = {
  version: typeof CHAT_VERSION;
  key: string;
  /** The page's id when written: the order the sessions were opened in. */
  id: number;
  title: string;
  /** `Session.startedAt`, in milliseconds. */
  startedAt: number;
  messages: StoredMessage[];
  log: LogRow[];
  attachments: Session["attachments"];
};

/** What a call left without a result is answered with when the session is read back. */
export const CLOSED_MID_CALL = "Stopped: the page was closed before the call finished.";

/** A session with nothing in it is not written: an empty one is what a new page opens with anyway. */
export const worthKeeping = (s: Session): boolean => s.messages.length > 0 || s.attachments.length > 0;

const storedImage = ({ full: _full, markup: _markup, ...image }: PromptImage): StoredImage => image;

export function toStored(s: Session, key: string): StoredChat {
  return {
    version: CHAT_VERSION,
    key,
    id: s.id,
    title: s.title,
    startedAt: s.startedAt.getTime(),
    messages: s.messages.map(({ images, ...m }) => (images ? { ...m, images: images.map(storedImage) } : m)),
    log: s.log,
    attachments: s.attachments,
  };
}

/** The store's pictures a session names: what deleting it may free, and what deleting anything else must keep. */
export function picturesOf(s: Session): Set<ImageId> {
  const ids = new Set<ImageId>();
  for (const m of s.messages)
    for (const image of m.images ?? []) {
      if (image.kept?.full) ids.add(image.kept.full);
      if (image.kept?.markup) ids.add(image.kept.markup.source).add(image.kept.markup.region);
    }
  return ids;
}

/** A stored value read as a session document of this page's schema, or null. */
function readStored(value: unknown): StoredChat | null {
  if (typeof value !== "object" || value === null) return null;
  const doc = value as Partial<StoredChat>;
  if (doc.version !== CHAT_VERSION || typeof doc.key !== "string" || typeof doc.id !== "number" || !Array.isArray(doc.messages)) return null;
  return { ...doc, title: doc.title ?? "", startedAt: doc.startedAt ?? Date.now(), log: doc.log ?? [], attachments: doc.attachments ?? [] } as StoredChat;
}

/**
 * The stored sessions, newest first and settled, with new ids from `counter` up
 * — above every message id they hold, since the page's one counter hands out
 * both — and the key of each by its new id. Their pictures are still in the
 * store (`withPictures`).
 */
export function restoreChats(values: unknown[], counter: number): { sessions: Session[]; keys: Map<number, string>; nextId: number } {
  const docs = values.map(readStored).filter((d): d is StoredChat => d !== null);
  const highest = Math.max(0, ...docs.flatMap((d) => d.messages.map((m) => m.id)));
  const { items, nextId } = restoreOrder(docs, Math.max(counter, highest + 1));
  let next = nextId;
  const newId = () => next++;
  const keys = new Map<number, string>();
  const sessions = items.map((doc) => {
    keys.set(doc.id, doc.key);
    return settle({ id: doc.id, title: doc.title, startedAt: new Date(doc.startedAt), messages: doc.messages, log: doc.log, attachments: doc.attachments }, newId);
  });
  return { sessions, keys, nextId: next };
}

const stopAgent = (run: AgentRun): AgentRun => ({
  ...run,
  status: run.status === "queued" || run.status === "running" ? "stopped" : run.status,
  ...(run.web ? { web: run.web.map(stopWeb) } : {}),
  ...(run.local ? { local: run.local.map(stopLocal) } : {}),
});
const stopWeb = (run: WebRun): WebRun => (run.status === "running" ? { ...run, status: "stopped" } : run);
const stopLocal = (run: LocalRun): LocalRun => (run.status === "running" || run.status === "checking" ? { ...run, status: "stopped" } : run);

function settleMessage(m: Message): Message {
  const { queued: _queued, ...rest } = m;
  return {
    ...rest,
    streaming: false,
    ...(m.switch?.status === "switching" ? { switch: { ...m.switch, status: "stopped" } } : {}),
    ...(m.agents ? { agents: m.agents.map(stopAgent) } : {}),
    ...(m.web ? { web: m.web.map(stopWeb) } : {}),
    ...(m.local ? { local: m.local.map(stopLocal) } : {}),
    ...(m.questions ? { questions: m.questions.map((q) => (q.status === "waiting" ? { ...q, status: "skipped" as const } : q)) } : {}),
    ...(m.imageRuns
      ? { imageRuns: m.imageRuns.map((r) => (r.status === "done" || r.status === "failed" || r.status === "stopped" ? r : { ...r, status: "stopped" as const })) }
      : {}),
  };
}

/** Anything in `m` a page that went away left running. */
const unsettled = (m: Message): boolean =>
  m.streaming ||
  m.queued === true ||
  m.switch?.status === "switching" ||
  (m.agents ?? []).some((a) => a.status === "queued" || a.status === "running" || (a.web ?? []).some((w) => w.status === "running") || (a.local ?? []).some((l) => l.status === "running" || l.status === "checking")) ||
  (m.web ?? []).some((w) => w.status === "running") ||
  (m.local ?? []).some((l) => l.status === "running" || l.status === "checking") ||
  (m.questions ?? []).some((q) => q.status === "waiting") ||
  (m.imageRuns ?? []).some((r) => r.status !== "done" && r.status !== "failed" && r.status !== "stopped");

/**
 * `s` as a page that is no longer there left it: a reply streaming is stopped
 * with the text it had, a run or a switch under way is stopped, a question
 * waiting is skipped, and every call left without a result is answered right
 * after the results it has — the history the next turn sends must hold a
 * result for each call, as the turn loop keeps it. A settled session is
 * handed back as it was.
 */
export function settle(s: Session, newId: () => number): Session {
  let changed = false;
  const messages: Message[] = [];
  for (let i = 0; i < s.messages.length; i++) {
    const m = s.messages[i];
    if (unsettled(m)) changed = true;
    messages.push(unsettled(m) ? settleMessage(m) : m);
    if (m.role !== "assistant" || !m.toolCalls?.length) continue;
    const answered = new Set<string>();
    while (s.messages[i + 1]?.role === "tool") {
      const result = s.messages[++i];
      if (result.toolCallId) answered.add(result.toolCallId);
      messages.push(result);
    }
    for (const call of m.toolCalls)
      if (!answered.has(call.id)) {
        changed = true;
        messages.push({ id: newId(), role: "tool", content: CLOSED_MID_CALL, toolCallId: call.id, reasoning: "", streaming: false });
      }
  }
  return changed ? { ...s, messages } : s;
}

/** `image` holds a picture beside its redraw that the store does not keep yet. */
const unkept = (image: PromptImage): boolean => (image.full !== undefined && !image.kept?.full) || (image.markup !== undefined && !image.kept?.markup);

/**
 * A sent prompt's images with the pictures beside each redraw put in the store
 * by `keep`, and named on the image; null when none came out kept — nothing to
 * keep, or the store refused it, and the session is kept with the redraw alone.
 */
export async function keepPictures(images: PromptImage[], keep: (blob: Blob) => Promise<ImageId>): Promise<PromptImage[] | null> {
  let any = false;
  const kept = await Promise.all(
    images.map(async (image) => {
      if (!unkept(image)) return image;
      try {
        const full = image.full ? await keep(image.full) : undefined;
        const markup = image.markup
          ? { source: await keep(image.markup.source), region: await keep(image.markup.region), marks: image.markup.marks, keepOutside: image.markup.keepOutside }
          : undefined;
        any = true;
        return { ...image, kept: { ...(full ? { full } : {}), ...(markup ? { markup } : {}) } };
      } catch {
        return image;
      }
    }),
  );
  return any ? kept : null;
}

/** `s` with the pictures its images keep in the store read back beside them; one the store no longer has stays without. */
export async function withPictures(s: Session, read: (id: ImageId) => Promise<Blob | null>): Promise<Session> {
  if (picturesOf(s).size === 0) return s;
  const readImage = async (image: PromptImage): Promise<PromptImage> => {
    const { kept } = image;
    if (!kept) return image;
    const [full, source, region] = await Promise.all([
      kept.full ? read(kept.full) : null,
      kept.markup ? read(kept.markup.source) : null,
      kept.markup ? read(kept.markup.region) : null,
    ]);
    return {
      ...image,
      ...(full ? { full } : {}),
      ...(kept.markup && source && region ? { markup: { source, region, marks: kept.markup.marks, keepOutside: kept.markup.keepOutside } } : {}),
    };
  };
  const messages = await Promise.all(s.messages.map(async (m) => (m.images ? { ...m, images: await Promise.all(m.images.map(readImage)) } : m)));
  return { ...s, messages };
}
