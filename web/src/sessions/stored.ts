import type { PromptImage } from "../conversation/images.ts";
import { restoreOrder, type Stamp } from "../store/docs.ts";
import { type ImageId, unreferenced } from "../store/imageStore.ts";
import type { AgentRun } from "../tools/agents/agents.ts";
import type { Question } from "../tools/ask/ask.ts";
import type { ImageRun } from "../tools/image/image.ts";
import type { LocalRun } from "../tools/local/local.ts";
import type { WebRun } from "../tools/web/web.ts";
import type { LogRow, Message, Session, WorkspaceMove } from "./sessions.ts";

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
  /** Where it sorts (store/docs.ts). */
  order: number;
  title: string;
  /** `Session.startedAt`, in milliseconds. */
  startedAt: number;
  messages: StoredMessage[];
  log: LogRow[];
  attachments: Session["attachments"];
  /** `Session.workspaceId`, absent while the chat has none. */
  workspaceId?: string;
  /** `Session.moved`, absent while the chat never moved. */
  moved?: WorkspaceMove;
};

/** What a call left without a result is answered with when the session is read back. */
export const CLOSED_MID_CALL = "Stopped: the page was closed before the call finished.";

/** A session with nothing in it is not written: an empty one is what a new page opens with anyway. */
export const worthKeeping = (s: Session): boolean => s.messages.length > 0 || s.attachments.length > 0 || s.workspaceId !== undefined;

const storedImage = ({ full: _full, markup: _markup, ...image }: PromptImage): StoredImage => image;

export function toStored(s: Session, { key, order }: Stamp): StoredChat {
  return {
    version: CHAT_VERSION,
    key,
    order,
    title: s.title,
    startedAt: s.startedAt.getTime(),
    messages: s.messages.map(({ images, ...m }) => (images ? { ...m, images: images.map(storedImage) } : m)),
    log: s.log,
    attachments: s.attachments,
    ...(s.workspaceId !== undefined ? { workspaceId: s.workspaceId } : {}),
    ...(s.moved ? { moved: s.moved } : {}),
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

/**
 * The pictures deleting `gone` frees: those no session `left` names, nor
 * anything `elsewhere` holds (the Image view's sessions). None while
 * `elsewhere` cannot say yet — the other view has not read its sessions, and a
 * picture only they name would go.
 */
export function freedBy(gone: Session, left: Session[], elsewhere: Iterable<ImageId> | null): ImageId[] {
  if (elsewhere === null) return [];
  return unreferenced(picturesOf(gone), new Set<ImageId>([...left.flatMap((s) => [...picturesOf(s)]), ...elsewhere]));
}

const isMove = (value: unknown): value is WorkspaceMove => {
  const move = value as Partial<WorkspaceMove> | null | undefined;
  return typeof move?.from === "string" && typeof move.at === "number" && (move.name === undefined || typeof move.name === "string");
};

/** A stored value read as a session document of this page's schema, or null. */
function readStored(value: unknown): StoredChat | null {
  if (typeof value !== "object" || value === null) return null;
  const { workspaceId, moved, ...doc } = value as Partial<StoredChat>;
  if (doc.version !== CHAT_VERSION || typeof doc.key !== "string" || typeof doc.order !== "number" || !Array.isArray(doc.messages)) return null;
  const workspace = { ...(typeof workspaceId === "string" ? { workspaceId } : {}), ...(isMove(moved) ? { moved } : {}) };
  return { ...doc, title: doc.title ?? "", startedAt: doc.startedAt ?? Date.now(), log: doc.log ?? [], attachments: doc.attachments ?? [], ...workspace } as StoredChat;
}

/**
 * The stored sessions, newest first and settled, with new ids from the page's
 * `nextId` up — and above every message id they hold, since the page's one
 * counter hands out both — and the stamp of each by its new id. Their pictures
 * are still in the store (`withPictures`).
 */
export function restoreChats(values: unknown[], nextId: number): { sessions: Session[]; stamps: Map<number, Stamp>; nextId: number } {
  const docs = values.map(readStored).filter((d): d is StoredChat => d !== null);
  const highest = Math.max(0, ...docs.flatMap((d) => d.messages.map((m) => m.id)));
  const ordered = restoreOrder(docs, Math.max(nextId, highest + 1));
  const { items } = ordered;
  let next = ordered.nextId;
  const newId = () => next++;
  const stamps = new Map<number, Stamp>();
  const sessions = items.map((doc) => {
    stamps.set(doc.id, { key: doc.key, order: doc.order });
    const workspace = { ...(doc.workspaceId !== undefined ? { workspaceId: doc.workspaceId } : {}), ...(doc.moved ? { moved: doc.moved } : {}) };
    return settle({ id: doc.id, title: doc.title, startedAt: new Date(doc.startedAt), messages: doc.messages, log: doc.log, attachments: doc.attachments, ...workspace }, newId);
  });
  return { sessions, stamps, nextId: next };
}

/** `items` with `stop` applied to each, or `items` itself when it changed none. */
function mapSame<T>(items: T[] | undefined, stop: (item: T) => T): T[] | undefined {
  const next = items?.map(stop);
  return next && next.some((item, i) => item !== items![i]) ? next : items;
}

const stopWeb = (run: WebRun): WebRun => (run.status === "running" ? { ...run, status: "stopped" } : run);
const stopLocal = (run: LocalRun): LocalRun => (run.status === "running" || run.status === "checking" ? { ...run, status: "stopped" } : run);
const skipQuestion = (q: Question): Question => (q.status === "waiting" ? { ...q, status: "skipped" } : q);
const stopPicture = (run: ImageRun): ImageRun => (run.status === "done" || run.status === "failed" || run.status === "stopped" ? run : { ...run, status: "stopped" });

function stopAgent(run: AgentRun): AgentRun {
  const web = mapSame(run.web, stopWeb);
  const local = mapSame(run.local, stopLocal);
  const status = run.status === "queued" || run.status === "running" ? "stopped" : run.status;
  if (status === run.status && web === run.web && local === run.local) return run;
  return { ...run, status, ...(web ? { web } : {}), ...(local ? { local } : {}) };
}

/** `m` with nothing left running, or `m` itself when nothing was. */
function settleMessage(m: Message): Message {
  const agents = mapSame(m.agents, stopAgent);
  const web = mapSame(m.web, stopWeb);
  const local = mapSame(m.local, stopLocal);
  const questions = mapSame(m.questions, skipQuestion);
  const imageRuns = mapSame(m.imageRuns, stopPicture);
  const switching = m.switch?.status === "switching";
  const runs = agents === m.agents && web === m.web && local === m.local && questions === m.questions && imageRuns === m.imageRuns;
  if (!m.streaming && !m.queued && !switching && runs) return m;
  const { queued: _queued, ...rest } = m;
  return {
    ...rest,
    streaming: false,
    ...(switching && m.switch ? { switch: { ...m.switch, status: "stopped" } } : {}),
    ...(agents ? { agents } : {}),
    ...(web ? { web } : {}),
    ...(local ? { local } : {}),
    ...(questions ? { questions } : {}),
    ...(imageRuns ? { imageRuns } : {}),
  };
}

/**
 * `s` as a page that is no longer there left it: a reply streaming is stopped
 * with the text it had, a run or a switch under way is stopped, a question
 * waiting is skipped, and every call left without a result is answered right
 * after the results it has — the history the next turn sends must hold a
 * result for each call, as the turn loop keeps it. A reply that failed keeps
 * its calls unanswered, as the loop leaves them: the history drops it
 * (`conversationTurns`). A settled session is handed back as it was.
 */
export function settle(s: Session, newId: () => number): Session {
  let changed = false;
  const messages: Message[] = [];
  for (let i = 0; i < s.messages.length; i++) {
    const m = s.messages[i];
    const settled = settleMessage(m);
    if (settled !== m) changed = true;
    messages.push(settled);
    if (m.role !== "assistant" || !m.toolCalls?.length || m.error !== undefined) continue;
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
