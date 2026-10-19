import { restoreOrder } from "../store/docs.ts";
import { reserveUids } from "./model.ts";
import { type Decision, isUntouched } from "./sessions.ts";

// A decision as the page's store keeps it (spec playground-enhancements/03):
// a document of `decide-sessions` under a key of its own (store/docs.ts), the
// chat's rules over the Decide tab's sessions. A decision is plain data
// already — its pictures are data URIs, its file a name and a size beside the
// text — so it is kept whole, the evidence a loaded log brought with it too.
//
// Read back, a decision is settled: a send the page that wrote it had in
// flight never answered here, and the question uids it holds are reserved so
// the next question added to it is not numbered as one of them.

export const DECIDE_SESSIONS = "decide-sessions";

/** The schema this page writes. A document of a later one was written by a newer page, and is left alone. */
export const DECIDE_VERSION = 1;

export type StoredDecision = Omit<Decision, "id"> & {
  version: typeof DECIDE_VERSION;
  key: string;
  /** The page's id when written: the order the decisions were opened in. */
  id: number;
};

/** Why a decision that was being asked when the page went away has no new answers. */
export const CLOSED_MID_RUN = "The page was closed before the answers came back.";

/** An untouched decision is not written: an empty one is what a new page opens with anyway. */
export const worthKeeping = (d: Decision): boolean => !isUntouched(d);

export function toStored(d: Decision, key: string): StoredDecision {
  return { ...d, version: DECIDE_VERSION, key };
}

function readStored(value: unknown): StoredDecision | null {
  if (typeof value !== "object" || value === null) return null;
  const doc = value as Partial<StoredDecision>;
  if (doc.version !== DECIDE_VERSION || typeof doc.key !== "string" || typeof doc.id !== "number" || !doc.draft || !doc.spare) return null;
  return { answer: null, refusal: null, running: false, ...doc } as StoredDecision;
}

/**
 * The stored decisions, newest first and settled, with new ids from the page's
 * `nextId` up, and the key of each by its new id. The question uids they hold
 * are reserved on the way (`reserveUids`): the page's count is a module's.
 */
export function restoreDecisions(values: unknown[], nextId: number): { decisions: Decision[]; keys: Map<number, string>; nextId: number } {
  const docs = values.map(readStored).filter((d): d is StoredDecision => d !== null);
  const ordered = restoreOrder(docs, nextId);
  const { items } = ordered;
  const keys = new Map<number, string>();
  const decisions = items.map(({ version: _version, key, ...d }): Decision => {
    keys.set(d.id, key);
    reserveUids([...d.draft.questions, ...(d.answer?.draft.questions ?? [])].map((q) => q.uid));
    return d.running ? { ...d, running: false, refusal: CLOSED_MID_RUN } : d;
  });
  return { decisions, keys, nextId: ordered.nextId };
}
