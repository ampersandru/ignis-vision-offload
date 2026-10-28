import type { ImageId, ImageStore } from "../store/imageStore.ts";
import { newKey } from "../store/docs.ts";
import { pageStore } from "../store/index.ts";
import {
  type Author,
  authorLabel,
  type ChatAuthor,
  cleanName,
  type FileKind,
  freedBy,
  KIND_LABEL,
  kindOf,
  latest,
  present,
  uniqueName,
  USER,
  type Version,
  type Workspace,
  WORKSPACE_FORMAT,
  type WorkspaceFile,
  type WorkspaceId,
} from "./files.ts";

// The workspace module (spec #170): the only writer of workspaces, their
// files and their versions, in the page's store (ADR 0073) — IndexedDB, or
// the in-memory store without it. It has two entrances: the user's actions
// (the files a prompt attaches, and the panel's: its files, hand edits and
// restores) and the model's tool
// calls (tools.ts, through the local tools' router). Both come here, so every
// change is a version with its author, and every change is announced to the
// page through `subscribe`.
//
// A workspace is a document of `workspaces`; its files are documents of a
// collection of their own, so that reading one workspace reads none of the
// others. Writes run one after another, the page over: a version's number is
// read, counted up and written back, and two writes at once would both take
// the same one.
//
// The guard: on the model's side every chat, and every sub-agent of a chat,
// is a reader, and the module keeps for each reader and each file the version
// that reader last read or wrote. A reader's write over an existing file, its
// edit or its deletion is refused when the file is at another version now, so
// that no reader overwrites a change it has not seen: the user's, another
// chat's or another sub-agent's. The check runs in the queue of writes, with
// the change it allows. A chat's reader starts from what its history read
// (tools.ts, seenIn), which is kept with the chat; a sub-agent's starts empty
// and lasts its run.

export const WORKSPACES = "workspaces";

/** The collection holding a workspace's files, one document per file, keyed by the file's id. */
export const filesCollection = (workspace: WorkspaceId) => `workspace-files/${workspace}`;

/** A version a write or an addition made, as a reply's card names it: by the file's id, with its name then. */
export type Written = { workspace: WorkspaceId; file: string; name: string; version: number };

/** A file the user brings: its text inline, or its bytes with the text derived from them. */
export type Incoming = { name: string; kind: FileKind; text?: string; blob?: Blob; derived?: string };

/**
 * On the model's side, who reads and writes the files: a chat, or a sub-agent
 * of a chat. Only the module makes readers, and only it keeps what each has
 * seen; a reader made anywhere else has seen nothing.
 */
export type Reader = { readonly author: ChatAuthor };

/** Who changes a file: the user, whom the guard never refuses, or a reader, under the guard. */
export type Changer = typeof USER | Reader;

/** A version of a file a reader has seen: it read it, or made it. */
export type Seen = { file: string; version: number };

/** What a chat's turn holds of the chat's workspace. */
export type ChatBinding = {
  /** The chat's key in the store: the chat as an author and a reader. */
  readonly key: string;
  /** The workspace the chat is bound to, if it is. */
  bound(): WorkspaceId | undefined;
  /** The chat's workspace, born now and named after the chat when it has none: once, however many calls ask at once. */
  workspace(): Promise<WorkspaceId>;
  /** The chat's model as a reader: it has seen what its history read or made, and what it reads or makes in this turn. */
  readonly reader: Reader;
  /** A sub-agent of the chat, started by the `agent` call `agent.call`: a reader of its own, which has seen nothing yet. */
  agent(agent: { call: string; name: string }): Reader;
};

export type BindingOptions = {
  bound: () => WorkspaceId | undefined;
  /** The new workspace's name: the chat's title as it is now. */
  name: () => string;
  /** The workspace born for the chat: the chat records it, and never changes it after. */
  onBorn: (id: WorkspaceId) => void;
  /** The versions the chat's history read or made (tools.ts, seenIn): what its reader has seen when the turn starts. */
  seen?: Iterable<Seen>;
};

export type WriteOutcome = { ok: true; written: Written; version: Version; created: boolean; workspace: Workspace } | { ok: false; error: string };

/** What a change to one file did: the version it made, or why it made none. */
export type FileOutcome = { ok: true; written: Written } | { ok: false; error: string };

export type Workspaces = {
  /** Every workspace, the last changed first. */
  list(): Promise<Workspace[]>;
  get(id: WorkspaceId): Promise<Workspace | null>;
  /** A workspace's files, deleted ones included, by name. */
  files(id: WorkspaceId): Promise<WorkspaceFile[]>;
  file(workspace: WorkspaceId, id: string): Promise<WorkspaceFile | null>;
  /** A version's content, as a download gives it: its text, or its bytes; null when the page holds no bytes of it. */
  content(version: Version): Promise<Blob | null>;
  create(name: string): Promise<Workspace>;
  rename(id: WorkspaceId, name: string): Promise<{ ok: true; workspace: Workspace } | { ok: false; error: string }>;
  /**
   * Deletes a workspace with its files and every version, and frees the bytes
   * nothing else holds: no other workspace, nor among the pictures `elsewhere`
   * names (the chats' and the image sessions'; null while not known, and then
   * no picture is freed). The chats that worked in it are the caller's.
   */
  remove(id: WorkspaceId, elsewhere: Iterable<ImageId> | null): Promise<{ files: number }>;
  /** A chat's hold on its workspace, for its turns. */
  chat(key: string, options: BindingOptions): ChatBinding;
  /** The file's newest version, as `reader` receives it: from then on the guard holds it as the version that reader read. */
  read(reader: Reader, file: WorkspaceFile): Version;
  /** Creates a text file, or replaces one whole with a new version. */
  writeText(workspace: WorkspaceId, name: string, text: string, by: Changer): Promise<WriteOutcome>;
  /** Replaces the one occurrence of `old` in the text file `file` (its id) with `replacement`, as a new version. */
  editText(workspace: WorkspaceId, file: string, old: string, replacement: string, by: Changer): Promise<WriteOutcome>;
  /** Files the user brings, each a new file under a name no other file of the workspace holds. */
  add(workspace: WorkspaceId, files: Incoming[], author: Author): Promise<Written[]>;
  /**
   * Deletes a file: a new version by `by` marking it deleted, the versions
   * before it kept in its history. The one way to delete, the user's and the
   * model's; a reader's deletion is under the guard.
   */
  deleteFile(workspace: WorkspaceId, file: string, by: Changer): Promise<FileOutcome>;
  /**
   * Gives a file another name, unique in its workspace, deleted files' names
   * included: their history keeps them. The content is not changed, so no
   * version is made; the file keeps its id and its kind.
   */
  renameFile(workspace: WorkspaceId, file: string, name: string): Promise<FileOutcome>;
  /**
   * The user's restore of version `n` of a file: a new newest version with
   * that version's content (its text, its bytes and the text derived from
   * them), authored by the user, the versions between kept. A deleted file is
   * restored the same way, from a version before its deletion.
   */
  restore(workspace: WorkspaceId, file: string, n: number): Promise<FileOutcome>;
  /** Called after every change, with the workspace and the file it touched. */
  subscribe(listener: (change: { workspace: WorkspaceId; file?: string }) => void): () => void;
};

const isWorkspace = (value: unknown): value is Workspace =>
  typeof value === "object" && value !== null && (value as Workspace).format === WORKSPACE_FORMAT && typeof (value as Workspace).id === "string";

const isFile = (value: unknown): value is WorkspaceFile =>
  typeof value === "object" && value !== null && (value as WorkspaceFile).format === WORKSPACE_FORMAT && Array.isArray((value as WorkspaceFile).versions);

const isReader = (by: Changer): by is Reader => "author" in by;

const authorOf = (by: Changer): Author => (isReader(by) ? by.author : USER);

/** The file of the workspace named `name`: the one in the workspace, else a deleted one. */
const named = (files: WorkspaceFile[], name: string) =>
  files.find((f) => f.name === name && present(f)) ?? files.find((f) => f.name === name);

/** Where `old` occurs in `text`, overlapping occurrences included. */
function occurrences(text: string, old: string): number[] {
  const at: number[] = [];
  for (let i = text.indexOf(old); i !== -1; i = text.indexOf(old, i + 1)) at.push(i);
  return at;
}

const lineAt = (text: string, index: number) => text.slice(0, index).split("\n").length;

/** "3", "3 and 9", "3, 9 and 14". */
const listed = (items: number[]) => (items.length < 2 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`);

export function createWorkspaces(store: ImageStore, options: { now?: () => number; newId?: () => string } = {}): Workspaces {
  const now = options.now ?? Date.now;
  const newId = options.newId ?? newKey;
  const listeners = new Set<(change: { workspace: WorkspaceId; file?: string }) => void>();
  // The workspace each chat's first file is creating, by the chat's key, kept once born.
  const births = new Map<string, Promise<WorkspaceId>>();
  // The same once born, read without waiting: deleting a workspace forgets the births into it.
  const bornTo = new Map<string, WorkspaceId>();
  // The guard's memory: for each reader, the newest version of each file (by its id) it read or made.
  const seen = new WeakMap<Reader, Map<string, number>>();
  let queue: Promise<unknown> = Promise.resolve();

  function newReader(author: ChatAuthor, from: Iterable<Seen> = []): Reader {
    const reader: Reader = Object.freeze({ author });
    seen.set(reader, new Map());
    for (const s of from) saw(reader, s.file, s.version);
    return reader;
  }

  /** `by` has seen version `n` of `file`: a version it read or made is never one it has not seen, whatever it saw since. */
  function saw(by: Changer, file: string, n: number) {
    const versions = isReader(by) ? seen.get(by) : undefined;
    versions?.set(file, Math.max(versions.get(file) ?? 0, n));
  }

  /** Why `by` may not change `file` as it is now, or null when it may: the user always may, a reader only over the version it last read or wrote. */
  function guard(by: Changer, file: WorkspaceFile): string | null {
    if (!isReader(by)) return null;
    const newest = latest(file);
    const had = seen.get(by)?.get(file.id);
    if (had === newest.n) return null;
    const who = authorLabel(newest.author, by.author);
    if (had === undefined) {
      return `${file.name} is a file you have not read: it is at version ${newest.n}, by ${who}. Read it with read_file before you change it.`;
    }
    if (newest.deleted) {
      return `${file.name} was deleted by ${who} since you last read it (version ${newest.n}). Read it again with read_file before you write it.`;
    }
    return `${file.name} changed since you last read it: it is at version ${newest.n}, by ${who}, and you last read version ${had}. Read it again with read_file before you change it.`;
  }

  /** `job` after every write before it; a failure fails that write alone. */
  function serial<T>(job: () => Promise<T>): Promise<T> {
    const next = queue.then(job);
    queue = next.catch(() => {});
    return next;
  }

  const notify = (change: { workspace: WorkspaceId; file?: string }) => {
    for (const listener of listeners) listener(change);
  };

  async function get(id: WorkspaceId): Promise<Workspace | null> {
    const value = await store.getDoc<unknown>(WORKSPACES, id);
    return isWorkspace(value) ? value : null;
  }

  async function files(id: WorkspaceId): Promise<WorkspaceFile[]> {
    const values = await store.listDocs<unknown>(filesCollection(id));
    return values.filter(isFile).sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  }

  async function touch(id: WorkspaceId, at: number): Promise<Workspace> {
    const workspace = await get(id);
    if (!workspace) throw new Error(`no workspace has the id ${id}`);
    const changed = { ...workspace, updated: at };
    await store.putDoc(WORKSPACES, id, changed);
    return changed;
  }

  async function create(name: string): Promise<Workspace> {
    return serial(async () => {
      const at = now();
      const workspace: Workspace = { format: WORKSPACE_FORMAT, id: newId(), name: name.trim() || "Workspace", created: at, updated: at };
      await store.putDoc(WORKSPACES, workspace.id, workspace);
      notify({ workspace: workspace.id });
      return workspace;
    });
  }

  /** Keeps the file with its new version `version`, made by `by`, and says what it wrote. */
  async function commit(file: WorkspaceFile, version: Version, by: Changer, created: boolean): Promise<WriteOutcome> {
    await store.putDoc(filesCollection(file.workspace), file.id, file);
    const changed = await touch(file.workspace, version.at);
    saw(by, file.id, version.n);
    notify({ workspace: file.workspace, file: file.id });
    return { ok: true, written: { workspace: file.workspace, file: file.id, name: file.name, version: version.n }, version, created, workspace: changed };
  }

  /** The file `id` of `workspace` as it is now, with the version `by` adds to it, or why `by` may not change it. */
  async function next(workspace: WorkspaceId, id: string, by: Changer): Promise<{ file: WorkspaceFile; n: number } | { error: string }> {
    const value = await store.getDoc<unknown>(filesCollection(workspace), id);
    if (!isFile(value)) return { error: "The workspace has no such file." };
    const newest = latest(value);
    if (newest.deleted) {
      return { error: isReader(by) ? `${value.name} was deleted by ${authorLabel(newest.author, by.author)} (version ${newest.n}).` : `${value.name} is already deleted.` };
    }
    const refused = guard(by, value);
    return refused ? { error: refused } : { file: value, n: newest.n + 1 };
  }

  return {
    async list() {
      const values = await store.listDocs<unknown>(WORKSPACES);
      return values.filter(isWorkspace).sort((a, b) => b.updated - a.updated);
    },
    get,
    files,
    async file(workspace, id) {
      const value = await store.getDoc<unknown>(filesCollection(workspace), id);
      return isFile(value) ? value : null;
    },
    async content(version) {
      if (version.text !== undefined) return new Blob([version.text], { type: "text/plain;charset=utf-8" });
      if (!version.bytes) return null;
      return version.bytes.store === "image" ? store.getImage(version.bytes.id) : store.getBlob(version.bytes.id);
    },
    create,
    rename(id, raw) {
      return serial(async () => {
        const name = raw.trim();
        if (!name) return { ok: false as const, error: "A workspace needs a name." };
        const workspace = await get(id);
        if (!workspace) return { ok: false as const, error: "That workspace was deleted." };
        const renamed = { ...workspace, name, updated: now() };
        await store.putDoc(WORKSPACES, id, renamed);
        notify({ workspace: id });
        return { ok: true as const, workspace: renamed };
      });
    },
    remove(id, elsewhere) {
      return serial(async () => {
        const gone = await files(id);
        const others: WorkspaceFile[] = [];
        for (const w of await store.listDocs<unknown>(WORKSPACES)) if (isWorkspace(w) && w.id !== id) others.push(...(await files(w.id)));
        const freed = freedBy(gone, others, elsewhere);
        for (const file of gone) await store.deleteDoc(filesCollection(id), file.id);
        await store.deleteDoc(WORKSPACES, id);
        // A chat born into it starts again with its next file.
        for (const [key, born] of bornTo) if (born === id) births.delete(key), bornTo.delete(key);
        // What the store fails to free stays in it, unnamed; the workspace is gone either way.
        for (const blob of freed.blobs) await store.deleteBlob(blob).catch(() => {});
        for (const image of freed.images) await store.deleteImage(image).catch(() => {});
        for (const file of gone) notify({ workspace: id, file: file.id });
        notify({ workspace: id });
        return { files: gone.length };
      });
    },
    chat(key, binding) {
      return {
        key,
        reader: newReader({ by: "chat", chat: key }, binding.seen),
        agent: (agent) => newReader({ by: "chat", chat: key, agent: { call: agent.call, name: agent.name } }),
        bound: binding.bound,
        workspace() {
          const bound = binding.bound();
          if (bound) return Promise.resolve(bound);
          let birth = births.get(key);
          if (!birth) {
            birth = create(binding.name()).then((workspace) => {
              bornTo.set(key, workspace.id);
              binding.onBorn(workspace.id);
              return workspace.id;
            });
            births.set(key, birth);
            // A birth the store refused is tried again by the next call.
            birth.catch(() => births.delete(key));
          }
          return birth;
        },
      };
    },
    read(reader, file) {
      const version = latest(file);
      saw(reader, file.id, version.n);
      return version;
    },
    writeText(workspace, raw, text, by) {
      return serial(async (): Promise<WriteOutcome> => {
        const name = cleanName(raw);
        if (!name) return { ok: false, error: `"${raw}" is not a file name.` };
        const kind = kindOf(name);
        const had = named(await files(workspace), name);
        if (had ? had.kind !== "text" : kind !== "text") {
          return { ok: false, error: `${name} is a ${KIND_LABEL[had?.kind ?? kind]} file, and only a text file can be written whole.` };
        }
        // A deleted file the reader never knew of is a new file to it, and a new file has no guard.
        const refused = had && (present(had) || (isReader(by) && seen.get(by)?.has(had.id))) ? guard(by, had) : null;
        if (refused) return { ok: false, error: refused };
        const version: Version = { n: had ? latest(had).n + 1 : 1, author: authorOf(by), at: now(), text };
        const file: WorkspaceFile = had
          ? { ...had, versions: [...had.versions, version] }
          : { format: WORKSPACE_FORMAT, id: newId(), workspace, name, kind: "text", versions: [version] };
        return commit(file, version, by, !had || !present(had));
      });
    },
    editText(workspace, id, old, replacement, by) {
      return serial(async (): Promise<WriteOutcome> => {
        const found = await next(workspace, id, by);
        if ("error" in found) return { ok: false, error: found.error };
        const { file, n } = found;
        if (file.kind !== "text") return { ok: false, error: `${file.name} is a ${KIND_LABEL[file.kind]} file, and only a text file can be edited.` };
        if (old === "") return { ok: false, error: 'The passage to replace ("old") is empty.' };
        if (old === replacement) return { ok: false, error: 'The passage to replace ("old") and its replacement ("new") are the same: the edit would change nothing.' };
        const text = latest(file).text ?? "";
        let at = occurrences(text, old);
        let swap = [old, replacement];
        // read_file shows a file's lines without their "\r": a passage over several lines of a file with CRLF line endings matches as it would have read.
        if (at.length === 0 && text.includes("\r\n") && old.includes("\n")) {
          swap = [old, replacement].map((s) => s.replace(/\r?\n/g, "\r\n"));
          at = occurrences(text, swap[0]);
        }
        if (at.length === 0) {
          return { ok: false, error: `The passage to replace ("old") occurs nowhere in ${file.name}. Copy it exactly as read_file shows it, without the line numbers.` };
        }
        if (at.length > 1) {
          const lines = [...new Set(at.map((i) => lineAt(text, i)))];
          return {
            ok: false,
            error: `The passage to replace ("old") occurs ${at.length} times in ${file.name}, on ${lines.length === 1 ? "line" : "lines"} ${listed(lines)}: give more of the text around the one to change, so that it occurs once.`,
          };
        }
        const version: Version = {
          n,
          author: authorOf(by),
          at: now(),
          text: text.slice(0, at[0]) + swap[1] + text.slice(at[0] + swap[0].length),
        };
        return commit({ ...file, versions: [...file.versions, version] }, version, by, false);
      });
    },
    add(workspace, incoming, author) {
      return serial(async () => {
        const taken = (await files(workspace)).map((f) => f.name);
        const written: Written[] = [];
        for (const item of incoming) {
          const at = now();
          const name = uniqueName(cleanName(item.name) ?? "file", taken);
          taken.push(name);
          const stored = item.blob ? await store.putBlob(item.blob) : null;
          const version: Version = {
            n: 1,
            author,
            at,
            ...(item.text !== undefined ? { text: item.text } : {}),
            ...(stored ? { bytes: { store: "blob" as const, id: stored.id, type: stored.type, size: stored.bytes } } : {}),
            ...(item.derived !== undefined ? { derived: item.derived } : {}),
          };
          const file: WorkspaceFile = { format: WORKSPACE_FORMAT, id: newId(), workspace, name, kind: item.kind, versions: [version] };
          await store.putDoc(filesCollection(workspace), file.id, file);
          written.push({ workspace, file: file.id, name, version: 1 });
        }
        if (written.length > 0) await touch(workspace, now());
        for (const w of written) notify({ workspace, file: w.file });
        return written;
      });
    },
    deleteFile(workspace, id, by) {
      return serial(async (): Promise<FileOutcome> => {
        const found = await next(workspace, id, by);
        if ("error" in found) return { ok: false, error: found.error };
        const { file, n } = found;
        const version: Version = { n, author: authorOf(by), at: now(), deleted: true };
        const outcome = await commit({ ...file, versions: [...file.versions, version] }, version, by, false);
        return outcome.ok ? { ok: true, written: outcome.written } : outcome;
      });
    },
    renameFile(workspace, id, raw) {
      return serial(async (): Promise<FileOutcome> => {
        const name = cleanName(raw);
        if (!name) return { ok: false, error: `"${raw.trim()}" is not a file name.` };
        const all = await files(workspace);
        const file = all.find((f) => f.id === id);
        if (!file) return { ok: false, error: "The workspace has no such file." };
        if (!present(file)) return { ok: false, error: `${file.name} is deleted.` };
        const written: Written = { workspace, file: id, name, version: latest(file).n };
        if (name === file.name) return { ok: true, written };
        const holder = all.find((f) => f.name === name);
        if (holder) {
          return { ok: false, error: present(holder) ? `A file named ${name} is already in the workspace.` : `A deleted file named ${name} keeps that name in its history.` };
        }
        await store.putDoc(filesCollection(workspace), id, { ...file, name });
        await touch(workspace, now());
        notify({ workspace, file: id });
        return { ok: true, written };
      });
    },
    restore(workspace, id, n) {
      return serial(async (): Promise<FileOutcome> => {
        const value = await store.getDoc<unknown>(filesCollection(workspace), id);
        if (!isFile(value)) return { ok: false, error: "The workspace has no such file." };
        const old = value.versions.find((v) => v.n === n);
        const newest = latest(value);
        if (!old) return { ok: false, error: `${value.name} has no version ${n}.` };
        if (old.deleted) return { ok: false, error: `Version ${n} of ${value.name} is its deletion, which has no content to restore.` };
        if (old === newest) return { ok: false, error: `Version ${n} is already the newest of ${value.name}.` };
        const version: Version = {
          n: newest.n + 1,
          author: USER,
          at: now(),
          ...(old.text !== undefined ? { text: old.text } : {}),
          ...(old.bytes ? { bytes: old.bytes } : {}),
          ...(old.derived !== undefined ? { derived: old.derived } : {}),
        };
        const outcome = await commit({ ...value, versions: [...value.versions, version] }, version, USER, false);
        return outcome.ok ? { ok: true, written: outcome.written } : outcome;
      });
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

let page: Workspaces | null = null;

/** The page's one workspace module, on the page's store: one queue of writes and one set of listeners for every chat. */
export function pageWorkspaces(): Workspaces {
  page ??= createWorkspaces(pageStore());
  return page;
}
