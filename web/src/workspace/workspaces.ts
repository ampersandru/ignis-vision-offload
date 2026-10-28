import type { ImageId, ImageStore } from "../store/imageStore.ts";
import { newKey } from "../store/docs.ts";
import { pageStore } from "../store/index.ts";
import {
  type Author,
  cleanName,
  type FileKind,
  freedBy,
  KIND_LABEL,
  kindOf,
  latest,
  present,
  uniqueName,
  type Version,
  type Workspace,
  WORKSPACE_FORMAT,
  type WorkspaceFile,
  type WorkspaceId,
} from "./files.ts";

// The workspace module (spec #170): the only writer of workspaces, their
// files and their versions, in the page's store (ADR 0073) — IndexedDB, or
// the in-memory store without it. It has two entrances: the user's actions
// (the files a prompt attaches, and the panel to come) and the model's tool
// calls (tools.ts, through the local tools' router). Both come here, so every
// change is a version with its author, and every change is announced to the
// page through `subscribe`.
//
// A workspace is a document of `workspaces`; its files are documents of a
// collection of their own, so that reading one workspace reads none of the
// others. Writes run one after another, the page over: a version's number is
// read, counted up and written back, and two writes at once would both take
// the same one.

export const WORKSPACES = "workspaces";

/** The collection holding a workspace's files, one document per file, keyed by the file's id. */
export const filesCollection = (workspace: WorkspaceId) => `workspace-files/${workspace}`;

/** A version a write or an addition made, as a reply's card names it: by the file's id, with its name then. */
export type Written = { workspace: WorkspaceId; file: string; name: string; version: number };

/** A file the user brings: its text inline, or its bytes with the text derived from them. */
export type Incoming = { name: string; kind: FileKind; text?: string; blob?: Blob; derived?: string };

/** What a chat's turn holds of the chat's workspace. */
export type ChatBinding = {
  /** The chat's key in the store: the chat as an author and a reader. */
  readonly key: string;
  /** The workspace the chat is bound to, if it is. */
  bound(): WorkspaceId | undefined;
  /** The chat's workspace, born now and named after the chat when it has none: once, however many calls ask at once. */
  workspace(): Promise<WorkspaceId>;
};

export type BindingOptions = {
  bound: () => WorkspaceId | undefined;
  /** The new workspace's name: the chat's title as it is now. */
  name: () => string;
  /** The workspace born for the chat: the chat records it, and never changes it after. */
  onBorn: (id: WorkspaceId) => void;
};

export type WriteOutcome = { ok: true; written: Written; created: boolean; workspace: Workspace } | { ok: false; error: string };

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
  /** Creates a text file, or replaces one whole with a new version. */
  writeText(workspace: WorkspaceId, name: string, text: string, author: Author): Promise<WriteOutcome>;
  /** Files the user brings, each a new file under a name no other file of the workspace holds. */
  add(workspace: WorkspaceId, files: Incoming[], author: Author): Promise<Written[]>;
  /**
   * Deletes a file: a new version by `author` marking it deleted, the versions
   * before it kept in its history. The one way to delete, the user's and the model's.
   */
  deleteFile(workspace: WorkspaceId, file: string, author: Author): Promise<FileOutcome>;
  /**
   * Gives a file another name, unique in its workspace, deleted files' names
   * included: their history keeps them. The content is not changed, so no
   * version is made; the file keeps its id and its kind.
   */
  renameFile(workspace: WorkspaceId, file: string, name: string): Promise<FileOutcome>;
  /** Called after every change, with the workspace and the file it touched. */
  subscribe(listener: (change: { workspace: WorkspaceId; file?: string }) => void): () => void;
};

const isWorkspace = (value: unknown): value is Workspace =>
  typeof value === "object" && value !== null && (value as Workspace).format === WORKSPACE_FORMAT && typeof (value as Workspace).id === "string";

const isFile = (value: unknown): value is WorkspaceFile =>
  typeof value === "object" && value !== null && (value as WorkspaceFile).format === WORKSPACE_FORMAT && Array.isArray((value as WorkspaceFile).versions);

export function createWorkspaces(store: ImageStore, options: { now?: () => number; newId?: () => string } = {}): Workspaces {
  const now = options.now ?? Date.now;
  const newId = options.newId ?? newKey;
  const listeners = new Set<(change: { workspace: WorkspaceId; file?: string }) => void>();
  // The workspace each chat's first file is creating, by the chat's key, kept once born.
  const births = new Map<string, Promise<WorkspaceId>>();
  // The same once born, read without waiting: deleting a workspace forgets the births into it.
  const bornTo = new Map<string, WorkspaceId>();
  let queue: Promise<unknown> = Promise.resolve();

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
    writeText(workspace, raw, text, author) {
      return serial(async (): Promise<WriteOutcome> => {
        const name = cleanName(raw);
        if (!name) return { ok: false, error: `"${raw}" is not a file name.` };
        const kind = kindOf(name);
        const had = (await files(workspace)).find((f) => f.name === name);
        if (had ? had.kind !== "text" : kind !== "text") {
          return { ok: false, error: `${name} is a ${KIND_LABEL[had?.kind ?? kind]} file, and only a text file can be written whole.` };
        }
        const at = now();
        const version: Version = { n: had ? latest(had).n + 1 : 1, author, at, text };
        const file: WorkspaceFile = had
          ? { ...had, versions: [...had.versions, version] }
          : { format: WORKSPACE_FORMAT, id: newId(), workspace, name, kind: "text", versions: [version] };
        await store.putDoc(filesCollection(workspace), file.id, file);
        const changed = await touch(workspace, at);
        notify({ workspace, file: file.id });
        return { ok: true, written: { workspace, file: file.id, name, version: version.n }, created: !had, workspace: changed };
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
    deleteFile(workspace, id, author) {
      return serial(async (): Promise<FileOutcome> => {
        const value = await store.getDoc<unknown>(filesCollection(workspace), id);
        if (!isFile(value)) return { ok: false, error: "The workspace has no such file." };
        if (!present(value)) return { ok: false, error: `${value.name} is already deleted.` };
        const at = now();
        const version: Version = { n: latest(value).n + 1, author, at, deleted: true };
        await store.putDoc(filesCollection(workspace), id, { ...value, versions: [...value.versions, version] });
        await touch(workspace, at);
        notify({ workspace, file: id });
        return { ok: true, written: { workspace, file: id, name: value.name, version: version.n } };
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
