import type { ImageStore } from "../store/imageStore.ts";
import { newKey } from "../store/docs.ts";
import { pageStore } from "../store/index.ts";
import {
  type Author,
  cleanName,
  type FileKind,
  KIND_LABEL,
  kindOf,
  latest,
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

export type Workspaces = {
  /** Every workspace, the last changed first. */
  list(): Promise<Workspace[]>;
  get(id: WorkspaceId): Promise<Workspace | null>;
  /** A workspace's files, deleted ones included, by name. */
  files(id: WorkspaceId): Promise<WorkspaceFile[]>;
  file(workspace: WorkspaceId, id: string): Promise<WorkspaceFile | null>;
  create(name: string): Promise<Workspace>;
  /** A chat's hold on its workspace, for its turns. */
  chat(key: string, options: BindingOptions): ChatBinding;
  /** Creates a text file, or replaces one whole with a new version. */
  writeText(workspace: WorkspaceId, name: string, text: string, author: Author): Promise<WriteOutcome>;
  /** Files the user brings, each a new file under a name no other file of the workspace holds. */
  add(workspace: WorkspaceId, files: Incoming[], author: Author): Promise<Written[]>;
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
    create,
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
