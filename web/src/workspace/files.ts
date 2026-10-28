import { type BlobId, type ImageId, unreferenced } from "../store/imageStore.ts";

// What a workspace is made of (spec #170, ADR 0073): a named set of files the
// user and the model work on together, kept by the page. A file has a name
// unique in its workspace, which may hold `/` (there are no folders as
// objects), a kind, and its versions; a version has a number, its author, its
// time and its content, or marks the file deleted. Plain data, as the page's
// store keeps it; workspaces.ts is the only writer.

export type WorkspaceId = string;

/** The schema these records are written in. A record of another one was written by a newer page, and is left alone. */
export const WORKSPACE_FORMAT = 1;

export type Workspace = {
  format: typeof WORKSPACE_FORMAT;
  id: WorkspaceId;
  name: string;
  /** Milliseconds. */
  created: number;
  /** The last change to the workspace or any of its files, in milliseconds. */
  updated: number;
};

/** Decided from the content type and the name: what the model can read of a file, and how the page shows it. */
export type FileKind = "text" | "pdf" | "picture" | "audio" | "other";

/**
 * Who made a version: the user, a chat (by its store key, which never
 * changes), or a sub-agent of a chat (by the `agent` call that started it).
 * A chat or a sub-agent is also a reader, the unit that reads and writes a
 * file on the model's side.
 */
export type Author = { by: "user" } | { by: "chat"; chat: string; agent?: { call: string; name: string } };

/** An author on the model's side: a chat, or a sub-agent of a chat. */
export type ChatAuthor = Extract<Author, { by: "chat" }>;

export const USER: { by: "user" } = { by: "user" };

/** `author` as the reader `reader` hears it named: the user, you, the chat or one of its sub-agents, or another chat. */
export function authorLabel(author: Author, reader: ChatAuthor): string {
  if (author.by === "user") return "the user";
  if (author.chat !== reader.chat) return "another chat";
  if ((author.agent?.call ?? null) === (reader.agent?.call ?? null)) return "you";
  return author.agent ? `the sub-agent ${author.agent.name}` : "the conversation that started you";
}

/**
 * Bytes kept in the page's store rather than inline: a picture in the image
 * store image sessions use (#394), so that a picture shared with one is one
 * copy, with its size in pixels as it was measured there; anything else as a blob.
 */
export type StoredBytes =
  | { store: "blob"; id: BlobId; type: string; size: number }
  | { store: "image"; id: ImageId; type: string; size: number; width?: number; height?: number };

export type Version = {
  /** 1 for the file's first version, counting up. */
  n: number;
  author: Author;
  /** Milliseconds. */
  at: number;
  /** A text file's content, inline. */
  text?: string;
  /** Any other file's content. */
  bytes?: StoredBytes;
  /** Text derived from the bytes, computed on the version's first read (or when a PDF is attached) and kept with it: a PDF's text, an audio file's transcript (#395). */
  derived?: string;
  /** This version marks the file deleted; the history keeps the versions before it. */
  deleted?: true;
};

export type WorkspaceFile = {
  format: typeof WORKSPACE_FORMAT;
  /** Its own id: a rename changes the name, never the id. */
  id: string;
  workspace: WorkspaceId;
  name: string;
  kind: FileKind;
  /** Oldest first, never empty. */
  versions: Version[];
};

/**
 * A workspace file as a chat knows it, in the workspace it is in: what a
 * later turn's notice compares the files with (workspace/notice.ts).
 */
export type FileMark = { id: string; workspace: WorkspaceId; name: string; version: number };

/**
 * A chat's move out of a workspace its model knew (#404): the one it left,
 * by its id and by the name it had then (none once deleted), and when.
 */
export type WorkspaceMove = { from: WorkspaceId; name?: string; at: number };

/** The file's newest version. */
export const latest = (file: WorkspaceFile): Version => file.versions[file.versions.length - 1];

/** The file is in the workspace: its newest version does not delete it. */
export const present = (file: WorkspaceFile): boolean => !latest(file).deleted;

const PICTURE = /\.(png|jpe?g|gif|webp|bmp|avif|heic|heif|tiff?|ico)$/i;
const AUDIO = /\.(wav|mp3|m4a|aac|flac|ogg|oga|opus|weba)$/i;
const BINARY = /\.(zip|gz|tgz|tar|7z|rar|bz2|xz|zst|xlsx?|docx?|pptx?|odt|ods|odp|exe|dll|so|dylib|bin|wasm|class|jar|o|a|lib|mp4|mov|avi|mkv|webm|woff2?|ttf|otf|eot|sqlite|db|iso|dmg|psd)$/i;
const TEXT_TYPE = /^(text\/|image\/svg\+xml$|application\/(json|xml|javascript|ecmascript|x-sh|x-yaml|yaml|toml|sql|x-tex|ld\+json|xhtml\+xml)\b)/;

/**
 * A file's kind from its name and its content type. A file nothing marks as
 * anything else is text (`Makefile`, `notes.txt`, `app.ts`). A browser's type
 * is a guess (it calls a `.ts` file a video), so whoever brings the bytes of
 * an `other` file may still find them to be text.
 */
export function kindOf(name: string, type = ""): FileKind {
  const t = type.toLowerCase().split(";")[0].trim();
  if (/\.pdf$/i.test(name) || t === "application/pdf") return "pdf";
  if (/\.svg$/i.test(name) || TEXT_TYPE.test(t)) return "text";
  if (PICTURE.test(name) || t.startsWith("image/")) return "picture";
  if (AUDIO.test(name) || t.startsWith("audio/")) return "audio";
  if (BINARY.test(name) || (t !== "" && t !== "application/octet-stream")) return "other";
  return "text";
}

const pad = (n: number) => String(n).padStart(2, "0");

/** A moment as the workspace's lines give it, a file's version or a workspace's last change: the local date and time to the minute. */
export function timeLabel(at: number): string {
  const d = new Date(at);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** What the page says when the browser's store refused a change. */
export const refusal = (err: unknown) => `Not saved in this browser: ${err instanceof Error ? err.message : String(err)}`;

/** The kind as a sentence names it. */
export const KIND_LABEL: Record<FileKind, string> = { text: "text", pdf: "PDF", picture: "picture", audio: "audio", other: "other" };

/**
 * A file name as a workspace keeps it, or null when nothing is left of it:
 * `/` separates folders (a `\` becomes one), empty and `.` segments go, a
 * `..` segment refuses the name, and the characters a file system refuses
 * become `_`, so that a zip of the workspace unpacks anywhere.
 */
export function cleanName(raw: string): string | null {
  const segments = raw.replace(/\\/g, "/").split("/").map((s) => s.trim());
  if (segments.includes("..")) return null;
  const kept = segments.filter((s) => s !== "" && s !== ".").map((s) => s.replace(/[\u0000-\u001f\u007f:*?"<>|]+/g, "_"));
  return kept.length > 0 ? kept.join("/") : null;
}

/** `name`, or `name (2).ext` when a file of that name is already taken. */
export function uniqueName(name: string, taken: Iterable<string>): string {
  const names = new Set(taken);
  if (!names.has(name)) return name;
  const slash = name.lastIndexOf("/") + 1;
  const dot = name.lastIndexOf(".");
  const cut = dot > slash ? dot : name.length;
  for (let n = 2; ; n++) {
    const candidate = `${name.slice(0, cut)} (${n})${name.slice(cut)}`;
    if (!names.has(candidate)) return candidate;
  }
}

/** The bytes files hold in the page's store, every version's: what deleting them may free, and what deleting anything else must keep. */
export function bytesOf(files: Iterable<WorkspaceFile>): { blobs: Set<BlobId>; images: Set<ImageId> } {
  const held = { blobs: new Set<BlobId>(), images: new Set<ImageId>() };
  for (const file of files)
    for (const v of file.versions) if (v.bytes) (v.bytes.store === "image" ? held.images : held.blobs).add(v.bytes.id);
  return held;
}

/**
 * The bytes deleting the files `gone` frees: those no file of the `others`
 * holds. Blobs belong to workspaces alone; a picture may also be named by a
 * chat or an image session, which `elsewhere` lists, and none is freed while
 * `elsewhere` cannot say yet (null).
 */
export function freedBy(gone: Iterable<WorkspaceFile>, others: Iterable<WorkspaceFile>, elsewhere: Iterable<ImageId> | null): { blobs: BlobId[]; images: ImageId[] } {
  const own = bytesOf(gone);
  const kept = bytesOf(others);
  return {
    blobs: [...own.blobs].filter((id) => !kept.blobs.has(id)),
    images: elsewhere === null ? [] : unreferenced(own.images, new Set([...kept.images, ...elsewhere])),
  };
}

/** A text's lines as a reader counts them: a final newline ends the last line rather than starting another. */
export function linesOf(text: string): string[] {
  if (text === "") return [];
  const lines = text.split(/\r?\n/);
  if (lines[lines.length - 1] === "") lines.pop();
  return lines;
}
