import type { BlobId, ImageId } from "../store/imageStore.ts";

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

export const USER: Author = { by: "user" };

/** Bytes kept in the page's store rather than inline: a picture in the image store (#394), anything else as a blob. */
export type StoredBytes = { store: "blob"; id: BlobId; type: string; size: number } | { store: "image"; id: ImageId; type: string; size: number };

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
  /** Text derived from the bytes, computed once and kept with the version: a PDF's text, an audio file's transcript (#395). */
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

/** A text's lines as a reader counts them: a final newline ends the last line rather than starting another. */
export function linesOf(text: string): string[] {
  if (text === "") return [];
  const lines = text.split(/\r?\n/);
  if (lines[lines.length - 1] === "") lines.pop();
  return lines;
}

/** The text the model reads of a version: a text file's own, or the text derived from its bytes. */
export const readableText = (version: Version): string | undefined => version.text ?? version.derived;
