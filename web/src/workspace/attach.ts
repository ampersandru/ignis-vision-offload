import type { Attachment } from "../tools/local/attachments.ts";
import { kindOf, refusal, uniqueName, USER } from "./files.ts";
import type { ChatBinding, Incoming, Workspaces } from "./workspaces.ts";

// Files the user attaches in the prompt box (spec #170). They wait in the box
// until the prompt is sent, read there so that a file that cannot be read is
// said at once; sending adds them to the chat's workspace, born if the chat
// has none, and the prompt names them. A PDF keeps its bytes and the text
// pdf.js reads from it, computed once. Audio keeps its bytes alone: its
// transcript is the server's, asked at the model's first read (#395). An
// older chat's attachments, kept with the chat as text, take the same way
// when the chat is resumed.

/** A file waiting in the prompt box: what the workspace will keep, and its size. */
export type AttachedFile = Incoming & { size: number };

/** What the prompt box shows of a file it will send: a text's characters (a PDF's text), audio's bytes. */
export type AttachedChip = { name: string; kind: "text" | "pdf" | "audio"; characters: number; size: number };

export const attachedChip = (file: AttachedFile): AttachedChip => ({
  name: file.name,
  kind: file.kind === "pdf" || file.kind === "audio" ? file.kind : "text",
  characters: (file.text ?? file.derived ?? "").length,
  size: file.size,
});

/** A text, PDF or audio file for the prompt, under a name the box does not hold yet, or why it cannot be one. */
export async function readAttached(
  file: File,
  taken: string[],
  readPdf: (data: ArrayBuffer) => Promise<string> = pdfText,
): Promise<{ ok: true; file: AttachedFile } | { ok: false; error: string }> {
  const kind = kindOf(file.name, file.type);
  const name = uniqueName(file.name, taken);
  try {
    if (kind === "pdf") {
      const blob = new Blob([await file.arrayBuffer()], { type: "application/pdf" });
      return { ok: true, file: { name, kind, size: file.size, blob, derived: await readPdf(await blob.arrayBuffer()) } };
    }
    if (kind === "audio") return { ok: true, file: { name, kind, size: file.size, blob: new Blob([await file.arrayBuffer()], { type: file.type }) } };
    const text = kind === "text" || kind === "other" ? await file.text() : null;
    if (text === null || text.includes("\u0000")) return { ok: false, error: `${file.name} is not a text, PDF or audio file.` };
    return { ok: true, file: { name, kind: "text", size: file.size, text } };
  } catch (err) {
    return { ok: false, error: `${file.name} could not be read: ${err instanceof Error ? err.message : String(err)}` };
  }
}

/**
 * A file the workspace panel takes in, whatever its kind: text inline, a PDF
 * with its text (none when pdf.js cannot read it), anything else as its bytes.
 * Text a file turns out not to be (a NUL in it) is kept as bytes too.
 */
export async function readUpload(file: File, readPdf: (data: ArrayBuffer) => Promise<string> = pdfText): Promise<Incoming> {
  const kind = kindOf(file.name, file.type);
  const blob = () => new Blob([file], { type: file.type });
  if (kind === "pdf") {
    const bytes = new Blob([await file.arrayBuffer()], { type: "application/pdf" });
    const derived = await readPdf(await bytes.arrayBuffer()).catch(() => undefined);
    return { name: file.name, kind, blob: bytes, ...(derived !== undefined ? { derived } : {}) };
  }
  // A name that says nothing else is tried as text, whatever the browser's type guessed (it calls a `.ts` file a video).
  if (kind === "text" || kindOf(file.name) === "text") {
    const text = await file.text();
    if (!text.includes("\u0000")) return { name: file.name, kind: "text", text };
    return { name: file.name, kind: "other", blob: blob() };
  }
  return { name: file.name, kind, blob: blob() };
}

/** An older chat's attachment as a workspace file: its text, or a PDF's text with no bytes, which that chat never kept. */
export const fromLegacy = (a: Attachment): AttachedFile =>
  a.kind === "pdf" ? { name: a.name, kind: "pdf", size: a.size, derived: a.text } : { name: a.name, kind: "text", size: a.size, text: a.text };

/** Adds `files` to the chat's workspace, born now if it has none, as the user's; the names they were given there. */
export async function commitAttached(workspaces: Workspaces, chat: ChatBinding, files: Incoming[]): Promise<string[]> {
  if (files.length === 0) return [];
  const written = await workspaces.add(await chat.workspace(), files, USER);
  return written.map((w) => w.name);
}

/** What a turn takes into the chat's workspace before it is sent: its files, and its pictures apart. */
export type TurnAdds = { files: AttachedFile[]; pictures: AttachedFile[]; clearsLegacy: boolean };

/**
 * What a turn takes into the chat's workspace: the box's files and pictures,
 * and an older chat's attachments ahead of them, which then leave the chat.
 * Nothing while the Workspace switch is off, when the model has no tool to
 * read them: an older chat's attachments stay with it until the first turn
 * with the switch on, whose prompt names them.
 */
export function turnAdds(workspaceOn: boolean, legacy: Attachment[], box: { files: AttachedFile[]; pictures: AttachedFile[] }): TurnAdds {
  if (!workspaceOn) return { files: [], pictures: [], clearsLegacy: false };
  return { files: [...legacy.map(fromLegacy), ...box.files], pictures: box.pictures, clearsLegacy: legacy.length > 0 };
}

/** What taking a turn's files did: the names its prompt gives them, and what the page says of a refusal; or that the turn is not sent. */
export type Taken = { sent: true; named: string[]; said?: string } | { sent: false; said: string };

/** What the page says of files the store refused for the workspace: they were not saved there, what became of them, and why. */
const notSaved = (names: string[], then: string, err: unknown) =>
  `${names.join(", ")} ${names.length === 1 ? "was" : "were"} not saved in the workspace: ${then} ${refusal(err)}`;

/**
 * Takes a turn's files into the chat's workspace, before the turn is sent.
 * A refusal never costs the user their content, and the page says it was not
 * saved in the workspace: files the store refuses, all of them, leave the
 * turn unsent, its prompt back in the box; a picture it refuses still reaches
 * the model inline, in this prompt only, as make_image's picture stays in its
 * reply.
 */
export async function takeTurnFiles(workspaces: Workspaces, chat: ChatBinding, adds: TurnAdds): Promise<Taken> {
  let named: string[] = [];
  try {
    named = await commitAttached(workspaces, chat, adds.files);
  } catch (err) {
    return { sent: false, said: notSaved(adds.files.map((f) => f.name), "nothing was sent, and nothing is lost.", err) };
  }
  try {
    return { sent: true, named: [...named, ...(await commitAttached(workspaces, chat, adds.pictures))] };
  } catch (err) {
    const then = `the model sees ${adds.pictures.length === 1 ? "it" : "them"} in this prompt only.`;
    return { sent: true, named, said: notSaved(adds.pictures.map((f) => f.name), then, err) };
  }
}

/** A prompt as it goes on the wire: its text, then the files it attached to the workspace. */
export function withAttached(content: string, names: string[] | undefined): string {
  if (!names?.length) return content;
  const line = `[Attached to the workspace: ${names.join(", ")}]`;
  return content === "" ? line : `${content}\n\n${line}`;
}

/** A PDF's text, page by page (pdf.js, loaded on first use). */
export async function pdfText(data: ArrayBuffer): Promise<string> {
  const pdfjs = await import("pdfjs-dist");
  const worker = await import("pdfjs-dist/build/pdf.worker.min.mjs?url");
  pdfjs.GlobalWorkerOptions.workerSrc = worker.default;
  const doc = await pdfjs.getDocument({ data }).promise;
  const pages: string[] = [];
  for (let i = 1; i <= doc.numPages; i++) {
    const content = await (await doc.getPage(i)).getTextContent();
    pages.push(content.items.map((item) => ("str" in item ? item.str : "")).join(" "));
  }
  return pages.map((text, i) => `[Page ${i + 1}]\n${text}`).join("\n\n");
}
