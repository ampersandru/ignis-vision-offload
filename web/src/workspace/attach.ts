import type { Attachment } from "../tools/local/attachments.ts";
import { kindOf, uniqueName, USER } from "./files.ts";
import type { ChatBinding, Incoming, Workspaces } from "./workspaces.ts";

// Files the user attaches in the prompt box (spec #170). They wait in the box
// until the prompt is sent, read there so that a file that cannot be read is
// said at once; sending adds them to the chat's workspace, born if the chat
// has none, and the prompt names them. A PDF keeps its bytes and the text
// pdf.js reads from it, computed once. An older chat's attachments, kept with
// the chat as text, take the same way when the chat is resumed.

/** A file waiting in the prompt box: what the workspace will keep, and its size. */
export type AttachedFile = Incoming & { size: number };

/** What the prompt box shows of a file it will send. */
export type AttachedChip = { name: string; kind: "text" | "pdf"; characters: number };

export const attachedChip = (file: Incoming): AttachedChip => ({
  name: file.name,
  kind: file.kind === "pdf" ? "pdf" : "text",
  characters: (file.text ?? file.derived ?? "").length,
});

/** A text or PDF file for the prompt, under a name the box does not hold yet, or why it cannot be one. */
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
    const text = kind === "text" || kind === "other" ? await file.text() : null;
    if (text === null || text.includes("\u0000")) return { ok: false, error: `${file.name} is not a text or PDF file.` };
    return { ok: true, file: { name, kind: "text", size: file.size, text } };
  } catch (err) {
    return { ok: false, error: `${file.name} could not be read: ${err instanceof Error ? err.message : String(err)}` };
  }
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

/** A prompt as it goes on the wire: its text, then the files it attached to the workspace. */
export function withAttached(content: string, names: string[] | undefined): string {
  if (!names?.length) return content;
  const line = `[Attached to the workspace: ${names.join(", ")}]`;
  return content === "" ? line : `${content}\n\n${line}`;
}

/** A PDF's text, page by page (pdf.js, loaded on first use). */
async function pdfText(data: ArrayBuffer): Promise<string> {
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
