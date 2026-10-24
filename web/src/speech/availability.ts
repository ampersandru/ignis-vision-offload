import type { SpeechStatus } from "../api/model.ts";

// Why the mic cannot start (spec speech/04, the "Unavailable" table): one
// sentence for the hint line, never a tooltip. Code spans are in backticks;
// the hint line sets them in the mono face.
//
// The page's own cause comes first: outside https and localhost the browser
// gives no page a microphone, whatever the server says (Q18). Then the
// server's, from GET /v1/models' `speech` object (Q55), in the spec's words
// where the table has a row and in the server's where it has none. A mic the
// browser blocked is known only once pressed, and is the one cause pressing it
// again can clear.

export const SECURE_CONTEXT =
  "The microphone needs https or localhost. Open the Playground through an https proxy, or start the server with `--server-expose cloudflare-quick`.";
export const TURNED_OFF = "Speech is turned off on this server (`speech.model = off`).";
export const NO_CPU_PATH = "Speech is unavailable on this machine: the CPU lacks AVX2 and the card has no free room to lend.";
export const MIC_BLOCKED = "Microphone access is blocked for this site. Allow it in the browser's site settings, then press the mic again.";

/** The download command the server's reason names, else one built from the id it names. */
const noModel = (command: string) => `No speech model on this server. Download one with \`${command}\`.`;

/** The server's reason, in the spec's sentence when the table has one. */
function serverSentence(reason: string): string {
  if (reason.includes("turned off")) return TURNED_OFF;
  if (reason.includes("no speech model files")) {
    const command = /`(ignis-server model download [^`]+)`/.exec(reason)?.[1];
    const id = /`([^`]+)`/.exec(reason)?.[1];
    return noModel(command ?? `ignis-server model download ${id ?? "<id>"}`);
  }
  if (reason.includes("AVX2")) return NO_CPU_PATH;
  return `Speech is unavailable on this server: ${reason.replace(/\.$/, "")}.`;
}

/**
 * Why the mic cannot start now, or `null` when it can. `speech` absent is a
 * server that has not said yet: the press finds out.
 */
export function unavailable({ secure, speech, blocked }: { secure: boolean; speech?: SpeechStatus; blocked: boolean }): { sentence: string; blocked: boolean } | null {
  if (!secure) return { sentence: SECURE_CONTEXT, blocked: false };
  if (speech && !speech.available) return { sentence: serverSentence(speech.reason ?? "no reason given"), blocked: false };
  if (blocked) return { sentence: MIC_BLOCKED, blocked: true };
  return null;
}
