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
/** The Transcribe view holds the page's speech (spec speech/07, Q85). */
export const LISTENING_ELSEWHERE = "Listening in Transcribe.";

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
 * server that has not said yet: the press finds out. A listening in the
 * Transcribe view comes first: it holds the page's speech whatever else holds.
 */
export function unavailable({
  secure,
  speech,
  blocked,
  listening = false,
}: {
  secure: boolean;
  speech?: SpeechStatus;
  blocked: boolean;
  listening?: boolean;
}): { sentence: string; blocked: boolean } | null {
  if (listening) return { sentence: LISTENING_ELSEWHERE, blocked: false };
  if (!secure) return { sentence: SECURE_CONTEXT, blocked: false };
  if (speech && !speech.available) return { sentence: serverSentence(speech.reason ?? "no reason given"), blocked: false };
  if (blocked) return { sentence: MIC_BLOCKED, blocked: true };
  return null;
}

/**
 * Why the Transcribe view's Speakers toggle is struck (spec speech/07, Q112), or `null` when the diarizer can
 * serve: GET /v1/models' `speech.diarizer` (spec speech/08). A server that says nothing of one is older than the
 * diarizer, and would refuse `session.ignis.diarization` as a field it does not know.
 */
export function speakersUnavailable(speech: SpeechStatus): string | null {
  const { diarizer } = speech;
  if (!diarizer) return "This server has no diarizer: Speakers needs a newer ignis.";
  if (diarizer.available) return null;
  const reason = diarizer.reason ?? "no reason given";
  if (reason.includes("turned off")) return "Speakers are turned off on this server (`diarizer.model = off`).";
  const command = /`(ignis-server model download [^`]+)`/.exec(reason)?.[1];
  if (reason.includes("no diarizer files") && command) return `No diarizer on this server. Download one with \`${command}\`.`;
  return `Speakers are unavailable on this server: ${reason.replace(/\.$/, "")}.`;
}
