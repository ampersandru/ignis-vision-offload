import { authHeaders, keyRequired } from "./auth.ts";
import { apiErrorMessage } from "./errors.ts";

// The server's transcription route, `POST /v1/audio/transcriptions`
// (docs/user/api/transcriptions.md), for an audio file of a workspace (#170):
// the file whole, as multipart `file`, and its transcript back as JSON. The
// route decodes WAV, FLAC, MP3 and AAC (ADTS or MP4/M4A), and Ogg or Matroska
// holding FLAC, not Opus; it clips at `speech.max_clip_seconds` and does not
// diarize, so a transcript has no speakers. It serves through a switch. The
// page's live speech goes through the Realtime socket instead (speech/realtime.ts).

export const TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions";

/** `audio`'s transcript, sent under `name`. Rejects with the server's reason, or with why ignis could not be reached. */
export async function transcribe(audio: Blob, name: string, signal?: AbortSignal, fetcher: typeof fetch = fetch): Promise<string> {
  const form = new FormData();
  form.append("file", audio, name);
  form.append("response_format", "json");
  let res: Response;
  try {
    res = await fetcher(TRANSCRIPTIONS_PATH, { method: "POST", headers: authHeaders(), body: form, signal });
  } catch (err) {
    if (signal?.aborted) throw err;
    throw new Error(`Could not reach ignis: ${String(err)}`);
  }
  if (res.status === 401) keyRequired();
  const body = await res.text();
  if (!res.ok) throw new Error(apiErrorMessage(res.status, body));
  let text: unknown;
  try {
    text = (JSON.parse(body) as { text?: unknown }).text;
  } catch {
    // Not JSON: said below.
  }
  if (typeof text !== "string") throw new Error("the server's answer holds no transcript");
  return text;
}
