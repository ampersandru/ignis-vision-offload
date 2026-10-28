import { readConfig } from "../server/routes.ts";
import { type Capture, MicBlocked, startCapture } from "../speech/capture.ts";
import { levelOf, TARGET_RATE } from "../speech/pcm.ts";
import { refusal } from "./files.ts";

// Recording from the microphone into the workspace (#395): the dictation's
// capture (speech/capture.ts), whose worklet gives 16 kHz mono PCM16, kept
// until Stop and saved as a WAV file. A browser's own recorder writes WebM or
// Ogg holding Opus, which the server's transcription route does not decode;
// WAV it does, so the model can read the recording as its transcript. A
// recording stops and saves itself at the longest clip the route transcribes,
// `speech.max_clip_seconds`, read from the server when it starts.

export type RecorderState =
  /** `blocked`: the browser refused the microphone, the cause pressing Record again can clear. `note`: how the last recording ended, when that needs saying. */
  | { phase: "idle"; blocked: boolean; note?: string }
  | { phase: "starting" }
  /** `seconds` recorded, the last chunk's `level` (0 to 1), and the `limit` it stops at, null when the server did not say. */
  | { phase: "recording"; seconds: number; level: number; limit: number | null }
  | { phase: "saving" };

export type Recorder = {
  /** Starts recording; `save` takes the file once it ends. Called inside the press, which the microphone needs. */
  start(save: (file: File) => Promise<void>): Promise<void>;
  /** Ends the recording and saves it; before the microphone started, there is nothing to save, and it cancels. */
  stop(): Promise<void>;
  /** Ends the recording and keeps nothing. */
  cancel(): void;
  getState(): RecorderState;
  subscribe(listener: () => void): () => void;
};

export type RecorderDeps = {
  capture?: (onChunk: (pcm: Int16Array) => void) => Promise<Capture>;
  now?: () => Date;
  /** The longest clip the server transcribes, in seconds; null when it cannot be read. */
  limit?: () => Promise<number | null>;
};

const pad = (n: number) => String(n).padStart(2, "0");

/** A recording's file name, from the moment it started: `recording 2026-10-28 14.05.09.wav`. */
export const recordingName = (d: Date) =>
  `recording ${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}.${pad(d.getMinutes())}.${pad(d.getSeconds())}.wav`;

/** Seconds as a recording's clock shows them: `m:ss`. */
export const clock = (seconds: number) => `${Math.floor(seconds / 60)}:${pad(Math.floor(seconds % 60))}`;

/** The samples, 16 kHz mono PCM16, as a WAV file: the 44-byte RIFF header, then the samples little-endian. */
export function wavFile(chunks: Int16Array[], name: string, rate = TARGET_RATE): File {
  const samples = chunks.reduce((n, c) => n + c.length, 0);
  const bytes = new Uint8Array(44 + samples * 2);
  const view = new DataView(bytes.buffer);
  const ascii = (at: number, text: string) => [...text].forEach((c, i) => view.setUint8(at + i, c.charCodeAt(0)));
  ascii(0, "RIFF");
  view.setUint32(4, 36 + samples * 2, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, samples * 2, true);
  let at = 44;
  for (const chunk of chunks)
    for (const sample of chunk) {
      view.setInt16(at, sample, true);
      at += 2;
    }
  return new File([bytes], name, { type: "audio/wav" });
}

/** `speech.max_clip_seconds`, as the server's configuration says it. */
async function clipLimit(): Promise<number | null> {
  const answer = await readConfig();
  const value = answer.ok ? answer.value.speech?.max_clip_seconds : undefined;
  return typeof value === "number" && value > 0 ? value : null;
}

/** The samples up to `limit` seconds, all of them for none. */
function upTo(chunks: Int16Array[], limit: number | null): Int16Array[] {
  if (limit === null) return chunks;
  let room = Math.floor(limit * TARGET_RATE);
  const kept: Int16Array[] = [];
  for (const chunk of chunks) {
    if (room <= 0) break;
    kept.push(chunk.length <= room ? chunk : chunk.subarray(0, room));
    room -= chunk.length;
  }
  return kept;
}

type Take = {
  name: string;
  save: (file: File) => Promise<void>;
  chunks: Int16Array[];
  samples: number;
  limit: number | null;
  capture: Capture | null;
  /** Saving: the chunks the flush hands over are still kept, nothing else changes. */
  stopping: boolean;
  /** Saved or cancelled: nothing more is kept. */
  ended: boolean;
};

export function createRecorder(deps: RecorderDeps = {}): Recorder {
  const capture = deps.capture ?? ((onChunk) => startCapture(onChunk));
  const now = deps.now ?? (() => new Date());
  const limitOf = deps.limit ?? clipLimit;
  let state: RecorderState = { phase: "idle", blocked: false };
  const listeners = new Set<() => void>();
  let take: Take | null = null;

  const set = (next: RecorderState) => {
    state = next;
    for (const listener of listeners) listener();
  };

  function cancel() {
    const current = take;
    if (!current) return;
    take = null;
    current.stopping = true;
    current.ended = true;
    current.capture?.stop();
    set({ phase: "idle", blocked: false });
  }

  async function finish(current: Take, note?: string) {
    if (current.stopping) return;
    current.stopping = true;
    set({ phase: "saving" });
    if (current.capture) {
      await current.capture.flush();
      current.capture.stop();
    }
    current.ended = true;
    if (take === current) take = null;
    try {
      await current.save(wavFile(upTo(current.chunks, current.limit), current.name));
      set({ phase: "idle", blocked: false, ...(note ? { note } : {}) });
    } catch (err) {
      set({ phase: "idle", blocked: false, note: refusal(err) });
    }
  }

  return {
    async start(save) {
      if (state.phase !== "idle") return;
      const current: Take = { name: recordingName(now()), save, chunks: [], samples: 0, limit: null, capture: null, stopping: false, ended: false };
      take = current;
      set({ phase: "starting" });
      // Both asked before the first wait: the microphone's audio context must be made inside the press.
      const limit = limitOf().catch(() => null);
      const starting = capture((pcm) => {
        if (current.ended) return;
        current.chunks.push(pcm);
        current.samples += pcm.length;
        if (current.stopping) return;
        const seconds = current.samples / TARGET_RATE;
        set({ phase: "recording", seconds, level: levelOf(pcm), limit: current.limit });
        if (current.limit !== null && seconds >= current.limit) {
          void finish(current, `Saved at ${clock(current.limit)}, the longest clip the server transcribes (\`speech.max_clip_seconds\`).`);
        }
      });
      let started: Capture;
      try {
        started = await starting;
      } catch (err) {
        if (take === current) take = null;
        if (current.ended) return;
        current.ended = true;
        set(
          err instanceof MicBlocked
            ? { phase: "idle", blocked: true }
            : { phase: "idle", blocked: false, note: `The microphone could not start: ${err instanceof Error ? err.message : String(err)}` },
        );
        return;
      }
      current.capture = started;
      current.limit = await limit;
      // Cancelled while the microphone started.
      if (current.ended) return started.stop();
      if (!current.stopping) set({ phase: "recording", seconds: current.samples / TARGET_RATE, level: 0, limit: current.limit });
    },
    async stop() {
      if (!take || take.ended) return;
      if (!take.capture) return cancel();
      await finish(take);
    },
    cancel,
    getState: () => state,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
