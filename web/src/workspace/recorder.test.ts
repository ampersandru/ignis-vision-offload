import { describe, expect, it } from "vitest";
import { MicBlocked } from "../speech/capture.ts";
import { createMemoryStore } from "../store/memory.ts";
import { latest } from "./files.ts";
import { uploadFiles } from "./panel.ts";
import { createRecorder, recordingName, wavFile } from "./recorder.ts";
import { createWorkspaces } from "./workspaces.ts";

// Recording into the workspace (#395): the microphone's PCM16, as the
// dictation's worklet gives it (16 kHz mono), kept until Stop and saved as a
// WAV file, a format the server's transcription route decodes. The capture is
// injected: no browser here.

/** A microphone whose chunks the test hands over, and whose release it sees. */
function mic() {
  let send: (pcm: Int16Array) => void = () => {};
  const state = { stopped: false, flushed: 0 };
  const capture = async (onChunk: (pcm: Int16Array) => void) => {
    send = onChunk;
    return {
      flush: async () => void state.flushed++,
      stop: () => void (state.stopped = true),
    };
  };
  return { capture, state, chunk: (pcm: Int16Array) => send(pcm) };
}

const at = new Date(2026, 9, 28, 14, 5, 9);

describe("a WAV file of the samples", () => {
  it("is a 16-bit mono RIFF file at 16 kHz, its samples little-endian after a 44-byte header", async () => {
    const file = wavFile([new Int16Array([1, -2]), new Int16Array([0x1234])], "memo.wav");
    expect([file.name, file.type, file.size]).toEqual(["memo.wav", "audio/wav", 50]);
    const bytes = new Uint8Array(await file.arrayBuffer());
    const view = new DataView(bytes.buffer);
    const ascii = (from: number) => String.fromCharCode(...bytes.slice(from, from + 4));
    expect([ascii(0), view.getUint32(4, true), ascii(8), ascii(12), ascii(36), view.getUint32(40, true)]).toEqual(["RIFF", 42, "WAVE", "fmt ", "data", 6]);
    // fmt: chunk size 16, PCM, one channel, 16,000 Hz, 32,000 bytes a second, 2 bytes a frame, 16 bits.
    expect([view.getUint32(16, true), view.getUint16(20, true), view.getUint16(22, true), view.getUint32(24, true), view.getUint32(28, true), view.getUint16(32, true), view.getUint16(34, true)]).toEqual([
      16, 1, 1, 16_000, 32_000, 2, 16,
    ]);
    expect([...bytes.slice(44)]).toEqual([0x01, 0x00, 0xfe, 0xff, 0x34, 0x12]);
  });

  it("is named after the moment the recording started", () => {
    expect(recordingName(at)).toBe("recording 2026-10-28 14.05.09.wav");
  });
});

describe("the recorder", () => {
  it("records from the microphone into a new audio file of the workspace, and lets the microphone go", async () => {
    const m = mic();
    const workspaces = createWorkspaces(createMemoryStore());
    let bound: string | undefined;
    const chat = workspaces.chat("chat-a", { bound: () => bound, name: () => "Standup", onBorn: (id) => (bound = id) });
    const recorder = createRecorder({ capture: m.capture, now: () => at, limit: async () => null });
    expect(recorder.getState()).toEqual({ phase: "idle", blocked: false });
    await recorder.start(async (file) => void (await uploadFiles(workspaces, chat, [file])));
    m.chunk(new Int16Array(1_600).fill(8_000));
    m.chunk(new Int16Array(1_600));
    expect(recorder.getState()).toMatchObject({ phase: "recording", seconds: 0.2 });
    await recorder.stop();
    expect(m.state).toEqual({ stopped: true, flushed: 1 });
    expect(recorder.getState()).toEqual({ phase: "idle", blocked: false });
    const [file] = await workspaces.files(bound!);
    expect([file.name, file.kind]).toEqual(["recording 2026-10-28 14.05.09.wav", "audio"]);
    expect(latest(file)).toMatchObject({ author: { by: "user" }, bytes: { store: "blob", type: "audio/wav", size: 44 + 3_200 * 2 } });
  });

  it("keeps nothing when cancelled", async () => {
    const m = mic();
    const saved: File[] = [];
    const recorder = createRecorder({ capture: m.capture, now: () => at, limit: async () => null });
    await recorder.start(async (file) => void saved.push(file));
    m.chunk(new Int16Array(160));
    recorder.cancel();
    expect(m.state.stopped).toBe(true);
    expect(recorder.getState()).toEqual({ phase: "idle", blocked: false });
    await recorder.stop();
    expect(saved).toEqual([]);
  });

  it("stops and saves at the longest clip the server transcribes", async () => {
    const m = mic();
    const saved: File[] = [];
    const recorder = createRecorder({ capture: m.capture, now: () => at, limit: async () => 0.2 });
    await recorder.start(async (file) => void saved.push(file));
    expect(recorder.getState()).toMatchObject({ phase: "recording", limit: 0.2 });
    m.chunk(new Int16Array(1_600));
    m.chunk(new Int16Array(1_600));
    m.chunk(new Int16Array(1_600));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(saved.map((f) => f.size)).toEqual([44 + 3_200 * 2]);
    expect(recorder.getState()).toEqual({
      phase: "idle",
      blocked: false,
      note: "Saved at 0:00, the longest clip the server transcribes (`speech.max_clip_seconds`).",
    });
  });

  it("says when the microphone is blocked, and when the save is refused", async () => {
    const blocked = createRecorder({
      capture: async () => Promise.reject(new MicBlocked("Permission denied")),
      limit: async () => null,
    });
    await blocked.start(async () => {});
    expect(blocked.getState()).toEqual({ phase: "idle", blocked: true });
    const m = mic();
    const refused = createRecorder({ capture: m.capture, now: () => at, limit: async () => null });
    await refused.start(async () => Promise.reject(new Error("QuotaExceededError")));
    m.chunk(new Int16Array(16));
    await refused.stop();
    expect(refused.getState()).toEqual({ phase: "idle", blocked: false, note: "Not saved in this browser: QuotaExceededError" });
  });
});
