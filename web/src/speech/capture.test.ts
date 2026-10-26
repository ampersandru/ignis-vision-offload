import { describe, expect, it } from "vitest";
import { CallNotShared, type CaptureDeps, captureStream, MicBlocked, NoSharedAudio, pickSharedAudio, startCapture } from "./capture.ts";
import { CHUNK_SAMPLES, PROCESSOR, TARGET_RATE } from "./pcm.ts";

// The mic and the worklet behind it (spec speech/04, "Capture"), with fakes
// for the browser's media and audio objects: what is asked for, what comes
// out, and that the mic is let go on every way out (Q13).

function fakes(options: { refuse?: Error; failModule?: boolean; failContext?: boolean } = {}) {
  const log: string[] = [];
  let port: { onmessage: ((event: { data: unknown }) => void) | null; postMessage: (message: unknown) => void } | undefined;
  let nodeOptions: unknown;
  const track = { stop: () => log.push("track stopped") };
  const deps: CaptureDeps = {
    getUserMedia: async (constraints) => {
      log.push(`getUserMedia ${JSON.stringify(constraints)}`);
      if (options.refuse) throw options.refuse;
      return { getTracks: () => [track] } as unknown as MediaStream;
    },
    createContext: () => {
      if (options.failContext) throw new Error("no audio context");
      return {
        audioWorklet: {
          addModule: async (url: string) => {
            log.push(`addModule ${url}`);
            if (options.failModule) throw new Error("no worklet");
          },
        },
        createMediaStreamSource: () => ({ connect: () => log.push("source connected"), disconnect: () => log.push("source disconnected") }),
        destination: {},
        resume: async () => {},
        close: async () => log.push("context closed"),
      } as unknown as AudioContext;
    },
    createNode: (_context, name, init) => {
      log.push(`node ${name}`);
      nodeOptions = init;
      port = {
        onmessage: null,
        postMessage: (message) => {
          log.push(`posted ${String(message)}`);
          if (message === "flush") {
            port!.onmessage?.({ data: { pcm: new Int16Array([7]) } });
            port!.onmessage?.({ data: { flushed: true } });
          }
        },
      };
      return { port, connect: () => {}, disconnect: () => log.push("node disconnected") } as unknown as AudioWorkletNode;
    },
    moduleUrl: () => "blob:worklet",
  };
  return { deps, log, port: () => port!, nodeOptions: () => nodeOptions };
}

describe("startCapture", () => {
  it("asks for one mic channel, loads the worklet, and hands each chunk over", async () => {
    const f = fakes();
    const chunks: number[][] = [];
    await startCapture((pcm) => chunks.push([...pcm]), f.deps);
    expect(f.log[0]).toBe('getUserMedia {"audio":{"channelCount":1,"echoCancellation":true,"noiseSuppression":true,"autoGainControl":true}}');
    expect(f.log).toContain("addModule blob:worklet");
    expect(f.log).toContain(`node ${PROCESSOR}`);
    expect(f.nodeOptions()).toMatchObject({ processorOptions: { targetRate: TARGET_RATE, chunkSamples: CHUNK_SAMPLES } });
    f.port().onmessage?.({ data: { pcm: new Int16Array([1, 2]) } });
    expect(chunks).toEqual([[1, 2]]);
  });

  it("flushes the last part chunk before it resolves, and lets the mic go on stop", async () => {
    const f = fakes();
    const chunks: number[][] = [];
    const capture = await startCapture((pcm) => chunks.push([...pcm]), f.deps);
    await capture.flush();
    expect(chunks).toEqual([[7]]);
    capture.stop();
    capture.stop();
    expect(f.log.filter((line) => line === "track stopped")).toHaveLength(1);
    expect(f.log).toContain("context closed");
    // Stopped, there is nothing left to flush.
    await capture.flush();
  });

  it("names a mic the browser blocked, and lets a mic go when the worklet fails to load", async () => {
    const denied = fakes({ refuse: Object.assign(new Error("Permission denied"), { name: "NotAllowedError" }) });
    await expect(startCapture(() => {}, denied.deps)).rejects.toBeInstanceOf(MicBlocked);
    expect(denied.log).toContain("context closed");
    const broken = fakes({ failModule: true });
    await expect(startCapture(() => {}, broken.deps)).rejects.toThrow("no worklet");
    expect(broken.log).toContain("track stopped");
    expect(broken.log).toContain("context closed");
    const contextless = fakes({ failContext: true });
    await expect(startCapture(() => {}, contextless.deps)).rejects.toThrow("no audio context");
    // The context comes first, inside the press's activation: no mic is asked for without one.
    expect(contextless.log.some((line) => line.startsWith("getUserMedia"))).toBe(false);
  });
});

// A call's other side (spec speech/07, "Sources", Q93): one getDisplayMedia pick, its audio run through the same
// worklet. The pick outlives a capture of it: a Pause lets the worklet go and keeps what was shared.
describe("a call's shared audio", () => {
  /** A picked surface: `audio` audio tracks beside its video, each stop logged. */
  function picked(log: string[], audio = 1) {
    const ended: (() => void)[] = [];
    const track = (kind: string) => ({
      kind,
      stop: () => log.push(`${kind} stopped`),
      addEventListener: (_type: string, listener: () => void) => ended.push(listener),
    });
    const tracks = [track("video"), ...Array.from({ length: audio }, () => track("audio"))];
    const stream = { getTracks: () => tracks, getAudioTracks: () => tracks.filter((t) => t.kind === "audio") } as unknown as MediaStream;
    return { stream, end: () => ended.forEach((listener) => listener()) };
  }

  it("asks for one pick with its audio, a tab's, the screen's system audio or a window's, and says when the sharing ends", async () => {
    const log: string[] = [];
    const surface = picked(log);
    let asked: unknown;
    let ended = 0;
    const stream = await pickSharedAudio(() => ended++, {
      getDisplayMedia: async (options) => {
        asked = options;
        return surface.stream;
      },
    });
    expect(stream).toBe(surface.stream);
    expect(asked).toMatchObject({ video: true, audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }, systemAudio: "include", windowAudio: "window" });
    surface.end();
    expect(ended).toBe(1);
  });

  it("refuses a pick that shares no audio, and lets it go; names a pick the person cancelled", async () => {
    const log: string[] = [];
    const silent = picked(log, 0);
    await expect(pickSharedAudio(() => {}, { getDisplayMedia: async () => silent.stream })).rejects.toBeInstanceOf(NoSharedAudio);
    expect(log).toEqual(["video stopped"]);
    const cancel = Object.assign(new Error("Permission denied by user"), { name: "NotAllowedError" });
    await expect(pickSharedAudio(() => {}, { getDisplayMedia: async () => Promise.reject(cancel) })).rejects.toBeInstanceOf(CallNotShared);
  });

  it("runs a stream through the worklet, and its stop leaves the stream to whoever picked it", async () => {
    const f = fakes();
    const log: string[] = [];
    const surface = picked(log);
    const chunks: number[][] = [];
    const capture = await captureStream(surface.stream, (pcm) => chunks.push([...pcm]), f.deps);
    f.port().onmessage?.({ data: { pcm: new Int16Array([5]) } });
    capture.stop();
    expect(chunks).toEqual([[5]]);
    expect(f.log).toContain("context closed");
    expect(f.log.some((line) => line.startsWith("getUserMedia"))).toBe(false);
    expect(log).toEqual([]);
  });
});
