import { CHUNK_SAMPLES, PROCESSOR, TARGET_RATE, WORKLET_SOURCE } from "./pcm.ts";

// The mic, for one dictation (spec speech/04, "Capture"): asked for at the
// press, run through the worklet in pcm.ts at the context's own rate, and let
// go on every way out — Done, Esc, a closed connection, the page leaving — so
// the browser's recording indicator never outlives a dictation (Q13). The
// browser's objects are injectable, so the cleanup is testable without one.

/** The browser, or the person, refused the mic: the cause pressing it again can clear. */
export class MicBlocked extends Error {}

export type Capture = {
  /** Hands over the part chunk still in the worklet, then resolves. */
  flush: () => Promise<void>;
  /** Lets the mic go. */
  stop: () => void;
};

export type CaptureDeps = {
  getUserMedia?: (constraints: MediaStreamConstraints) => Promise<MediaStream>;
  createContext?: () => AudioContext;
  createNode?: (context: AudioContext, name: string, options: AudioWorkletNodeOptions) => AudioWorkletNode;
  /** A URL the worklet's source loads from; revoked once loaded. */
  moduleUrl?: () => string;
};

export async function startCapture(onChunk: (pcm: Int16Array) => void, deps: CaptureDeps = {}): Promise<Capture> {
  const getUserMedia = deps.getUserMedia ?? ((constraints) => navigator.mediaDevices.getUserMedia(constraints));
  const createContext = deps.createContext ?? (() => new AudioContext());
  const createNode = deps.createNode ?? ((context, name, options) => new AudioWorkletNode(context, name, options));
  const moduleUrl = deps.moduleUrl ?? (() => URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: "text/javascript" })));

  let stream: MediaStream | null = null;
  let context: AudioContext | null = null;
  let stopped = false;
  let flushed: (() => void) | null = null;
  const stop = () => {
    if (stopped) return;
    stopped = true;
    for (const track of stream?.getTracks() ?? []) track.stop();
    void context?.close();
    flushed?.();
  };
  try {
    // Made before the first await, inside the press's user activation: a context made later can stay suspended.
    context = createContext();
    try {
      stream = await getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    } catch (err) {
      if (err instanceof Error && (err.name === "NotAllowedError" || err.name === "SecurityError")) throw new MicBlocked(err.message);
      throw err;
    }
    const url = moduleUrl();
    try {
      await context.audioWorklet.addModule(url);
    } finally {
      if (!deps.moduleUrl) URL.revokeObjectURL(url);
    }
    const source = context.createMediaStreamSource(stream);
    // One output, to the destination: a node nothing pulls is not processed. It writes silence.
    const node = createNode(context, PROCESSOR, {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      processorOptions: { targetRate: TARGET_RATE, chunkSamples: CHUNK_SAMPLES },
    });
    node.port.onmessage = (event: MessageEvent<{ pcm?: Int16Array; flushed?: true }>) => {
      if (event.data.pcm && !stopped) onChunk(event.data.pcm);
      if (event.data.flushed) flushed?.();
    };
    source.connect(node);
    node.connect(context.destination);
    void context.resume();
    return {
      flush: () =>
        stopped
          ? Promise.resolve()
          : new Promise<void>((resolve) => {
              flushed = () => {
                flushed = null;
                resolve();
              };
              node.port.postMessage("flush");
            }),
      stop,
    };
  } catch (err) {
    stop();
    throw err;
  }
}
