import { describe, expect, it } from "vitest";
import { CHUNK_SAMPLES, levelOf, pcmBase64, PROCESSOR, TARGET_RATE, WORKLET_SOURCE } from "./pcm.ts";

// The mic's audio on its way to the server (spec speech/04): the AudioWorklet
// downsamples it to 16 kHz mono PCM16 and hands it over in 100 ms chunks. The
// worklet's source is evaluated here as the browser would, with the three
// globals an AudioWorkletGlobalScope gives it.

type Posted = { pcm?: Int16Array; flushed?: true };
type Processor = { port: { postMessage: (message: Posted) => void; onmessage: ((event: { data: unknown }) => void) | null }; process: (inputs: Float32Array[][]) => boolean };

/** The processor the source registers, built for a context running at `rate`, and what it posts. */
function processorAt(rate: number) {
  const posted: Posted[] = [];
  let registered: (new (options: unknown) => Processor) | undefined;
  class AudioWorkletProcessor {
    port = { postMessage: (message: Posted) => posted.push(message), onmessage: null };
  }
  new Function("AudioWorkletProcessor", "registerProcessor", "sampleRate", WORKLET_SOURCE)(
    AudioWorkletProcessor,
    (name: string, processor: new (options: unknown) => Processor) => {
      expect(name).toBe(PROCESSOR);
      registered = processor;
    },
    rate,
  );
  const processor = new registered!({ processorOptions: { targetRate: TARGET_RATE, chunkSamples: CHUNK_SAMPLES } });
  return { processor, posted };
}

/** `seconds` of a sine at `hz`, amplitude `amp`, fed in 128-frame render quanta on `channels` channels. */
function feed(processor: Processor, rate: number, seconds: number, hz = 440, amp = 0.5, channels = 1) {
  const frames = Math.round(rate * seconds);
  for (let at = 0; at < frames; at += 128) {
    const quantum = new Float32Array(Math.min(128, frames - at)).map((_, i) => amp * Math.sin((2 * Math.PI * hz * (at + i)) / rate));
    processor.process([Array.from({ length: channels }, () => quantum)]);
  }
}

const samples = (posted: Posted[]) => posted.reduce((n, p) => n + (p.pcm?.length ?? 0), 0);

describe("the capture worklet", () => {
  it("posts 100 ms chunks of 16 kHz PCM16 from a 48 kHz mic", () => {
    const { processor, posted } = processorAt(48000);
    feed(processor, 48000, 1);
    expect(posted.map((p) => p.pcm?.length)).toEqual(Array(10).fill(1600));
    // The tone keeps its loudness: a 0.5 sine peaks near 16384.
    const peak = Math.max(...posted.flatMap((p) => [...p.pcm!].map(Math.abs)));
    expect(peak).toBeGreaterThan(15500);
    expect(peak).toBeLessThan(16500);
  });

  it("keeps 16 kHz from a 44.1 kHz mic, and from one already at 16 kHz", () => {
    const at441 = processorAt(44100);
    feed(at441.processor, 44100, 2);
    at441.processor.port.onmessage!({ data: "flush" });
    expect(Math.abs(samples(at441.posted) - 32000)).toBeLessThanOrEqual(1);
    const at16 = processorAt(16000);
    feed(at16.processor, 16000, 0.5);
    expect(samples(at16.posted)).toBe(8000);
  });

  it("mixes several channels down to mono", () => {
    const { processor, posted } = processorAt(48000);
    feed(processor, 48000, 0.1, 440, 0.5, 2);
    const peak = Math.max(...[...posted[0].pcm!].map(Math.abs));
    expect(peak).toBeGreaterThan(15500);
    expect(peak).toBeLessThan(16500);
  });

  it("hands over the part chunk on a flush, then says it has", () => {
    const { processor, posted } = processorAt(48000);
    feed(processor, 48000, 0.15);
    processor.port.onmessage!({ data: "flush" });
    expect(posted.map((p) => p.pcm?.length ?? "flushed")).toEqual([1600, 800, "flushed"]);
    // Nothing left: a second flush only says so.
    processor.port.onmessage!({ data: "flush" });
    expect(posted.at(-1)).toEqual({ flushed: true });
    expect(posted).toHaveLength(4);
  });
});

describe("pcmBase64", () => {
  it("is the little-endian bytes of the samples, as input_audio_buffer.append carries them", () => {
    // 1 = 01 00, -2 = fe ff, 256 = 00 01: bytes 01 00 fe ff 00 01.
    expect(pcmBase64(new Int16Array([1, -2, 256]))).toBe("AQD+/wAB");
    expect(pcmBase64(new Int16Array(0))).toBe("");
  });
});

describe("levelOf", () => {
  it("is near nothing for silence and full for a loud chunk", () => {
    expect(levelOf(new Int16Array(1600))).toBe(0);
    expect(levelOf(new Int16Array(1600).fill(16000))).toBe(1);
    const quiet = levelOf(new Int16Array(1600).fill(100));
    expect(quiet).toBeGreaterThan(0);
    expect(quiet).toBeLessThan(0.05);
  });
});
