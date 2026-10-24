// The mic's audio on its way to the server (spec speech/04, "Capture"): an
// AudioWorklet downsamples it to 16 kHz mono and hands over 100 ms of PCM16 at
// a time, each the body of one `input_audio_buffer.append` (owner,
// 2026-10-24). 16 kHz is the model's own rate, ignis's declared extension
// (spec speech/03, Q21), so the server resamples nothing.
//
// The worklet is plain JavaScript in a string, loaded from a Blob URL: a
// worklet runs in its own global scope, and a string is the same code in the
// dev server, the build and the tests, which evaluate it with the globals that
// scope gives (pcm.test.ts).

/** The rate the server is told the audio is at. */
export const TARGET_RATE = 16_000;

/** 100 ms at TARGET_RATE: one append. */
export const CHUNK_SAMPLES = 1_600;

/** The name the worklet registers its processor under. */
export const PROCESSOR = "ignis-dictation-capture";

/**
 * The worklet. Each output sample is the mean of the input samples its span
 * covers — a box filter, enough against aliasing for speech — after the
 * channels are mixed to mono; a context already at 16 kHz passes through.
 * A full chunk is posted as `{pcm}` (its buffer transferred); the message
 * "flush" posts the part chunk, if any, then `{flushed: true}`.
 */
export const WORKLET_SOURCE = `
class DictationCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = (options && options.processorOptions) || {};
    this.step = sampleRate / (o.targetRate || ${TARGET_RATE});
    this.size = o.chunkSamples || ${CHUNK_SAMPLES};
    this.chunk = new Int16Array(this.size);
    this.filled = 0;
    this.sum = 0;
    this.count = 0;
    this.span = 0;
    this.port.onmessage = (event) => {
      if (event.data !== "flush") return;
      this.post();
      this.port.postMessage({ flushed: true });
    };
  }
  post() {
    if (this.filled === 0) return;
    const pcm = this.chunk.slice(0, this.filled);
    this.filled = 0;
    this.port.postMessage({ pcm }, [pcm.buffer]);
  }
  push(value) {
    const s = Math.max(-1, Math.min(1, value));
    this.chunk[this.filled++] = Math.round(s < 0 ? s * 32768 : s * 32767);
    if (this.filled === this.size) this.post();
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const frames = input[0].length;
    for (let i = 0; i < frames; i++) {
      let x = 0;
      for (let c = 0; c < input.length; c++) x += input[c][i];
      this.sum += x / input.length;
      this.count += 1;
      this.span += 1;
      if (this.span >= this.step) {
        this.push(this.sum / this.count);
        this.sum = 0;
        this.count = 0;
        this.span -= this.step;
      }
    }
    return true;
  }
}
registerProcessor("${PROCESSOR}", DictationCapture);
`;

/** The samples as base64 of their 16-bit little-endian bytes: `input_audio_buffer.append`'s `audio`. */
export function pcmBase64(pcm: Int16Array): string {
  const bytes = new Uint8Array(pcm.length * 2);
  const view = new DataView(bytes.buffer);
  pcm.forEach((sample, i) => view.setInt16(i * 2, sample, true));
  let binary = "";
  // In slices: one String.fromCharCode over a whole chunk can overflow the argument limit.
  for (let at = 0; at < bytes.length; at += 0x8000) binary += String.fromCharCode(...bytes.subarray(at, at + 0x8000));
  return btoa(binary);
}

/** How loud a chunk is, 0 to 1, for the level meter: its RMS, scaled so speech fills most of the bar. */
export function levelOf(pcm: Int16Array): number {
  if (pcm.length === 0) return 0;
  let sum = 0;
  for (const sample of pcm) sum += sample * sample;
  return Math.min(1, (Math.sqrt(sum / pcm.length) / 32768) * 5);
}
