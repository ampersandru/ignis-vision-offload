import { TARGET_RATE } from "../speech/pcm.ts";

// A recording the user picks (spec speech/07, "Sources", Q94): wav, mp3, ogg,
// m4a, webm, an mp4's audio — whatever the browser decodes. `decodeAudioData`
// on an OfflineAudioContext at 16 kHz resamples it to the model's own rate;
// its channels are mixed to mono here, as the microphone's worklet does, and
// written as the PCM16 a Realtime append carries. Nothing of it is kept.

/** Why a file cannot be transcribed: the browser could not decode it. */
export const UNDECODABLE = "The browser could not decode this file as audio.";

type DecodedAudio = { numberOfChannels: number; length: number; getChannelData: (channel: number) => Float32Array };

export type DecodeDeps = {
  createContext?: (channels: number, length: number, rate: number) => { decodeAudioData: (data: ArrayBuffer) => Promise<DecodedAudio> };
};

export async function decodeFile(file: Blob, deps: DecodeDeps = {}): Promise<Int16Array> {
  const createContext = deps.createContext ?? ((channels, length, rate) => new OfflineAudioContext(channels, length, rate));
  let audio: DecodedAudio;
  try {
    audio = await createContext(1, 1, TARGET_RATE).decodeAudioData(await file.arrayBuffer());
  } catch (err) {
    throw new Error(`${UNDECODABLE} ${err instanceof Error ? err.message : String(err)}`.trim());
  }
  const channels = Array.from({ length: audio.numberOfChannels }, (_, c) => audio.getChannelData(c));
  const pcm = new Int16Array(audio.length);
  for (let i = 0; i < audio.length; i++) {
    let x = 0;
    for (const channel of channels) x += channel[i];
    const s = Math.max(-1, Math.min(1, x / channels.length));
    pcm[i] = Math.round(s < 0 ? s * 32768 : s * 32767);
  }
  return pcm;
}
