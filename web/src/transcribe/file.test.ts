import { describe, expect, it } from "vitest";
import { decodeFile, UNDECODABLE } from "./file.ts";

// A recording the user picks (spec speech/07, "Sources", Q94): decoded in the
// browser into an OfflineAudioContext at 16 kHz, its channels mixed to mono,
// as PCM16 the page sends over Realtime.

/** A context that decodes to `channels`, at the rate it was made with. */
function decoding(channels: number[][], asked: { rate?: number; channels?: number } = {}) {
  return (channelCount: number, _length: number, rate: number) => {
    asked.rate = rate;
    asked.channels = channelCount;
    return {
      decodeAudioData: async (_data: ArrayBuffer) => ({
        numberOfChannels: channels.length,
        length: channels[0].length,
        getChannelData: (c: number) => Float32Array.from(channels[c]),
      }),
    };
  };
}

describe("decodeFile", () => {
  it("decodes at 16 kHz, mixes the channels to mono and writes PCM16", async () => {
    const asked: { rate?: number; channels?: number } = {};
    const pcm = await decodeFile(new Blob([new Uint8Array([1, 2])]), { createContext: decoding([[1, 0.5, -1], [0, -0.5, -1]], asked) });
    expect(asked).toEqual({ rate: 16_000, channels: 1 });
    expect([...pcm]).toEqual([16384, 0, -32768]);
  });

  it("says when the browser cannot decode the file", async () => {
    const failing = () => ({ decodeAudioData: async () => Promise.reject(new DOMException("unknown content type", "EncodingError")) });
    await expect(decodeFile(new Blob([]), { createContext: failing })).rejects.toThrow(UNDECODABLE);
  });
});
