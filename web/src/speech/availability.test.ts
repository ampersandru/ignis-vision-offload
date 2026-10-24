import { describe, expect, it } from "vitest";
import { unavailable } from "./availability.ts";

// Why the mic cannot start (spec speech/04, Q18, Q35, Q39, Q17): one sentence
// for the hint line. The server's reasons are the ones speech/mod.rs writes.

const OFF = "the speech model is turned off (`speech.model` is `off`)";
const NO_FILES = "no speech model files for `qwen3-asr-1.7b` under download.path; fetch them with `ignis-server model download qwen3-asr-1.7b`";
const NO_AVX2 = "the CPU lacks AVX2, which the speech model needs";
const NOT_IN_CATALOG = "`speech.model` names `qwen3-asr-9b`, which is not in the model catalog";

describe("unavailable", () => {
  it("is nothing while the speech model can serve, or before the server has said", () => {
    expect(unavailable({ secure: true, speech: { available: true }, blocked: false })).toBeNull();
    expect(unavailable({ secure: true, blocked: false })).toBeNull();
  });

  it("asks for https or localhost first: no server answer helps a page the browser keeps from the mic", () => {
    expect(unavailable({ secure: false, speech: { available: false, reason: OFF }, blocked: false })).toEqual({
      sentence: "The microphone needs https or localhost. Open the Playground through an https proxy, or start the server with `--server-expose cloudflare-quick`.",
      blocked: false,
    });
  });

  it("says each server cause in the spec's words, the download command filled in", () => {
    const say = (reason: string) => unavailable({ secure: true, speech: { available: false, reason }, blocked: false })?.sentence;
    expect(say(OFF)).toBe("Speech is turned off on this server (`speech.model = off`).");
    expect(say(NO_FILES)).toBe("No speech model on this server. Download one with `ignis-server model download qwen3-asr-1.7b`.");
    expect(say(NO_AVX2)).toBe("Speech is unavailable on this machine: the CPU lacks AVX2 and the card has no free room to lend.");
    // A cause the table has no row for is the server's own words.
    expect(say(NOT_IN_CATALOG)).toBe("Speech is unavailable on this server: `speech.model` names `qwen3-asr-9b`, which is not in the model catalog.");
    expect(say("this build has no speech executor")).toBe("Speech is unavailable on this server: this build has no speech executor.");
  });

  it("names a blocked mic as the one cause the mic can be pressed again for", () => {
    expect(unavailable({ secure: true, speech: { available: true }, blocked: true })).toEqual({
      sentence: "Microphone access is blocked for this site. Allow it in the browser's site settings, then press the mic again.",
      blocked: true,
    });
  });
});
