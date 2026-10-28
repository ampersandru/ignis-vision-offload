import { describe, expect, it } from "vitest";
import { transcribe, TRANSCRIPTIONS_PATH } from "./transcriptions.ts";

// The page's client of `POST /v1/audio/transcriptions`: the file as multipart
// `file`, its transcript out of the JSON answer, and the server's own words
// when it refuses.

/** A fetch that answers `status` with `body`, keeping what it was asked. */
function server(status: number, body: string) {
  const asked: { path: string; init: RequestInit }[] = [];
  const fetcher = (async (path: string, init: RequestInit) => {
    asked.push({ path, init });
    return new Response(body, { status });
  }) as unknown as typeof fetch;
  return { fetcher, asked };
}

const clip = () => new Blob([new Uint8Array(44)], { type: "audio/wav" });

describe("the transcription client", () => {
  it("posts the file as multipart `file` and returns the transcript", async () => {
    const { fetcher, asked } = server(200, JSON.stringify({ text: "Buy bread.", usage: { type: "duration", seconds: 1.2 } }));
    expect(await transcribe(clip(), "memo.wav", undefined, fetcher)).toBe("Buy bread.");
    expect(asked).toHaveLength(1);
    expect(asked[0].path).toBe(TRANSCRIPTIONS_PATH);
    expect(asked[0].init.method).toBe("POST");
    const form = asked[0].init.body as FormData;
    const file = form.get("file") as File;
    expect([file.name, file.size, file.type]).toEqual(["memo.wav", 44, "audio/wav"]);
    expect(form.get("response_format")).toBe("json");
  });

  it("rejects with the server's reason", async () => {
    const refusal = {
      error: { message: "`file` is not a format this server transcribes (the codec is not decoded (Opus is not): unsupported)", code: "invalid_value" },
    };
    const { fetcher } = server(400, JSON.stringify(refusal));
    await expect(transcribe(clip(), "memo.opus", undefined, fetcher)).rejects.toThrow(
      "400: `file` is not a format this server transcribes (the codec is not decoded (Opus is not): unsupported)",
    );
    const busy = server(503, JSON.stringify({ error: { message: "the speech queue is full", code: "engine_full" } }));
    await expect(transcribe(clip(), "memo.wav", undefined, busy.fetcher)).rejects.toThrow("503: the speech queue is full");
  });

  it("rejects with why ignis could not be reached", async () => {
    const down = (async () => Promise.reject(new TypeError("Failed to fetch"))) as unknown as typeof fetch;
    await expect(transcribe(clip(), "memo.wav", undefined, down)).rejects.toThrow("Could not reach ignis: TypeError: Failed to fetch");
  });
});
