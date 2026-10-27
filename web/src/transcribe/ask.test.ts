import { describe, expect, it } from "vitest";
import type { StreamOptions, StreamResult } from "../api/stream.ts";
import { ASK_TASK, Asker, askRequest, readAnswered, withAnswer } from "./ask.ts";
import { ASSIST_SYSTEM } from "./assist.ts";
import { addUtterance, createTranscript, type Transcript } from "./transcript.ts";

// Ask about this conversation (spec speech/07, Q139): the transcript, the
// question last, on the interactive lane with thinking off; the answer
// streams, and lands on the Transcript asked about.

function heard(): Transcript {
  let t = { ...createTranscript(3), title: "27 Oct 15:42 · Call" };
  t = addUtterance(t, { stream: "s", startMs: 4_000, endMs: 6_000, text: "Il backup si fa venerdì.", voice: { kind: "me" } });
  return t;
}

describe("a question's request", () => {
  it("is the task, then the transcript and the question last", () => {
    const request = askRequest(heard(), "  Quando il backup? ", null);
    expect(request.settings).toMatchObject({ systemPrompt: `${ASSIST_SYSTEM}\n\n${ASK_TASK}`, reasoningEffort: "none", laneTag: "interactive" });
    expect(request.turns[0].content).toBe("27 Oct 15:42 · Call\n\n[00:04] Me: Il backup si fa venerdì.\n\nQuestion: Quando il backup?");
  });
});

describe("the answers", () => {
  it("are kept latest first, and read back those this page can read", () => {
    const one = withAnswer(undefined, "Quando?", "Venerdì [00:04].", 1);
    const two = withAnswer(one, "Chi?", "Me.", 2);
    expect(two.map((a) => [a.id, a.question])).toEqual([
      [2, "Chi?"],
      [1, "Quando?"],
    ]);
    expect(readAnswered([...two, { id: "x" }])).toEqual(two);
    expect(readAnswered("x")).toBeUndefined();
  });

  it("stream, and land on the Transcript asked about", async () => {
    const answered: [number, string, string][] = [];
    const seen: string[] = [];
    const stream = async (options: StreamOptions): Promise<StreamResult> => {
      for (const text of ["Venerdì ", "[00:04]."]) options.onEvent({ kind: "content", text });
      return { ok: true, timeline: { sentAt: 0, stopped: false } };
    };
    const asker = new Asker({ stream, onAnswered: (id, q, a) => answered.push([id, q, a]) });
    asker.subscribe(() => seen.push(asker.getState().text));
    await asker.ask(heard(), "Quando il backup?", null);
    expect(seen).toContain("Venerdì [00:04].");
    expect(answered).toEqual([[3, "Quando il backup?", "Venerdì [00:04]."]]);
    expect(asker.getState().writing).toBe(false);
  });
});
