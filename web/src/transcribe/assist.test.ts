import { describe, expect, it } from "vitest";
import type { StreamOptions, StreamResult } from "../api/stream.ts";
import { ASSIST_SYSTEM, askModel, assistRequest, settle, transcriptInput } from "./assist.ts";
import { addUtterance, createTranscript, type Transcript } from "./transcript.ts";

// What the Transcribe view asks the text model (spec speech/07, Q125, Q134):
// the preamble and the task as the system prompt, the transcript the user
// turn, which a listening only appends to; a full engine waited out.

function call(lines: [string, string][]): Transcript {
  let t = { ...createTranscript(1, { source: "call", at: new Date(2026, 9, 27, 15, 42) }), title: "27 Oct 15:42 · Call" };
  lines.forEach(([who, text], i) => {
    t = addUtterance(t, { stream: "s", startMs: i * 10_000, endMs: i * 10_000 + 4_000, text, voice: who === "me" ? { kind: "me" } : { kind: "others" } });
  });
  return t;
}

describe("the transcript the model reads", () => {
  it("is the title and the .txt, so a later request only appends", () => {
    const t = call([
      ["me", "Partiamo dal budget."],
      ["others", "Va bene."],
    ]);
    expect(transcriptInput(t, null, 1_500)).toEqual({ content: "27 Oct 15:42 · Call\n\n[00:00] Me: Partiamo dal budget.\n[00:10] Others: Va bene.\n", fromMs: null });
    const longer = transcriptInput(addUtterance(t, { stream: "s", startMs: 20_000, endMs: 21_000, text: "Poi?", voice: { kind: "me" } }), null, 1_500);
    expect(longer.content.startsWith(transcriptInput(t, null, 1_500).content)).toBe(true);
  });

  it("is its tail when the whole would not fit the model's context with the answer, and says where it starts", () => {
    const t = call([
      ["me", "a".repeat(3_000)],
      ["others", "b".repeat(3_000)],
      ["me", "c".repeat(3_000)],
    ]);
    // 1,500 tokens for the answer and 1,000 for the prompt leave 2,500 tokens: about 7,500 characters.
    const { content, fromMs } = transcriptInput(t, 5_000, 1_500);
    expect(fromMs).toBe(10_000);
    expect(content).not.toContain("aaa");
    expect(content).toContain("[00:10] Others: bbb");
    expect(transcriptInput(t, 100_000, 1_500).fromMs).toBeNull();
  });

  it("is the user turn, after the preamble and the task, so a later request of the task begins with an earlier one", () => {
    const settings = { temperature: 0.2, topP: 0.9, maxTokens: 10, reasoningEffort: "none" as const, thinkingBudget: null, laneTag: "agent" as const };
    expect(assistRequest(settings, "Do it.", "T\n")).toEqual({ settings: { ...settings, systemPrompt: `${ASSIST_SYSTEM}\n\nDo it.` }, turns: [{ role: "user", content: "T\n" }] });
  });
});

describe("an answer settled", () => {
  it("is its text tidied, or why there is none, or stopped", () => {
    expect(settle({ ok: true, text: "  Sì. " })).toEqual({ text: "Sì." });
    expect(settle({ ok: true, text: "   " })).toEqual({ failed: "The model wrote nothing." });
    expect(settle({ ok: false, message: "ignis answered 500" })).toEqual({ failed: "ignis answered 500" });
    expect(settle({ ok: "stopped" })).toBe("stopped");
    expect(settle({ ok: true, text: "```\nx\n```" }, (text) => text.replaceAll("`", "").trim())).toEqual({ text: "x" });
  });
});

describe("asking the model", () => {
  const request = assistRequest({ temperature: 0.2, topP: 0.9, maxTokens: 10, reasoningEffort: "none", thinkingBudget: null, laneTag: "agent" }, "Do it.", "T\n");

  it("streams the text and waits out a full engine, a second at a time", async () => {
    const answers: StreamResult[] = [
      { ok: false, message: "engine_full: all lanes in use", timeline: { sentAt: 0, stopped: false } },
      { ok: true, timeline: { sentAt: 0, stopped: false } },
    ];
    const slept: number[] = [];
    const seen: string[] = [];
    const stream = async (options: StreamOptions) => {
      if (answers.length === 1) for (const text of ["Ciao", " a tutti"]) options.onEvent({ kind: "content", text });
      return answers.shift()!;
    };
    const asked = await askModel(request, new AbortController().signal, { stream, sleep: async (ms) => void slept.push(ms) }, (text) => seen.push(text));
    expect(asked).toEqual({ ok: true, text: "Ciao a tutti" });
    expect(slept).toEqual([1_000]);
    expect(seen).toEqual(["Ciao", "Ciao a tutti"]);
  });

  it("says why it failed, or that it was stopped", async () => {
    const failed = await askModel(request, new AbortController().signal, { stream: async () => ({ ok: false, message: "ignis answered 500", timeline: { sentAt: 0, stopped: false } }) });
    expect(failed).toEqual({ ok: false, message: "ignis answered 500" });
    const abort = new AbortController();
    const stopped = await askModel(request, abort.signal, {
      stream: async () => {
        abort.abort();
        return { ok: true, timeline: { sentAt: 0, stopped: true } };
      },
    });
    expect(stopped).toEqual({ ok: "stopped" });
  });
});
