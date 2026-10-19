import { describe, expect, it } from "vitest";
import { jsonString } from "./json.ts";
import { type Draft, EMPTY_DRAFT, EMPTY_SPARE, newQuestion } from "./model.ts";
import type { DecideResponse, Run } from "./request.ts";
import { createDecision, type Decision } from "./sessions.ts";
import { CLOSED_MID_RUN, DECIDE_VERSION, restoreDecisions, toStored, worthKeeping } from "./stored.ts";

// A decision as the page's store keeps it (spec playground-enhancements/03),
// the chat's rules over the Decide tab's sessions: plain data, read back newest
// first with new ids, nothing left running.

const asking = (text: string): Draft => ({
  ...EMPTY_DRAFT,
  evidence: { mode: "text", text: "The payout failed twice.", file: { name: "prod.log", size: 24 } },
  questions: [{ ...newQuestion("noul", "urgent"), instructions: jsonString(text) }],
});

function ran(): Run {
  const response: DecideResponse = { model: "m", answers: {}, usage: { input_tokens: 1, output_tokens: 0 } };
  return { response, elapsedMs: 1, raw: JSON.stringify(response) };
}

function answered(id = 3): Decision {
  const draft = asking("Is it urgent?");
  return { ...createDecision(id), draft, spare: { ...EMPTY_SPARE, json: "{}" }, answer: { run: ran(), draft }, refusal: null };
}

describe("a decision in the store", () => {
  it("is plain data, and comes back as it was", () => {
    const original = answered();
    const stored = toStored(original, { key: "key-1", order: 1 });
    expect(JSON.parse(JSON.stringify(stored))).toEqual(stored);
    expect(stored).toMatchObject({ version: DECIDE_VERSION, key: "key-1", order: 1 });
    expect(stored).not.toHaveProperty("id");
    const [back] = restoreDecisions([stored], 1).decisions;
    expect(back).toEqual({ ...original, id: back.id });
  });

  it("is worth keeping once anything is written in it or asked of it", () => {
    expect(worthKeeping(createDecision(1))).toBe(false);
    expect(worthKeeping(answered())).toBe(true);
  });
});

describe("reading the decisions back", () => {
  it("lists them newest first, with new ids from the page's next one up", () => {
    const older = toStored({ ...answered(9) }, { key: "a", order: 2 });
    const newer = toStored({ ...answered(2) }, { key: "b", order: 9 });
    const { decisions, stamps, nextId } = restoreDecisions([newer, older], 5);
    expect(decisions.map((d) => d.id)).toEqual([6, 5]);
    expect(stamps.get(6)).toEqual({ key: "b", order: 9 });
    expect(stamps.get(5)).toEqual({ key: "a", order: 2 });
    expect(nextId).toBe(7);
  });

  it("brings a decision that was being asked back stopped, its earlier answers kept, and says why", () => {
    const { decisions } = restoreDecisions([toStored({ ...answered(), running: true }, { key: "a", order: 1 })], 1);
    expect(decisions[0].running).toBe(false);
    expect(decisions[0].refusal).toBe(CLOSED_MID_RUN);
    expect(decisions[0].answer).not.toBeNull();
  });

  it("numbers the next question past the ones read back", () => {
    const stored = toStored(answered(), { key: "a", order: 1 });
    const uid = Number(stored.draft.questions[0].uid.slice(1));
    const far = { ...stored, draft: { ...stored.draft, questions: [{ ...stored.draft.questions[0], uid: `q${uid + 1000}` }] } };
    restoreDecisions([far], 1);
    expect(Number(newQuestion("noul", "b").uid.slice(1))).toBeGreaterThan(uid + 1000);
  });

  it("leaves out what it cannot read: a newer page's schema, or something that is not a decision", () => {
    const newer = { ...toStored(answered(), { key: "a", order: 1 }), version: DECIDE_VERSION + 1 };
    expect(restoreDecisions([newer, null, 7, { version: 1 }, toStored(answered(), { key: "b", order: 1 })], 1).decisions).toHaveLength(1);
  });
});
