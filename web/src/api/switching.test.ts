import { describe, expect, it } from "vitest";
import type { LastSwitch } from "./model.ts";
import { switchAndWait } from "./switching.ts";

// The switch the page waits out (api/switching.ts): none when the model
// already serves, the server's refusal in words, a failed load from the
// switch's record — and never an older record taken for this switch's.

type Served = { id: string; kind: string; status: string; last_switch?: LastSwitch };

/** A server whose GET /v1/models answers from `reads` in turn (the last one repeating), and that records its switches. */
function server(reads: Served[], switchAnswer: { status: number; body: object } = { status: 200, body: { from: "a", to: "b" } }) {
  const log: string[] = [];
  let i = 0;
  const json = (body: object, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  const fetcher = (async (path: string, init: RequestInit = {}) => {
    log.push(`${init.method ?? "GET"} ${path}`);
    if (path === "/v1/models") {
      const read = reads[Math.min(i++, reads.length - 1)];
      return json({ data: [{ id: read.id, kind: read.kind }], status: read.status, ...(read.last_switch ? { last_switch: read.last_switch } : {}) });
    }
    if (path === "/v1/models/switch") return json(switchAnswer.body, switchAnswer.status);
    throw new Error(`unexpected ${path}`);
  }) as unknown as typeof fetch;
  return { fetcher, log };
}

const options = (fetcher: typeof fetch) => ({ fetch: fetcher, wait: async () => {} });
const image = { id: "ignis-image-2.1-turbo", kind: "image", status: "ready" };
const text = { id: "qwen3.8-27b", kind: "text", status: "ready" };

describe("a switch the page waits out", () => {
  it("sends nothing when the model already serves, loaded or unloaded while idle", async () => {
    const s = server([{ ...text, status: "unloaded" }]);
    expect(await switchAndWait("qwen3.8-27b", options(s.fetcher))).toBeNull();
    expect(s.log).toEqual(["GET /v1/models"]);
  });

  it("switches, then follows the model until it serves", async () => {
    const s = server([image, image, text]);
    expect(await switchAndWait("qwen3.8-27b", options(s.fetcher))).toBeNull();
    expect(s.log.filter((l) => l === "POST /v1/models/switch")).toHaveLength(1);
  });

  it("says the server's refusal", async () => {
    const s = server([image], { status: 404, body: { error: { message: "no model qwen3.8-9b", code: "model_not_found" } } });
    expect(await switchAndWait("qwen3.8-9b", options(s.fetcher))).toMatch(/no model qwen3.8-9b/i);
  });

  it("ends on a new record of this switch that failed, but not on an older one", async () => {
    const older: LastSwitch = { from: "qwen3.8-27b", to: "qwen3.8-flash-next", outcome: "refused", reason: "short of RAM" };
    const now: LastSwitch = { from: "ignis-image-2.1-turbo", to: "qwen3.8-flash-next", outcome: "restored", reason: "the load failed" };
    const s = server([
      { ...image, last_switch: older },
      { ...image, last_switch: older },
      { ...image, last_switch: now },
    ]);
    const answer = await switchAndWait("qwen3.8-flash-next", options(s.fetcher));
    expect(answer).toContain("the load failed");
    expect(s.log.filter((l) => l === "GET /v1/models")).toHaveLength(3);
  });

  it("stops waiting when the signal says so", async () => {
    const stop = new AbortController();
    const s = server([image]);
    const answer = switchAndWait("qwen3.8-27b", { ...options(s.fetcher), signal: stop.signal, wait: async () => stop.abort() });
    expect(await answer).toBe("stopped");
  });
});
