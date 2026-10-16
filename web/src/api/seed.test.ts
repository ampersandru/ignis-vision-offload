import { describe, expect, it } from "vitest";
import { MAX_SEED, nthSeed, parseKeepingSeeds, seedFrom, seedOf, stringifyWithSeed } from "./seed.ts";

// Seeds as u64 digits (api/seed.ts): read from the server's JSON without
// rounding, written back as a number of any size, and added to as the server
// does for each image of a job.

const BIG = "5936113633719656123";

describe("a seed", () => {
  it("is read from the field as digits up to 2^64 - 1, blank as none", () => {
    expect(seedOf(" 7 ")).toBe("7");
    expect(seedOf(BIG)).toBe(BIG);
    expect(seedOf(MAX_SEED.toString())).toBe("18446744073709551615");
    expect(seedOf("18446744073709551616")).toBeUndefined();
    expect(seedOf("")).toBeNull();
    expect(seedOf("7a")).toBeUndefined();
    expect(seedOf("-1")).toBeUndefined();
  });

  it("is taken as digits or a safe number from what the server or the model wrote", () => {
    expect(seedFrom(BIG)).toBe(BIG);
    expect(seedFrom(41)).toBe("41");
    expect(seedFrom(5936113633719656000)).toBeUndefined();
    expect(seedFrom(-3)).toBeUndefined();
    expect(seedFrom("x")).toBeUndefined();
  });

  it("comes out of the server's JSON whole, and nothing else changes", () => {
    const text = `{"seed": ${BIG}, "data":[{"b64_json":"AA==","revised_prompt":"a \\"seed\\": 5 note"}], "n": 2}`;
    expect(parseKeepingSeeds(text)).toEqual({ seed: BIG, data: [{ b64_json: "AA==", revised_prompt: 'a "seed": 5 note' }], n: 2 });
    expect(String(JSON.parse(`{"seed": ${BIG}}`).seed)).not.toBe(BIG); // what the page did before: rounded
  });

  it("goes back to the server as a number of any size", () => {
    expect(stringifyWithSeed({ prompt: "a fox", seed: BIG })).toBe(`{"prompt":"a fox","seed":${BIG}}`);
    expect(stringifyWithSeed({ prompt: "a fox" })).toBe('{"prompt":"a fox"}');
  });

  it("of the job's image k is the job's plus k, wrapping at 2^64 as the server's does", () => {
    expect(nthSeed(BIG, 2)).toBe("5936113633719656125");
    expect(nthSeed(MAX_SEED.toString(), 1)).toBe("0");
  });
});
