// An image job's seed is a u64 on the server (crates/server/src/images/
// routes.rs), drawn over the whole range when the request names none. A
// JavaScript number holds integers exactly only up to 2^53, so a drawn seed
// such as 5936113633719656123 came back from JSON.parse as …656000: shown
// wrong, refused by the seed field, and a different picture if sent back. The
// page keeps every seed as its decimal digits, reads the server's JSON without
// turning a seed into a number, and writes one back as a number of any size.

/** A seed as its decimal digits: 0 to 2^64 - 1. */
export type Seed = string;

export const MAX_SEED = 2n ** 64n - 1n;

/** The text of a seed field as a seed: blank leaves it to the server (`null`); digits within u64; else `undefined`. */
export function seedOf(text: string): Seed | null | undefined {
  const t = text.trim();
  if (t === "") return null;
  if (!/^\d+$/.test(t)) return undefined;
  const n = BigInt(t);
  return n <= MAX_SEED ? n.toString() : undefined;
}

/** A seed as the server or the model wrote it — digits or a number — or `undefined` when it is not one. */
export function seedFrom(value: unknown): Seed | undefined {
  if (typeof value === "string") return seedOf(value) ?? undefined;
  if (typeof value === "number" && Number.isSafeInteger(value) && value >= 0) return String(value);
  return undefined;
}

/** The seed image `index` of a job started from: the server's `seed.wrapping_add(index)`. */
export function nthSeed(seed: Seed, index: number): Seed {
  return ((BigInt(seed) + BigInt(index)) & MAX_SEED).toString();
}

/** A JSON text read with every `"seed": <number>` kept as its digits. A key inside a string value has escaped quotes, so it is not matched. */
export function parseKeepingSeeds(text: string): unknown {
  return JSON.parse(text.replace(/("seed"\s*:\s*)(\d+)/g, '$1"$2"'));
}

/** A request body as JSON, its `seed` (digits) written as a number of any size, as the server's u64 takes it. */
export function stringifyWithSeed(body: object): string {
  return JSON.stringify(body).replace(/"seed":"(\d+)"/, '"seed":$1');
}
