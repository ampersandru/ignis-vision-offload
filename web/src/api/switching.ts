import { lastSwitchLine, refusalLines } from "../server/outcome.ts";
import { switchTo } from "../server/routes.ts";
import { type LastSwitch, type ModelState, readModel } from "./model.ts";

// A switch the page makes on the owner's behalf and waits out before going on:
// the chat's message to a model other than the one loaded (the composer's
// model picker), and the picture tool's visit to the image model and back
// (tools/image/image.ts). It goes through POST /v1/models/switch, so no chat
// request ever names a model (GitHub #305's ping-pong), and it follows
// GET /v1/models each second until the model serves or the switch's record
// says how it ended.

export type SwitchOptions = {
  /** Stops the wait (not the switch, which the server finishes on its own). */
  signal?: AbortSignal;
  /** Read the model again, so the page's header follows the switch. */
  onModelChange?: () => void;
  fetch?: typeof fetch;
  /** Waits between two reads of a switch under way. */
  wait?: (ms: number) => Promise<void>;
  /** How long a switch may take before the wait gives up on it. */
  timeoutMs?: number;
};

const TIMEOUT_MS = 10 * 60_000;

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/** `id` serves: loaded, or unloaded while idle, which the next request loads again. */
export const serves = (model: ModelState, id: string) => (model.state === "ready" || model.state === "unloaded") && model.id === id;

const recordOf = (model: ModelState): LastSwitch | undefined => (model.state === "reading" ? undefined : model.lastSwitch);

/**
 * Switches to `id` and waits until it serves. Resolves with `null` once it does (at once when it already did),
 * `"stopped"` when the signal ended the wait, or the reason it did not land, in words.
 */
export async function switchAndWait(id: string, options: SwitchOptions = {}): Promise<string | null> {
  const doFetch = options.fetch ?? fetch.bind(globalThis);
  const before = await readModel(doFetch);
  if (serves(before, id)) return null;
  if (options.signal?.aborted) return "stopped";
  const earlier = JSON.stringify(recordOf(before));
  const answer = await switchTo(id, doFetch);
  options.onModelChange?.();
  // A 409 is a switch already under way: it is followed like this one.
  if (!answer.ok && answer.status !== 409) return refusalLines(answer)[0].text;
  const wait = options.wait ?? sleep;
  const timeoutMs = options.timeoutMs ?? TIMEOUT_MS;
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (options.signal?.aborted) return "stopped";
    await wait(1000);
    const now = await readModel(doFetch);
    if (serves(now, id)) {
      options.onModelChange?.();
      return null;
    }
    // A new record for this switch that is not a success, once nothing loads any more: how it ended.
    const last = recordOf(now);
    if (last && last.to === id && last.outcome !== "switched" && JSON.stringify(last) !== earlier && now.state !== "switching" && now.state !== "loading") {
      options.onModelChange?.();
      return lastSwitchLine(last).text;
    }
    if (Date.now() > deadline) return `The switch to ${id} did not finish within ${Math.round(timeoutMs / 60_000)} minutes.`;
  }
}
