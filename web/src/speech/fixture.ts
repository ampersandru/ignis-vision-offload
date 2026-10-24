import { type DictationHandle, type DictationState, IDLE_STATE } from "./dictation.ts";
import type { Dictation } from "./useDictation.ts";

/** A dictation fixed in `state`, for drawing a field in any of its states without a mic or a server. */
export function fixedDictation(state: Partial<DictationState> = {}): Dictation {
  const fixed: DictationState = { ...IDLE_STATE, ...state };
  const none = () => {};
  const controller: DictationHandle = {
    getState: () => fixed,
    subscribe: () => none,
    toggle: none,
    done: async () => {},
    discard: none,
    insertHeard: none,
    keyDown: none,
    keyUp: none,
    active: fixed.phase !== "idle",
  };
  return { controller, field: { current: null } };
}
