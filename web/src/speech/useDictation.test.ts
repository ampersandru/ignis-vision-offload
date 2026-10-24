import type { KeyboardEvent } from "react";
import { describe, expect, it } from "vitest";
import { fixedDictation } from "./fixture.ts";
import { dictationKeys } from "./useDictation.ts";

// The keys a field's container hands its dictation (spec speech/04): the
// shortcut and Esc, only where the field takes dictation, and only from inside
// the container's own DOM.

function recorded() {
  const calls: string[] = [];
  const dictation = fixedDictation();
  dictation.controller.keyDown = () => calls.push("down");
  dictation.controller.keyUp = () => calls.push("up");
  return { calls, dictation };
}

/** A key event from inside the container's DOM, or from a portal React bubbles through it. */
const event = (inside: boolean) => ({ currentTarget: { contains: () => inside }, target: {} }) as unknown as KeyboardEvent;

describe("dictationKeys", () => {
  it("hands the container's keys to the dictation", () => {
    const { calls, dictation } = recorded();
    const keys = dictationKeys(dictation);
    keys.onKeyDown?.(event(true));
    keys.onKeyUp?.(event(true));
    expect(calls).toEqual(["down", "up"]);
  });

  it("hands none from a field that takes no dictation: the chord never starts one with nowhere to land", () => {
    const { dictation } = recorded();
    expect(dictationKeys(dictation, false)).toEqual({});
  });

  it("leaves a portalled menu's keys to the menu", () => {
    const { calls, dictation } = recorded();
    dictationKeys(dictation).onKeyDown?.(event(false));
    expect(calls).toEqual([]);
  });
});
