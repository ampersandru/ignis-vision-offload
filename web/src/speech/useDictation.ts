import { type KeyboardEvent, type RefObject, useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { keyRequired } from "../api/auth.ts";
import type { SpeechStatus } from "../api/model.ts";
import { startCapture } from "./capture.ts";
import { DictationController, type DictationHandle, type DictationState } from "./dictation.ts";
import { openTranscription } from "./realtime.ts";

/** A field's dictation: what drives it, and the textarea it writes into. */
export type Dictation = { controller: DictationHandle; field: RefObject<HTMLTextAreaElement | null> };

/**
 * A dictation for one field, made by whoever owns the field's text. It does
 * not re-render its owner: the mic and the strip read it (`useDictationState`),
 * so a level meter moving ten times a second redraws the composer, not the
 * page around it.
 */
export function useDictation(options: {
  /** The field's new text: a dictation only ever writes it, never sends it. */
  onChange: (value: string) => void;
  /** Read at the press. */
  context: () => string;
  speech?: SpeechStatus;
  onUnavailable?: () => void;
}): Dictation {
  const field = useRef<HTMLTextAreaElement | null>(null);
  const latest = useRef(options);
  useLayoutEffect(() => {
    latest.current = options;
  });
  const [controller] = useState(
    () =>
      new DictationController(
        {
          read: () => {
            const element = field.current;
            return { value: element?.value ?? "", at: element?.selectionEnd ?? element?.value.length ?? 0 };
          },
          write: (value) => latest.current.onChange(value),
          context: () => latest.current.context(),
        },
        {
          open: (context, handlers) => openTranscription(context, handlers),
          capture: (onChunk) => startCapture(onChunk),
          onKeyRequired: keyRequired,
          onUnavailable: () => latest.current.onUnavailable?.(),
        },
      ),
  );
  // Outside https and localhost the browser gives no page a microphone (Q18); a test has no such flag.
  const secure = typeof globalThis.isSecureContext === "boolean" ? globalThis.isSecureContext : true;
  const { speech } = options;
  useEffect(() => controller.setAvailability({ secure, speech }), [controller, secure, speech]);
  useEffect(() => () => controller.dispose(), [controller]);
  return useMemo(() => ({ controller, field }), [controller]);
}

/**
 * What the dictation shows now, for the component that draws the mic and the
 * strip. An insertion selects its text in the field once the field holds it.
 */
export function useDictationState({ controller, field }: Dictation): DictationState {
  const state = useSyncExternalStore(controller.subscribe, controller.getState, controller.getState);
  const selection = state.selection;
  // Only an insertion made while this component shows: one from before it mounted is not selected again.
  const applied = useRef(selection?.seq);
  useLayoutEffect(() => {
    const element = field.current;
    if (!element || !selection || selection.seq === applied.current) return;
    applied.current = selection.seq;
    element.focus();
    element.setSelectionRange(selection.start, selection.end);
  }, [selection, field]);
  return state;
}

/**
 * The key handlers a field's container hands its dictation — Ctrl+Shift+Space and Esc — or none while the field
 * takes no dictation (a JSON field, a file), so the chord never starts one with nowhere to land. Only keys pressed
 * in the container's own DOM count: a portalled menu's Esc is the menu's, though React bubbles it here.
 */
export function dictationKeys({ controller }: Dictation, on = true): { onKeyDown?: (event: KeyboardEvent) => void; onKeyUp?: (event: KeyboardEvent) => void } {
  if (!on) return {};
  const inside = (event: KeyboardEvent) => event.currentTarget.contains(event.target as Node);
  return {
    onKeyDown: (event) => inside(event) && controller.keyDown(event),
    onKeyUp: (event) => inside(event) && controller.keyUp(event),
  };
}
