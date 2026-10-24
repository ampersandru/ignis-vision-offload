import { Fragment } from "react";
import { IconMic, IconStopSquare } from "../ui/icons.tsx";
import type { DictationState, Tone } from "./dictation.ts";
import type { Dictation } from "./useDictation.ts";

// The mic and the strip (spec speech/04, "What the user sees"; the Q17
// mockup). One pair serves the chat, the image studio and Decide's prose
// fields; `compact` is a Decide field's size.

/** The mic's tooltip in a prompt box. */
export const MIC_TITLE = "Dictate. Or hold Ctrl+Shift+Space while you speak.";

/**
 * The mic beside Send, in the ash of the prompt box's other icons; the ember
 * Done chip, in Send's colour, while listening; a disabled Finishing after.
 * While it cannot start it is struck through and dimmed, its reason in the
 * hint line: still pressable, which says the reason in a Decide field and asks
 * a blocked mic again.
 */
export function DictationMic({
  dictation,
  state,
  label = "Dictate",
  title = MIC_TITLE,
  compact = false,
}: {
  dictation: Dictation;
  state: DictationState;
  /** "Dictate", or a Decide field's "Dictate into Question". */
  label?: string;
  title?: string;
  compact?: boolean;
}) {
  const { controller } = dictation;
  const chip = compact ? "gap-1.5 px-3 py-1.5 text-[13px] [--cut-size:8px]" : "gap-2 px-4 py-2.5 text-sm [--cut-size:10px]";
  if (state.phase === "listening") {
    return (
      <button type="button" className={`cut flex shrink-0 items-center bg-[#ff5a1f] font-display font-semibold text-[#1c2026] hover:bg-[#ff7a45] ${chip}`} onClick={() => controller.toggle()}>
        <IconStopSquare />
        Done
      </button>
    );
  }
  if (state.phase === "finishing") {
    return (
      <button type="button" disabled className={`cut shrink-0 cursor-default bg-line font-display font-semibold text-ash ${chip}`}>
        Finishing
      </button>
    );
  }
  const size = compact ? "size-8" : "size-10 [&>svg]:size-[18px]";
  if (state.unavailable) {
    return (
      <button type="button" aria-label={`${label} (unavailable)`} aria-disabled="true" className={`grid shrink-0 cursor-default place-items-center text-ash opacity-45 ${size}`} onClick={() => controller.toggle()}>
        <IconMic struck />
      </button>
    );
  }
  // A busy wait or a stopped dictation is under way: the mic waits for it to end.
  const waiting = state.phase !== "idle";
  return (
    <button
      type="button"
      aria-label={label}
      title={title}
      disabled={waiting}
      className={`grid shrink-0 place-items-center text-ash hover:text-ink disabled:cursor-default disabled:opacity-45 disabled:hover:text-ash ${size}`}
      onClick={() => controller.toggle()}
    >
      <IconMic />
    </button>
  );
}

/** The most characters of what was heard the strip's two lines are given: the newest. */
const STRIP_CHARS = 140;

/** What the strip shows of the settled words and the tail: the newest `max` characters, the cut marked. */
export function visibleText(stable: string, tail: string, max = STRIP_CHARS): { stable: string; tail: string } {
  const drop = stable.length + tail.length - max;
  if (drop <= 0) return { stable, tail };
  if (drop < stable.length) return { stable: `…${stable.slice(drop)}`, tail };
  return { stable: "…", tail: tail.slice(drop - stable.length) };
}

const clock = (seconds: number) => `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;

/**
 * The strip above the prompt box (under a Decide field): on the kiln's dark,
 * the level meter, what was heard — the settled words light, the provisional
 * tail grey — the timer, Discard, and the running heat bar at its foot. After
 * a stop it keeps what was heard, to insert or discard. A polite live region;
 * its motion stops under `prefers-reduced-motion` (styles.css).
 */
export function DictationStrip({ dictation, state, compact = false }: { dictation: Dictation; state: DictationState; compact?: boolean }) {
  const { phase } = state;
  if (phase !== "listening" && phase !== "finishing" && phase !== "failed") return null;
  const { controller } = dictation;
  const live = phase !== "failed";
  const shown = visibleText(state.stable, state.tail);
  const heard = (state.stable + state.tail).trim() !== "";
  return (
    <div
      role="status"
      aria-live="polite"
      className={`cut relative flex items-center bg-kiln text-[#eae8e4] ${compact ? "gap-3 pt-2 pr-2.5 pb-2.5 pl-3 [--cut-size:8px]" : "gap-3.5 pt-2.5 pr-3 pb-3 pl-3.5 [--cut-size:10px]"}`}
    >
      <div aria-hidden="true" className={`flex shrink-0 items-center gap-0.5 ${compact ? "h-[22px]" : "h-7"}`}>
        {state.levels.map((level, i) => (
          <div
            key={i}
            className="dictation-bar w-0.5"
            style={{ height: `${Math.max(2, Math.round(level * (compact ? 20 : 26)))}px`, background: live && level > 0.12 ? "#ff5a1f" : "#4a525c" }}
          />
        ))}
      </div>
      <div className={`line-clamp-2 min-w-0 flex-1 leading-[1.45] ${compact ? "text-sm" : "text-[15px]"}`}>
        {heard ? (
          <>
            <span className="text-[#eae8e4]">{shown.stable}</span>
            <span className="text-[#939ba4]">{shown.tail}</span>
          </>
        ) : (
          phase === "listening" && <span className="text-[#939ba4]">Listening…</span>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-2.5">
        {/* Hidden from the live region: a reader would hear the clock every second. */}
        {phase === "listening" && (
          <span aria-hidden="true" className="flex items-center gap-1.5 font-display text-[13px] tabular-nums text-[#b9bec4]">
            <span className="dictation-dot inline-block size-[7px] bg-[#ff5a1f]" />
            {clock(state.elapsed)}
          </span>
        )}
        {phase === "finishing" && <span className="font-display text-[13px] text-[#b9bec4]">Finishing</span>}
        {phase === "failed" && heard && (
          <button type="button" className="px-1.5 py-1 font-display text-xs font-semibold text-[#ff8a4c] hover:text-[#ffa877]" onClick={() => controller.insertHeard()}>
            Insert what was heard
          </button>
        )}
        <button type="button" className="px-1.5 py-1 font-display text-xs text-[#939ba4] hover:text-[#eae8e4]" onClick={() => controller.discard()}>
          Discard <span className="text-[#6b737d]">Esc</span>
        </button>
      </div>
      {live && <span aria-hidden="true" className="dictation-heat absolute inset-x-0 bottom-0 h-0.5" />}
    </div>
  );
}

/** The colour of a hint-line sentence: ash, warn, fault. */
export function toneClass(tone: Tone): string {
  return tone === "fault" ? "text-fault" : tone === "warn" ? "text-warn" : "text-ash";
}

/** A hint-line sentence, its backticked spans set as code. */
export function HintText({ text }: { text: string }) {
  return (
    <>
      {text.split("`").map((part, i) =>
        i % 2 === 1 ? (
          <code key={i} className="font-mono text-[11px]">
            {part}
          </code>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        ),
      )}
    </>
  );
}
