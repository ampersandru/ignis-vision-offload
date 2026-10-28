import { useRef } from "react";
import { createPortal } from "react-dom";
import type { SpeechStatus } from "../api/model.ts";
import { SECURE_CONTEXT, unavailable } from "../speech/availability.ts";
import { HintText, toneClass } from "../speech/Dictation.tsx";
import { useAnchoredMenu } from "../ui/anchoredMenu.ts";
import { IconChevron, IconPause, IconPlay, IconStopSquare } from "../ui/icons.tsx";
import { asksFirst, type CallContext, contextOf, contextSummary, fromDefault, isBlank, sameContext } from "./context.ts";
import { clock } from "./export.ts";
import type { LaneName, ListeningState } from "./listening.ts";
import { titleOf } from "./sessions.ts";
import type { SourceKind, Transcript } from "./transcript.ts";
import type { TranscribeActions, TranscribeState } from "./useTranscribe.ts";
import { viewerChoice } from "./viewerChoice.ts";

// The Transcribe view's deck, at the foot of the sheet as the chat's composer
// is (spec speech/07): before a listening, Start with what it listens to
// beside it; while one runs, the dictation's kiln strip — each source's
// level, the clock, Pause or Resume, Stop — or a file's progress and the time
// left. The hint line says what the server or the browser refused, in spec
// 04's words, and before a call that headphones keep the far voices out of
// the microphone (Q93). What it is told and how it listens are set in the
// panel (the owner's word: the deck's switches came before the panel did);
// the first Start of a Transcript whose context was neither applied nor
// opened asks first, over Start, and a second press goes on.

/** A second press this soon after the first is the same double click, not an answer to the question it raised. */
export const ANSWER_AFTER_MS = 500;

/** What a press on Start does: raise the first Start's question, start, or nothing, the double click that raised it. */
export function startPress({ asks, checking, checkedAt, now }: { asks: boolean; checking: boolean; checkedAt: number; now: number }): "check" | "start" | null {
  if (!asks) return "start";
  if (!checking) return "check";
  return now - checkedAt < ANSWER_AFTER_MS ? null : "start";
}

/** Said before a call starts (Q93). */
export const HEADPHONES = "Use headphones on a call: through speakers, the far voices reach the microphone and are transcribed twice.";

/** The files a pick offers: what `decodeAudioData` reads in a browser, an mp4's audio included (Q94). */
export const AUDIO_FILES = "audio/*,video/mp4,video/webm,.m4a,.mp3,.ogg,.opus,.wav,.webm,.mp4";

const SOURCES: { kind: SourceKind; label: string; detail: string }[] = [
  { kind: "microphone", label: "Microphone", detail: "The room" },
  { kind: "call", label: "Call", detail: "Your mic and a shared tab, window or screen" },
  { kind: "file", label: "File", detail: "A recording, faster than real time" },
];

/** What Start listens to, as this browser last picked it. */
export const SOURCE_KEY = "ignis.transcribeSource";

const sourceChoice = viewerChoice<SourceKind>({ key: SOURCE_KEY, choices: SOURCES.map((s) => s.kind), fallback: "microphone" });

/** Why Start cannot start a listening from `source`, or `null`. A file needs no microphone, so no secure context. */
export function startBlocked(source: SourceKind, { secure, speech }: { secure: boolean; speech?: SpeechStatus }): string | null {
  if (source !== "file" && !secure) return SECURE_CONTEXT;
  return unavailable({ secure: true, speech, blocked: false })?.sentence ?? null;
}

/**
 * The hint line: what the listening last said, else, before one, why Start cannot start, or the headphones
 * sentence before a call (Q93).
 */
export function deckHint({ notice, idle, blocked, source }: { notice: ListeningState["notice"]; idle: boolean; blocked: string | null; source: SourceKind }): ListeningState["notice"] {
  if (notice) return notice;
  if (!idle) return null;
  if (blocked) return { text: blocked, tone: "ash" };
  return source === "call" ? { text: HEADPHONES, tone: "ash" } : null;
}

export function Deck({
  state,
  listening,
  actions,
  speech,
  secure,
  here,
  saved = null,
  checking = false,
  onCheck = () => {},
  onEditContext = () => {},
}: {
  state: TranscribeState;
  listening: ListeningState;
  actions: TranscribeActions;
  speech?: SpeechStatus;
  secure: boolean;
  /** The listening, if any, appends to the Transcript open. */
  here: boolean;
  /** This browser's default context, `null` when none is saved. */
  saved?: CallContext | null;
  /** The first Start's question stands for the Transcript open. */
  checking?: boolean;
  /** Raises it: the view shows the Call tab, its context framed. */
  onCheck?: () => void;
  /** Opens the Call tab on the context's draft. */
  onEditContext?: () => void;
}) {
  const source = sourceChoice.use();
  const picker = useRef<HTMLInputElement>(null);
  const checkedAt = useRef(0);
  const idle = listening.phase === "idle";
  const blocked = startBlocked(source, { secure, speech });
  const continuing = state.active.utterances.length > 0;
  const asks = asksFirst(state.active, state.draft);

  function start() {
    if (blocked) return;
    const press = startPress({ asks, checking, checkedAt: checkedAt.current, now: Date.now() });
    if (press === "check") {
      checkedAt.current = Date.now();
      return onCheck();
    }
    if (press === null) return;
    if (source === "file") return picker.current?.click();
    actions.start(source === "call" ? { kind: "call" } : { kind: "microphone" });
  }

  const hint = deckHint({ notice: listening.notice, idle, blocked, source });
  const hintLine = hint && (
    <p role="status" className={`min-w-0 text-xs leading-snug ${toneClass(hint.tone)} ${idle ? "flex-1 basis-60" : ""}`}>
      <HintText text={hint.text} />
    </p>
  );
  return (
    // The sheet's gutter, kept for its scrollbar, so Start lines up with the field above it.
    <div className="overflow-y-hidden border-t border-line bg-ground px-4 pt-3 pb-3 [scrollbar-gutter:stable] md:px-10">
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-2.5">
        {idle && checking && state.active.source === null && <ContextCheck transcript={state.active} list={state.list.transcripts} draft={state.draft} saved={saved} actions={actions} onEdit={onEditContext} />}
        {idle ? (
          <div className="flex flex-wrap items-center justify-end gap-x-5 gap-y-2.5">
            {hintLine}
            <StartButton
              label={`${source === "file" ? (continuing ? "Continue with a file" : "Choose a file") : continuing ? "Continue" : "Start"}${asks && checking ? " anyway" : ""}`}
              source={source}
              blocked={blocked !== null}
              onStart={start}
              onSource={sourceChoice.choose}
            />
            <input
              ref={picker}
              type="file"
              accept={AUDIO_FILES}
              hidden
              onChange={(e) => {
                const file = e.currentTarget.files?.[0];
                e.currentTarget.value = "";
                if (file) actions.start({ kind: "file", file });
              }}
            />
          </div>
        ) : (
          <>
            <Strip listening={listening} here={here} actions={actions} />
            {hintLine}
          </>
        )}
      </div>
    </div>
  );
}

/** What the first Start's question says the call would be told: the draft being edited, the default, or nothing. */
export function contextQuestion(t: Transcript, draft: CallContext | null, saved: CallContext | null): string {
  if (draft !== null) return `Being edited in the Call tab: ${contextSummary(draft)}. Start applies it.`;
  const c = contextOf(t, saved);
  if (isBlank(c)) return contextSummary(c);
  return `${fromDefault(t, saved) ? "Your default context" : "Told"}: ${contextSummary(c)}. Is it this call's?`;
}

/**
 * The first Start's question, over Start (the owner's word): what the call would be told, the context of the latest
 * Transcript that had another to take as a draft, and the draft to edit; the Call tab frames the context meanwhile.
 */
export function ContextCheck({
  transcript: t,
  list,
  draft,
  saved,
  actions,
  onEdit,
}: {
  transcript: Transcript;
  list: Transcript[];
  draft: CallContext | null;
  saved: CallContext | null;
  actions: Pick<TranscribeActions, "copyContext">;
  onEdit: () => void;
}) {
  const told = contextOf(t, saved);
  const theirs = draft === null ? list.find((x) => x.id !== t.id && x.context !== undefined && !isBlank(x.context) && !sameContext(x.context, told)) : undefined;
  return (
    <div role="status" className="flex flex-col gap-1 border-l-2 border-ember pl-3">
      <p className="text-[13px] leading-snug text-ink">{contextQuestion(t, draft, saved)}</p>
      <div className="flex flex-wrap gap-x-4 gap-y-1 font-display text-xs font-semibold text-ember">
        {theirs && (
          <button type="button" onClick={() => actions.copyContext(theirs.id)} className="hover:underline">
            Use the context of {titleOf(theirs)}
          </button>
        )}
        <button type="button" onClick={onEdit} className="hover:underline">
          Edit the context
        </button>
      </div>
    </div>
  );
}

/**
 * Start, and beside it what it listens to: a menu over the deck, this browser's pick, in a portal since the deck
 * clips what overflows it. The menu opens while Start cannot start, so a page without a microphone can still pick a
 * file.
 */
export function StartButton({
  label,
  source,
  blocked,
  onStart,
  onSource,
}: {
  label: string;
  source: SourceKind;
  blocked: boolean;
  onStart: () => void;
  onSource: (source: SourceKind) => void;
}) {
  const menu = useAnchoredMenu();
  const current = SOURCES.find((s) => s.kind === source) ?? SOURCES[0];
  const tone = blocked ? "bg-line text-ash" : "bg-[#ff5a1f] text-[#1c2026] hover:bg-[#ff7a45]";
  return (
    <div className="flex items-stretch">
      <button
        type="button"
        disabled={blocked}
        onClick={onStart}
        className={`cut flex items-center gap-2 py-2.5 pr-4 pl-5 font-display text-sm font-semibold [--cut-size:10px] disabled:cursor-not-allowed ${tone}`}
      >
        <IconPlay />
        {label}
      </button>
      <button
        ref={menu.button}
        type="button"
        aria-haspopup="menu"
        aria-expanded={menu.open}
        aria-label={`Source: ${current.label}`}
        title="What Start listens to"
        onClick={menu.toggle}
        className={`flex items-center gap-1.5 border-l border-[#1c2026]/25 px-3 font-display text-[13px] font-medium ${blocked ? "bg-line text-ink hover:bg-surface" : tone}`}
      >
        {current.label}
        <IconChevron className={`transition-transform motion-reduce:transition-none ${menu.open ? "rotate-180" : ""}`} />
      </button>
      {menu.open &&
        createPortal(
          <div
            ref={menu.menu}
            role="menu"
            aria-label="Source"
            style={{ right: menu.at.right, bottom: menu.at.bottom }}
            className="fixed z-40 flex w-64 max-w-[calc(100vw-2rem)] flex-col border border-line bg-surface py-1 shadow-[0_8px_24px_rgba(28,32,38,0.28)]"
            onClick={menu.close}
          >
            {SOURCES.map((s) => (
              <button
                key={s.kind}
                type="button"
                role="menuitemradio"
                aria-checked={source === s.kind}
                onClick={() => onSource(s.kind)}
                className={`flex flex-col gap-0.5 border-l-2 py-1.5 pr-3 pl-2.5 text-left hover:bg-line ${source === s.kind ? "border-ember" : "border-transparent"}`}
              >
                <span className="font-display text-[13px] font-medium text-ink">{s.label}</span>
                <span className="text-xs leading-snug text-ash">{s.detail}</span>
              </button>
            ))}
          </div>,
          document.body,
        )}
    </div>
  );
}

const LANE_LABELS: Record<LaneName, string> = { mic: "Mic", shared: "Call", file: "File" };

/** What the strip says the listening is doing. */
export function stripStatus(listening: ListeningState): string {
  switch (listening.phase) {
    case "paused":
      return "Paused";
    case "finishing":
      return "Finishing the last words";
    case "waiting":
      return "Waiting for a free session";
    default:
      return listening.file ? `Transcribing ${listening.file.name}` : "Listening";
  }
}

/** The time left as a reader says it. */
export function timeLeft(ms: number): string {
  if (ms >= 60_000) return `about ${Math.round(ms / 60_000)} min left`;
  return `about ${Math.max(1, Math.round(ms / 1_000))} s left`;
}

/** The dictation's kiln strip, for a listening: levels or a file's progress, the clock, Pause or Resume, Stop. */
export function Strip({ listening, here, actions }: { listening: ListeningState; here: boolean; actions: Pick<TranscribeActions, "pause" | "resume" | "stop"> }) {
  const { phase, file } = listening;
  const running = phase === "listening";
  return (
    <div className="cut relative flex flex-wrap items-center gap-x-4 gap-y-2 bg-kiln pt-2.5 pr-3 pb-3 pl-3.5 text-[#eae8e4] [--cut-size:10px]">
      {file ? (
        <div className="flex min-w-[10rem] flex-1 flex-col gap-1.5">
          <div
            role="progressbar"
            aria-label={`Transcribed of ${file.name}`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(file.fraction * 100)}
            className="h-1.5 w-full bg-kiln-line"
          >
            <div className="h-full bg-[#ff5a1f]" style={{ width: `${file.fraction * 100}%` }} />
          </div>
          <span className="font-display text-xs text-[#b9bec4] tabular-nums">
            {Math.round(file.fraction * 100)}% {file.leftMs !== null && file.leftMs > 0 && <span className="ml-3">{timeLeft(file.leftMs)}</span>}
          </span>
        </div>
      ) : (
        <div className="flex items-center gap-4">
          {Object.entries(listening.levels).map(([lane, levels]) => (
            <div key={lane} className="flex items-center gap-1.5">
              <span className="font-display text-xs text-[#939ba4]">
                {LANE_LABELS[lane as LaneName]}
                {lane === "mic" && listening.held && <span className="ml-1 text-[#ff8a4c]">held</span>}
              </span>
              <div aria-hidden="true" className="flex h-6 items-center gap-0.5">
                {levels.map((level, i) => (
                  <div
                    key={i}
                    className="dictation-bar w-0.5"
                    style={{ height: `${Math.max(2, Math.round(level * 22))}px`, background: running && level > 0.12 ? "#ff5a1f" : "#4a525c" }}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      <span className="font-display text-[13px] text-[#b9bec4]">
        {stripStatus(listening)}
        {!here && <span className="ml-2 text-[#939ba4]">in another transcript</span>}
      </span>
      <div className="ml-auto flex items-center gap-2.5">
        {/* Hidden from a reader: the clock would be read out every second. */}
        <span aria-hidden="true" className="flex items-center gap-1.5 font-display text-[13px] text-[#b9bec4] tabular-nums">
          {running && <span className="dictation-dot inline-block size-[7px] bg-[#ff5a1f]" />}
          {clock(listening.clockMs)}
        </span>
        {phase === "listening" && (
          <button type="button" onClick={actions.pause} className="flex items-center gap-1.5 px-2 py-1 font-display text-xs font-semibold text-[#eae8e4] hover:text-white">
            <IconPause />
            Pause
          </button>
        )}
        {phase === "paused" && (
          <button type="button" onClick={actions.resume} className="flex items-center gap-1.5 px-2 py-1 font-display text-xs font-semibold text-[#ff8a4c] hover:text-[#ffa877]">
            <IconPlay />
            Resume
          </button>
        )}
        {phase !== "waiting" && (
          <button type="button" onClick={actions.stop} className="flex items-center gap-1.5 px-2 py-1 font-display text-xs font-semibold text-[#eae8e4] hover:text-white">
            <IconStopSquare />
            {phase === "finishing" ? "Stop now" : "Stop"}
          </button>
        )}
        {phase === "waiting" && (
          <button type="button" onClick={actions.stop} className="px-2 py-1 font-display text-xs text-[#939ba4] hover:text-[#eae8e4]">
            Cancel
          </button>
        )}
      </div>
      {running && <span aria-hidden="true" className="dictation-heat absolute inset-x-0 bottom-0 h-0.5" />}
    </div>
  );
}
