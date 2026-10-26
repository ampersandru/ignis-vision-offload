import { useId, useRef, useState } from "react";
import type { SpeechStatus } from "../api/model.ts";
import { SECURE_CONTEXT, unavailable } from "../speech/availability.ts";
import { HintText, toneClass } from "../speech/Dictation.tsx";
import { caption, field } from "../ui/classes.ts";
import { IconPause, IconPlay, IconStopSquare } from "../ui/icons.tsx";
import { Switch } from "../ui/Switch.tsx";
import { clock } from "./export.ts";
import type { LaneName, ListeningState } from "./listening.ts";
import type { SourceKind } from "./transcript.ts";
import type { TranscribeActions, TranscribeState } from "./useTranscribe.ts";

// The Transcribe view's deck, at the foot of the sheet as the chat's composer
// is (spec speech/07): the context field, then, before a listening, the source,
// the Speakers toggle and Start; while one runs, the dictation's kiln strip —
// each source's level, the clock, Pause or Resume, Stop — or a file's
// progress and the time left. The hint line under it says what the server or
// the browser refused, in spec 04's words, and before a call that headphones
// keep the far voices out of the microphone (Q93).

/** Said before a call starts (Q93). */
export const HEADPHONES = "Use headphones on a call: through speakers, the far voices reach the microphone and are transcribed twice.";

/** The files a pick offers: what `decodeAudioData` reads in a browser, an mp4's audio included (Q94). */
export const AUDIO_FILES = "audio/*,video/mp4,video/webm,.m4a,.mp3,.ogg,.opus,.wav,.webm,.mp4";

const SOURCES: [SourceKind, string][] = [
  ["microphone", "Microphone"],
  ["call", "Call"],
  ["file", "File"],
];

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
}: {
  state: TranscribeState;
  listening: ListeningState;
  actions: TranscribeActions;
  speech?: SpeechStatus;
  secure: boolean;
  /** The listening, if any, appends to the Transcript open. */
  here: boolean;
}) {
  const [source, setSource] = useState<SourceKind>("microphone");
  const picker = useRef<HTMLInputElement>(null);
  const contextId = useId();
  const speakersId = useId();
  const idle = listening.phase === "idle";
  const blocked = startBlocked(source, { secure, speech });
  const continuing = state.active.utterances.length > 0;

  function start() {
    if (blocked) return;
    if (source === "file") return picker.current?.click();
    actions.start(source === "call" ? { kind: "call" } : { kind: "microphone" });
  }

  const hint = deckHint({ notice: listening.notice, idle, blocked, source });
  return (
    <div className="border-t border-line bg-ground px-4 pt-3 pb-3 md:px-10">
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-2.5">
        <label htmlFor={contextId} className="flex flex-col gap-1">
          <span className={caption}>Context</span>
          <textarea
            id={contextId}
            rows={1}
            value={state.context}
            onChange={(e) => actions.setContext(e.currentTarget.value)}
            placeholder="The topic, the names, the jargon: what the transcription should spell right."
            className={`${field} max-h-24 resize-none [field-sizing:content]`}
          />
        </label>

        {idle ? (
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2.5">
            <div role="radiogroup" aria-label="Source" className="flex">
              {SOURCES.map(([kind, label]) => (
                <button
                  key={kind}
                  type="button"
                  role="radio"
                  aria-checked={source === kind}
                  onClick={() => setSource(kind)}
                  className={`border border-line px-3 py-1.5 font-display text-[13px] font-medium not-first:-ml-px ${source === kind ? "relative z-10 border-ember bg-surface text-ink" : "bg-ground text-ash hover:text-ink"}`}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2.5">
              {state.speakersUnavailable ? (
                <span aria-disabled="true" className="pointer-events-none flex opacity-45">
                  <Switch on={false} onChange={() => {}} labelledBy={speakersId} />
                </span>
              ) : (
                <Switch on={state.speakers} onChange={actions.setSpeakers} labelledBy={speakersId} />
              )}
              <span id={speakersId} className={`font-display text-[13px] font-medium ${state.speakersUnavailable ? "text-ash line-through" : "text-ink"}`}>
                Speakers
              </span>
            </div>
            <button
              type="button"
              disabled={blocked !== null}
              onClick={start}
              className="cut ml-auto flex items-center gap-2 bg-[#ff5a1f] px-5 py-2.5 font-display text-sm font-semibold text-[#1c2026] [--cut-size:10px] hover:bg-[#ff7a45] disabled:cursor-not-allowed disabled:bg-line disabled:text-ash"
            >
              <IconPlay />
              {source === "file" ? (continuing ? "Continue with a file" : "Choose a file") : continuing ? "Continue" : "Start"}
            </button>
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
          <Strip listening={listening} here={here} actions={actions} />
        )}

        {(hint || (idle && state.speakersUnavailable)) && (
          <div className="flex flex-col gap-0.5 text-xs leading-snug">
            {hint && (
              <p role="status" className={toneClass(hint.tone)}>
                <HintText text={hint.text} />
              </p>
            )}
            {idle && state.speakersUnavailable && (
              <p className="text-ash">
                <HintText text={state.speakersUnavailable} />
              </p>
            )}
          </div>
        )}
      </div>
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
              <span className="font-display text-xs text-[#939ba4]">{LANE_LABELS[lane as LaneName]}</span>
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
