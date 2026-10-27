import { memo, type ReactNode, useLayoutEffect, useRef, useState } from "react";
import { isAtBottom } from "../ui/scroll.ts";
import { clock, startedLine, startOf } from "./export.ts";
import type { LiveUtterance, ListeningState } from "./listening.ts";
import { titleOf } from "./sessions.ts";
import type { TranscribeActions } from "./useTranscribe.ts";
import { fixedWordsOf, type Speaker, type Transcript, type Utterance } from "./transcript.ts";

// The Transcript as it is heard (spec speech/07): its heading (drawn by the
// view above the sheet, with the exports, so it stays in sight), then its
// utterances hung on a time spine — each one's mm:ss to the left of the
// rule, its speaker's hue as a tick along it — and the utterances under way at
// the foot, their provisional tail light (Q82), the spine burning with the
// ember there while it listens. The view follows the newest words until the
// reader scrolls up, and offers Jump to live then (Q102). Every completed
// utterance's text is editable inline, and its speaker a menu: a move, a new
// speaker, and a split at the cursor (Q92, Q96, Q98).

/** A speaker's hue: Me in ink, Others in ash, each voice one of six (styles.css). */
export function speakerColor(speaker: Speaker | undefined): string {
  if (!speaker) return "var(--line)";
  if (speaker.role === "me") return "var(--ink)";
  if (speaker.role === "others") return "var(--ash)";
  return `var(--speaker-${((speaker.id - 1) % 6) + 1})`;
}

/** The label of a call's utterance under way, before the server names its speaker. */
const laneLabel = (lane: LiveUtterance["lane"], call: boolean) => (call ? (lane === "mic" ? "Me" : "Call") : null);

/**
 * The sheet. It is handed only what it draws, not the listening's whole state, and redraws only when that changes:
 * the levels move ten times a second and the clock every second.
 */
export const Sheet = memo(function Sheet({
  transcript,
  live,
  phase,
  call,
  here,
  actions,
}: {
  transcript: Transcript;
  /** The utterances under way of the listening appending here; none when it appends elsewhere. */
  live: LiveUtterance[];
  phase: ListeningState["phase"];
  call: boolean;
  /** The listening appends to this Transcript. */
  here: boolean;
  actions: Pick<TranscribeActions, "moveUtterance" | "splitUtterance" | "editText" | "undoFixes">;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const [following, setFollowing] = useState(true);
  const speakers = new Map(transcript.speakers.map((s) => [s.id, s]));

  useLayoutEffect(() => setFollowing(true), [transcript.id]);
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && following) el.scrollTop = el.scrollHeight;
  }, [transcript.utterances, live, following]);

  const jump = () => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
    setFollowing(true);
  };

  const empty = transcript.utterances.length === 0 && live.length === 0;
  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <div ref={scroller} onScroll={(e) => setFollowing(isAtBottom(e.currentTarget))} className="min-h-0 flex-1 overflow-y-auto px-4 py-6 md:px-10">
        <div className="mx-auto w-full max-w-3xl">
          {empty ? (
            <Empty listening={here && phase !== "idle"} />
          ) : (
            <ol aria-label="Utterances">
              {transcript.utterances.map((u) => (
                <UtteranceRow key={u.id} utterance={u} speaker={u.speaker === null ? undefined : speakers.get(u.speaker)} speakers={transcript.speakers} actions={actions} />
              ))}
              {live.map((u) => (
                <LiveRow key={u.key} utterance={u} label={laneLabel(u.lane, call)} burning={phase === "listening" || phase === "finishing"} />
              ))}
            </ol>
          )}
        </div>
      </div>
      {!following && here && phase !== "idle" && (
        <button
          type="button"
          onClick={jump}
          className="cut absolute bottom-3 left-1/2 -translate-x-1/2 bg-ember px-4 py-2 font-display text-[13px] font-semibold text-[#1c2026] [--cut-size:8px] hover:bg-[#ff7a45]"
        >
          Jump to live
        </button>
      )}
    </div>
  );
});

/** The Transcript's title, editable, its start, what was heard and by how many; `children` at the title's right. */
export function Heading({ transcript, onRename, children }: { transcript: Transcript; onRename: (title: string) => void; children?: ReactNode }) {
  const speaking = transcript.speakers.length;
  return (
    <header className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <input
          key={`${transcript.id}:${transcript.title}`}
          aria-label="Transcript title"
          defaultValue={titleOf(transcript)}
          onBlur={(e) => e.currentTarget.value.trim() !== titleOf(transcript) && onRename(e.currentTarget.value)}
          onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
          className="-ml-1 min-w-[12rem] flex-1 rounded-[2px] border border-transparent bg-transparent px-1 font-display text-2xl font-semibold text-ink hover:border-line focus:border-ember focus:outline-none"
        />
        {children}
      </div>
      <p className="flex flex-wrap gap-x-5 gap-y-1 font-display text-[13px] text-ash">
        {transcript.startedAt !== null && <span>{startedLine(transcript.startedAt)}</span>}
        {transcript.clockMs > 0 && <span className="tabular-nums">{clock(transcript.clockMs)} heard</span>}
        {speaking > 0 && <span>{speaking === 1 ? "1 speaker" : `${speaking} speakers`}</span>}
      </p>
    </header>
  );
}

function Empty({ listening }: { listening: boolean }) {
  return (
    <p className="mt-10 max-w-[60ch] text-[15px] leading-relaxed text-ash">
      {listening
        ? "Listening. Each utterance shows here as it is heard, and settles when its text is complete."
        : "Nothing heard yet. Pick a source below and start: the microphone hears the room, a call adds the audio of a tab, a window or the screen, and a file is transcribed faster than real time."}
    </p>
  );
}

/** The spine and the time beside it: one row of the sheet. */
function Row({ time, tick, label, children }: { time: string; tick: { color?: string; burning?: boolean }; label: ReactNode; children: ReactNode }) {
  return (
    <li className="grid grid-cols-[3.75rem_1fr] gap-x-4 sm:grid-cols-[4.5rem_1fr]">
      <span className="pt-[3px] text-right font-display text-[13px] text-ash tabular-nums">{time}</span>
      <div className="relative border-l-2 border-line pb-5 pl-4">
        <span
          aria-hidden="true"
          className={`absolute top-1 bottom-5 -left-[3px] w-1 ${tick.burning ? "spine-live" : ""}`}
          style={tick.burning ? undefined : { background: tick.color }}
        />
        {label}
        {children}
      </div>
    </li>
  );
}

/** `text` with the words fixes put in it marked (Q133), each where it stands as words of its own. */
export function marked(text: string, fixed: string[]): ReactNode {
  if (fixed.length === 0) return text;
  // The pattern's one group makes every odd part a word it matched.
  return text.split(fixedWordsOf(fixed)).map((part, i) =>
    i % 2 === 1 ? (
      <mark key={i} className="bg-transparent text-ink underline decoration-ember decoration-dotted decoration-2 underline-offset-4">
        {part}
      </mark>
    ) : (
      part
    ),
  );
}

/** An utterance of the sheet: drawn again only when it, its speaker or the speakers change. */
const UtteranceRow = memo(function UtteranceRow({
  utterance: u,
  speaker,
  speakers,
  actions,
}: {
  utterance: Utterance;
  speaker: Speaker | undefined;
  speakers: Speaker[];
  actions: Pick<TranscribeActions, "moveUtterance" | "splitUtterance" | "editText" | "undoFixes">;
}) {
  const [editing, setEditing] = useState(false);
  const cursor = useRef(0);
  const color = speakerColor(speaker);
  const label = speaker && (
    <SpeakerMenu
      value={speaker.id}
      speakers={speakers}
      color={color}
      ariaLabel={`Speaker of the utterance at ${startOf(u)}`}
      onChoose={(to) => actions.moveUtterance(u.id, to)}
    />
  );
  return (
    <Row time={startOf(u)} tick={{ color }} label={label}>
      {editing ? (
        <div className="flex flex-col gap-1.5">
          <textarea
            aria-label={`Text of the utterance at ${startOf(u)}`}
            autoFocus
            value={u.text}
            onChange={(e) => actions.editText(u.id, e.currentTarget.value)}
            onSelect={(e) => (cursor.current = e.currentTarget.selectionStart)}
            onBlur={(e) => {
              // Moving to the split menu keeps the editor open: the cursor it splits at is here.
              if (!e.currentTarget.parentElement?.contains(e.relatedTarget as Node | null)) setEditing(false);
            }}
            onKeyDown={(e) => e.key === "Escape" && setEditing(false)}
            className="w-full resize-none rounded-[2px] border border-line bg-surface px-2.5 py-1.5 text-[15px] leading-relaxed text-ink [field-sizing:content] focus:border-ember focus:outline-none"
          />
          <SpeakerMenu
            value={null}
            speakers={speakers}
            prompt="Split at the cursor, the rest to…"
            ariaLabel="Split this utterance at the cursor"
            onChoose={(to) => {
              actions.splitUtterance(u.id, cursor.current, to);
              setEditing(false);
            }}
          />
        </div>
      ) : (
        <div>
          <p
            tabIndex={0}
            title="Edit the text"
            onClick={() => setEditing(true)}
            onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), setEditing(true))}
            className="cursor-text rounded-[2px] text-[15px] leading-relaxed whitespace-pre-wrap text-ink hover:bg-surface/70"
          >
            {u.text ? marked(u.text, u.fixed ?? []) : <span className="text-ash">(no text)</span>}
          </p>
          {u.heard !== undefined && (
            <button
              type="button"
              title={`Heard as: ${u.heard}`}
              onClick={() => actions.undoFixes(u.id)}
              className="font-display text-xs text-ash hover:text-ember"
            >
              Undo fix
            </button>
          )}
        </div>
      )}
    </Row>
  );
});

/**
 * A speaker as a native menu: the speakers, then a new one. Native, so it opens over a clipped `.cut` box and on a
 * phone. With `value` null it is an action menu that resets after each choice.
 */
function SpeakerMenu({
  value,
  speakers,
  color,
  prompt,
  ariaLabel,
  onChoose,
}: {
  value: number | null;
  speakers: Speaker[];
  color?: string;
  prompt?: string;
  ariaLabel: string;
  onChoose: (to: number | "new") => void;
}) {
  return (
    <select
      aria-label={ariaLabel}
      value={value === null ? "" : String(value)}
      onChange={(e) => {
        const chosen = e.currentTarget.value;
        if (chosen !== "") onChoose(chosen === "new" ? "new" : Number(chosen));
      }}
      style={color ? { color } : undefined}
      className={`max-w-full cursor-pointer appearance-none rounded-[2px] bg-transparent py-0.5 pr-1 font-display text-[13px] font-semibold hover:bg-surface focus:outline-none focus-visible:outline-2 focus-visible:outline-ember ${color ? "" : "self-start border border-line px-2 text-ash"}`}
    >
      {value === null && (
        <option value="" disabled>
          {prompt}
        </option>
      )}
      {speakers.map((s) => (
        <option key={s.id} value={s.id}>
          {s.name}
        </option>
      ))}
      <option value="new">New speaker</option>
    </select>
  );
}

function LiveRow({ utterance: u, label, burning }: { utterance: LiveUtterance; label: string | null; burning: boolean }) {
  const heard = (u.stable + u.tail).trim() !== "";
  return (
    <Row time={clock(u.startMs)} tick={{ burning }} label={label && <span className="font-display text-[13px] font-semibold text-ash">{label}</span>}>
      <p aria-live="polite" className="text-[15px] leading-relaxed">
        {heard ? (
          <>
            <span className="text-ink">{u.stable}</span>
            <span className="text-ash">{u.tail}</span>
          </>
        ) : (
          <span className="text-ash">…</span>
        )}
      </p>
    </Row>
  );
}
