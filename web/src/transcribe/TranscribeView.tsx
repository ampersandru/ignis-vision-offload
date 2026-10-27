import { type ReactNode, useEffect, useState } from "react";
import type { SpeechStatus } from "../api/model.ts";
import { caption } from "../ui/classes.ts";
import { IconDownload, IconPlus, IconTrash } from "../ui/icons.tsx";
import { Deck } from "./Deck.tsx";
import { Notes } from "./Notes.tsx";
import type { NotesHandle } from "./notesWriter.ts";
import { toMarkdown, toSrt, toText, toVtt } from "./export.ts";
import { chooseFilePace, FILE_PACE_CHOICES, useFilePace } from "./filePace.ts";
import { SILENCE_CHOICES, useUtteranceSilence } from "./silence.ts";
import { type ListeningHandle, type ListeningState, type LiveUtterance, useListening } from "./listening.ts";
import { Sheet, speakerColor } from "./Sheet.tsx";
import { titleOf, type TranscriptList } from "./sessions.ts";
import { sessionsWithSpeakers, type Transcript } from "./transcript.ts";
import type { TranscribeActions, TranscribeState } from "./useTranscribe.ts";

// The Transcribe view (spec speech/07), in the Decide tab's three-column
// shape: the Transcripts on the left, the one open in the middle with the deck
// at its foot, and on the right two tabs: its notes (Notes.tsx), and its
// speakers, its exports, Open in chat and the view's settings.
// Below `lg` the two side columns are drawers the header opens. The state is
// the page's (useTranscribe.ts): a listening goes on while the view is hidden.

/** The utterances under way of a listening that appends elsewhere: one array, so the sheet is not drawn again. */
const NONE_LIVE: LiveUtterance[] = [];

/** Which side drawer is out, below `lg`. */
export type Drawer = "sessions" | "settings" | null;

export function TranscribeView({
  state,
  actions,
  listening: handle,
  notes,
  speech,
  drawer,
  onDrawer,
  onOpenInChat,
}: {
  state: TranscribeState;
  actions: TranscribeActions;
  listening: ListeningHandle;
  notes: NotesHandle;
  speech?: SpeechStatus;
  drawer: Drawer;
  onDrawer: (drawer: Drawer) => void;
  /** A new chat, its composer seeded with the Transcript as Markdown; nothing sent (Q89). */
  onOpenInChat: (markdown: string) => void;
}) {
  // Outside https and localhost the browser gives no page a microphone; a test has no such flag.
  const secure = typeof globalThis.isSecureContext === "boolean" ? globalThis.isSecureContext : true;
  const listening = useListening(handle);
  const here = listening.transcript === state.active.id;

  useEffect(() => {
    if (!drawer) return;
    const close = (e: globalThis.KeyboardEvent) => e.key === "Escape" && onDrawer(null);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [drawer, onDrawer]);

  return (
    <div className="relative flex min-h-0 flex-1">
      {drawer && <div className="fixed inset-0 z-30 bg-[#1c2026]/60 lg:hidden" onClick={() => onDrawer(null)} aria-hidden />}

      <Transcripts
        list={state.list}
        listening={listening}
        open={drawer === "sessions"}
        notice={state.storeError}
        onNew={() => {
          actions.newTranscript();
          onDrawer(null);
        }}
        onSelect={(id) => {
          actions.select(id);
          onDrawer(null);
        }}
        onRemove={actions.remove}
      />

      <main aria-label="Transcript" className="flex min-w-0 flex-1 flex-col">
        <Sheet transcript={state.active} live={here ? listening.live : NONE_LIVE} phase={listening.phase} call={listening.source === "call"} here={here} actions={actions} />
        <Deck state={state} listening={listening} actions={actions} speech={speech} secure={secure} here={here} />
      </main>

      <Side open={drawer === "settings"}>
        {(tab) =>
          tab === "notes" ? (
            <Notes transcript={state.active} state={state} actions={actions} notes={notes} />
          ) : (
            <Tools transcript={state.active} actions={actions} onOpenInChat={onOpenInChat} />
          )
        }
      </Side>
    </div>
  );
}

/** What a Transcript's row says under its title. */
export function transcriptSummary(t: Transcript, listening: ListeningState): string {
  if (listening.transcript === t.id) return listening.phase === "paused" ? "Paused" : "Listening";
  const n = t.utterances.length;
  return n === 0 ? "Nothing heard yet" : n === 1 ? "1 utterance" : `${n} utterances`;
}

function Transcripts({
  list,
  listening,
  open,
  notice,
  onNew,
  onSelect,
  onRemove,
}: {
  list: TranscriptList;
  listening: ListeningState;
  open: boolean;
  notice: string | null;
  onNew: () => void;
  onSelect: (id: number) => void;
  onRemove: (id: number) => void;
}) {
  return (
    <aside
      aria-label="Transcripts"
      className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-line bg-ground transition-transform motion-reduce:transition-none lg:static lg:z-auto lg:w-60 lg:translate-x-0 ${open ? "translate-x-0" : "max-lg:invisible max-lg:-translate-x-full"}`}
    >
      <div className="p-3">
        <button type="button" className="cut flex w-full items-center gap-2 bg-surface px-3 py-2.5 font-display text-sm font-medium text-ink hover:bg-line" onClick={onNew}>
          <IconPlus />
          New transcript
        </button>
      </div>
      <ul className="flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto px-3 pb-3">
        {list.transcripts.map((t) => {
          const active = t.id === list.activeId;
          const heard = listening.transcript === t.id;
          return (
            <li key={t.id} className={`group relative flex items-stretch ${active ? "bg-surface shadow-[inset_2px_0_0_var(--ember)]" : "hover:bg-surface/60"}`}>
              <button type="button" aria-current={active ? "true" : undefined} className="flex min-w-0 flex-1 flex-col gap-0.5 px-3 py-2 text-left" onClick={() => onSelect(t.id)}>
                <span className={`truncate text-sm ${active ? "text-ink" : "text-ink/85"}`}>{titleOf(t)}</span>
                <span className="flex items-center gap-1.5 font-display text-xs text-ash">
                  {heard && <span className="cut size-2 bg-ember [--cut-size:3px]" aria-hidden />}
                  {transcriptSummary(t, listening)}
                </span>
              </button>
              <button
                type="button"
                aria-label={`Delete transcript: ${titleOf(t)}`}
                title="Delete transcript"
                disabled={heard}
                className={`grid w-8 shrink-0 place-items-center text-ash hover:text-fault focus-visible:opacity-100 disabled:hidden ${active ? "" : "opacity-0 group-hover:opacity-100 max-lg:opacity-100"}`}
                onClick={() => onRemove(t.id)}
              >
                <IconTrash />
              </button>
            </li>
          );
        })}
      </ul>
      {notice && (
        <p role="status" className="border-t border-line px-3 py-2 text-xs text-fault">
          {notice}
        </p>
      )}
    </aside>
  );
}

/** A file name from a title: what the downloads are saved as. */
export const fileStem = (title: string) => title.replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-+|-+$/g, "").slice(0, 60) || "transcript";

const EXPORTS: { label: string; extension: string; type: string; write: (t: Transcript) => string }[] = [
  { label: ".txt", extension: "txt", type: "text/plain", write: toText },
  { label: ".md", extension: "md", type: "text/markdown", write: toMarkdown },
  { label: ".srt", extension: "srt", type: "application/x-subrip", write: toSrt },
  { label: ".vtt", extension: "vtt", type: "text/vtt", write: toVtt },
];

function download(t: Transcript, { extension, type, write }: (typeof EXPORTS)[number]) {
  const url = URL.createObjectURL(new Blob([write(t)], { type: `${type};charset=utf-8` }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `${fileStem(titleOf(t))}.${extension}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

type SideTab = "notes" | "tools";

/** The right panel, in two tabs as the chat's settings are: Notes and Tools. A drawer below `lg`, shown when `open`. */
function Side({ open, children }: { open: boolean; children: (tab: SideTab) => ReactNode }) {
  const [tab, setTab] = useState<SideTab>("notes");
  return (
    <aside
      aria-label="Notes and tools"
      className={`fixed inset-y-0 right-0 z-40 flex w-80 flex-col overflow-y-auto border-l border-line bg-ground px-5 pb-5 transition-transform motion-reduce:transition-none lg:static lg:z-auto lg:translate-x-0 xl:w-96 ${open ? "translate-x-0" : "max-lg:invisible max-lg:translate-x-full"}`}
    >
      <div role="tablist" aria-label="Panel sections" className="sticky top-0 z-10 -mx-5 mb-5 flex gap-5 border-b border-line bg-ground px-5 pt-4">
        {(["notes", "tools"] as const).map((t) => (
          <button
            key={t}
            type="button"
            role="tab"
            id={`transcribe-tab-${t}`}
            aria-selected={tab === t}
            aria-controls="transcribe-tab-panel"
            onClick={() => setTab(t)}
            className={`-mb-px border-b-2 pb-2 font-display text-[13px] font-semibold ${tab === t ? "border-ember text-ink" : "border-transparent text-ash hover:text-ink"}`}
          >
            {t === "notes" ? "Notes" : "Tools"}
          </button>
        ))}
      </div>
      <div id="transcribe-tab-panel" role="tabpanel" aria-labelledby={`transcribe-tab-${tab}`} className="flex flex-1 flex-col gap-6">
        {children(tab)}
      </div>
    </aside>
  );
}

function Tools({ transcript: t, actions, onOpenInChat }: { transcript: Transcript; actions: TranscribeActions; onOpenInChat: (markdown: string) => void }) {
  const [copied, setCopied] = useState(false);
  const pace = useFilePace();
  const silence = useUtteranceSilence();
  const nothing = t.utterances.every((u) => u.text.trim() === "");
  const counts = new Map<number, number>();
  for (const u of t.utterances) if (u.speaker !== null) counts.set(u.speaker, (counts.get(u.speaker) ?? 0) + 1);

  return (
    <>
      <section aria-labelledby="transcribe-speakers" className="flex flex-col gap-2.5">
        <h2 id="transcribe-speakers" className={caption}>
          Speakers
        </h2>
        {t.speakers.length === 0 ? (
          <p className="text-xs leading-snug text-ash">Each voice the diarizer tells apart gets a name here, which you can change.</p>
        ) : (
          <ul className="flex flex-col gap-2">
            {t.speakers.map((s) => (
              <li key={s.id} className="flex items-center gap-2">
                <span aria-hidden="true" className="h-6 w-1 shrink-0" style={{ background: speakerColor(s) }} />
                <input
                  key={`${s.id}:${s.name}`}
                  aria-label={`Name of ${s.name}`}
                  defaultValue={s.name}
                  onBlur={(e) => e.currentTarget.value.trim() !== s.name && actions.renameSpeaker(s.id, e.currentTarget.value)}
                  onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
                  className="min-w-0 flex-1 rounded-[2px] border border-transparent bg-transparent px-1.5 py-1 font-display text-[13px] font-semibold text-ink hover:border-line focus:border-ember focus:outline-none"
                />
                <span className="shrink-0 font-display text-xs text-ash tabular-nums" title="Utterances">
                  {counts.get(s.id) ?? 0}
                </span>
                {t.speakers.length > 1 && (
                  <select
                    aria-label={`Merge ${s.name} into another speaker`}
                    value=""
                    onChange={(e) => e.currentTarget.value !== "" && actions.mergeSpeakers(s.id, Number(e.currentTarget.value))}
                    className="w-[5.5rem] shrink-0 cursor-pointer rounded-[2px] border border-line bg-ground px-1 py-1 font-display text-xs text-ash hover:text-ink focus:border-ember focus:outline-none"
                  >
                    <option value="" disabled>
                      Merge into
                    </option>
                    {t.speakers
                      .filter((other) => other.id !== s.id)
                      .map((other) => (
                        <option key={other.id} value={other.id}>
                          {other.name}
                        </option>
                      ))}
                  </select>
                )}
              </li>
            ))}
          </ul>
        )}
        {sessionsWithSpeakers(t) > 1 && (
          <p className="text-xs leading-snug text-ash">A Resume or a Continue numbers the voices anew: merge a speaker who came back into the one they were.</p>
        )}
      </section>

      <section aria-labelledby="transcribe-export" className="flex flex-col gap-2.5">
        <h2 id="transcribe-export" className={caption}>
          Export
        </h2>
        <div className="flex flex-wrap gap-1.5">
          <button
            type="button"
            disabled={nothing}
            onClick={() => {
              void navigator.clipboard?.writeText(toText(t)).then(() => {
                setCopied(true);
                setTimeout(() => setCopied(false), 2_000);
              });
            }}
            className="cut bg-ink px-3 py-1.5 font-display text-[13px] font-semibold text-ground [--cut-size:7px] hover:bg-ember disabled:cursor-not-allowed disabled:opacity-50"
          >
            {copied ? "Copied" : "Copy"}
          </button>
          {EXPORTS.map((format) => (
            <button
              key={format.extension}
              type="button"
              disabled={nothing}
              title={`Download as ${format.label}`}
              onClick={() => download(t, format)}
              className="flex items-center gap-1 border border-line px-2.5 py-1.5 font-display text-[13px] font-medium text-ink hover:border-ember disabled:cursor-not-allowed disabled:opacity-50"
            >
              <IconDownload />
              {format.label}
            </button>
          ))}
        </div>
        <button
          type="button"
          disabled={nothing}
          onClick={() => onOpenInChat(toMarkdown(t))}
          className="self-start font-display text-[13px] font-semibold text-ember hover:underline disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:no-underline"
        >
          Open in chat
        </button>
        <p className="text-xs leading-snug text-ash">A new chat opens with the transcript in its prompt box. Nothing is sent until you press Send.</p>
      </section>

      <section aria-labelledby="transcribe-silence" className="flex flex-col gap-2">
        <h2 id="transcribe-silence" className={caption}>
          Utterance ends after
        </h2>
        <div role="radiogroup" aria-labelledby="transcribe-silence" className="flex">
          {SILENCE_CHOICES.map((ms) => (
            <button
              key={ms}
              type="button"
              role="radio"
              aria-checked={silence === ms}
              onClick={() => actions.setSilence(ms)}
              className={`border border-line px-3 py-1 font-display text-[13px] not-first:-ml-px ${silence === ms ? "relative z-10 border-ember bg-surface text-ink" : "text-ash hover:text-ink"}`}
            >
              {ms / 1_000} s
            </button>
          ))}
        </div>
        <p className="text-xs leading-snug text-ash">The silence that ends an utterance. Shorter cuts a speaker at every breath, and the model guesses the language of each piece on its own.</p>
      </section>

      <section aria-labelledby="transcribe-file-pace" className="flex flex-col gap-2">
        <h2 id="transcribe-file-pace" className={caption}>
          File pace
        </h2>
        <div role="radiogroup" aria-labelledby="transcribe-file-pace" className="flex">
          {FILE_PACE_CHOICES.map((seconds) => (
            <button
              key={seconds}
              type="button"
              role="radio"
              aria-checked={pace === seconds}
              onClick={() => chooseFilePace(seconds)}
              className={`border border-line px-3 py-1 font-display text-[13px] not-first:-ml-px ${pace === seconds ? "relative z-10 border-ember bg-surface text-ink" : "text-ash hover:text-ink"}`}
            >
              {seconds} s
            </button>
          ))}
        </div>
        <p className="text-xs leading-snug text-ash">How far ahead of its transcript a file is sent. Further ahead finishes sooner and holds more of the server's queue.</p>
      </section>
    </>
  );
}
