import { useId, useState } from "react";
import { SPEECH_LANGUAGES, languageName } from "../speech/languages.ts";
import { caption, field } from "../ui/classes.ts";
import { IconClose } from "../ui/icons.tsx";
import { Segmented } from "../ui/Segmented.tsx";
import {
  addTerms,
  type CallContext,
  contextOf,
  isBlank,
  languagesLine,
  languagesUnder,
  type LanguageMode,
  pickLanguage,
  sameContext,
  termProblem,
  told,
  toldTokens,
} from "./context.ts";
import { dayAndTime, type Transcript } from "./transcript.ts";
import type { TranscribeActions } from "./useTranscribe.ts";

// The Call tab's context (spec speech/07, Q128): what the Transcript's
// transcription is told, applied or being edited. Applied, it reads as a
// record, with when it was applied; Edit opens a draft, framed in the ember,
// that nothing sends until Apply — one update, from the next utterance on —
// and that a Start applies itself. A Transcript never started shows this
// browser's default, saved from here (the owner's "Save as default").

/** "Applied at 15:44", and how much of the server's context it takes (`speech.context_max_tokens`, `maxTokens`). */
export function appliedLine(t: Transcript, maxTokens: number, saved: CallContext | null = null): string {
  const c = contextOf(t, saved);
  const tokens = toldTokens(told(c, t.speakers));
  const at = c.appliedAt == null ? "Not applied yet" : `Applied at ${dayAndTime(new Date(c.appliedAt)).time}`;
  return `${at} · about ${tokens} of ${maxTokens} tokens`;
}

/** Save as default, for `c`; that it is the default already; or, for nothing told, Clear the default. */
function SaveDefault({ context: c, saved, onSave }: { context: CallContext; saved: CallContext | null; onSave: (c: CallContext) => void }) {
  if (saved && sameContext(c, saved)) return <span className="font-display text-xs text-ash">Your default</span>;
  if (isBlank(c) && !saved) return null;
  return (
    <button type="button" onClick={() => onSave(c)} title="What a new transcript is told, in this browser" className="font-display text-xs font-semibold text-ember hover:underline">
      {isBlank(c) ? "Clear the default" : "Save as default"}
    </button>
  );
}

export function ContextCard({
  transcript: t,
  draft,
  maxTokens,
  live,
  saved = null,
  flagged = false,
  actions,
}: {
  transcript: Transcript;
  draft: CallContext | null;
  /** What the server reads of a context (`speech.context_max_tokens`). */
  maxTokens: number;
  /** A listening appends to this Transcript: an Apply holds from its next utterance. */
  live: boolean;
  /** This browser's default context, `null` when none is saved. */
  saved?: CallContext | null;
  /** The first Start asks about it: the card is framed in the ember while the question stands. */
  flagged?: boolean;
  actions: Pick<TranscribeActions, "editContext" | "changeContext" | "applyContext" | "cancelContext" | "saveDefaultContext">;
}) {
  const headingId = useId();
  const c = contextOf(t, saved);
  return (
    <section aria-labelledby={headingId} className={`flex flex-col gap-2.5 ${flagged && draft === null ? "outline-2 outline-offset-[6px] outline-ember" : ""}`}>
      <div className="flex items-baseline justify-between gap-3">
        <h2 id={headingId} className={caption}>
          Context
        </h2>
        {draft === null && (
          <button type="button" onClick={actions.editContext} className="font-display text-[13px] font-semibold text-ember hover:underline">
            {isBlank(c) ? "Add" : "Edit"}
          </button>
        )}
      </div>
      {draft !== null ? (
        <ContextForm transcript={t} draft={draft} maxTokens={maxTokens} live={live} saved={saved} actions={actions} />
      ) : isBlank(c) ? (
        <p className="text-xs leading-snug text-ash">
          Nothing told yet. Tell the transcription the topic and the words to spell right: names, products, jargon. It applies from the next utterance.
        </p>
      ) : (
        <>
          <dl className="grid grid-cols-[auto_1fr] items-baseline gap-x-3 gap-y-2 border-l-2 border-line pl-3 text-[13px]">
            {languagesLine(c.languages) !== null && (
              <>
                <dt className="font-display text-xs text-ash">Languages</dt>
                <dd className="text-ink">{languagesLine(c.languages)}</dd>
              </>
            )}
            {c.topic.trim() !== "" && (
              <>
                <dt className="font-display text-xs text-ash">Topic</dt>
                <dd className="text-ink">{c.topic}</dd>
              </>
            )}
            {c.terms.length > 0 && (
              <>
                <dt className="font-display text-xs text-ash">Terms</dt>
                <dd className="flex flex-wrap gap-1">
                  {c.terms.map((term) => (
                    <span key={term} className="border border-line bg-surface px-1.5 py-px font-display text-xs text-ink">
                      {term}
                    </span>
                  ))}
                </dd>
              </>
            )}
            {c.other.trim() !== "" && (
              <>
                <dt className="font-display text-xs text-ash">More</dt>
                <dd className="line-clamp-4 whitespace-pre-line text-ink">{c.other}</dd>
              </>
            )}
          </dl>
          <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
            <p className="text-xs text-ash">{appliedLine(t, maxTokens, saved)}</p>
            <SaveDefault context={c} saved={saved} onSave={actions.saveDefaultContext} />
          </div>
        </>
      )}
    </section>
  );
}

function ContextForm({
  transcript: t,
  draft,
  maxTokens,
  live,
  saved,
  actions,
}: {
  transcript: Transcript;
  draft: CallContext;
  maxTokens: number;
  live: boolean;
  saved: CallContext | null;
  actions: Pick<TranscribeActions, "changeContext" | "applyContext" | "cancelContext" | "saveDefaultContext">;
}) {
  const topicId = useId();
  const termsId = useId();
  const otherId = useId();
  const [typed, setTyped] = useState("");
  const problem = typed.split(",").map((term) => termProblem(term.trim())).find((p) => p !== null) ?? null;
  const tokens = toldTokens(told(draft, t.speakers));
  const set = (patch: Partial<CallContext>) => actions.changeContext({ ...draft, ...patch });
  const add = () => {
    if (typed.trim() === "" || problem) return;
    set({ terms: addTerms(draft.terms, typed) });
    setTyped("");
  };

  return (
    <div className="flex flex-col gap-3 border border-ember/70 bg-surface/40 p-3">
      <Languages languages={draft.languages} onChange={(languages) => set({ languages })} />

      <label htmlFor={topicId} className="flex flex-col gap-1">
        <span className="font-display text-xs text-ash">Topic</span>
        <input id={topicId} value={draft.topic} onChange={(e) => set({ topic: e.currentTarget.value })} placeholder="An informal IT support call" className={field} />
      </label>

      <div className="flex flex-col gap-1">
        <label htmlFor={termsId} className="font-display text-xs text-ash">
          Terms to spell right
        </label>
        {draft.terms.length > 0 && (
          <ul aria-label="Terms" className="flex flex-wrap gap-1">
            {draft.terms.map((term) => (
              <li key={term} className="flex items-center border border-line bg-surface font-display text-xs text-ink">
                <span className="py-px pl-1.5">{term}</span>
                <button type="button" aria-label={`Remove ${term}`} onClick={() => set({ terms: draft.terms.filter((x) => x !== term) })} className="px-1 text-ash hover:text-fault">
                  <IconClose />
                </button>
              </li>
            ))}
          </ul>
        )}
        <input
          id={termsId}
          value={typed}
          onChange={(e) => setTyped(e.currentTarget.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              add();
            }
          }}
          onBlur={add}
          placeholder="VPN, PostgreSQL, Kubernetes"
          className={field}
        />
        <span className={`text-xs leading-snug ${problem ? "text-fault" : "text-ash"}`}>
          {problem ?? "Enter or a comma adds one. A speaker's name you give goes with them."}
        </span>
      </div>

      <label htmlFor={otherId} className="flex flex-col gap-1">
        <span className="font-display text-xs text-ash">Anything else</span>
        <textarea
          id={otherId}
          rows={2}
          value={draft.other}
          onChange={(e) => set({ other: e.currentTarget.value })}
          placeholder="Who is on the call, what it is about."
          className={`${field} max-h-40 resize-none [field-sizing:content]`}
        />
      </label>

      <p className={`text-xs leading-snug ${tokens > maxTokens ? "text-fault" : "text-ash"}`}>
        {tokens > maxTokens ? `About ${tokens} tokens: the server reads the last ${maxTokens}, so the start is lost.` : `About ${tokens} of the ${maxTokens} tokens the server reads.`}
      </p>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={actions.applyContext}
          className="cut bg-ink px-3.5 py-1.5 font-display text-[13px] font-semibold text-ground [--cut-size:7px] hover:bg-ember"
        >
          Apply
        </button>
        <button type="button" onClick={actions.cancelContext} className="px-2 py-1.5 font-display text-[13px] text-ash hover:text-ink">
          Cancel
        </button>
        <span className="text-xs text-ash">{live ? "Holds from the next utterance." : "A Start applies it too."}</span>
        <span className="ml-auto">
          <SaveDefault context={draft} saved={saved} onSave={actions.saveDefaultContext} />
        </span>
      </div>
    </div>
  );
}

const MODES: { value: LanguageMode; label: string }[] = [
  { value: "free", label: "Detect" },
  { value: "force", label: "Force one" },
];

/** What each mode does, as the field says it. */
export function languagesHint(languages: CallContext["languages"]): string {
  if (languages.mode === "force") {
    const name = languages.codes[0] ? languageName(languages.codes[0]) : "one language";
    return `Every utterance is transcribed as ${name}: words of another language come out in its spelling.`;
  }
  return "The model detects each utterance's language; the ones listed go into the context as a hint.";
}

/** The languages: Detect, with the expected ones, or Force one. */
function Languages({ languages, onChange }: { languages: CallContext["languages"]; onChange: (languages: CallContext["languages"]) => void }) {
  const pickId = useId();
  const offered = SPEECH_LANGUAGES.filter((l) => languages.mode === "force" || !languages.codes.includes(l.code));
  return (
    <div className="flex flex-col gap-1.5">
      <span className="font-display text-xs text-ash">Languages</span>
      <Segmented legend="Languages" legendHidden name="transcribe-language-mode" value={languages.mode} options={MODES} onChange={(mode) => onChange(languagesUnder(languages, mode))} />
      <div className="flex flex-wrap items-center gap-1">
        {languages.codes.map((code) => (
          <span key={code} className="flex items-center border border-line bg-surface font-display text-xs text-ink">
            <span className="py-px pl-1.5">{languageName(code)}</span>
            <button
              type="button"
              aria-label={`Remove ${languageName(code)}`}
              onClick={() => onChange({ ...languages, codes: languages.codes.filter((c) => c !== code) })}
              className="px-1 text-ash hover:text-fault"
            >
              <IconClose />
            </button>
          </span>
        ))}
        <label htmlFor={pickId} className="sr-only">
          {languages.mode === "force" ? "The language to force" : "Add a language"}
        </label>
        <select
          id={pickId}
          value=""
          onChange={(e) => e.currentTarget.value !== "" && onChange(pickLanguage(languages, e.currentTarget.value))}
          className="cursor-pointer rounded-[2px] border border-line bg-ground px-1.5 py-0.5 font-display text-xs text-ash hover:text-ink focus:border-ember focus:outline-none"
        >
          <option value="" disabled>
            {languages.mode === "force" ? (languages.codes.length ? "Another" : "Pick one") : "Add"}
          </option>
          {offered.map((l) => (
            <option key={l.code} value={l.code}>
              {l.name}
            </option>
          ))}
        </select>
      </div>
      <span className="text-xs leading-snug text-ash">{languagesHint(languages)}</span>
    </div>
  );
}
