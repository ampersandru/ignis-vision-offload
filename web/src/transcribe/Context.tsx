import { useId, useState } from "react";
import { caption, field } from "../ui/classes.ts";
import { IconClose } from "../ui/icons.tsx";
import { addTerms, type CallContext, CONTEXT_MAX_TOKENS, EMPTY_CONTEXT, isBlank, termProblem, told, toldTokens } from "./context.ts";
import { dayAndTime, type Transcript } from "./transcript.ts";
import type { TranscribeActions } from "./useTranscribe.ts";

// The Call tab's context (spec speech/07, Q128): what the Transcript's
// transcription is told, applied or being edited. Applied, it reads as a
// record, with when it was applied; Edit opens a draft, framed in the ember,
// that nothing sends until Apply — one update, from the next utterance on —
// and that a Start applies itself.

/** "Applied at 15:44", and how much of the server's context it takes. */
export function appliedLine(t: Transcript): string {
  const tokens = toldTokens(told(t.context, t.speakers));
  const at = t.context?.appliedAt == null ? "Not applied yet" : `Applied at ${dayAndTime(new Date(t.context.appliedAt)).time}`;
  return `${at} · about ${tokens} of ${CONTEXT_MAX_TOKENS} tokens`;
}

export function ContextCard({
  transcript: t,
  draft,
  live,
  actions,
}: {
  transcript: Transcript;
  draft: CallContext | null;
  /** A listening appends to this Transcript: an Apply holds from its next utterance. */
  live: boolean;
  actions: Pick<TranscribeActions, "editContext" | "changeContext" | "applyContext" | "cancelContext">;
}) {
  const headingId = useId();
  const c = t.context ?? EMPTY_CONTEXT;
  return (
    <section aria-labelledby={headingId} className="flex flex-col gap-2.5">
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
        <ContextForm transcript={t} draft={draft} live={live} actions={actions} />
      ) : isBlank(c) ? (
        <p className="text-xs leading-snug text-ash">
          Nothing told yet. Tell the transcription the topic and the words to spell right: names, products, jargon. It applies from the next utterance.
        </p>
      ) : (
        <>
          <dl className="grid grid-cols-[auto_1fr] items-baseline gap-x-3 gap-y-2 border-l-2 border-line pl-3 text-[13px]">
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
          <p className="text-xs text-ash">{appliedLine(t)}</p>
        </>
      )}
    </section>
  );
}

function ContextForm({
  transcript: t,
  draft,
  live,
  actions,
}: {
  transcript: Transcript;
  draft: CallContext;
  live: boolean;
  actions: Pick<TranscribeActions, "changeContext" | "applyContext" | "cancelContext">;
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
          placeholder="AnyDesk, VPN, Jira"
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

      <p className={`text-xs leading-snug ${tokens > CONTEXT_MAX_TOKENS ? "text-fault" : "text-ash"}`}>
        {tokens > CONTEXT_MAX_TOKENS
          ? `About ${tokens} tokens: the server reads the last ${CONTEXT_MAX_TOKENS}, so the start is lost.`
          : `About ${tokens} of the ${CONTEXT_MAX_TOKENS} tokens the server reads.`}
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
      </div>
    </div>
  );
}
