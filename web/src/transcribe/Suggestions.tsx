import { useId } from "react";
import { caption } from "../ui/classes.ts";
import { IconClose } from "../ui/icons.tsx";
import type { AnalystHandle, AnalystState } from "./analyst.ts";
import { useWatched } from "./assist.ts";
import { languageName } from "../speech/languages.ts";
import { fixWhere, openSuggestions, type Suggestion } from "./suggestions.ts";
import type { Transcript } from "./transcript.ts";
import type { TranscribeActions, TranscribeState } from "./useTranscribe.ts";

// The suggestions waiting in the Call tab (spec speech/07, Q132): each a card
// with what it proposes and why, its verb, Always (accept, and every later one
// of its kind in this Transcript) and Dismiss. Suggest now reads the
// Transcript on the spot, a listening or not.

/** What a card proposes, and the verb that accepts it. */
export function suggestionText(s: Suggestion): { headline: string; verb: string } {
  switch (s.kind) {
    case "languages": {
      const names = s.languages.codes.map(languageName).join(" and ");
      return { headline: s.languages.mode === "force" ? `Force ${names}` : `Detect ${names}`, verb: "Apply" };
    }
    case "topic":
      return { headline: `Topic: ${s.topic}`, verb: "Use" };
    case "title":
      return { headline: `Rename to “${s.title}”`, verb: "Rename" };
    case "terms":
      return { headline: s.terms.length === 1 ? "A term to spell right" : "Terms to spell right", verb: s.terms.length === 1 ? "Add it" : "Add them" };
    case "fix":
      return { headline: `“${s.from}” is ${s.to}`, verb: "Fix" };
  }
}

type SuggestionActions = Pick<TranscribeActions, "acceptSuggestion" | "dismissSuggestion" | "dropSuggestedTerm" | "suggestNow" | "stopSuggesting">;

export function Suggestions(props: { transcript: Transcript; state: TranscribeState; actions: SuggestionActions; analyst: AnalystHandle }) {
  return <SuggestionsPanel {...props} looking={useWatched(props.analyst)} />;
}

export function SuggestionsPanel({
  transcript: t,
  state,
  actions,
  looking,
}: {
  transcript: Transcript;
  state: TranscribeState;
  actions: SuggestionActions;
  looking: AnalystState;
}) {
  const headingId = useId();
  const open = openSuggestions(t);
  const here = looking.looking === t.id;
  const nothing = t.utterances.every((u) => u.text.trim() === "");
  const failed = looking.failed?.transcript === t.id && !here ? looking.failed.message : null;
  return (
    <section aria-labelledby={headingId} className="flex flex-col gap-2.5">
      <div className="flex items-baseline justify-between gap-3">
        <h2 id={headingId} className={caption}>
          Suggestions
        </h2>
        <button
          type="button"
          disabled={!here && (state.textBlocked !== null || nothing || looking.looking !== null)}
          onClick={here ? actions.stopSuggesting : actions.suggestNow}
          className="font-display text-[13px] font-semibold text-ember hover:underline disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:no-underline"
        >
          {here ? "Stop" : "Suggest now"}
        </button>
      </div>
      {open.length === 0 ? (
        <p className="text-xs leading-snug text-ash">
          {here
            ? "The text model is reading the transcript."
            : state.suggestionsOn
              ? "While listening, the languages come from the server, and about every minute and a half the text model suggests the topic, a title and terms to spell right."
              : "Suggestions while listening are off in Settings. Suggest now still reads the transcript."}
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {open.map((s) => (
            <li key={s.id}>
              <Card transcript={t} suggestion={s} actions={actions} />
            </li>
          ))}
        </ul>
      )}
      {failed && (
        <p role="status" className="border-l-2 border-fault pl-2 text-xs leading-snug text-fault">
          No suggestions: {failed}
        </p>
      )}
    </section>
  );
}

function Card({ transcript, suggestion: s, actions }: { transcript: Transcript; suggestion: Suggestion; actions: SuggestionActions }) {
  const { headline, verb } = suggestionText(s);
  return (
    <article className="flex flex-col gap-2 border-l-2 border-ember bg-surface px-3 py-2.5">
      <p className="font-display text-[13px] font-semibold leading-snug text-ink">{headline}</p>
      {s.kind === "languages" && <p className="text-xs leading-snug text-ash">{s.why}</p>}
      {s.kind === "fix" && (
        <p className="text-xs leading-snug text-ash">Misheard {fixWhere(transcript, s)}. Fixed, it is marked in the transcript and can be undone; {s.to} becomes a term.</p>
      )}
      {s.kind === "terms" && (
        <ul aria-label="Suggested terms" className="flex flex-wrap gap-1">
          {s.terms.map((term) => (
            <li key={term} className="flex items-center border border-line bg-ground font-display text-xs text-ink">
              <span className="py-px pl-1.5">{term}</span>
              <button type="button" aria-label={`Not ${term}`} onClick={() => actions.dropSuggestedTerm(s.id, term)} className="px-1 text-ash hover:text-fault">
                <IconClose />
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="flex flex-wrap items-center gap-1.5">
        <button
          type="button"
          onClick={() => actions.acceptSuggestion(s.id)}
          className="cut bg-ink px-3 py-1 font-display text-xs font-semibold text-ground [--cut-size:6px] hover:bg-ember"
        >
          {verb}
        </button>
        <button
          type="button"
          title="Accept it, and every later suggestion of this kind in this transcript, without asking"
          onClick={() => actions.acceptSuggestion(s.id, true)}
          className="border border-line px-2.5 py-[3px] font-display text-xs font-medium text-ink hover:border-ember"
        >
          Always
        </button>
        <button type="button" onClick={() => actions.dismissSuggestion(s.id)} className="px-1.5 py-1 font-display text-xs text-ash hover:text-ink">
          Dismiss
        </button>
      </div>
    </article>
  );
}
