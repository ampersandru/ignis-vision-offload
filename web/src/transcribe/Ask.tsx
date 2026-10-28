import { useState } from "react";
import { Markdown } from "../ui/Markdown.tsx";
import { type AskerState, quickQuestions } from "./ask.ts";
import { clock } from "./export.ts";
import { jumpOnClick, linkTimes } from "./jump.ts";
import type { Transcript } from "./transcript.ts";
import type { TranscribeActions, TranscribeState } from "./useTranscribe.ts";

// Ask about the conversation, in its flow (spec speech/07, Q139, Q141): a bar
// at the sheet's foot, where the eyes are during a call, with the questions a
// call asks most a tap away; the answer stands in the sheet where it was
// asked, on the time spine after the last utterance it read, so what it knew
// is plain, and the conversation goes on under it.

/** The bar under the sheet: the quick questions, the field, and why it cannot ask when it cannot. */
export function AskBar({
  transcript: t,
  state,
  actions,
  asking,
  onPreview = () => {},
}: {
  transcript: Transcript;
  state: TranscribeState;
  actions: Pick<TranscribeActions, "askAbout" | "stopAsking" | "clearMissed">;
  asking: AskerState;
  /** What did I miss? is pointed at or focused: the sheet tints what it would read. */
  onPreview?: (on: boolean) => void;
}) {
  const [question, setQuestion] = useState("");
  const here = asking.transcript === t.id;
  const writing = here && asking.writing;
  // One question at a time for the page: while another Transcript's is answered, this one waits, its question kept.
  const elsewhere = asking.writing && !here;
  const cannot = state.textBlocked !== null || elsewhere;
  // Nothing heard, nothing to ask about: the bar comes with the first utterance.
  if (t.utterances.every((u) => u.text.trim() === "")) return null;
  // The sheet's foot, not a band of its own: no rule (the deck's is the footer's one), and from `sm` the field in
  // the sheet's text column, where its answer will stand, the quick questions beside it. It keeps the gutter the
  // sheet keeps for its scrollbar, so the two columns line up.
  return (
    <section aria-label="Ask about the conversation" className="overflow-y-hidden bg-ground px-4 pt-1.5 pb-3 [scrollbar-gutter:stable] md:px-10">
      <div className="mx-auto w-full max-w-3xl sm:grid sm:grid-cols-[4.5rem_1fr] sm:gap-x-4">
        <div className="flex min-w-0 flex-col gap-2 sm:col-start-2 sm:pl-[18px]">
          {state.textBlocked && <p className="border-l-2 border-ember pl-2 text-xs leading-snug text-ink">{state.textBlocked}</p>}
          <div className="flex flex-wrap items-center gap-1.5">
            <form
              className="flex min-w-[min(100%,16rem)] flex-1 gap-1.5"
              onSubmit={(e) => {
                e.preventDefault();
                if (writing) return actions.stopAsking();
                if (question.trim() === "" || cannot) return;
                actions.askAbout(question);
                setQuestion("");
              }}
            >
              <input
                aria-label="Question about the conversation"
                value={question}
                onChange={(e) => setQuestion(e.currentTarget.value)}
                placeholder="Ask about the conversation"
                className="h-8 min-w-0 flex-1 rounded-[2px] border border-line bg-surface px-2.5 text-[14px] text-ink placeholder:text-ash focus:border-ember focus:outline-none"
              />
              <button
                type="submit"
                disabled={!writing && (cannot || question.trim() === "")}
                className={`cut h-8 px-3.5 font-display text-[13px] font-semibold [--cut-size:7px] disabled:cursor-not-allowed disabled:opacity-50 ${writing ? "bg-surface text-ink hover:bg-line" : "bg-ink text-ground hover:bg-ember"}`}
              >
                {writing ? "Stop" : "Ask"}
              </button>
            </form>
            {!state.textBlocked && (
              // On a phone one row under the field, scrolled sideways, so the sheet keeps its height.
              <div className="flex gap-1.5 max-sm:w-full max-sm:overflow-x-auto">
                {quickQuestions(t, state.missedFrom).map((q) => (
                  <button
                    key={q.label}
                    type="button"
                    title={q.missed ? `${q.question} A time on the left picks another moment; Escape goes back to the last 5 minutes.` : q.question}
                    disabled={writing || cannot}
                    onClick={() => {
                      actions.askAbout(q.question, q.sent);
                      // Asked: the moment it read from is spent, and the next reads the last five minutes again.
                      if (q.missed) actions.clearMissed();
                    }}
                    {...(q.missed && {
                      onMouseEnter: () => onPreview(true),
                      onMouseLeave: () => onPreview(false),
                      onFocus: () => onPreview(true),
                      onBlur: () => onPreview(false),
                    })}
                    className={`flex h-8 shrink-0 items-center border px-2.5 font-display text-xs font-medium text-ink hover:border-ember disabled:cursor-not-allowed disabled:opacity-50 ${q.missed && state.missedFrom ? "border-ember/60" : "border-line"}`}
                  >
                    {q.label}
                    {q.detail && <span className="ml-1.5 font-normal text-ash">{q.detail}</span>}
                  </button>
                ))}
              </div>
            )}
          </div>
          {elsewhere && <p className="text-xs leading-snug text-ash">A question about another transcript is being answered; ask this one when it is done.</p>}
          {here && asking.failed && (
            <p role="status" className="border-l-2 border-fault pl-2 text-xs leading-snug text-fault">
              No answer: {asking.failed}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

/**
 * An answer on the time spine: the moment it read up to in the time column, in the ember, a hollow diamond on the
 * rule, and the question with its answer beside them; the rule burns while it streams. Its `[mm:ss]` are links that
 * bring their utterance into view.
 */
export function AnswerAside({
  question,
  answer,
  untilMs,
  writing = false,
  onForget,
}: {
  question: string;
  answer: string;
  untilMs: number | undefined;
  writing?: boolean;
  onForget?: () => void;
}) {
  const until = untilMs === undefined ? null : clock(untilMs);
  return (
    <li aria-label={`Answer to: ${question}`} aria-busy={writing} className="grid grid-cols-[3.75rem_1fr] gap-x-4 sm:grid-cols-[4.5rem_1fr]">
      <span title={until === null ? undefined : `It read the conversation up to ${until}`} className="pt-[11px] text-right font-display text-[13px] text-ember tabular-nums">
        {until}
      </span>
      <div className="relative border-l-2 border-line pb-5 pl-4">
        <span aria-hidden="true" className="absolute top-[14px] -left-[6px] size-2.5 rotate-45 border-2 border-ember bg-ground" />
        {writing && <span aria-hidden="true" className="spine-live absolute top-7 bottom-5 -left-[3px] w-1" />}
        <article onClickCapture={jumpOnClick} className="cut flex flex-col gap-1.5 bg-surface px-3.5 pt-2.5 pb-3 [--cut-size:10px]">
          <div className="flex items-baseline gap-3">
            <h3 className="flex-1 font-display text-[14px] font-semibold leading-snug text-ink">{question}</h3>
            {onForget && (
              <button type="button" aria-label={`Forget: ${question}`} onClick={onForget} className="shrink-0 font-display text-xs text-ash hover:text-fault">
                Forget
              </button>
            )}
          </div>
          {answer === "" ? (
            <p className="text-xs text-ash">{until === null ? "Reading the conversation." : `Reading the conversation up to ${until}.`}</p>
          ) : (
            <div className="notes notes-answer">
              <Markdown text={linkTimes(answer)} streaming={writing} />
            </div>
          )}
        </article>
      </div>
    </li>
  );
}
