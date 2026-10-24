import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { ModelState } from "../api/model.ts";
import { DictationMic, DictationStrip, HintText } from "../speech/Dictation.tsx";
import { dictationHint } from "../speech/dictation.ts";
import { type Dictation, useDictationState } from "../speech/useDictation.ts";
import { useAnchoredMenu } from "../ui/anchoredMenu.ts";
import { IconBrush, IconClose, IconImage, IconPlus } from "../ui/icons.tsx";
import { ideasFor } from "./examples.ts";
import { Picture } from "./pictures.tsx";
import { inputNumbers, type ImageSession, pictureOf, wireImages } from "./sessions.ts";
import { ACCEPTED_IMAGES, paramsOf, parseSeed, runPlan } from "./studio.ts";
import type { ImageSessionsActions, ImageSessionsState } from "./useImageSessions.ts";

// The foot of the image view, as the chat's prompt box is the foot of the
// chat: the next step's inputs — each named as the prompt calls it, image 1,
// image 2 — the pool to add more from or a picture to upload, the prompt, and
// Generate or Edit (or the switch it makes first). Pictures dropped anywhere
// on it join the pool as inputs. The mic beside the run button dictates into
// the prompt, and the run waits until the dictation ends (spec speech/04).

export function StudioComposer({
  model,
  state,
  actions,
  dictation,
}: {
  model: ModelState;
  state: ImageSessionsState;
  actions: ImageSessionsActions;
  dictation: Dictation;
}) {
  const s = state.active;
  const { draft } = s;
  const { limits } = state;
  const [over, setOver] = useState(false);
  const busy = state.run !== null;
  const edit = wireImages(s, draft.inputs).length > 0;
  const plan = runPlan(model, limits, state.imageModel, draft.enhance ?? state.enhanceDefault, edit, busy);
  // Only whether it can be sent: the files are read from the store when it is.
  const sendable = paramsOf(draft, [], limits) !== null;
  const speech = useDictationState(dictation);
  const dictating = speech.phase !== "idle";
  const said = dictationHint(speech, "Generate", false);
  // A dictation is never sent by itself: Generate waits until it ends.
  const canRun = state.loaded && plan.ready && !busy && sendable && !dictating;
  const runningHere = state.run?.sessionId === s.id;
  const seedBad = parseSeed(draft.seedText) === undefined;
  return (
    <div
      className="shrink-0 px-4 pb-4 md:px-8"
      onKeyDown={(e) => dictation.controller.keyDown(e)}
      onKeyUp={(e) => dictation.controller.keyUp(e)}
      onDragOver={(e) => {
        if (![...e.dataTransfer.types].includes("Files")) return;
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        actions.addFiles(Array.from(e.dataTransfer.files));
      }}
    >
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-2">
        <Inputs session={s} max={limits.maxImages} actions={actions} />
        {draft.prompt.trim() === "" && ideasFor(draft.inputs.length).length > 0 && (
          <div aria-label="Ideas" className="flex flex-wrap items-center gap-1.5">
            <span className="pr-1 font-display text-xs text-ash">Ideas</span>
            {ideasFor(draft.inputs.length).map((idea) => (
              <button
                key={idea.id}
                type="button"
                title={idea.prompt}
                className="cut bg-surface px-2.5 py-1 font-display text-xs font-medium text-ink [--cut-size:5px] hover:bg-line hover:text-ember"
                onClick={() => actions.setDraft({ prompt: idea.prompt, ...(idea.background ? { background: idea.background } : {}) })}
              >
                {idea.label}
              </button>
            ))}
          </div>
        )}
        {dictating && <DictationStrip dictation={dictation} state={speech} />}
        <form
          aria-label="Next step"
          className={`cut flex items-end gap-2 bg-surface p-2 shadow-[inset_0_-2px_0_var(--line)] [--cut-size:14px] focus-within:shadow-[inset_0_-2px_0_var(--ember)] ${over ? "outline-2 outline-ember outline-dashed" : ""}`}
          onSubmit={(e) => {
            e.preventDefault();
            if (canRun) actions.run();
          }}
        >
          <PoolButton session={s} actions={actions} />
          <textarea
            ref={dictation.field}
            name="prompt"
            aria-label="Prompt"
            rows={1}
            value={draft.prompt}
            onChange={(e) => actions.setDraft({ prompt: e.target.value })}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                e.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder={edit ? (draft.inputs.length > 1 ? "Put the jacket from image 2 on the person in image 1" : "What should change in image 1") : "Describe the picture"}
            className="max-h-48 min-h-10 flex-1 resize-none bg-transparent px-2 py-2 text-[15px] leading-normal text-ink [field-sizing:content] placeholder:text-ash focus:outline-none"
          />
          <DictationMic dictation={dictation} state={speech} />
          {/* Two buttons, never one changing type: the run ends while Stop's click is still being dispatched, and a
              button that is a submit by then would send the form again — the step would start over. */}
          {runningHere ? (
            <button
              key="stop"
              type="button"
              className="cut bg-[#c8161d] px-5 py-2.5 font-display text-sm font-semibold text-white hover:bg-[#a8121a]"
              onClick={(e) => {
                e.preventDefault();
                actions.stop();
              }}
            >
              Stop
            </button>
          ) : (
            <button
              key="send"
              type="submit"
              disabled={!canRun}
              className="cut bg-[#ff5a1f] px-5 py-2.5 font-display text-sm font-semibold text-[#1c2026] hover:bg-[#ff7a45] disabled:cursor-default disabled:bg-line disabled:text-ash"
            >
              {plan.label}
            </button>
          )}
        </form>
        <p className="min-h-4 px-1 text-xs text-ash [overflow-wrap:anywhere]">
          {said ? (
            <span className={said.tone === "fault" ? "text-fault" : said.tone === "warn" ? "text-warn" : undefined}>
              <HintText text={said.text} />
            </span>
          ) : state.notice ? (
            <span className="text-warn">{state.notice}</span>
          ) : draft.sameSeed && edit ? (
            <span className="font-medium text-fault">Same seed between edits is on: not advised, the picture builds up grain. Turn it off in the settings.</span>
          ) : seedBad ? (
            <span className="text-fault">The seed in the settings is not a whole number.</span>
          ) : busy && !runningHere ? (
            "Another session is making pictures. Its step ends first."
          ) : speech.unavailable ? (
            <HintText text={speech.unavailable.sentence} />
          ) : (
            (plan.note ?? "Enter sends. Shift+Enter starts a new line. Drop pictures here to add them.")
          )}
        </p>
      </div>
    </div>
  );
}

/** The next step's inputs, as the prompt names them; a marked one can be marked again. */
function Inputs({ session: s, max, actions }: { session: ImageSession; max: number; actions: ImageSessionsActions }) {
  const numbers = inputNumbers(s, s.draft.inputs);
  if (numbers.length === 0) {
    return <p className="px-1 text-xs text-ash">No inputs: the next step makes a new picture. Add pictures from the pool, or drop them here, to edit them.</p>;
  }
  return (
    <div className="flex flex-wrap items-end gap-2" aria-label="Inputs">
      {numbers.map(({ n, picture, mask }) => {
        const p = pictureOf(s, n);
        if (!p) return null;
        return (
          <span key={n} className="cut flex items-center gap-2 bg-surface p-1 pr-1.5 [--cut-size:8px]">
            <span className="checker block h-12 overflow-hidden">
              <Picture id={p.id} alt={`#${n}`} className="block h-12 w-auto max-w-24 object-contain" />
            </span>
            <span className="flex flex-col">
              <span className="font-display text-[13px] leading-tight font-semibold text-ink">
                image {picture}
                {mask !== undefined && <span className="text-ash"> + mask {mask}</span>}
              </span>
              <span className="font-display text-[11px] leading-tight text-ash tabular-nums">
                #{n}
                {p.markup ? ", marked" : ""}
              </span>
            </span>
            <button type="button" aria-label={`Mark #${n}`} title="Mark the area to change" className={`grid size-7 place-items-center hover:text-ink ${p.markup ? "text-ember" : "text-ash"}`} onClick={() => actions.mark(n)}>
              <IconBrush />
            </button>
            <button type="button" aria-label={`Drop #${n} from the inputs`} className="grid size-7 place-items-center text-ash hover:text-fault" onClick={() => actions.toggleInput(n)}>
              <IconClose />
            </button>
          </span>
        );
      })}
      <span className="pb-1 font-display text-[11px] text-ash tabular-nums">
        {wireImages(s, s.draft.inputs).length} of {max}
      </span>
    </div>
  );
}

/** The `+` of the prompt box: the session's pool to pick inputs from, and an upload. */
function PoolButton({ session: s, actions }: { session: ImageSession; actions: ImageSessionsActions }) {
  const menu = useAnchoredMenu();
  const input = useRef<HTMLInputElement>(null);
  return (
    <div className="shrink-0">
      <button
        ref={menu.button}
        type="button"
        aria-label="Add inputs"
        title="Add inputs from the pool, or upload pictures"
        aria-haspopup="menu"
        aria-expanded={menu.open}
        className={`grid size-10 place-items-center hover:text-ink [&>svg]:size-[18px] ${menu.open ? "text-ink" : "text-ash"}`}
        onClick={menu.toggle}
      >
        <IconPlus />
      </button>
      <input
        ref={input}
        type="file"
        name="pictures"
        accept={ACCEPTED_IMAGES.join(",")}
        multiple
        hidden
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          e.target.value = "";
          if (files.length > 0) actions.addFiles(files);
        }}
      />
      {menu.open &&
        createPortal(
          <div
            ref={menu.menu}
            role="menu"
            aria-label="Pool"
            style={{ left: menu.at.left, bottom: menu.at.bottom }}
            className="cut fixed z-40 flex w-[24rem] max-w-[calc(100vw-2rem)] flex-col gap-2 bg-surface p-3 shadow-[0_8px_24px_rgba(28,32,38,0.28)] [--cut-size:10px]"
          >
            <p className="font-display text-[13px] font-medium text-ash">
              Pool <span className="tabular-nums">{s.pool.length}</span>
            </p>
            {s.pool.length > 0 ? (
              <ul className="grid max-h-80 grid-cols-4 gap-1.5 overflow-y-auto">
                {[...s.pool].reverse().map((p) => {
                  const on = s.draft.inputs.includes(p.n);
                  return (
                    <li key={p.n}>
                      <button
                        type="button"
                        role="menuitemcheckbox"
                        aria-checked={on}
                        aria-label={`#${p.n}`}
                        className={`checker relative block aspect-square w-full overflow-hidden border-2 ${on ? "border-ember" : "border-transparent hover:border-ash"}`}
                        onClick={() => actions.toggleInput(p.n)}
                      >
                        <Picture id={p.id} alt="" className="size-full object-cover" />
                        <span className="absolute bottom-0 left-0 bg-kiln px-1 font-display text-[11px] font-semibold text-[#eae8e4] tabular-nums">#{p.n}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p className="text-xs text-ash">The pool is empty: the pictures this session makes or is given land here.</p>
            )}
            <button
              type="button"
              role="menuitem"
              className="flex items-center gap-2 self-start py-1 font-display text-sm font-semibold text-ink hover:text-ember"
              onClick={() => {
                menu.close();
                input.current?.click();
              }}
            >
              <IconImage />
              Upload pictures
            </button>
          </div>,
          document.body,
        )}
    </div>
  );
}
