import { useEffect, useRef } from "react";
import type { ModelState } from "../api/model.ts";
import { MarkupEditor } from "../markup/MarkupEditor.tsx";
import { IconBrush, IconDownload, IconExpand, IconReference, IconRegenerate } from "../ui/icons.tsx";
import { Frame, RunLine, ShapeGlyph } from "./controls.tsx";
import { exampleDraft, PICTURE_EXAMPLES } from "./examples.ts";
import { Picture, usePictureUrl } from "./pictures.tsx";
import { type ImageSession, inputNumbers, pictureName, type PoolImage, pictureOf, type Step, wireImages } from "./sessions.ts";
import { StudioComposer } from "./StudioComposer.tsx";
import { StudioSessions, StudioSettings } from "./StudioPanels.tsx";
import { sidesOf } from "./studio.ts";
import type { ImageSessionsActions, ImageSessionsState, StudioRun } from "./useImageSessions.ts";
import { Viewer } from "./Viewer.tsx";

// The image view (image/sessions.ts says what a session is), laid out as the
// chat is: the sessions on the left, the next step's settings on the right,
// and between them the session's steps — each its prompt, the inputs it was
// sent with and the pictures it made, the last one large on the kiln as the
// stage — with the prompt box at the foot. A picture opens in the viewer, or
// in the markup editor to mark the area to change. With a text model loaded,
// Generate and Edit switch to the image model on their own (GitHub #324).

export type StudioDrawer = "sessions" | "settings" | null;

export function ImageStudio({
  model,
  state,
  actions,
  drawer,
  onDrawer,
  onSendToChat,
}: {
  model: ModelState;
  state: ImageSessionsState;
  actions: ImageSessionsActions;
  drawer: StudioDrawer;
  onDrawer: (drawer: StudioDrawer) => void;
  /** A pool picture into the chat's prompt box; absent, the viewer offers no way there. */
  onSendToChat?: (picture: PoolImage) => void;
}) {
  return (
    <div className="relative flex min-h-0 flex-1">
      {drawer && <div className="fixed inset-0 z-30 bg-[#1c2026]/60 lg:hidden" onClick={() => onDrawer(null)} aria-hidden />}
      <StudioSessions
        open={drawer === "sessions"}
        state={state}
        actions={{
          ...actions,
          select: (id) => {
            actions.select(id);
            onDrawer(null);
          },
        }}
      />
      <main aria-label="Image" className="flex min-w-0 flex-1 flex-col">
        <Timeline state={state} actions={actions} />
        <StudioComposer model={model} state={state} actions={actions} />
      </main>
      <StudioSettings open={drawer === "settings"} state={state} actions={actions} />
      {state.viewing !== null && <Viewer session={state.active} n={state.viewing} actions={actions} onSendToChat={onSendToChat} />}
      <Marking state={state} actions={actions} />
    </div>
  );
}

function Timeline({ state, actions }: { state: ImageSessionsState; actions: ImageSessionsActions }) {
  const s = state.active;
  const run = state.run?.sessionId === s.id ? state.run : null;
  const scroller = useRef<HTMLDivElement>(null);
  // A new step, a picture landing or the step moving on pulls the view down to it.
  const last = s.steps[s.steps.length - 1];
  const tail = `${s.id}:${s.steps.length}:${last?.outputs.length ?? 0}:${last?.status ?? ""}`;
  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [tail]);
  if (!state.loaded) {
    return <p className="m-auto font-display text-sm text-ash">Opening your sessions…</p>;
  }
  if (s.steps.length === 0) return <EmptyStudio state={state} actions={actions} />;
  return (
    <div ref={scroller} className="min-h-0 flex-1 overflow-y-auto">
      <ol className="mx-auto flex w-full max-w-5xl flex-col gap-10 px-4 py-6 md:px-8">
        {s.steps.map((step) => (
          <StepView key={step.n} session={s} step={step} last={step === last} run={step.status === "running" ? run : null} actions={actions} />
        ))}
      </ol>
    </div>
  );
}

/**
 * A session with no steps: what to do, and — with no inputs yet — pictures to start from, each a kind the model is
 * good at, shown by its shape. Picking one fills the prompt box and the settings; nothing is sent.
 */
function EmptyStudio({ state, actions }: { state: ImageSessionsState; actions: ImageSessionsActions }) {
  const s = state.active;
  const editing = s.draft.inputs.length > 0;
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto flex w-full max-w-4xl flex-col gap-8 px-4 py-10 md:px-8">
        <div className="flex flex-col gap-2">
          <p className="font-display text-3xl font-semibold tracking-tight text-ink">Start a picture</p>
          <p className="max-w-xl text-sm text-ash">
            {editing
              ? "Say below what should change in the inputs, or pick one of the ideas over the prompt box."
              : "Describe it below, or drop pictures on the prompt box to edit them. Each step stays in this session, and its last picture is the next step's input. Or start from one of these:"}
          </p>
        </div>
        {!editing && (
          <ul aria-label="Examples" className="grid gap-x-8 sm:grid-cols-2">
            {PICTURE_EXAMPLES.map((example) => (
              <li key={example.id}>
                <button
                  type="button"
                  title={example.prompt}
                  className="group flex w-full items-start gap-3.5 border-b border-line py-3 text-left hover:bg-surface"
                  onClick={() => actions.setDraft(exampleDraft(example, state.limits.sizes))}
                >
                  <span className={`mt-1 grid size-9 shrink-0 place-items-center text-ash group-hover:text-ember [&>span]:scale-150 ${example.background ? "checker" : ""}`}>
                    <ShapeGlyph size={example.size} />
                  </span>
                  <span className="flex min-w-0 flex-col gap-0.5">
                    <span className="font-display text-xs text-ash">
                      {example.kind}
                      {example.background ? ", transparent" : ""}
                    </span>
                    <span className="font-display text-[15px] font-semibold text-ink group-hover:text-ember">{example.title}</span>
                    <span className="line-clamp-2 text-[13px] leading-snug text-ash">{example.prompt}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function StepView({ session: s, step, last, run, actions }: { session: ImageSession; step: Step; last: boolean; run: StudioRun | null; actions: ImageSessionsActions }) {
  const numbers = inputNumbers(s, step.inputs);
  const waiting = run !== null && run.made === 0;
  const settings = step.settings;
  return (
    <li className="flex flex-col gap-3" aria-label={`Step ${step.n}`}>
      <div className="flex flex-wrap items-start gap-x-4 gap-y-2">
        <span className="pt-0.5 font-display text-xs font-semibold text-ash tabular-nums">Step {step.n}</span>
        {numbers.length > 0 && (
          <ul className="flex flex-wrap gap-1.5" aria-label="Inputs">
            {numbers.map(({ n, picture, mask }) => {
              const p = pictureOf(s, n);
              return (
                p && (
                  <li key={n} className="flex flex-col items-center gap-0.5">
                    <button type="button" className="checker block h-11 overflow-hidden border border-line hover:border-ember" onClick={() => actions.view(n)} aria-label={`Open #${n}`}>
                      <Picture id={p.id} alt={`#${n}`} className="block h-11 w-auto max-w-20 object-contain" />
                    </button>
                    <span className="font-display text-[10px] leading-none text-ash tabular-nums">
                      {mask !== undefined ? `image ${picture}+${mask}` : `image ${picture}`}
                    </span>
                  </li>
                )
              );
            })}
          </ul>
        )}
        <p className="min-w-0 flex-1 basis-64 text-[15px] leading-relaxed [overflow-wrap:anywhere]">{step.prompt}</p>
        <div className="flex items-center gap-3 font-display text-xs text-ash tabular-nums">
          <span>{settings.size === "auto" ? "Auto size" : sidesOf(settings.size)}</span>
          {settings.count > 1 && <span>{settings.count} pictures</span>}
          {settings.background !== "auto" && <span>{settings.background === "transparent" ? "Transparent" : "On white"}</span>}
          <button type="button" className="flex items-center gap-1 text-ash hover:text-ink" title="Put this step's prompt, inputs and settings back in the prompt box" onClick={() => actions.reuseStep(step.n)}>
            <IconRegenerate />
            Reuse
          </button>
        </div>
      </div>

      {step.revised && (
        <p className="border-l-2 border-line pl-3 text-sm text-ash [overflow-wrap:anywhere]">
          <span className="font-display font-medium text-ink">Rewritten as </span>
          {step.revised}
        </p>
      )}

      {waiting ? (
        <div className="monitor-kiln cut h-[min(56vh,36rem)] min-h-64 p-4 [--cut-size:18px]">
          <Frame run={run} size={run.size} />
        </div>
      ) : (
        step.outputs.length > 0 && (
          <ul className={`grid gap-3 ${step.outputs.length > 1 ? (last ? "sm:grid-cols-2" : "grid-cols-2 sm:grid-cols-4") : last ? "" : "max-w-sm"}`}>
            {step.outputs.map((n) => {
              const p = pictureOf(s, n);
              return p && <Output key={n} session={s} picture={p} large={last} actions={actions} />;
            })}
          </ul>
        )
      )}
      {run && !waiting && <RunLine run={run} />}

      {step.status === "failed" && step.error && (
        <p role="alert" className="border-l-2 border-fault pl-3 text-sm text-fault [overflow-wrap:anywhere]">
          {step.error}
        </p>
      )}
      {step.status === "stopped" && step.outputs.length === 0 && <p className="text-sm text-ash">Stopped before a picture was made.</p>}
      {step.notes?.map((note) => (
        <p key={note} className="border-l-2 border-warn pl-3 text-sm text-warn [overflow-wrap:anywhere]">
          {note}
        </p>
      ))}
    </li>
  );
}

/** A picture a step made, on the kiln: its name, whether it is an input of the next step, and what to do with it. */
function Output({ session: s, picture: p, large, actions }: { session: ImageSession; picture: PoolImage; large: boolean; actions: ImageSessionsActions }) {
  const url = usePictureUrl()(p.id, "full");
  const input = inputNumbers(s, s.draft.inputs).find((i) => i.n === p.n);
  const tool = "flex items-center gap-1.5 px-2 py-1 font-display text-xs font-medium text-[#b9bec4] hover:bg-[#2e343c] hover:text-white";
  return (
    <li className={`monitor-kiln cut flex flex-col text-[#eae8e4] [--cut-size:14px] ${input ? "outline-2 -outline-offset-2 outline-[#ff5a1f]" : ""}`}>
      <button type="button" className="flex justify-center p-2" onClick={() => actions.view(p.n)} aria-label={`Open #${p.n}`}>
        <Picture
          id={p.id}
          kind={large ? "full" : "thumb"}
          alt={p.revisedPrompt ?? `#${p.n}`}
          className={`stage-checker block max-w-full object-contain ${large ? "max-h-[min(60vh,40rem)]" : "max-h-56"}`}
        />
      </button>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-[#2e343c] px-2 py-1.5">
        <span className="pl-1 font-display text-xs font-semibold tabular-nums">#{p.n}</span>
        {input && <span className="font-display text-xs font-semibold text-[#ff8a4c]">image {input.picture} next</span>}
        {large && (
          <span className="font-display text-xs text-[#939ba4] tabular-nums">
            {p.width} × {p.height}
            {p.seed !== undefined ? `, seed ${p.seed}` : ""}
            {p.kept ? ", marked area only" : ""}
          </span>
        )}
        <span className="ml-auto flex items-center">
          <button type="button" className={tool} title={input ? "Drop it from the next step's inputs" : "Use it in the next step"} onClick={() => actions.toggleInput(p.n)}>
            <IconReference />
            {large && (input ? "Drop input" : "Use as input")}
          </button>
          <button type="button" className={tool} title="Mark the area to change" onClick={() => actions.mark(p.n)}>
            <IconBrush />
            {large && "Mark"}
          </button>
          {large && (
            <button type="button" className={tool} title="Open" onClick={() => actions.view(p.n)}>
              <IconExpand />
            </button>
          )}
          {url ? (
            <a className={tool} href={url} download={pictureName(p)} title="Download PNG">
              <IconDownload />
              {large && "PNG"}
            </a>
          ) : null}
        </span>
      </div>
    </li>
  );
}

/** The markup editor, on the picture being marked: a marked one is marked again from its source. */
function Marking({ state, actions }: { state: ImageSessionsState; actions: ImageSessionsActions }) {
  const url = usePictureUrl();
  const s = state.active;
  const p = state.marking !== null ? pictureOf(s, state.marking) : undefined;
  if (!p) return null;
  const source = p.markup ? pictureOf(s, p.markup.source) : p;
  const sourceUrl = source ? url(source.id, "full") : null;
  // The editor opens once the picture is read: the lookup renders this again then.
  if (!source || !sourceUrl) return null;
  const inputs = s.draft.inputs.includes(p.n) ? s.draft.inputs : [...s.draft.inputs, p.n];
  const numbered = inputNumbers(s, inputs).find((i) => i.n === p.n) ?? { n: p.n, picture: 1 };
  const room = p.markup?.mode === "mask" || wireImages(s, inputs).length < state.limits.maxImages;
  return (
    <MarkupEditor
      key={p.n}
      source={{ url: sourceUrl, ...(p.markup ? { marks: p.markup.marks, mode: p.markup.mode, keepOutside: p.markup.keepOutside } : {}) }}
      title={`Mark #${source.n}`}
      modes={["drawn", "mask"]}
      maskLabel={`As a mask, image ${numbered.picture + 1}`}
      maskRoom={room}
      onSave={(marked, color) => actions.saveMarks(p.n, marked, color)}
      onClose={() => actions.mark(null)}
    />
  );
}
