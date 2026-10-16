import { useId } from "react";
import { IconClose, IconPlus } from "../ui/icons.tsx";
import { Segmented } from "../ui/Segmented.tsx";
import { Switch } from "../ui/Switch.tsx";
import { caption, field } from "../ui/classes.ts";
import { BACKGROUND_HINT, BACKGROUNDS, ShapePicker } from "./controls.tsx";
import { Picture } from "./pictures.tsx";
import { coverOf, type ImageSession, storeLine, wireImages } from "./sessions.ts";
import { parseSeed } from "./studio.ts";
import type { ImageSessionsActions, ImageSessionsState } from "./useImageSessions.ts";

// The image view's two side columns, as the chat has them: the sessions on the
// left, each shown by its last picture, with what the store holds at the foot;
// the next step's settings on the right. Drawers below `lg`.

export function StudioSessions({ open, state, actions }: { open: boolean; state: ImageSessionsState; actions: ImageSessionsActions }) {
  return (
    <aside
      aria-label="Image sessions"
      className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-line bg-ground transition-transform motion-reduce:transition-none lg:static lg:z-auto lg:w-60 lg:translate-x-0 ${open ? "translate-x-0" : "max-lg:invisible max-lg:-translate-x-full"}`}
    >
      <div className="p-3">
        <button
          type="button"
          className="cut flex w-full items-center gap-2 bg-surface px-3 py-2.5 font-display text-sm font-medium text-ink hover:bg-line"
          onClick={actions.newSession}
        >
          <IconPlus />
          New session
        </button>
      </div>
      <ul className="flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto px-3 pb-3">
        {state.sessions.map((s) => (
          <SessionRow
            key={s.id}
            session={s}
            active={s.id === state.active.id}
            running={state.run?.sessionId === s.id}
            onSelect={() => actions.select(s.id)}
            onRemove={() => actions.remove(s.id)}
          />
        ))}
      </ul>
      <p className="border-t border-line px-4 py-3 text-xs leading-snug text-ash">{storeLine(state)}</p>
    </aside>
  );
}

function SessionRow({ session: s, active, running, onSelect, onRemove }: { session: ImageSession; active: boolean; running: boolean; onSelect: () => void; onRemove: () => void }) {
  const cover = coverOf(s);
  const made = s.pool.filter((p) => p.origin === "made").length;
  return (
    <li className={`group relative flex items-stretch ${active ? "bg-surface shadow-[inset_2px_0_0_var(--ember)]" : "hover:bg-surface/60"}`}>
      <button type="button" aria-current={active ? "true" : undefined} className="flex min-w-0 flex-1 items-center gap-2.5 px-2.5 py-2 text-left" onClick={onSelect}>
        <span className="checker grid size-10 shrink-0 place-items-center overflow-hidden">
          {cover ? <Picture id={cover.id} alt="" className="size-full object-cover" /> : <span className="size-2 bg-line" aria-hidden />}
        </span>
        <span className="flex min-w-0 flex-col gap-0.5">
          <span className={`truncate text-sm ${active ? "text-ink" : "text-ink/85"}`}>{s.title}</span>
          <span className="flex items-center gap-1.5 font-display text-xs text-ash">
            {running && <span className="cut size-2 bg-ember [--cut-size:3px]" aria-hidden />}
            {running ? "Making" : s.steps.length === 0 ? "No steps yet" : `${s.steps.length} ${s.steps.length === 1 ? "step" : "steps"}, ${made} made`}
          </span>
        </span>
      </button>
      <button
        type="button"
        aria-label={`Delete session: ${s.title}`}
        title="Delete session and its pictures"
        disabled={running}
        className={`grid w-8 shrink-0 place-items-center text-ash hover:text-fault focus-visible:opacity-100 disabled:hidden ${active ? "" : "opacity-0 group-hover:opacity-100 max-lg:opacity-100"}`}
        onClick={onRemove}
      >
        <IconClose />
      </button>
    </li>
  );
}

export function StudioSettings({ open, state, actions }: { open: boolean; state: ImageSessionsState; actions: ImageSessionsActions }) {
  const { draft } = state.active;
  const { limits } = state;
  const enhanceLabel = useId();
  const streamLabel = useId();
  const sameSeedLabel = useId();
  const seed = parseSeed(draft.seedText);
  const edit = wireImages(state.active, draft.inputs).length > 0;
  return (
    <aside
      aria-label="Picture settings"
      className={`fixed inset-y-0 right-0 z-40 flex w-80 flex-col gap-5 overflow-y-auto border-l border-line bg-ground px-5 py-4 transition-transform motion-reduce:transition-none lg:static lg:z-auto lg:w-80 lg:translate-x-0 ${open ? "translate-x-0" : "max-lg:invisible max-lg:translate-x-full"}`}
    >
      <ShapePicker sizes={limits.sizes} value={draft.size} edit={edit} onChange={(size) => actions.setDraft({ size })} />

      <div className="flex gap-4">
        <label className="flex w-24 flex-col gap-1.5">
          <span className={caption}>Pictures</span>
          <input
            name="n"
            type="number"
            min={1}
            max={limits.maxN}
            value={draft.count}
            onChange={(e) => actions.setDraft({ count: Number(e.target.value) || 1 })}
            className={`${field} tabular-nums`}
          />
        </label>
        <label className="flex min-w-0 flex-1 flex-col gap-1.5">
          <span className={caption}>Seed</span>
          <input
            name="seed"
            inputMode="numeric"
            value={draft.seedText}
            onChange={(e) => actions.setDraft({ seedText: e.target.value })}
            placeholder="Random"
            aria-invalid={seed === undefined}
            className={`${field} tabular-nums`}
          />
        </label>
      </div>
      {seed === undefined ? (
        <p className="-mt-3 text-sm text-fault">A seed is a whole number, or blank for a random one.</p>
      ) : (
        !draft.sameSeed && (
          <p className="-mt-3 text-[13px] text-ash">
            Blank after each step. An edit never reuses the seed behind its inputs: the same noise again builds up into grain.
          </p>
        )
      )}
      <div className={`-mt-2 flex flex-col gap-2 ${draft.sameSeed ? "border-l-2 border-fault bg-fault/10 py-2.5 pr-3 pl-3" : ""}`}>
        <div className="flex items-start gap-3">
          <Switch on={draft.sameSeed} onChange={(sameSeed) => actions.setDraft({ sameSeed })} labelledBy={sameSeedLabel} />
          <div className="flex flex-col gap-0.5 text-sm">
            <span id={sameSeedLabel} className="font-display font-medium">
              Same seed between edits
            </span>
            {!draft.sameSeed && <span className="text-ash">Off: each edit starts from fresh noise.</span>}
          </div>
        </div>
        {draft.sameSeed && (
          <p role="alert" className="text-[13px] leading-snug text-fault">
            <span className="font-display font-semibold">Not advised.</span> Each step keeps the last picture's seed, and an edit is sent the seed
            that made its own input. The model finds that noise's faint imprint in the picture and reinforces it: after two or three edits the
            picture turns to grain, and asking to remove the noise makes it worse. Use it only to reproduce a picture, or to see this happen.
          </p>
        )}
      </div>

      <div className="flex flex-col gap-1.5">
        <Segmented legend="Background" name="background" value={draft.background} options={BACKGROUNDS} onChange={(background) => actions.setDraft({ background })} />
        <p className="text-[13px] text-ash">{BACKGROUND_HINT[draft.background]}</p>
      </div>

      <div className="flex items-start gap-3">
        <Switch on={draft.enhance ?? state.enhanceDefault} onChange={(enhance) => actions.setDraft({ enhance })} labelledBy={enhanceLabel} />
        <div className="flex flex-col gap-0.5 text-sm">
          <span id={enhanceLabel} className="font-display font-medium">
            Enhance the prompt
          </span>
          <span className="text-ash">
            {limits.enhancer} rewrites it first. With the image model loaded that is a visit there and back: two model loads before each step.
            Remembered for the next session.
          </span>
        </div>
      </div>
      <div className="flex items-start gap-3">
        <Switch on={draft.stream} onChange={(stream) => actions.setDraft({ stream })} labelledBy={streamLabel} />
        <div className="flex flex-col gap-0.5 text-sm">
          <span id={streamLabel} className="font-display font-medium">
            Show progress
          </span>
          <span className="text-ash">Streams the phases and denoising steps, and each picture as it is made.</span>
        </div>
      </div>
    </aside>
  );
}
