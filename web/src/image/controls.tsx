import { useId } from "react";
import type { ImageBackground, ImageProgress } from "../api/images.ts";
import { caption } from "../ui/classes.ts";
import { AUTO_SIZE, glyphOf, progressLine, progressShare, shapeOf, sidesOf } from "./studio.ts";

// The image view's pieces that are not about sessions: the shape picker drawn
// at each size's ratio, the backgrounds and what each does, and the frame of a
// picture being made — filling with heat as the stream reports the steps —
// with the line that says where the run is.

/** What the frame and the run line need to know of a run. */
export type RunShape = { n: number; edit: boolean; switches: boolean; progress: ImageProgress | null; size: string };

export const BACKGROUNDS: { value: ImageBackground; label: string }[] = [
  { value: "auto", label: "As made" },
  { value: "transparent", label: "Transparent" },
  { value: "opaque", label: "On white" },
];

export const BACKGROUND_HINT: Record<ImageBackground, string> = {
  auto: "The alpha the model makes, usually none.",
  transparent: "Wraps the prompt in the model's own wording for a transparent picture.",
  opaque: "Any transparency is flattened onto white.",
};

/** The sizes the server serves, each drawn at its own ratio; `auto` first. */
export function ShapePicker({ sizes, value, edit, onChange }: { sizes: string[]; value: string; edit: boolean; onChange: (size: string) => void }) {
  const name = useId();
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className={`${caption} mb-2`}>Shape</legend>
      <div className="grid grid-cols-3 gap-1.5">
        {[AUTO_SIZE, ...sizes].map((size) => {
          const shape = shapeOf(size);
          return (
            <label key={size} className="relative">
              <input type="radio" name={name} value={size} className="peer sr-only" checked={value === size} onChange={() => onChange(size)} />
              <span className="flex h-full cursor-pointer items-center gap-2 border border-line bg-surface px-2 py-1.5 text-ash peer-checked:border-ink peer-checked:bg-ink peer-checked:text-ground peer-focus-visible:outline-2 peer-focus-visible:outline-ember hover:border-ash">
                <ShapeGlyph size={size} />
                <span className="flex min-w-0 flex-col">
                  <span className="font-display text-[13px] leading-tight font-semibold">{shape ? shape.ratio : "Auto"}</span>
                  <span className="truncate text-[11px] leading-tight tabular-nums opacity-80">{shape ? sidesOf(size) : edit ? "like image 1" : "2048 × 2048"}</span>
                </span>
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

export function ShapeGlyph({ size }: { size: string }) {
  const shape = shapeOf(size);
  if (!shape) return <span aria-hidden className="block size-[18px] shrink-0 border border-dashed border-current" />;
  const glyph = glyphOf(shape, 18);
  return (
    <span aria-hidden className="grid size-[18px] shrink-0 place-items-center">
      <span className="block border-[1.5px] border-current" style={{ width: glyph.width, height: glyph.height }} />
    </span>
  );
}

/** The shape about to be made, drawn at its ratio: empty at rest, filling with heat while the run fires. */
export function Frame({ run, size, idle }: { run: RunShape | null; size: string; idle?: string }) {
  const shape = shapeOf(size) ?? { width: 1, height: 1, ratio: "1:1" };
  const share = run?.progress ? progressShare(run.progress, run.n) : 0;
  const ratio = shape.width / shape.height;
  return (
    <div className="stage-frame flex size-full items-center justify-center">
      <div
        className="cut relative flex flex-col items-center justify-center gap-2 border border-dashed border-[#3e454f] [--cut-size:14px]"
        style={{ width: `min(100cqw, calc(100cqh * ${ratio}))`, aspectRatio: `${shape.width} / ${shape.height}` }}
      >
        {run && <div aria-hidden className="stage-heat absolute inset-x-0 bottom-0" style={{ height: `${Math.round(share * 100)}%` }} />}
        <span className="relative font-display text-2xl font-semibold text-[#eae8e4]/80 tabular-nums md:text-3xl">{shapeOf(size) ? sidesOf(size) : "Auto"}</span>
        <span className="relative px-6 text-center text-sm text-[#939ba4]">{run ? <RunLine run={run} bare /> : idle}</span>
        {run?.progress?.phase === "denoising" && run.progress.total > 0 && <Steps step={run.progress.step} total={run.progress.total} />}
      </div>
    </div>
  );
}

/** One mark per denoising step: lit up to the one under way. */
export function Steps({ step, total }: { step: number; total: number }) {
  return (
    <div aria-hidden className="absolute inset-x-4 bottom-4 flex gap-1">
      {Array.from({ length: total }, (_, i) => (
        <span key={i} className={`h-1 flex-1 ${i < step ? "bg-[#ff5a1f]" : "bg-[#3e454f]"}`} />
      ))}
    </div>
  );
}

/** Where the run is, in words; outside the frame, with its bar. */
export function RunLine({ run, bare = false }: { run: RunShape; bare?: boolean }) {
  const words = run.progress
    ? progressLine(run.progress, run.n)
    : `Sending the ${run.edit ? "edit" : "request"}${run.switches ? ": the image model is loaded before the job…" : "…"}`;
  if (bare) {
    return (
      <span role="status" aria-live="polite">
        {words}
      </span>
    );
  }
  const share = run.progress ? progressShare(run.progress, run.n) : 0;
  return (
    <div role="status" aria-live="polite" className="flex flex-col gap-1.5">
      <p className="font-display text-[13px] font-medium text-[#eae8e4]">{words}</p>
      <div className="h-1 overflow-hidden bg-[#2e343c]" aria-hidden>
        <div className="h-full bg-[#ff5a1f] transition-[width] motion-reduce:transition-none" style={{ width: `${Math.round(share * 100)}%` }} />
      </div>
    </div>
  );
}
