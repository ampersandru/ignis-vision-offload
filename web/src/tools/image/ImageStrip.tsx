import type { ChatPicture, PictureAction } from "../../conversation/images.ts";
import { downloadImage, progressShare, sidesOf } from "../../image/studio.ts";
import { PictureTools } from "../../conversation/UserTurn.tsx";
import { IconDownload } from "../../ui/icons.tsx";
import { StatusMark } from "../agents/status.tsx";
import { type ImageRun, type ImageStatus, imageSummary, type MadePicture } from "./image.ts";

// The pictures a reply asked for, inside the reply: each call on the kiln's
// dark plate, as on the Image view's stage — the switch and the steps while
// it works, then the pictures with their names, which is how the model and
// the user both refer to them afterwards.

const STATUS_WORDS: Record<ImageStatus, string> = {
  queued: "Waiting for the other tools",
  switching: "Switching to the image model",
  making: "Making",
  returning: "Loading the text model again",
  done: "Done",
  failed: "Failed",
  stopped: "Stopped",
};

const markOf = (status: ImageStatus) =>
  status === "switching" || status === "making" || status === "returning" ? "running" : status;

type OnPicture = (picture: ChatPicture, action: PictureAction) => void;

export function ImageStrip({ runs, onPicture }: { runs: ImageRun[]; onPicture?: OnPicture }) {
  return (
    <section aria-label="Pictures" className="flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h3 className="font-display text-[13px] font-semibold text-ink">Pictures</h3>
        <p className="font-display text-xs text-ash">{imageSummary(runs)}</p>
      </div>
      <ul className="flex flex-col gap-3">
        {runs.map((run, i) => (
          <PictureCall key={`${i}:${run.callId}`} run={run} onPicture={onPicture} />
        ))}
      </ul>
    </section>
  );
}

function PictureCall({ run, onPicture }: { run: ImageRun; onPicture?: OnPicture }) {
  const made = run.made ?? [];
  const working = run.status === "switching" || run.status === "making" || run.status === "returning" || run.status === "queued";
  const share = run.progress ? progressShare(run.progress, run.count) : 0;
  return (
    <li className="monitor-kiln cut @container flex flex-col text-[#eae8e4] [--cut-size:14px]">
      <div className="flex min-w-0 items-center gap-2 px-3 py-2">
        <StatusMark status={markOf(run.status)} />
        <p className="min-w-0 flex-1 truncate text-sm" title={run.prompt}>
          {run.prompt || "make_image"}
        </p>
        <span className={`shrink-0 font-display text-xs ${run.status === "failed" ? "text-[#ff7b6b]" : "text-[#939ba4]"}`}>
          {run.line ?? STATUS_WORDS[run.status]}
        </span>
      </div>
      {working && (
        <div className="h-0.5 bg-[#2e343c]" aria-hidden>
          <div className="h-full bg-[#ff5a1f] transition-[width] motion-reduce:transition-none" style={{ width: `${Math.round(share * 100)}%` }} />
        </div>
      )}
      {(run.references.length > 0 || run.transparent || run.aspect) && (
        <p className="px-3 pb-2 text-xs text-[#939ba4]">
          {[run.references.length > 0 ? `From ${run.references.join(", ")}` : "", run.aspect ?? "", run.transparent ? "transparent" : ""]
            .filter(Boolean)
            .join(", ")}
        </p>
      )}
      {made.length > 0 && (
        // Side by side only when the reply itself is wide enough, whatever the window: the side columns narrow it.
        <ul className={`grid gap-px border-t border-[#2e343c] ${made.length > 1 ? "@2xl:grid-cols-2" : ""}`}>
          {made.map((picture) => (
            <Picture key={picture.name} picture={picture} onPicture={onPicture} />
          ))}
        </ul>
      )}
      {run.error && <p className="border-t border-[#2e343c] px-3 py-2 text-sm text-[#ff7b6b] [overflow-wrap:anywhere]">{run.error}</p>}
    </li>
  );
}

// Marking and the Image view sit over the picture, as on the user's own
// (PictureTools); the line under it wraps a whole piece at a time, so two
// pictures side by side never break a name or a seed.
function Picture({ picture, onPicture }: { picture: MadePicture; onPicture?: OnPicture }) {
  const chat: ChatPicture = { name: `${picture.name}.png`, url: picture.url };
  return (
    <li className="flex min-w-0 flex-col">
      <div className="flex justify-center bg-[#23282f] p-2">
        <span className="group/pic relative block max-w-full">
          <img src={picture.url} alt={picture.revisedPrompt ?? picture.name} className="stage-checker block max-h-[28rem] max-w-full object-contain" />
          {onPicture && <PictureTools picture={chat} onPicture={onPicture} />}
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-1.5 text-xs whitespace-nowrap text-[#939ba4] tabular-nums">
        <span className="font-display font-semibold text-[#eae8e4]">{picture.name}</span>
        <span>{sidesOf(picture.size)}</span>
        {picture.seed !== undefined && <span>Seed {picture.seed}</span>}
        {picture.kept && <span>Marked area only</span>}
        <button
          type="button"
          className="ml-auto flex items-center gap-1.5 font-display font-medium text-[#b9bec4] hover:text-white"
          onClick={() =>
            downloadImage({ id: 0, url: picture.url, prompt: picture.name, size: picture.size, edit: false, ...(picture.seed !== undefined ? { seed: picture.seed } : {}) })
          }
        >
          <IconDownload />
          PNG
        </button>
      </div>
    </li>
  );
}
