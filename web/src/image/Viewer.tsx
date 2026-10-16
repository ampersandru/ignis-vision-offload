import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { IconBrush, IconChevron, IconClose, IconDownload, IconReference, IconRegenerate } from "../ui/icons.tsx";
import { Picture, usePictureUrl } from "./pictures.tsx";
import { type ImageSession, inputNumbers, pictureName, type PoolImage, pictureOf } from "./sessions.ts";
import type { ImageSessionsActions } from "./useImageSessions.ts";

// One pool picture full-screen on the kiln, at its own resolution: what made
// it, and the ways on from it. The arrows walk the pool; Escape closes. A
// portal on the body, for the `.cut` clip-path the rest of the page lives in.

export function Viewer({
  session: s,
  n,
  actions,
  onSendToChat,
}: {
  session: ImageSession;
  n: number;
  actions: ImageSessionsActions;
  onSendToChat?: (picture: PoolImage) => void;
}) {
  const p = pictureOf(s, n);
  const url = usePictureUrl();
  const dialog = useRef<HTMLDivElement>(null);
  const index = s.pool.findIndex((x) => x.n === n);
  const go = (step: number) => {
    const next = s.pool[(index + step + s.pool.length) % s.pool.length];
    if (next) actions.view(next.n);
  };

  useEffect(() => dialog.current?.focus(), []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") actions.view(null);
      if (e.key === "ArrowLeft") go(-1);
      if (e.key === "ArrowRight") go(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (!p) return null;
  const step = p.step !== undefined ? s.steps.find((st) => st.n === p.step) : undefined;
  const input = inputNumbers(s, s.draft.inputs).find((i) => i.n === p.n);
  const full = url(p.id, "full");
  const origin = { upload: p.name ? `Uploaded: ${p.name}` : "Uploaded", made: step ? `Made by step ${step.n}` : "Made", marked: p.markup ? `#${p.markup.source} marked` : "Marked", chat: "From the chat" }[p.origin];
  const tool = "flex items-center gap-1.5 px-3 py-2 font-display text-[13px] font-medium text-[#b9bec4] hover:bg-[#2e343c] hover:text-white";
  return createPortal(
    <div ref={dialog} role="dialog" aria-modal="true" aria-label={`#${p.n}`} tabIndex={-1} className="monitor-kiln fixed inset-0 z-50 flex flex-col text-[#eae8e4] outline-none">
      <div className="flex items-center gap-4 border-b border-[#2e343c] px-4 py-3 md:px-6">
        <h2 className="font-display text-lg font-semibold tabular-nums">#{p.n}</h2>
        <p className="min-w-0 flex-1 truncate text-sm text-[#b9bec4]">
          {origin}
          {step ? `: ${step.prompt}` : ""}
        </p>
        <span className="font-display text-xs text-[#939ba4] tabular-nums">
          {p.width} × {p.height}
          {p.seed !== undefined ? `, seed ${p.seed}` : ""}
        </span>
        <button type="button" aria-label="Close" className="grid size-9 place-items-center hover:bg-[#2e343c]" onClick={() => actions.view(null)}>
          <IconClose />
        </button>
      </div>
      <div className="relative flex min-h-0 flex-1 items-center justify-center p-4">
        <Picture id={p.id} kind="full" alt={p.revisedPrompt ?? step?.prompt ?? `#${p.n}`} className="stage-checker block max-h-full max-w-full object-contain" />
        {s.pool.length > 1 && (
          <>
            <button type="button" aria-label="Previous picture" className="absolute top-1/2 left-3 grid size-10 -translate-y-1/2 place-items-center bg-[#1c2026]/70 hover:bg-[#2e343c]" onClick={() => go(-1)}>
              <IconChevron className="-rotate-90" />
            </button>
            <button type="button" aria-label="Next picture" className="absolute top-1/2 right-3 grid size-10 -translate-y-1/2 place-items-center bg-[#1c2026]/70 hover:bg-[#2e343c]" onClick={() => go(1)}>
              <IconChevron className="rotate-90" />
            </button>
          </>
        )}
      </div>
      <div className="flex flex-wrap items-center justify-center gap-1 border-t border-[#2e343c] px-4 py-2">
        <button type="button" className={tool} onClick={() => actions.toggleInput(p.n)}>
          <IconReference />
          {input ? `Drop from the inputs (image ${input.picture})` : "Use as input"}
        </button>
        <button
          type="button"
          className={tool}
          onClick={() => {
            actions.view(null);
            actions.mark(p.n);
          }}
        >
          <IconBrush />
          Mark
        </button>
        {p.origin === "made" && (
          <button type="button" className={tool} onClick={() => actions.reusePicture(p.n)}>
            <IconRegenerate />
            Reuse prompt and seed
          </button>
        )}
        {full && (
          <a className={tool} href={full} download={pictureName(p)}>
            <IconDownload />
            Download PNG
          </a>
        )}
        {onSendToChat && (
          <button
            type="button"
            className={tool}
            onClick={() => {
              actions.view(null);
              onSendToChat(p);
            }}
          >
            Send to the chat
          </button>
        )}
      </div>
    </div>,
    document.body,
  );
}
