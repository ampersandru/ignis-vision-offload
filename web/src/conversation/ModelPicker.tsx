import { createPortal } from "react-dom";
import { loadUnderWay, type ModelState } from "../api/model.ts";
import { type CatalogRow, unavailableReason } from "../server/routes.ts";
import { useAnchoredMenu } from "../ui/anchoredMenu.ts";
import { IconChevron } from "../ui/icons.tsx";
import { type NextModel, nextLine, rowLine, sizeOf, usable } from "./nextModel.ts";

// Under the prompt box: the model the next message goes to, and what sending
// does first — nothing, a load, or a switch. Pressing it lists the text
// models of the catalog; choosing one makes it the next message's, and the
// message switches to it when it is sent (conversation/nextModel.ts). The
// header says what is loaded now; this says what will answer.

export function ModelPicker({
  model,
  next,
  rows,
  onChoose,
}: {
  model: ModelState;
  next: NextModel;
  rows: CatalogRow[];
  onChoose: (id: string) => void;
}) {
  const menu = useAnchoredMenu();
  const busy = loadUnderWay(model);
  const label =
    next.id ??
    (model.state === "switching"
      ? `Switching to ${model.to}…`
      : model.state === "loading"
        ? `Loading ${model.to}…`
        : model.state === "reading"
          ? "Reading the model…"
          : "No text model");
  const dot = busy
    ? "bg-ember animate-pulse motion-reduce:animate-none"
    : next.switches
      ? "bg-ember"
      : next.why === "loaded"
        ? "bg-good"
        : next.why === "unloaded"
          ? "bg-ash"
          : "bg-line";
  const line = nextLine(next);
  return (
    <div className="flex min-w-0 items-center gap-2">
      <button
        ref={menu.button}
        type="button"
        aria-haspopup="menu"
        aria-expanded={menu.open}
        aria-label={`Model for the next message: ${label}`}
        title="Choose the model for the next message"
        disabled={rows.length === 0}
        className="cut flex min-w-0 items-center gap-2 bg-surface py-1 pr-2 pl-2.5 font-display text-[13px] font-semibold text-ink [--cut-size:6px] hover:bg-line disabled:cursor-default disabled:hover:bg-surface"
        onClick={menu.toggle}
      >
        <span className={`cut size-2 shrink-0 [--cut-size:3px] ${dot}`} aria-hidden />
        <span className="min-w-0 truncate">{label}</span>
        {rows.length > 0 && <IconChevron className={`size-3.5 shrink-0 text-ash transition-transform motion-reduce:transition-none ${menu.open ? "rotate-180" : ""}`} />}
      </button>
      {line && <span className={`hidden min-w-0 truncate text-xs sm:inline ${next.switches ? "text-ember" : "text-ash"}`}>{line}</span>}
      {menu.open &&
        createPortal(
          <div
            ref={menu.menu}
            role="menu"
            aria-label="Model for the next message"
            style={{ left: menu.at.left, bottom: menu.at.bottom }}
            className="cut fixed z-40 flex w-[22rem] max-w-[calc(100vw-2rem)] flex-col bg-surface py-1.5 shadow-[0_8px_24px_rgba(28,32,38,0.28)] [--cut-size:10px]"
            onClick={menu.close}
          >
            <p className="px-3 pt-1 pb-2 font-display text-[13px] font-medium text-ash">Model for the next message</p>
            {rows.map((row) => {
              const on = row.id === next.id;
              return (
                <button
                  key={row.id}
                  type="button"
                  role="menuitemradio"
                  aria-checked={on}
                  disabled={!usable(row)}
                  title={unavailableReason(row) ?? undefined}
                  className="flex w-full items-start gap-2.5 px-3 py-2 text-left hover:bg-line disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent"
                  onClick={() => onChoose(row.id)}
                >
                  <span className={`cut mt-1 size-2.5 shrink-0 [--cut-size:3px] ${on ? "bg-ember" : "border border-line"}`} aria-hidden />
                  <span className="min-w-0 flex-1">
                    <span className="flex items-baseline justify-between gap-3">
                      <span className="truncate font-display text-sm font-semibold text-ink">{row.id}</span>
                      <span className="shrink-0 font-display text-xs text-ash tabular-nums">{sizeOf(row.bytes)}</span>
                    </span>
                    <span className="block text-xs leading-snug text-ash">{rowLine(row, model)}</span>
                  </span>
                </button>
              );
            })}
            <p className="mt-1 border-t border-line px-3 pt-2 pb-1 text-xs leading-snug text-ash">
              Pictures come from the Image tab, or from the model through the Pictures tool.
            </p>
          </div>,
          document.body,
        )}
    </div>
  );
}
