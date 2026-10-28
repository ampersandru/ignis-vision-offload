import { type KeyboardEvent, type ReactNode, type RefObject, useEffect, useRef } from "react";
import { IconClose } from "./icons.tsx";

// The full-screen sheet the HTML preview, a workspace file's text and the diff
// between two of its versions open on: the kiln's dark over the whole window,
// a header with the title, the sheet's controls and Close, and a body that
// scrolls under it. It is modal: the focus moves into it when it opens, the
// page behind it is inert while it is open, so Tab cannot reach what it
// covers, and the focus goes back to what opened it once it closes. Esc
// closes it and nothing else: the sheet takes the key, and the page's own Esc
// listeners leave a key taken (`escapeFree`). The caller puts it in a portal
// on the body, since the reply's chamfered boxes clip fixed children.

/** A toggle of a sheet's header, in the radiogroups' face. */
export const SHEET_BUTTON = "font-display text-[13px] font-medium disabled:cursor-not-allowed disabled:opacity-40";

/** A choice of a sheet's header, underlined in ember while it is on. */
export const underlined = (on: boolean) => `border-b-2 pb-0.5 ${on ? "border-[#ff5a1f] text-[#eae8e4]" : "border-transparent text-[#939ba4] hover:text-[#eae8e4]"}`;

/** An Esc nothing took yet: the page's own Esc listeners act on it alone, and leave the one a sheet closed on. */
export const escapeFree = (event: { key: string; defaultPrevented: boolean }) => event.key === "Escape" && !event.defaultPrevented;

/** A key as the sheet reads it. */
export type SheetKeyEvent = { key: string; defaultPrevented: boolean; preventDefault: () => void };

/** A key pressed in the sheet: an Esc nothing in it took closes it, and is taken; any other key is the sheet's own (`onKey`). */
export function sheetKeyDown<E extends SheetKeyEvent>(event: E, sheet: { onClose: () => void; onKey?: (event: E) => void }) {
  if (event.key !== "Escape") return sheet.onKey?.(event);
  if (event.defaultPrevented) return;
  event.preventDefault();
  sheet.onClose();
}

/** A node of the page, as the sheet's hold reads it: an element that can be inert, its parent and its children. */
export type PageNode = { inert?: boolean; parentElement: PageNode | null; children: ArrayLike<PageNode> };

/**
 * Makes the page around `sheet` inert while it is open: everything under the
 * page's `body` but the sheet, whose focus it takes. Gives back what lets the
 * page go again, the focus back on what had it (`activeElement`), leaving
 * inert what already was.
 */
export function holdFocus(sheet: PageNode & { focus: () => void }, page: { body: PageNode; activeElement: unknown }): () => void {
  const opener = page.activeElement as { focus?: () => void } | null;
  const held: PageNode[] = [];
  for (let node: PageNode = sheet; node !== page.body && node.parentElement; node = node.parentElement) {
    for (const other of Array.from(node.parentElement.children)) {
      if (other === node || other.inert !== false) continue;
      other.inert = true;
      held.push(other);
    }
  }
  sheet.focus();
  return () => {
    for (const node of held) node.inert = false;
    opener?.focus?.();
  };
}

/** The sheet: its title and controls in the header, then Close; its body below, scrolled. `onKey` takes the keys but Esc. */
export function Sheet({
  label,
  title,
  titleFills = false,
  controls,
  closeLabel,
  onClose,
  onKey,
  bodyRef,
  bodyClass = "",
  children,
}: {
  /** What the dialog is called. */
  label: string;
  title: ReactNode;
  /** The title takes the header's free width, pushing the controls to the right. */
  titleFills?: boolean;
  controls?: ReactNode;
  closeLabel: string;
  onClose: () => void;
  onKey?: (event: KeyboardEvent<HTMLDivElement>) => void;
  bodyRef?: RefObject<HTMLDivElement | null>;
  /** The body's own classes beside the sheet's. */
  bodyClass?: string;
  children: ReactNode;
}) {
  const sheet = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const node = sheet.current;
    return node ? holdFocus(node, document) : undefined;
  }, []);
  return (
    <div
      ref={sheet}
      role="dialog"
      aria-modal="true"
      aria-label={label}
      tabIndex={-1}
      onKeyDown={(e) => sheetKeyDown(e, { onClose, onKey })}
      className="memory-sheet fixed inset-0 z-[60] flex flex-col bg-kiln text-[#eae8e4] outline-none"
    >
      <header className="flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-kiln-line px-4 py-2.5 md:px-6">
        <h2 className={`min-w-0 ${titleFills ? "flex-1 " : ""}truncate font-display text-[15px] font-semibold`}>{title}</h2>
        {controls}
        <button
          type="button"
          aria-label={closeLabel}
          title="Close (Esc)"
          onClick={onClose}
          className="-mr-2 ml-auto grid size-9 place-items-center text-[#939ba4] hover:bg-kiln-line hover:text-white"
        >
          <IconClose />
        </button>
      </header>
      <div ref={bodyRef} className={`min-h-0 flex-1 overflow-auto bg-[#12151a] p-3 md:p-5 ${bodyClass}`.trimEnd()}>
        {children}
      </div>
    </div>
  );
}
