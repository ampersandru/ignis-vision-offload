import { type ReactNode, useEffect } from "react";
import { createPortal } from "react-dom";
import { IconClose } from "./icons.tsx";

/**
 * A picture at its own size, over the page. Escape and a click outside close
 * it; nothing under it moves, so going back is exactly where you were. In a
 * portal: the reply's chamfered boxes clip fixed children, so it lives on the
 * body.
 */
export function Lightbox({ onClose, children }: { onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    const key = (e: globalThis.KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);
  return createPortal(
    <div className="fixed inset-0 z-50 grid place-items-center bg-[#1c2026]/88 p-4" onClick={onClose} role="presentation">
      <div className="relative max-h-full max-w-full" onClick={(e) => e.stopPropagation()}>
        {children}
        <button
          type="button"
          onClick={onClose}
          aria-label="Close the picture"
          className="absolute right-0 top-0 grid size-8 place-items-center bg-kiln text-[#eae8e4] hover:bg-ember"
        >
          <IconClose />
        </button>
      </div>
    </div>,
    document.body,
  );
}
