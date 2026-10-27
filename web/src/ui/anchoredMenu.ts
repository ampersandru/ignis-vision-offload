import { useEffect, useRef, useState } from "react";

/**
 * A menu that opens upwards from a button at the foot of the page, in a portal placed by hand. It cannot live beside
 * its button: the prompt box is a `.cut`, and a clip-path cuts its descendants too, so a menu standing above the box
 * is clipped away entirely — it opens and nothing is on screen. It closes on Escape and on a mousedown anywhere but
 * the button and the menu: the menu is not inside the button's subtree any more, so it has to be asked as well —
 * closing on its own mousedown would take the item away before its click could land.
 */
export function useAnchoredMenu<B extends HTMLElement = HTMLButtonElement>() {
  const [open, setOpen] = useState(false);
  const [at, setAt] = useState({ left: 0, right: 0, bottom: 0 });
  const button = useRef<B>(null);
  const menu = useRef<HTMLDivElement>(null);

  function place() {
    const box = button.current?.getBoundingClientRect();
    // `right` for a menu that ends where its button does, at the page's right.
    if (box) setAt({ left: box.left, right: window.innerWidth - box.right, bottom: window.innerHeight - box.top + 8 });
  }

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (!button.current?.contains(target) && !menu.current?.contains(target)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", place);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", place);
    };
  }, [open]);

  return {
    open,
    close: () => setOpen(false),
    toggle: () => {
      place();
      setOpen((o) => !o);
    },
    at,
    button,
    menu,
  };
}
