// The workspace panel's shared looks: its buttons and the block a file's text shows in.

export const BUTTON = "cut flex shrink-0 items-center gap-1.5 bg-surface px-2.5 py-1.5 font-display text-xs font-semibold text-ink [--cut-size:6px] hover:bg-line";

/** A file's text as it is, wrapped, scrolled past 60% of the screen's height. */
export const Source = ({ text }: { text: string }) => (
  <pre className="max-h-[60vh] overflow-auto bg-surface px-2.5 py-2 font-mono text-xs leading-relaxed whitespace-pre-wrap break-words text-ink">{text}</pre>
);
