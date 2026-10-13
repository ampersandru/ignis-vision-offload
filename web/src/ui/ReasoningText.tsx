import { splitRewinds } from "./rewinds.ts";

/**
 * A reasoning text as the reasoning box shows it: what the loop guard took
 * back struck through, its tag marked (`rewinds.ts`).
 */
export function ReasoningText({ text }: { text: string }) {
  if (!text.includes("[LOOP DETECTED")) return <>{text}</>;
  return (
    <>
      {splitRewinds(text).map((piece, i) => {
        switch (piece.kind) {
          case "text":
            return <span key={i}>{piece.text}</span>;
          case "struck":
            return (
              <del key={i} className="opacity-60 decoration-ember/70" title="Taken back: the reasoning was looping">
                {piece.text}
              </del>
            );
          case "tag":
            return (
              <span
                key={i}
                className="font-display text-xs font-medium text-ember"
                title={`ignis found a loop and took back ${piece.tokens} tokens (${piece.chars} characters)`}>
                {piece.text}
              </span>
            );
        }
      })}
    </>
  );
}
