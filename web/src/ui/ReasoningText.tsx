import { Fragment } from "react";
import { splitRewinds } from "./rewinds.ts";

/**
 * A reasoning text as the reasoning box shows it: what the loop guard took
 * back struck through, its tag marked (`rewinds.ts`). The tag ends its line:
 * after a kept line break the server sends the nudge with no blank line of
 * its own, and it would otherwise run on after the tag.
 */
export function ReasoningText({ text }: { text: string }) {
  if (!text.includes("[LOOP DETECTED")) return <>{text}</>;
  const pieces = splitRewinds(text);
  return (
    <>
      {pieces.map((piece, i) => {
        switch (piece.kind) {
          case "text":
            return <span key={i}>{piece.text}</span>;
          case "struck":
            return (
              <del key={i} className="opacity-60 decoration-ember/70" title="Taken back: the reasoning was looping">
                {piece.text}
              </del>
            );
          case "tag": {
            const next = pieces[i + 1];
            return (
              <Fragment key={i}>
                <span
                  className="font-display text-xs font-medium text-ember"
                  title={`ignis found a loop and took back ${piece.tokens} tokens (${piece.chars} characters)`}>
                  {piece.text}
                </span>
                {next !== undefined && !next.text.startsWith("\n") ? "\n" : null}
              </Fragment>
            );
          }
        }
      })}
    </>
  );
}
