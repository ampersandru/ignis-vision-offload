import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ReasoningText } from "./ReasoningText.tsx";

// How the reasoning box shows a rewind (ADR 0052): the struck text, then the
// tag on a line of its own, then the nudge.

describe("ReasoningText", () => {
  it("ends the tag's line when the nudge follows it with no line break of its own", () => {
    const html = renderToStaticMarkup(
      <ReasoningText text={"Kept.\nloop loop\n\n[LOOP DETECTED, REWINDING 3 TOKENS / 10 CHARS · 0a1b2c]Wait, I'm starting to repeat myself."} />,
    );
    expect(html).toContain("<del");
    expect(html).toMatch(/0a1b2c\]<\/span>\n<span>Wait, I&#x27;m starting/);
  });

  it("adds no line break when the text after the tag opens with one, or nothing follows", () => {
    const old = renderToStaticMarkup(<ReasoningText text={"loop loop\n\n[LOOP DETECTED, REWINDING 3 TOKENS / 9 CHARS]\n\nWait."} />);
    expect(old).toMatch(/CHARS\]<\/span><span>\n\nWait\./);
    const last = renderToStaticMarkup(<ReasoningText text={"loop loop\n\n[LOOP DETECTED, REWINDING 3 TOKENS / 9 CHARS · 0a1b2c]"} />);
    expect(last).toMatch(/0a1b2c\]<\/span>$/);
  });
});
