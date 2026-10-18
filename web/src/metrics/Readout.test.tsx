import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { Figures } from "./figures.ts";
import { Readout } from "./Readout.tsx";

const figures: Figures = { ttftMs: 250, decodeTokensPerSec: 25, durationMs: 2300, promptTokens: 100, completionTokens: 51, finishReason: "stop", partial: false };

// GitHub #330: the tokens a loop-guard rewind took back show in the turn's line, and only then.
describe("Readout", () => {
  it("says how many tokens a rewind took back, and nothing without one", () => {
    const html = renderToStaticMarkup(<Readout figures={{ ...figures, rewoundTokens: 512 }} />).replace(/<!-- -->/g, "");
    expect(html).toContain("Rewound");
    expect(html).toContain("512 tokens");
    expect(renderToStaticMarkup(<Readout figures={figures} />)).not.toContain("Rewound");
  });
});
