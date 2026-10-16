import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ModelState } from "../api/model.ts";
import { DEFAULT_LIMITS } from "./studio.ts";
import { SwitchBand } from "./SwitchBand.tsx";

// The explicit switch to the image model (not shown for now): what it says
// while a text model serves, while a switch runs, and when it cannot switch.

const text: ModelState = { state: "ready", id: "qwen3.8-27b", contextLimit: 786432, kind: "text" };
type BandState = Parameters<typeof SwitchBand>[0]["state"];

const base: BandState = {
  imageModel: { id: "ignis-image-2.1-turbo", source: "built-in", on_disk: "complete", family: "qwenimage21", loaded: false, switchable: true },
  limits: DEFAULT_LIMITS,
  switching: false,
  switchRefusal: null,
  run: null,
};
const band = (model: ModelState, change: Partial<BandState> = {}) =>
  renderToStaticMarkup(<SwitchBand model={model} state={{ ...base, ...change }} onSwitch={() => {}} />);

describe("SwitchBand", () => {
  it("offers the explicit switch while a text model is loaded, and says Generate switches too", () => {
    const html = band(text);
    expect(html).toContain("qwen3.8-27b is loaded");
    expect(html).toMatch(/<button[^>]*>Switch to the image model<\/button>/);
    expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>Switch to the image model/);
    expect(html).toContain("Generate switches to it on its own");
    expect(band(text, { limits: { ...DEFAULT_LIMITS, allowImplicit: false } })).not.toContain("Generate switches to it on its own");
  });

  it("does not take the switch button while a run is under way", () => {
    const run = { sessionId: "s", step: 1, edit: false, n: 1, switches: true, progress: null, size: "auto", made: 0 };
    expect(band(text, { run })).toMatch(/<button[^>]*disabled=""[^>]*>Switch to the image model<\/button>/);
  });

  it("shows the switch under way, not pressable", () => {
    const html = band(text, { switching: true });
    expect(html).toContain("Switching to the image model…");
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Switch to the image model<\/button>/);
    expect(band({ state: "switching", from: "qwen3.8-27b", to: "ignis-image-2.1-turbo", since: 0 })).toContain("Switching to the image model…");
  });

  it("says why the switch is refused, and where the model comes from when none is listed", () => {
    expect(band(text, { switchRefusal: "the artifact is missing" })).toContain("the artifact is missing");
    const none = band(text, { imageModel: null });
    expect(none).toContain("ignis lists no image model");
    expect(none).toMatch(/<button[^>]*disabled=""[^>]*>Switch to the image model<\/button>/);
  });
});
