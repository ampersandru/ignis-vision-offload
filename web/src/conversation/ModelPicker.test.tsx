import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ModelState } from "../api/model.ts";
import type { CatalogRow } from "../server/routes.ts";
import { ModelPicker } from "./ModelPicker.tsx";
import { nextModel } from "./nextModel.ts";
import { SwitchLine } from "./UserTurn.tsx";

// The composer's model picker and the switch line a turn leaves on its
// prompt: what the next message goes to and what sending does first.

const rows: CatalogRow[] = [
  { id: "qwen3.8-27b", source: "built-in", on_disk: "complete", bytes: 19406942468, family: "qwen38", loaded: false, switchable: true },
  { id: "qwen3.8-flash-next", source: "built-in", on_disk: "complete", family: "qwen38flashnext", loaded: false, switchable: true },
];
const text: ModelState = { state: "ready", id: "qwen3.8-27b", contextLimit: 524288, kind: "text" };
const image: ModelState = { state: "ready", id: "ignis-image-2.1-turbo", contextLimit: null, kind: "image" };

const render = (model: ModelState, choice: string | null = null, list = rows) =>
  renderToStaticMarkup(<ModelPicker model={model} next={nextModel(model, choice, null, list)} rows={list} onChoose={() => {}} />);

describe("ModelPicker", () => {
  it("names the model loaded as the next message's, and says it is loaded", () => {
    const html = render(text);
    expect(html).toContain('aria-label="Model for the next message: qwen3.8-27b"');
    expect(html).toContain(">Loaded<");
    expect(html).toContain("bg-good");
  });

  it("with the image model loaded, names the text model the message goes back to, and the switch first", () => {
    const html = render(image);
    expect(html).toContain(">qwen3.8-27b<");
    expect(html).toContain("Switches from ignis-image-2.1-turbo when you send");
  });

  it("names the owner's choice and the switch it makes", () => {
    expect(render(text, "qwen3.8-flash-next")).toContain("Switches from qwen3.8-27b when you send");
  });

  it("says a switch under way, and is closed while the catalog lists no text model", () => {
    expect(render({ state: "switching", from: "a", to: "qwen3.8-27b", since: 0 })).toContain("Switching to qwen3.8-27b…");
    expect(render(text, null, [])).toMatch(/<button[^>]*disabled=""/);
  });
});

describe("SwitchLine", () => {
  it("says the switch under way, done, stopped or refused", () => {
    expect(renderToStaticMarkup(<SwitchLine change={{ to: "qwen3.8-27b", from: "ignis-image-2.1-turbo", status: "switching" }} />)).toContain(
      "Switching to qwen3.8-27b from ignis-image-2.1-turbo: the reply starts once it is loaded.",
    );
    expect(renderToStaticMarkup(<SwitchLine change={{ to: "qwen3.8-27b", status: "switched" }} />)).toContain("Sent to qwen3.8-27b, switched.");
    expect(renderToStaticMarkup(<SwitchLine change={{ to: "x", status: "stopped" }} />)).toContain("Stopped while switching to x");
    const failed = renderToStaticMarkup(<SwitchLine change={{ to: "x", status: "failed", error: "short of RAM" }} />);
    expect(failed).toContain('role="alert"');
    expect(failed).toContain("Not sent: the switch to x did not land. short of RAM");
  });
});
