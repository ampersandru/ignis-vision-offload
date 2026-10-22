import { describe, expect, it } from "vitest";
import type { ModelState } from "../api/model.ts";
import type { CatalogRow } from "../server/routes.ts";
import { nextLine, nextModel, rowLine, sizeOf, textModels, unavailableReason, usable } from "./nextModel.ts";

// The model the chat's next message goes to (conversation/nextModel.ts): the
// one loaded unless the owner chose another, the way back to a text model when
// the image model is loaded, and the picker's words for each.

const row = (id: string, change: Partial<CatalogRow> = {}): CatalogRow => ({
  id,
  source: "built-in",
  on_disk: "complete",
  family: id.startsWith("ignis-image") ? "qwenimage21" : "qwen38",
  loaded: false,
  switchable: true,
  ...change,
});

const catalog = [row("qwen3.8-27b"), row("qwen3.8-27b-abliterated"), row("qwen3.8-flash-next", { family: "qwen38flashnext" }), row("ignis-image-2.1-turbo")];
const rows = textModels(catalog);

const text: ModelState = { state: "ready", id: "qwen3.8-27b", contextLimit: 524288, kind: "text" };
const image: ModelState = { state: "ready", id: "ignis-image-2.1-turbo", contextLimit: null, kind: "image" };

describe("the next message's model", () => {
  it("lists the text models only", () => {
    expect(rows.map((r) => r.id)).toEqual(["qwen3.8-27b", "qwen3.8-27b-abliterated", "qwen3.8-flash-next"]);
  });

  it("is the model loaded while nothing else is chosen, unloaded or not", () => {
    expect(nextModel(text, null, null, rows)).toEqual({ id: "qwen3.8-27b", switches: false, why: "loaded" });
    expect(nextModel({ ...text, state: "unloaded" }, null, null, rows)).toEqual({ id: "qwen3.8-27b", switches: false, why: "unloaded" });
    expect(nextModel(text, "qwen3.8-27b", null, rows)).toMatchObject({ switches: false, why: "loaded" });
  });

  it("is the owner's choice, switching from the model loaded", () => {
    expect(nextModel(text, "qwen3.8-flash-next", null, rows)).toEqual({ id: "qwen3.8-flash-next", switches: true, why: "chosen", from: "qwen3.8-27b" });
    expect(nextModel(image, "qwen3.8-flash-next", "qwen3.8-27b", rows)).toMatchObject({ id: "qwen3.8-flash-next", why: "chosen" });
  });

  it("goes back from the image model to the text model the page last saw, else the one the server switched away from, else the first", () => {
    expect(nextModel(image, null, "qwen3.8-27b-abliterated", rows)).toEqual({
      id: "qwen3.8-27b-abliterated",
      switches: true,
      why: "after-image",
      from: "ignis-image-2.1-turbo",
    });
    const record = { from: "qwen3.8-flash-next", to: "ignis-image-2.1-turbo", outcome: "switched" as const };
    expect(nextModel({ ...image, lastSwitch: record }, null, null, rows).id).toBe("qwen3.8-flash-next");
    expect(nextModel(image, null, null, rows).id).toBe("qwen3.8-27b");
    // A remembered model that is no longer on disk is passed over.
    expect(nextModel(image, null, "qwen3.8-flash-next", textModels([row("qwen3.8-27b"), row("qwen3.8-flash-next", { on_disk: "partial" })])).id).toBe(
      "qwen3.8-27b",
    );
  });

  it("names none while a switch runs, nothing is read yet, or no text model is known", () => {
    expect(nextModel({ state: "switching", from: "a", to: "b", since: 0 }, null, null, rows).id).toBeNull();
    expect(nextModel({ state: "reading" }, "qwen3.8-27b", null, rows).id).toBeNull();
    expect(nextModel(image, null, null, []).id).toBeNull();
  });

  it("says what sending does first", () => {
    expect(nextLine(nextModel(text, null, null, rows))).toBe("Loaded");
    expect(nextLine(nextModel({ ...text, state: "unloaded" }, null, null, rows))).toBe("Loads when you send");
    expect(nextLine(nextModel(image, null, null, rows))).toBe("Switches from ignis-image-2.1-turbo when you send");
  });

  it("says, for each model in the menu, what choosing it means now", () => {
    expect(rowLine(row("qwen3.8-27b"), text)).toBe("Loaded");
    expect(rowLine(row("qwen3.8-27b"), { ...text, state: "unloaded" })).toContain("the message loads it");
    expect(rowLine(row("qwen3.8-flash-next"), text)).toBe("The message switches from qwen3.8-27b first");
    expect(rowLine(row("x", { on_disk: "absent" }), text)).toContain("Not on disk");
    expect(rowLine(row("x", { switchable: false }), text)).toContain("Not one the server can switch to");
    expect(sizeOf(19406942468)).toBe("19.4 GB");
  });

  // ADR 0064: a model this card's class cannot load is listed, disabled, with
  // the server's reason as its line and its tooltip — whatever is on disk.
  it("offers no model above the card's class, and says why in the server's words", () => {
    const reason = "qwen3.8-flash-next needs a 32 GB card or larger (its catalog entry's card_class), and this card is of the 16 GB class";
    const above = row("qwen3.8-flash-next", { family: "qwen38flashnext", switchable: false, available: false, reason });
    expect(usable(above)).toBe(false);
    expect(usable({ ...above, switchable: true })).toBe(false);
    expect(unavailableReason(above)).toBe(reason);
    expect(unavailableReason(row("qwen3.8-27b"))).toBeNull();
    expect(unavailableReason(row("qwen3.8-27b", { available: true }))).toBeNull();
    expect(rowLine(above, text)).toBe(`Not for this card: ${reason}`);
    expect(rowLine({ ...above, on_disk: "absent" }, text)).toBe(`Not for this card: ${reason}`);
    // Never the way back from the image model either.
    const sixteen = [above, row("qwen3.8-27b-16g")];
    expect(nextModel(image, null, "qwen3.8-flash-next", sixteen).id).toBe("qwen3.8-27b-16g");
  });
});
