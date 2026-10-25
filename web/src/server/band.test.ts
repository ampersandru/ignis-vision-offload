import { describe, expect, it } from "vitest";
import type { ModelState } from "../api/model.ts";
import { diskLabel, downloadCommand, elapsed, rowAction, runningFamily, sourceLabel } from "./band.ts";
import { CATALOG } from "./fixture.ts";
import type { CatalogRow } from "./routes.ts";

// The model band's rows (spec playground/05 §Model band): one per catalog
// row, with where it is from, what is on disk, and the one action it offers.

const row = (id: string): CatalogRow => CATALOG.find((r) => r.id === id)!;
const ready: ModelState = { state: "ready", id: "qwen3.8-27b", contextLimit: 786432 };
const switching: ModelState = { state: "switching", from: "qwen3.8-27b", to: "qwen3.8-flash-next", since: 0 };

describe("rowAction", () => {
  it("is Loaded on the loaded row, Load where a switch can begin, and Download where the entry is not wholly on disk", () => {
    expect(rowAction(row("qwen3.8-27b"), ready)).toEqual({ kind: "loaded" });
    expect(rowAction(row("qwen3.8-flash-next"), ready)).toEqual({ kind: "load", enabled: true });
    expect(rowAction(row("qwen3.8-27b-abliterated"), ready)).toEqual({ kind: "download", enabled: true });
    expect(rowAction(row("qwen3.8-27b-custom"), ready)).toEqual({ kind: "download", enabled: true });
  });

  // GitHub #376: one download at a time, followed on its row; where the
  // server downloads nothing from the page, the row copies the command.
  it("shows a running download on its row and keeps every other Download out of reach meanwhile", () => {
    const busy = { fromPage: true, busy: true };
    expect(rowAction({ ...row("qwen3.8-27b-abliterated"), download: "running" }, ready, busy)).toEqual({ kind: "downloading" });
    expect(rowAction(row("qwen3.8-27b-custom"), ready, busy)).toEqual({ kind: "download", enabled: false });
    expect(rowAction(row("qwen3.8-flash-next"), ready, busy)).toEqual({ kind: "load", enabled: true });
  });

  it("offers Download again after a failure, and none while no model serves or a switch runs", () => {
    expect(rowAction({ ...row("qwen3.8-27b-custom"), download: "failed", download_reason: "401" }, ready)).toEqual({ kind: "download", enabled: true });
    expect(rowAction(row("qwen3.8-27b-custom"), switching)).toEqual({ kind: "download", enabled: false });
    expect(rowAction(row("qwen3.8-27b-custom"), { state: "error", message: "x", failed: true })).toEqual({ kind: "download", enabled: false });
    expect(rowAction(row("qwen3.8-27b-custom"), { state: "unloaded", id: "qwen3.8-27b", contextLimit: null })).toEqual({ kind: "download", enabled: true });
  });

  it("copies the download command where download.enabled or download.from_ui is off", () => {
    const off = { fromPage: false, busy: false };
    expect(rowAction(row("qwen3.8-27b-custom"), ready, off)).toEqual({ kind: "copy" });
    expect(rowAction(row("qwen3.8-27b-abliterated"), ready, off)).toEqual({ kind: "copy" });
    expect(rowAction(row("qwen3.8-flash-next"), ready, off)).toEqual({ kind: "load", enabled: true });
  });

  it("switches to an entry its download just finished, and offers no Download for it", () => {
    const landed = { ...row("qwen3.8-27b-custom"), on_disk: "complete" as const, switchable: true, download: "none" as const, bytes_on_disk: 19406949449 };
    expect(rowAction(landed, ready)).toEqual({ kind: "load", enabled: true });
  });

  // ADR 0064: an entry above this card's class offers neither Load nor
  // Download, whatever is on disk; the server's reason is its tooltip.
  it("offers nothing for an entry above the card's class, and gives the server's reason", () => {
    const reason = "qwen3.8-flash-next needs a 32 GB card or larger (its catalog entry's card_class), and this card is of the 16 GB class";
    const above = { ...row("qwen3.8-flash-next"), switchable: false, available: false, reason };
    expect(rowAction(above, ready)).toEqual({ kind: "unavailable", reason });
    expect(rowAction({ ...above, on_disk: "absent" }, ready)).toEqual({ kind: "unavailable", reason });
    expect(rowAction({ ...above, on_disk: "absent" }, ready, { fromPage: false, busy: false })).toEqual({ kind: "unavailable", reason });
    expect(rowAction({ ...above, loaded: true }, ready)).toEqual({ kind: "loaded" });
    expect(rowAction({ ...row("qwen3.8-flash-next"), available: true }, ready)).toEqual({ kind: "load", enabled: true });
  });

  it("offers no download for a known model, which is no catalog entry: its artifact is just missing", () => {
    expect(rowAction({ ...row("qwen3.8-27b-fail"), on_disk: "absent", switchable: false }, ready)).toEqual({ kind: "missing" });
  });

  it("follows the server's `switchable` before what is on disk: a catalog entry the server can switch to is Load", () => {
    expect(rowAction({ ...row("qwen3.8-27b-custom"), on_disk: "absent", switchable: true }, ready)).toEqual({ kind: "load", enabled: true });
    expect(rowAction({ ...row("qwen3.8-27b-custom"), source: "built-in", on_disk: "absent", switchable: true }, ready)).toEqual({ kind: "load", enabled: true });
  });

  it("offers Load on every row, the torn-down model's included, while nothing serves", () => {
    const dead: ModelState = { state: "error", message: "Switch failed: x", failed: true };
    const torn = CATALOG.map((r) => ({ ...r, loaded: false, switchable: r.on_disk === "complete" }));
    expect(rowAction(torn[0], dead)).toEqual({ kind: "load", enabled: true });
    expect(torn.map((r) => rowAction(r, dead).kind)).not.toContain("loaded");
  });

  it("marks the model a switch is loading, and offers no Load while one runs", () => {
    expect(rowAction(row("qwen3.8-flash-next"), switching)).toEqual({ kind: "loading" });
    expect(rowAction(row("qwen3.8-27b-fail"), switching)).toEqual({ kind: "load", enabled: false });
    // A failed switch is left by switching again.
    expect(rowAction(row("qwen3.8-27b-fail"), { state: "error", message: "Switch failed: x", failed: true })).toEqual({ kind: "load", enabled: true });
  });
  it("offers Load on the unloaded model, marks it loading while it loads, and offers no Load meanwhile (spec model-switch/02)", () => {
    const unloaded: ModelState = { state: "unloaded", id: "qwen3.8-27b", contextLimit: 786432 };
    const unloadedRow = { ...row("qwen3.8-27b"), loaded: false, switchable: true };
    expect(rowAction(unloadedRow, unloaded)).toEqual({ kind: "load", enabled: true });
    const loading: ModelState = { state: "loading", to: "qwen3.8-27b", since: 0 };
    expect(rowAction(unloadedRow, loading)).toEqual({ kind: "loading" });
    expect(rowAction(row("qwen3.8-flash-next"), loading)).toEqual({ kind: "load", enabled: false });
  });
});

describe("row labels", () => {
  it("says where a model is from and what of it is on disk, in GiB", () => {
    expect(sourceLabel("built-in")).toBe("Built-in catalog");
    expect(sourceLabel("operator")).toBe("Operator catalog");
    expect(sourceLabel("known")).toBe("Known model");
    expect(diskLabel(row("qwen3.8-27b"))).toBe("On disk, 18.07 GiB");
    expect(diskLabel(row("qwen3.8-flash-next"))).toBe("On disk, 67.8 GiB");
    expect(diskLabel(row("qwen3.8-27b-abliterated"))).toBe("Partly downloaded, 18.07 GiB in all");
    expect(diskLabel(row("qwen3.8-27b-custom"))).toBe("Not downloaded, 18.07 GiB");
    expect(diskLabel({ id: "x", source: "known", on_disk: "absent", loaded: false, switchable: false })).toBe("Artifact not found");
  });

  it("copies the command that fetches a model", () => {
    expect(downloadCommand("qwen3.8-27b-abliterated")).toBe("ignis-server model download qwen3.8-27b-abliterated");
  });
});

describe("runningFamily", () => {
  it("is the loaded row's family, or unknown without a catalog", () => {
    expect(runningFamily(CATALOG)).toBe("qwen38");
    expect(runningFamily(null)).toBeNull();
  });
});

describe("elapsed", () => {
  it("counts whole seconds since the switch was first seen", () => {
    expect(elapsed(1_000, 1_400)).toBe("0 s");
    expect(elapsed(1_000, 13_200)).toBe("12 s");
    expect(elapsed(0, 72_000)).toBe("1m 12s");
  });
});
