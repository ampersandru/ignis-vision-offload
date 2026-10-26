import { describe, expect, it } from "vitest";
import type { ModelState } from "../api/model.ts";
import { CATALOG, CONFIG } from "./fixture.ts";
import { afterRemoval, removalPrompt, removalsFromPage, removeAction } from "./removal.ts";
import type { CatalogRow } from "./routes.ts";

// A removal from the Server view (GitHub #380, ADR 0071): which rows offer
// Remove, what its confirmation says, and how the answer is read.

const row = (id: string): CatalogRow => CATALOG.find((r) => r.id === id)!;
const ready: ModelState = { state: "ready", id: "qwen3.8-27b", contextLimit: 786432 };
const free = { fromPage: true, busy: false };

describe("removeAction", () => {
  it("is offered on a catalog entry on disk or partly downloaded that nothing uses, and on no other row", () => {
    expect(removeAction(row("qwen3.8-flash-next"), ready, free)).toEqual({ enabled: true });
    expect(removeAction(row("qwen3.8-27b-abliterated"), ready, free)).toEqual({ enabled: true });
    // The loaded model, a known model's path, an entry not downloaded, a running download (Cancel is its action).
    expect(removeAction(row("qwen3.8-27b"), ready, free)).toBeNull();
    expect(removeAction(row("qwen3.8-27b-dead"), ready, free)).toBeNull();
    expect(removeAction(row("qwen3.8-27b-custom"), ready, free)).toBeNull();
    expect(removeAction({ ...row("qwen3.8-27b-abliterated"), download: "running" }, ready, free)).toBeNull();
    // Above the card's class it is never loaded here, and what is on disk of it can still go.
    expect(removeAction({ ...row("qwen3.8-flash-next"), available: false, reason: "needs a 32 GB card" }, ready, free)).toEqual({ enabled: true });
  });

  it("is not offered for the model unloaded while idle, nor the one a switch or a load is loading", () => {
    const unloaded: ModelState = { state: "unloaded", id: "qwen3.8-flash-next", contextLimit: 589824 };
    expect(removeAction({ ...row("qwen3.8-flash-next"), loaded: false }, unloaded, free)).toBeNull();
    const switching: ModelState = { state: "switching", from: "qwen3.8-27b", to: "qwen3.8-flash-next", since: 0 };
    expect(removeAction(row("qwen3.8-flash-next"), switching, free)).toBeNull();
    const loading: ModelState = { state: "loading", to: "qwen3.8-flash-next", since: 0 };
    expect(removeAction(row("qwen3.8-flash-next"), loading, free)).toBeNull();
  });

  it("is out of reach while a removal is sent or no model serves, and not offered where the page removes nothing", () => {
    expect(removeAction(row("qwen3.8-flash-next"), ready, { fromPage: true, busy: true })).toEqual({ enabled: false });
    const switching: ModelState = { state: "switching", from: "qwen3.8-27b", to: "qwen3.8-27b-fail", since: 0 };
    expect(removeAction(row("qwen3.8-flash-next"), switching, free)).toEqual({ enabled: false });
    expect(removeAction(row("qwen3.8-flash-next"), { state: "error", message: "x", failed: true }, free)).toEqual({ enabled: false });
    expect(removeAction(row("qwen3.8-flash-next"), ready, { fromPage: false, busy: false })).toBeNull();
  });
});

describe("removalsFromPage", () => {
  it("follows download.from_ui alone: download.enabled false still frees the disk", () => {
    expect(removalsFromPage(CONFIG)).toBe(true);
    expect(removalsFromPage(null)).toBe(true);
    expect(removalsFromPage({ ...CONFIG, download: { ...CONFIG.download, from_ui: false } })).toBe(false);
    expect(removalsFromPage({ ...CONFIG, download: { ...CONFIG.download, enabled: false } })).toBe(true);
  });
});

describe("removalPrompt", () => {
  it("names the entry and what of it is on disk, partial files included", () => {
    expect(removalPrompt(row("qwen3.8-flash-next"), CONFIG)).toEqual([
      "Remove qwen3.8-flash-next? Its files under download.path go, 67.8 GiB; it can be downloaded again.",
    ]);
    expect(removalPrompt(row("qwen3.8-27b-abliterated"), CONFIG)).toEqual([
      "Remove qwen3.8-27b-abliterated? Its files under download.path go, 6 GiB, its unfinished download included; it can be downloaded again.",
    ]);
  });

  it("says when it is the model the server started on, and what the next start does without it", () => {
    const start = { ...row("qwen3.8-flash-next"), start: true };
    expect(removalPrompt(start, CONFIG)[1]).toBe(
      "The server started on it: the next start downloads it again (asking first on a terminal), or, when model.artifact names its file, refuses to start.",
    );
    const offline = { ...CONFIG, download: { ...CONFIG.download, enabled: false } };
    expect(removalPrompt(start, offline)[1]).toBe(
      "The server started on it: with download.enabled false the next start does not download it again and comes up with no model, on the placeholder template, or, when model.artifact names its file, refuses to start.",
    );
  });

  it("says when it is a known model, which no switch reaches once it is gone", () => {
    const known = { ...row("qwen3.8-flash-next"), known: true };
    expect(removalPrompt(known, CONFIG)[1]).toBe(
      "It is a known model (switch.known_models names its file): until it is downloaded again, a switch to it is refused, after the next start too.",
    );
  });
});

describe("afterRemoval", () => {
  it("says what was removed and freed, and which files stayed because another entry lists them", () => {
    const removed = { status: "removed", model: "qwen3.8-flash-next", bytes_freed: 72799980571, files_removed: ["a", "b"], files_kept: [] };
    expect(afterRemoval({ ok: true, status: 200, value: removed })).toEqual({ tone: "done", text: "Removed qwen3.8-flash-next: 67.8 GiB freed." });
    const kept = { ...removed, bytes_freed: 5, files_kept: [{ file: "shared.bin", by: "acme-b" }] };
    expect(afterRemoval({ ok: true, status: 200, value: kept })).toEqual({
      tone: "done",
      text: "Removed qwen3.8-flash-next: 5 B freed. Kept shared.bin, which acme-b also lists.",
    });
  });

  it("shows a refusal as the server words it", () => {
    const message = "qwen3.8-flash-next is the model loaded (qwen3.8-flash-next), which serves from its files; load another model before removing it";
    expect(afterRemoval({ ok: false, status: 409, message, code: "model_in_use" })).toEqual({ tone: "fault", text: message });
  });
});
