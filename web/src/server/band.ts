import { canSend, loadUnderWay, type ModelState } from "../api/model.ts";
import { formatBytes, formatSeconds } from "../monitor/format.ts";
import { type CatalogRow, unavailableReason } from "./routes.ts";

// The model band's rows (spec playground/05 §Model band): one per
// GET /v1/models/catalog row, switched to by id alone. A catalog entry not on
// disk is downloaded onto the server from its row (GitHub #376); where the
// server downloads nothing from the page, the row copies the command that
// fetches it.

const SOURCES: Record<CatalogRow["source"], string> = {
  "built-in": "Built-in catalog",
  operator: "Operator catalog",
  known: "Known model",
};

export function sourceLabel(source: CatalogRow["source"]): string {
  return SOURCES[source] ?? source;
}

/** What of a model is on disk; an entry's size is its total, whatever is there. */
export function diskLabel(row: CatalogRow): string {
  const size = row.bytes === undefined ? "" : formatBytes(row.bytes);
  if (row.on_disk === "complete") return size ? `On disk, ${size}` : "On disk";
  if (row.on_disk === "partial") return size ? `Partly downloaded, ${size} in all` : "Partly downloaded";
  // A known model is a path, not a catalog entry: there is nothing to download.
  if (row.source === "known") return "Artifact not found";
  return size ? `Not downloaded, ${size}` : "Not downloaded";
}

export function downloadCommand(id: string): string {
  return `ignis-server model download ${id}`;
}

export type RowAction =
  | { kind: "loaded" }
  /** The model the switch under way is loading. */
  | { kind: "loading" }
  | { kind: "load"; enabled: boolean }
  /** A catalog entry not (wholly) on disk: download it onto the server, resuming what is there. */
  | { kind: "download"; enabled: boolean }
  /** The same where the server downloads nothing from the page: copy `model download`. */
  | { kind: "copy" }
  /** Its download runs now. */
  | { kind: "downloading" }
  /** A known model whose artifact is not there. */
  | { kind: "missing" }
  /** A catalog entry above this card's class (ADR 0064): neither switched to nor downloaded; `reason` is the server's. */
  | { kind: "unavailable"; reason: string };

/** What a row's Download depends on beyond the row: whether the server downloads from the page, and whether a download runs or is being sent (one at a time). */
export type DownloadGate = { fromPage: boolean; busy: boolean };

/**
 * The row's one action. "Load" follows the server's `switchable`, and is out
 * of reach while a switch or a load runs whatever the last catalog read said;
 * a failed switch is left by switching again, so it stays in reach then, and
 * so does an unloaded model (spec model-switch/02), which "Load" loads.
 * "Download" is out of reach while any download runs, and wherever the
 * server would refuse it for its model's state: no model serving, or one on
 * its way.
 */
export function rowAction(row: CatalogRow, model: ModelState, gate: DownloadGate = { fromPage: true, busy: false }): RowAction {
  const loading = loadUnderWay(model);
  if (loading && row.id === model.to) return { kind: "loading" };
  if (row.loaded) return { kind: "loaded" };
  // The card's class decides before anything on disk: such an entry is never
  // switchable, and `model download` refuses it.
  const unavailable = unavailableReason(row);
  if (unavailable !== null) return { kind: "unavailable", reason: unavailable };
  if (row.download === "running") return { kind: "downloading" };
  // `switchable` decides first: the server knows a switch by id alone can begin
  // (a catalog entry fetched since the last read, the model torn down by a
  // failed restore), whatever `on_disk` last said.
  if (row.switchable || row.on_disk === "complete") {
    return { kind: "load", enabled: row.switchable && !loading && model.state !== "reading" };
  }
  if (row.source === "known") return { kind: "missing" };
  return gate.fromPage ? { kind: "download", enabled: !gate.busy && canSend(model) } : { kind: "copy" };
}

/** The family the loaded model runs as (its scope name), from the catalog's loaded row. */
export function runningFamily(catalog: CatalogRow[] | null): string | null {
  return catalog?.find((row) => row.loaded)?.family ?? null;
}

/** Whole seconds since a switch was first seen: `12 s`, `1m 12s`. */
export function elapsed(since: number, now: number): string {
  const seconds = Math.max(0, Math.floor((now - since) / 1000));
  return seconds < 60 ? `${seconds} s` : formatSeconds(seconds);
}
