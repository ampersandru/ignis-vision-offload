import type { ModelState } from "../api/model.ts";
import type { CatalogRow } from "../server/routes.ts";

// Which model the chat's next message goes to (the composer's model picker).
// By default the one loaded: the chat follows the server, whoever switched
// it. The owner may choose another text model from the catalog, and the
// message switches to it before it is sent; with the image model loaded the
// message goes back to the text model the page last saw, or the one the
// server last switched away from. The choice is the next message's only:
// once the model it named serves, the chat follows the server again
// (useNextModel), so no request ever names a model (GitHub #305).

/** The catalog's text models: everything but the image family. */
export function textModels(catalog: CatalogRow[] | null): CatalogRow[] {
  return (catalog ?? []).filter((row) => row.family !== "qwenimage21" && !/^ignis-image/.test(row.id));
}

/** A row the server can switch to: all its files on disk, switchable, and of a card class this card loads (ADR 0064). */
export const usable = (row: CatalogRow) => row.switchable && row.on_disk === "complete" && row.available !== false;

/** Why the card cannot load a row (ADR 0064), as the server words it; null when it can. */
export const unavailableReason = (row: CatalogRow): string | null => (row.available === false ? (row.reason ?? "Too large for this card") : null);

export type NextModel = {
  /** The model the next message goes to; null while none can be named (a switch under way, nothing read yet). */
  id: string | null;
  /** Sending switches to it first. */
  switches: boolean;
  /** Why it is that one: the model loaded, one unloaded while idle that the message loads, the owner's choice, or the way back from the image model. */
  why: "loaded" | "unloaded" | "chosen" | "after-image" | "none";
  /** The model a switch replaces. */
  from?: string;
};

const NONE: NextModel = { id: null, switches: false, why: "none" };

export function nextModel(model: ModelState, choice: string | null, lastText: string | null, rows: CatalogRow[]): NextModel {
  if (model.state !== "ready" && model.state !== "unloaded") return NONE;
  const loaded = model.id;
  if (choice !== null && choice !== loaded) return { id: choice, switches: true, why: "chosen", from: loaded };
  if (model.kind !== "image") return { id: loaded, switches: false, why: model.state === "unloaded" ? "unloaded" : "loaded" };
  const known = (id: string | null | undefined): id is string =>
    typeof id === "string" && (rows.length === 0 ? id !== loaded : rows.some((row) => row.id === id && usable(row)));
  const back = [lastText, model.lastSwitch?.from].find(known) ?? rows.find(usable)?.id;
  return back ? { id: back, switches: true, why: "after-image", from: loaded } : NONE;
}

/** The chip's word on what sending does first. */
export function nextLine(next: NextModel): string {
  switch (next.why) {
    case "loaded":
      return "Loaded";
    case "unloaded":
      return "Loads when you send";
    case "chosen":
    case "after-image":
      return `Switches from ${next.from} when you send`;
    case "none":
      return "";
  }
}

/** A menu row's line: what choosing it means now. */
export function rowLine(row: CatalogRow, model: ModelState): string {
  const loaded = model.state === "ready" || model.state === "unloaded" ? model.id : null;
  if (row.id === loaded) return model.state === "unloaded" ? "Unloaded while idle: the message loads it" : "Loaded";
  const unavailable = unavailableReason(row);
  if (unavailable) return `Not for this card: ${unavailable}`;
  if (row.on_disk !== "complete") return row.on_disk === "partial" ? "Partly on disk: download it from the Server view" : "Not on disk: download it from the Server view";
  if (!row.switchable) return "Not one the server can switch to";
  return loaded ? `The message switches from ${loaded} first` : "The message loads it first";
}

/** `19.4 GB`. */
export const sizeOf = (bytes: number | undefined) => (bytes === undefined ? "" : `${(bytes / 1e9).toFixed(1)} GB`);
