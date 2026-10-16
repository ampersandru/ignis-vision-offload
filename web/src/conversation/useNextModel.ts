import { useEffect, useRef, useState } from "react";
import type { ModelState } from "../api/model.ts";
import { type CatalogRow, readCatalog } from "../server/routes.ts";
import { type NextModel, nextModel, textModels } from "./nextModel.ts";

// The composer's model picker (conversation/nextModel.ts): the catalog's text
// models, read when the page opens and whenever the model loaded changes; the
// owner's choice for the next message; and the text model the page last saw
// loaded, for the way back from the image model.

export type NextModelState = {
  next: NextModel;
  /** The text models the picker offers. */
  rows: CatalogRow[];
  /** The owner chose `id` for the next message; the model loaded is no choice at all. */
  choose: (id: string) => void;
};

export function useNextModel(model: ModelState): NextModelState {
  const [catalog, setCatalog] = useState<CatalogRow[] | null>(null);
  const [choice, setChoice] = useState<string | null>(null);
  const serving = model.state === "ready" || model.state === "unloaded" ? model : null;
  const loaded = serving?.id ?? null;
  const loadedText = serving && serving.kind !== "image" ? serving.id : null;
  const lastText = useRef<string | null>(null);
  useEffect(() => {
    if (loadedText) lastText.current = loadedText;
  }, [loadedText]);

  useEffect(() => {
    if (model.state === "reading" || model.state === "error") return;
    void readCatalog().then((answer) => answer.ok && setCatalog(answer.value));
  }, [loaded, model.state]);

  // The choice is the next message's only: once its model serves, the chat follows the server again, whoever
  // switches it after — a choice that stayed would switch back on every message (GitHub #305's ping-pong).
  useEffect(() => {
    if (choice !== null && choice === loaded) setChoice(null);
  }, [choice, loaded]);

  const rows = textModels(catalog);
  return {
    next: nextModel(model, choice, loadedText ?? lastText.current, rows),
    rows,
    choose: (id) => setChoice(id === loadedText ? null : id),
  };
}
