import { canSend, loadUnderWay, type ModelState } from "../api/model.ts";
import { formatBytes } from "../monitor/format.ts";
import type { PageGate } from "./band.ts";
import type { Line } from "./outcome.ts";
import type { Answer, CatalogRow, ConfigDocument, ModelRemoved } from "./routes.ts";

// A removal from the Server view (GitHub #380, ADR 0071): the server deletes
// a catalog entry's files, finished or partial, from its models folder
// (POST /v1/models/remove), after the page asked for a confirmation that
// says what the entry is to this server. Remove sits beside the row's own
// action: Load on an entry on disk, Resume on a partial one.

/** Whether the server removes from the page: `download.from_ui`, as the running configuration says. `download.enabled` false keeps the server off the network, and a removal uses none. Unread, it may. */
export function removalsFromPage(config: ConfigDocument | null): boolean {
  return config?.download?.from_ui !== false;
}

/**
 * The row's Remove, or none: offered on a catalog entry with something of it on
 * disk that nothing uses — not the model loaded, nor the one unloaded while idle
 * (the next message loads it again), nor the one a switch or a load is loading,
 * nor one whose download runs, which Cancel stops first. Out of reach while a
 * removal is sent, and wherever the server would refuse it for its model's
 * state: no model serving, or one on its way.
 */
export function removeAction(row: CatalogRow, model: ModelState, gate: PageGate): { enabled: boolean } | null {
  if (!gate.fromPage || row.source === "known" || row.on_disk === "absent" || row.loaded || row.download === "running") return null;
  if (model.state === "unloaded" && model.id === row.id) return null;
  if (loadUnderWay(model) && model.to === row.id) return null;
  return { enabled: !gate.busy && canSend(model) };
}

/**
 * The confirmation's sentences: what goes, then what the entry is to this server — the model it started on, or known
 * models — and what that means once it is gone, naming each by the id it is served under when that is not the row's.
 */
export function removalPrompt(row: CatalogRow, config: ConfigDocument | null): string[] {
  const size = formatBytes(row.bytes_on_disk ?? row.bytes ?? 0);
  const partial = row.on_disk === "partial" ? ", its unfinished download included" : "";
  const lines = [`Remove ${row.id}? Its files under download.path go, ${size}${partial}; it can be downloaded again.`];
  if (row.start_model !== undefined) {
    const as = row.start_model === row.id ? "" : ` (as ${row.start_model})`;
    lines.push(
      config?.download?.enabled === false
        ? `The server started on it${as}: with download.enabled false the next start does not download it again and comes up with no model, on the placeholder template, or, when model.artifact names its file, refuses to start.`
        : `The server started on it${as}: the next start downloads it again (asking first on a terminal), or, when model.artifact names its file, refuses to start.`,
    );
  }
  const known = row.known_models ?? [];
  if (known.length === 1 && known[0] === row.id) {
    lines.push("It is a known model (switch.known_models names its file): until it is downloaded again, a switch to it is refused, after the next start too.");
  } else if (known.length > 0) {
    const them = known.length === 1 ? "it" : "any of them";
    lines.push(`switch.known_models names its file as ${known.join(", ")}: until it is downloaded again, a switch to ${them} is refused, after the next start too.`);
  }
  return lines;
}

/** The band's line after "Remove": what went and what it freed, the files another entry keeps, or the server's refusal. */
export function afterRemoval(answer: Answer<ModelRemoved>): Line {
  if (!answer.ok) return { tone: "fault", text: answer.message };
  const removed = answer.value;
  const kept = removed.files_kept.map(({ file, by }) => ` Kept ${file}, which ${by} also lists.`).join("");
  return { tone: "done", text: `Removed ${removed.model}: ${formatBytes(removed.bytes_freed)} freed.${kept}` };
}
