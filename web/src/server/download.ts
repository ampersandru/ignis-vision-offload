import { canSend, loadUnderWay, type ModelState } from "../api/model.ts";
import { formatBytes } from "../monitor/format.ts";
import type { Answer, CatalogRow, ConfigDocument } from "./routes.ts";

// A download from the Server view (GitHub #376, ADR 0071): the server fetches
// a catalog entry in the background (POST /v1/models/download) and the view
// follows it by reading GET /v1/models/catalog every second while one runs,
// as it follows a switch. Its speed is the page's own reading: the bytes on
// disk between two reads.

/** The row whose download runs now, if one does: one at a time, on the server's word. */
export const runningDownload = (rows: CatalogRow[] | null): CatalogRow | null => rows?.find((row) => row.download === "running") ?? null;

/**
 * Whether the server downloads from the page at all: `download.enabled` (off, it never
 * uses the network) and `download.from_ui`, as the running configuration says. Unread,
 * it may: the server's refusal then says why not.
 */
export function downloadsFromPage(config: ConfigDocument | null): boolean {
  const group = config?.download;
  return group?.enabled !== false && group?.from_ui !== false;
}

/** One reading of a running download: the bytes on disk, and when they were read (ms). */
export type Sample = { bytes: number; at: number };

/** The readings a speed is taken over: the last five seconds' of one download. */
export type Progress = { id: string; samples: Sample[] };

/** How long the readings a speed is taken over span, at most. */
const SPEED_WINDOW_MS = 5_000;

/** `progress` after a catalog read at `at`: the running row's reading added, a new download's readings started afresh, none without one running. */
export function nextProgress(progress: Progress | null, rows: CatalogRow[], at: number): Progress | null {
  const row = runningDownload(rows);
  if (!row) return null;
  const sample = { bytes: row.bytes_on_disk ?? 0, at };
  const earlier = progress?.id === row.id ? progress.samples.filter((s) => at - s.at <= SPEED_WINDOW_MS) : [];
  return { id: row.id, samples: [...earlier, sample] };
}

/** Bytes per second over the readings kept; null until two readings apart in time. */
export function speedOf(progress: Progress | null): number | null {
  const samples = progress?.samples ?? [];
  if (samples.length < 2) return null;
  const first = samples[0];
  const last = samples[samples.length - 1];
  if (last.at <= first.at) return null;
  return Math.max(0, ((last.bytes - first.bytes) * 1000) / (last.at - first.at));
}

/** A running row's disk column: what is on disk of its total, and the speed once there is one. */
export function progressLabel(row: CatalogRow, speed: number | null): string {
  const done = formatBytes(row.bytes_on_disk ?? 0);
  const of = row.bytes === undefined ? done : `${done} of ${formatBytes(row.bytes)}`;
  return speed === null ? `Downloading, ${of}` : `Downloading, ${of}, ${formatBytes(speed)}/s`;
}

/**
 * Whether the server takes "Cancel" in its model's state (GitHub #377): a cancel never touches the loaded model, so
 * it is open while a model serves or is unloaded, and during a switch or a load and after a failed one too. Not
 * before the page has read the model, nor with the server unreachable.
 */
export function cancelReachable(model: ModelState): boolean {
  return canSend(model) || loadUnderWay(model) || (model.state === "error" && model.failed === true);
}

/** Why a row's last download failed, as the server words it; null when it did not. */
export const downloadFailure = (row: CatalogRow): string | null => (row.download === "failed" ? (row.download_reason ?? "The download failed") : null);

/** What the view does with the answer to "Download": a refusal to show, and whether to follow a download (the one begun, or the one already running). */
export function afterDownload(answer: Answer<unknown>): { refusal: string | null; follow: boolean } {
  if (answer.ok) return { refusal: null, follow: true };
  // GitHub #378: a refusal for want of room is kept on the entry's row: read it there.
  if (answer.code === "download_no_room") return { refusal: null, follow: true };
  return { refusal: answer.message, follow: answer.code === "download_in_progress" };
}

/**
 * Follow the downloads: read the catalog every `every` ms and hand each answer on, until a
 * read shows none running. A read that failed is tried again a tick later. Returns the stop.
 */
export function followDownloads(read: () => Promise<Answer<CatalogRow[]>>, onAnswer: (answer: Answer<CatalogRow[]>) => void, every = 1_000): () => void {
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const tick = async () => {
    const answer = await read();
    if (stopped) return;
    onAnswer(answer);
    if (!answer.ok || runningDownload(answer.value)) timer = setTimeout(() => void tick(), every);
  };
  timer = setTimeout(() => void tick(), every);
  return () => {
    stopped = true;
    clearTimeout(timer);
  };
}
