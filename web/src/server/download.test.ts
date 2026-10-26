import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CATALOG, CONFIG } from "./fixture.ts";
import { afterDownload, downloadFailure, downloadsFromPage, followDownloads, nextProgress, progressLabel, runningDownload, speedOf } from "./download.ts";
import type { Answer, CatalogRow } from "./routes.ts";

// A download from the Server view (GitHub #376): which row runs, whether the
// page may download at all, the speed the page reads off the bytes on disk,
// and the catalog read each second while one runs.

const GIB = 1024 ** 3;
const MIB = 1024 ** 2;
const running = (bytes: number): CatalogRow => ({ ...CATALOG[1], download: "running", bytes_on_disk: bytes });
const withRunning = (bytes: number) => CATALOG.map((row, i) => (i === 1 ? running(bytes) : row));

describe("runningDownload", () => {
  it("is the row whose download runs, or none", () => {
    expect(runningDownload(CATALOG)).toBeNull();
    expect(runningDownload(null)).toBeNull();
    expect(runningDownload(withRunning(5))?.id).toBe("qwen3.8-27b-abliterated");
  });
});

describe("downloadsFromPage", () => {
  it("needs both download.enabled and download.from_ui, and assumes yes when the configuration is unread", () => {
    expect(downloadsFromPage(CONFIG)).toBe(true);
    expect(downloadsFromPage(null)).toBe(true);
    expect(downloadsFromPage({ ...CONFIG, download: { ...CONFIG.download, from_ui: false } })).toBe(false);
    expect(downloadsFromPage({ ...CONFIG, download: { ...CONFIG.download, enabled: false } })).toBe(false);
  });
});

describe("the speed", () => {
  it("is the bytes landed between the readings of the last five seconds, per second", () => {
    let progress = nextProgress(null, withRunning(1 * GIB), 0);
    expect(speedOf(progress)).toBeNull();
    progress = nextProgress(progress, withRunning(1 * GIB + 50 * MIB), 1_000);
    expect(speedOf(progress)).toBe(50 * MIB);
    progress = nextProgress(progress, withRunning(1 * GIB + 150 * MIB), 2_000);
    expect(speedOf(progress)).toBe(75 * MIB);
    // Readings older than five seconds drop out.
    progress = nextProgress(progress, withRunning(1 * GIB + 450 * MIB), 6_000);
    expect(progress?.samples.map((s) => s.at)).toEqual([1_000, 2_000, 6_000]);
    expect(speedOf(progress)).toBe(80 * MIB);
  });

  it("starts afresh for another download, and is gone once none runs", () => {
    const first = nextProgress(nextProgress(null, withRunning(0), 0), withRunning(MIB), 1_000);
    const other = CATALOG.map((row, i) => (i === 3 ? { ...row, download: "running" as const, bytes_on_disk: 7 } : row));
    expect(nextProgress(first, other, 2_000)).toEqual({ id: "qwen3.8-27b-custom", samples: [{ bytes: 7, at: 2_000 }] });
    expect(nextProgress(first, CATALOG, 2_000)).toBeNull();
  });
});

describe("the row's words", () => {
  it("says how much of the total is on disk, and the speed once read", () => {
    expect(progressLabel(running(1.5 * GIB), null)).toBe("Downloading, 1.5 GiB of 18.07 GiB");
    expect(progressLabel(running(1.5 * GIB), 85.3 * MIB)).toBe("Downloading, 1.5 GiB of 18.07 GiB, 85.3 MiB/s");
  });

  it("gives a failed download's reason, as the server words it", () => {
    const reason = "GET http://hf/acme/m/resolve/v1/m.ninfer: 401 Unauthorized: the endpoint wants a token and none was sent";
    expect(downloadFailure({ ...CATALOG[1], download: "failed", download_reason: reason })).toBe(reason);
    expect(downloadFailure(CATALOG[1])).toBeNull();
  });
});

describe("afterDownload", () => {
  it("follows a download that began, and the one already running when refused for it", () => {
    expect(afterDownload({ ok: true, status: 202, value: {} })).toEqual({ refusal: null, follow: true });
    const busy = { ok: false as const, status: 409, message: "a download of x is already running", code: "download_in_progress" };
    expect(afterDownload(busy)).toEqual({ refusal: busy.message, follow: true });
    const off = { ok: false as const, status: 403, message: "download.from_ui is false", code: "download_disabled" };
    expect(afterDownload(off)).toEqual({ refusal: off.message, follow: false });
  });

  // GitHub #378: the server keeps a refusal for want of room on the entry's row,
  // so the view reads the catalog to show it there, not in the band.
  it("reads the row that keeps a refusal for want of room, instead of showing it in the band", () => {
    const short = { ok: false as const, status: 507, message: "not enough room for x under ./models: 2147483648 bytes needed", code: "download_no_room" };
    expect(afterDownload(short)).toEqual({ refusal: null, follow: true });
  });
});

describe("followDownloads", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  const reads = (answers: Answer<CatalogRow[]>[]) => {
    const read = vi.fn(async () => answers.shift() ?? { ok: true as const, status: 200, value: CATALOG });
    return read;
  };

  it("reads the catalog every second while a download runs, and stops once none does", async () => {
    const read = reads([
      { ok: true, status: 200, value: withRunning(1) },
      { ok: true, status: 200, value: withRunning(2) },
      { ok: true, status: 200, value: CATALOG },
    ]);
    const seen: Answer<CatalogRow[]>[] = [];
    const stop = followDownloads(read, (answer) => seen.push(answer));
    await vi.advanceTimersByTimeAsync(999);
    expect(read).toHaveBeenCalledTimes(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(read).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(2_000);
    expect(read).toHaveBeenCalledTimes(3);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(read).toHaveBeenCalledTimes(3);
    expect(seen.map((answer) => answer.ok && runningDownload(answer.value)?.bytes_on_disk)).toEqual([1, 2, undefined]);
    stop();
  });

  it("tries a failed read again a second later, and reads nothing once stopped", async () => {
    const read = reads([{ ok: false, status: 0, message: "Could not reach ignis" }, { ok: true, status: 200, value: withRunning(3) }]);
    const stop = followDownloads(read, () => {});
    await vi.advanceTimersByTimeAsync(2_000);
    expect(read).toHaveBeenCalledTimes(2);
    stop();
    await vi.advanceTimersByTimeAsync(5_000);
    expect(read).toHaveBeenCalledTimes(2);
  });
});
