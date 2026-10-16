import type { ImageStore, StoredImage } from "./imageStore.ts";

// Pictures into the store, as the page makes them: measured, and with a
// thumbnail beside, so a strip of a hundred 2K pictures decodes a hundred
// small ones. Browser only: it needs createImageBitmap and a canvas.

/** The long side of a thumbnail. */
export const THUMB_EDGE = 384;

export async function measure(blob: Blob): Promise<{ width: number; height: number }> {
  const bitmap = await createImageBitmap(blob);
  try {
    return { width: bitmap.width, height: bitmap.height };
  } finally {
    bitmap.close();
  }
}

/** A WebP no longer than `edge` on its long side, its transparency kept; none when the browser cannot make one. */
export async function makeThumb(blob: Blob, edge = THUMB_EDGE): Promise<{ thumb?: Blob; width: number; height: number }> {
  const bitmap = await createImageBitmap(blob);
  try {
    const { width, height } = bitmap;
    const scale = Math.min(1, edge / Math.max(width, height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) return { width, height };
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const thumb = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/webp", 0.82));
    return { ...(thumb ? { thumb } : {}), width, height };
  } finally {
    bitmap.close();
  }
}

/** A picture into the store with its size and thumbnail. */
export async function keepPicture(store: ImageStore, blob: Blob): Promise<StoredImage> {
  const { thumb, width, height } = await makeThumb(blob);
  return store.putImage(blob, { width, height, ...(thumb ? { thumb } : {}) });
}

/** A picture with no thumbnail of its own worth making: a mask, an area. */
export async function keepPlain(store: ImageStore, blob: Blob): Promise<StoredImage> {
  return store.putImage(blob, await measure(blob));
}
