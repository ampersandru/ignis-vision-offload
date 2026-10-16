import { type ImageId, type ImageStore, sha256, type StoredImage } from "./imageStore.ts";

/** An `ImageStore` in memory: what the tests use, and the page's fallback where the browser has no IndexedDB. */
export function createMemoryStore(): ImageStore {
  const images = new Map<ImageId, { info: StoredImage; blob: Blob; thumb?: Blob }>();
  const docs = new Map<string, Map<string, unknown>>();
  const collection = (name: string) => {
    let c = docs.get(name);
    if (!c) docs.set(name, (c = new Map()));
    return c;
  };
  // A document goes in and comes out as a copy, as it would through IndexedDB: nobody holds the store's own.
  const copy = <T,>(value: T): T => structuredClone(value);
  return {
    kind: "memory",
    async putImage(blob, info) {
      const id = await sha256(blob);
      const had = images.get(id);
      if (had) {
        if (info.thumb && !had.thumb) had.thumb = info.thumb;
        return { ...had.info };
      }
      const stored: StoredImage = { id, type: blob.type || "application/octet-stream", bytes: blob.size, width: info.width, height: info.height, added: Date.now() };
      images.set(id, { info: stored, blob, ...(info.thumb ? { thumb: info.thumb } : {}) });
      return { ...stored };
    },
    getImage: async (id) => images.get(id)?.blob ?? null,
    getThumb: async (id) => images.get(id)?.thumb ?? null,
    imageInfo: async (id) => (images.has(id) ? { ...images.get(id)!.info } : null),
    listImages: async () => [...images.values()].map((i) => ({ ...i.info })),
    deleteImage: async (id) => void images.delete(id),
    putDoc: async (name, id, value) => void collection(name).set(id, copy(value)),
    getDoc: async <T,>(name: string, id: string) => (collection(name).has(id) ? copy(collection(name).get(id) as T) : null),
    listDocs: async <T,>(name: string) => [...collection(name).values()].map((v) => copy(v as T)),
    deleteDoc: async (name, id) => void collection(name).delete(id),
    usage: async () => {
      const all = [...images.values()];
      return { bytes: all.reduce((n, i) => n + i.info.bytes, 0), images: all.length };
    },
  };
}
