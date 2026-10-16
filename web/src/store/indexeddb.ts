import { type ImageId, type ImageStore, sha256, type StoredImage } from "./imageStore.ts";

// The `ImageStore` on IndexedDB. Four object stores: `images` holds what is
// known of each image (StoredImage) and `bytes` its bytes, so listing the
// images or adding up their size never reads a picture; `thumbs` the small
// copies; `docs` the documents, keyed [collection, id] with an index on the
// collection. Bytes are kept as an ArrayBuffer with their type, not as a
// Blob: every structured-clone implementation copies an ArrayBuffer, which is
// not true of a Blob.

const VERSION = 1;

type Bytes = { id: ImageId; type: string; data: ArrayBuffer };
type Doc = { collection: string; id: string; value: unknown };

const done = <T,>(request: IDBRequest<T>) =>
  new Promise<T>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });

const finished = (tx: IDBTransaction) =>
  new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error ?? new Error("the transaction was aborted"));
  });

function open(factory: IDBFactory, name: string): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = factory.open(name, VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      db.createObjectStore("images", { keyPath: "id" });
      db.createObjectStore("bytes", { keyPath: "id" });
      db.createObjectStore("thumbs", { keyPath: "id" });
      db.createObjectStore("docs", { keyPath: ["collection", "id"] }).createIndex("collection", "collection");
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

const asBlob = (bytes: Bytes | undefined) => (bytes ? new Blob([bytes.data], { type: bytes.type }) : null);

/** The store on `factory` (the browser's `indexedDB`), in the database `name`; opened at the first call, not before. */
export function openIndexedDbStore(factory: IDBFactory, name = "ignis-playground"): ImageStore {
  let db: Promise<IDBDatabase> | null = null;
  const connect = () => (db ??= open(factory, name));
  async function read<T>(store: string, key: IDBValidKey): Promise<T | undefined> {
    const tx = (await connect()).transaction(store, "readonly");
    return (await done(tx.objectStore(store).get(key))) as T | undefined;
  }
  async function listImages(): Promise<StoredImage[]> {
    const tx = (await connect()).transaction("images", "readonly");
    return (await done(tx.objectStore("images").getAll())) as StoredImage[];
  }
  return {
    kind: "indexeddb",
    async putImage(blob, info) {
      const id = await sha256(blob);
      const data = await blob.arrayBuffer();
      const thumb = info.thumb ? { id, type: info.thumb.type, data: await info.thumb.arrayBuffer() } : null;
      const tx = (await connect()).transaction(["images", "bytes", "thumbs"], "readwrite");
      const had = (await done(tx.objectStore("images").get(id))) as StoredImage | undefined;
      const stored: StoredImage = had ?? { id, type: blob.type || "application/octet-stream", bytes: blob.size, width: info.width, height: info.height, added: Date.now() };
      if (!had) {
        tx.objectStore("images").put(stored);
        tx.objectStore("bytes").put({ id, type: stored.type, data } satisfies Bytes);
      }
      if (thumb && !(await done(tx.objectStore("thumbs").count(id)))) tx.objectStore("thumbs").put(thumb);
      await finished(tx);
      return { ...stored };
    },
    getImage: async (id) => asBlob(await read<Bytes>("bytes", id)),
    getThumb: async (id) => asBlob(await read<Bytes>("thumbs", id)),
    imageInfo: async (id) => (await read<StoredImage>("images", id)) ?? null,
    listImages,
    async deleteImage(id) {
      const tx = (await connect()).transaction(["images", "bytes", "thumbs"], "readwrite");
      for (const store of ["images", "bytes", "thumbs"]) tx.objectStore(store).delete(id);
      await finished(tx);
    },
    async putDoc(collection, id, value) {
      const tx = (await connect()).transaction("docs", "readwrite");
      tx.objectStore("docs").put({ collection, id, value } satisfies Doc);
      await finished(tx);
    },
    getDoc: async <T,>(collection: string, id: string) => ((await read<Doc>("docs", [collection, id]))?.value as T | undefined) ?? null,
    async listDocs<T>(collection: string) {
      const tx = (await connect()).transaction("docs", "readonly");
      const docs = (await done(tx.objectStore("docs").index("collection").getAll(collection))) as Doc[];
      return docs.map((d) => d.value as T);
    },
    async deleteDoc(collection, id) {
      const tx = (await connect()).transaction("docs", "readwrite");
      tx.objectStore("docs").delete([collection, id]);
      await finished(tx);
    },
    async usage() {
      const images = await listImages();
      const estimate = await globalThis.navigator?.storage?.estimate?.().catch(() => undefined);
      return {
        bytes: images.reduce((n, i) => n + i.bytes, 0),
        images: images.length,
        ...(estimate?.quota ? { quota: estimate.quota } : {}),
      };
    },
  };
}
