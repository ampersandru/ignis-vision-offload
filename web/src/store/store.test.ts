import { IDBFactory } from "fake-indexeddb";
import { describe, expect, it } from "vitest";
import { type ImageStore, sha256, unreferenced } from "./imageStore.ts";
import { openIndexedDbStore } from "./indexeddb.ts";
import { createMemoryStore } from "./memory.ts";

// The contract every ImageStore keeps (store/imageStore.ts), run on each
// adapter: the memory one, and IndexedDB on fake-indexeddb with a fresh
// factory per case. A new adapter — a local folder, the server — joins the
// list below and passes the same cases.

const ADAPTERS: [string, () => ImageStore][] = [
  ["memory", createMemoryStore],
  ["indexeddb", () => openIndexedDbStore(new IDBFactory(), "test")],
];

const png = (...bytes: number[]) => new Blob([new Uint8Array(bytes)], { type: "image/png" });
const bytesOf = async (blob: Blob | null) => (blob ? [...new Uint8Array(await blob.arrayBuffer())] : null);

for (const [name, make] of ADAPTERS) {
  describe(`the ${name} store`, () => {
    it("keeps an image by the SHA-256 of its bytes, and gives the bytes and their type back", async () => {
      const store = make();
      const stored = await store.putImage(png(1, 2, 3), { width: 4, height: 3 });
      expect(stored.id).toBe(await sha256(png(1, 2, 3)));
      expect(stored).toMatchObject({ type: "image/png", bytes: 3, width: 4, height: 3 });
      const back = await store.getImage(stored.id);
      expect(back?.type).toBe("image/png");
      expect(await bytesOf(back)).toEqual([1, 2, 3]);
      expect(await store.imageInfo(stored.id)).toEqual(stored);
    });

    it("keeps the same bytes once, and the first one's time", async () => {
      const store = make();
      const first = await store.putImage(png(9), { width: 1, height: 1 });
      const again = await store.putImage(png(9), { width: 1, height: 1 });
      expect(again).toEqual(first);
      expect(await store.listImages()).toHaveLength(1);
      expect(await store.usage()).toMatchObject({ bytes: 1, images: 1 });
    });

    it("keeps a thumbnail beside the image, given once", async () => {
      const store = make();
      const { id } = await store.putImage(png(1), { width: 1, height: 1 });
      expect(await store.getThumb(id)).toBeNull();
      await store.putImage(png(1), { width: 1, height: 1, thumb: new Blob([new Uint8Array([7])], { type: "image/webp" }) });
      const thumb = await store.getThumb(id);
      expect(thumb?.type).toBe("image/webp");
      expect(await bytesOf(thumb)).toEqual([7]);
    });

    it("forgets an image, its bytes and its thumbnail", async () => {
      const store = make();
      const { id } = await store.putImage(png(5), { width: 1, height: 1, thumb: png(6) });
      await store.deleteImage(id);
      expect(await store.getImage(id)).toBeNull();
      expect(await store.getThumb(id)).toBeNull();
      expect(await store.imageInfo(id)).toBeNull();
      expect(await store.usage()).toMatchObject({ bytes: 0, images: 0 });
    });

    it("keeps documents by collection and id, as copies", async () => {
      const store = make();
      const doc = { title: "Fox", pool: [{ n: 1, id: "abc" }] };
      await store.putDoc("image-sessions", "s1", doc);
      await store.putDoc("image-sessions", "s2", { title: "Logo", pool: [] });
      await store.putDoc("other", "s1", { title: "elsewhere" });
      doc.title = "changed after";
      expect(await store.getDoc("image-sessions", "s1")).toEqual({ title: "Fox", pool: [{ n: 1, id: "abc" }] });
      expect((await store.listDocs<{ title: string }>("image-sessions")).map((d) => d.title).sort()).toEqual(["Fox", "Logo"]);
      await store.putDoc("image-sessions", "s1", { title: "Fox 2", pool: [] });
      expect(await store.getDoc<{ title: string }>("image-sessions", "s1")).toMatchObject({ title: "Fox 2" });
      await store.deleteDoc("image-sessions", "s1");
      expect(await store.getDoc("image-sessions", "s1")).toBeNull();
      expect(await store.listDocs("image-sessions")).toHaveLength(1);
      expect(await store.getDoc("other", "s1")).toEqual({ title: "elsewhere" });
    });
  });
}

describe("what can go", () => {
  it("is every image asked about that nothing keeps", () => {
    expect(unreferenced(["a", "b", "a", "c"], new Set(["b"]))).toEqual(["a", "c"]);
  });
});
