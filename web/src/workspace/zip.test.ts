import { describe, expect, it } from "vitest";
import { crc32, isDirectory, readZip, writeZip } from "./zip.ts";

// The zip writer and reader: a zip of ours reads back, and a zip made by
// another tool (Python's zipfile, streamed: deflate, data descriptors, a
// directory entry, UTF-8 names) reads too.

const bytes = (s: string) => new TextEncoder().encode(s);
const text = (b: Uint8Array) => new TextDecoder().decode(b);

// Written by Python's zipfile to a stream that cannot seek.
const FOREIGN =
  "UEsDBBQACAAAAAAAIQAAAAAAAAAAAAAAAAAFAAAAZG9jcy9QSwcIAAAAAAAAAAAAAAAAUEsDBBQACAgIABgtXF0AAAAAAAAAAAAAAAAS" +
  "AAAAZG9jcy9jYWZmw6ggw7wudHh0y0jNyclXKM8vyklRyBhlj7KpxAYAUEsHCA0urSUVAAAAWAIAAFBLAwQUAAgACAAYLVxdAAAAAAAA" +
  "AAAAAAAACQAAAHBsYWluLnR4dCsuyS9KTbEHAFBLBwjBWODzCQAAAAcAAABQSwECFAAUAAgAAAAAACEAAAAAAAAAAAAAAAAABQAAAAAA" +
  "AAAAAAAAgAEAAAAAZG9jcy9QSwECFAAUAAgICAAYLVxdDS6tJRUAAABYAgAAEgAAAAAAAAAAAAAAgAEzAAAAZG9jcy9jYWZmw6ggw7wu" +
  "dHh0UEsBAhQAFAAIAAgAGC1cXcFY4PMJAAAABwAAAAkAAAAAAAAAAAAAAIABiAAAAHBsYWluLnR4dFBLBQYAAAAAAwADAKoAAADIAAAA" +
  "AAA=";

describe("zip", () => {
  it("checksums as the standard CRC-32 does", () => {
    expect(crc32(bytes("123456789"))).toBe(0xcbf43926);
  });

  it("reads back what it wrote: deflated and stored entries, empty files, names with / and non-ASCII", async () => {
    const binary = Uint8Array.from({ length: 300 }, (_, i) => (i * 7919) % 256);
    const zip = await writeZip([
      { name: "site/index.html", data: bytes("<p>hello</p>".repeat(100)) },
      { name: "caffè/ü.txt", data: bytes("x") },
      { name: "empty.txt", data: new Uint8Array(0) },
      { name: "noise.bin", data: binary },
    ]);
    const back = await readZip(zip);
    expect(back.map((e) => e.name)).toEqual(["site/index.html", "caffè/ü.txt", "empty.txt", "noise.bin"]);
    expect(text(back[0].data)).toBe("<p>hello</p>".repeat(100));
    // The repeated text is deflated: the zip is far smaller than its 1200 bytes.
    expect(zip.length).toBeLessThan(900);
    expect(text(back[1].data)).toBe("x");
    expect(back[2].data.length).toBe(0);
    expect([...back[3].data]).toEqual([...binary]);
  });

  it("reads a zip another tool made: data descriptors, a directory entry, a UTF-8 name", async () => {
    const back = await readZip(Uint8Array.from(Buffer.from(FOREIGN, "base64")));
    expect(back.map((e) => e.name)).toEqual(["docs/", "docs/caffè ü.txt", "plain.txt"]);
    expect(isDirectory(back[0].name)).toBe(true);
    expect(text(back[1].data)).toBe("hello world ".repeat(50));
    expect(text(back[2].data)).toBe("stored?");
  });

  it("refuses what is not a zip, a damaged entry and an encrypted one, saying so", async () => {
    await expect(readZip(bytes("just some text, not a zip at all"))).rejects.toThrow(/not a zip ignis can read/);
    const zip = await writeZip([{ name: "a.txt", data: bytes("hello") }]);
    const flipped = zip.slice();
    flipped[30 + 5] ^= 0xff; // a byte of the stored content
    await expect(readZip(flipped)).rejects.toThrow(/a\.txt is damaged/);
    const encrypted = zip.slice();
    encrypted[zip.length - 22 - (46 + 5) + 8] |= 1;
    await expect(readZip(encrypted)).rejects.toThrow(/a\.txt is encrypted/);
  });
});
