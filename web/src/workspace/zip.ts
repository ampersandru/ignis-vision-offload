// A zip writer and reader of our own (spec #170, #397), pure apart from the
// platform's deflate streams: local headers, the central directory, CRC-32,
// and entries stored or deflated ("deflate-raw"). No zip64, no encryption.
// The reader takes what other tools make: data descriptors, directory
// entries, UTF-8 names; an entry it cannot read is refused by name.

export type ZipEntry = {
  name: string;
  data: Uint8Array;
  /** Milliseconds; a zip keeps a DOS date, two seconds apart from 1980. */
  modified?: number;
};

const SIG_LOCAL = 0x04034b50;
const SIG_CENTRAL = 0x02014b50;
const SIG_END = 0x06054b50;
const UTF8_FLAG = 0x0800;

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(data: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

async function through(data: Uint8Array, stream: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const piped = new Blob([data as BlobPart]).stream().pipeThrough(stream as unknown as ReadableWritablePair<Uint8Array, Uint8Array>);
  return new Uint8Array(await new Response(piped).arrayBuffer());
}

const deflate = (data: Uint8Array) => through(data, new CompressionStream("deflate-raw"));
const inflate = (data: Uint8Array) => through(data, new DecompressionStream("deflate-raw"));

function dosStamp(ms: number): { time: number; date: number } {
  const d = new Date(ms);
  const year = Math.min(Math.max(d.getFullYear(), 1980), 2107);
  return {
    time: (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1),
    date: ((year - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate(),
  };
}

/** The zip holding `entries`, each deflated when that makes it smaller. */
export async function writeZip(entries: ZipEntry[]): Promise<Uint8Array> {
  if (entries.length > 0xfffe) throw new Error("A zip holds at most 65,534 files.");
  const encoder = new TextEncoder();
  const parts: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;
  for (const entry of entries) {
    const name = encoder.encode(entry.name);
    const packed = await deflate(entry.data);
    const method = packed.length < entry.data.length ? 8 : 0;
    const body = method === 8 ? packed : entry.data;
    if (body.length > 0xfffffffe || offset > 0xfffffffe) throw new Error("A zip made here holds at most 4 GB.");
    const { time, date } = dosStamp(entry.modified ?? Date.now());
    const crc = crc32(entry.data);
    const version = method === 8 ? 20 : 10;
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, SIG_LOCAL, true);
    local.setUint16(4, version, true);
    local.setUint16(6, UTF8_FLAG, true);
    local.setUint16(8, method, true);
    local.setUint16(10, time, true);
    local.setUint16(12, date, true);
    local.setUint32(14, crc, true);
    local.setUint32(18, body.length, true);
    local.setUint32(22, entry.data.length, true);
    local.setUint16(26, name.length, true);
    const head = new DataView(new ArrayBuffer(46));
    head.setUint32(0, SIG_CENTRAL, true);
    head.setUint16(4, 20, true);
    head.setUint16(6, version, true);
    head.setUint16(8, UTF8_FLAG, true);
    head.setUint16(10, method, true);
    head.setUint16(12, time, true);
    head.setUint16(14, date, true);
    head.setUint32(16, crc, true);
    head.setUint32(20, body.length, true);
    head.setUint32(24, entry.data.length, true);
    head.setUint16(28, name.length, true);
    head.setUint32(42, offset, true);
    central.push(new Uint8Array(head.buffer), name);
    parts.push(new Uint8Array(local.buffer), name, body);
    offset += 30 + name.length + body.length;
  }
  const size = central.reduce((n, p) => n + p.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, SIG_END, true);
  end.setUint16(8, entries.length, true);
  end.setUint16(10, entries.length, true);
  end.setUint32(12, size, true);
  end.setUint32(16, offset, true);
  const all = [...parts, ...central, new Uint8Array(end.buffer)];
  const out = new Uint8Array(all.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of all) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

/** A directory entry: its name ends in `/`. */
export const isDirectory = (name: string) => name.endsWith("/");

/** The entries of `zip`, directories included with no data; rejects with an Error whose message the user can read when the zip cannot be read. */
export async function readZip(zip: Uint8Array): Promise<ZipEntry[]> {
  const view = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
  const bad = (why: string) => new Error(`This is not a zip ignis can read: ${why}.`);
  let end = -1;
  for (let i = zip.length - 22; i >= Math.max(0, zip.length - 22 - 0xffff); i--) {
    if (view.getUint32(i, true) === SIG_END) {
      end = i;
      break;
    }
  }
  if (end < 0) throw bad("it has no central directory");
  const count = view.getUint16(end + 10, true);
  const size = view.getUint32(end + 12, true);
  let at = view.getUint32(end + 16, true);
  if (count === 0xffff || size === 0xffffffff || at === 0xffffffff) throw bad("it needs zip64, which is not supported");
  const decoder = new TextDecoder("utf-8");
  const entries: ZipEntry[] = [];
  for (let n = 0; n < count; n++) {
    if (at + 46 > zip.length || view.getUint32(at, true) !== SIG_CENTRAL) throw bad("its central directory is damaged");
    const flags = view.getUint16(at + 8, true);
    const method = view.getUint16(at + 10, true);
    const crc = view.getUint32(at + 16, true);
    const packedSize = view.getUint32(at + 20, true);
    const fullSize = view.getUint32(at + 24, true);
    const nameLength = view.getUint16(at + 28, true);
    const extraLength = view.getUint16(at + 30, true);
    const commentLength = view.getUint16(at + 32, true);
    const local = view.getUint32(at + 42, true);
    const name = decoder.decode(zip.subarray(at + 46, at + 46 + nameLength));
    at += 46 + nameLength + extraLength + commentLength;
    if (isDirectory(name)) {
      entries.push({ name, data: new Uint8Array(0) });
      continue;
    }
    if (flags & 1) throw bad(`${name} is encrypted`);
    if (packedSize === 0xffffffff || fullSize === 0xffffffff || local === 0xffffffff) throw bad("it needs zip64, which is not supported");
    if (method !== 0 && method !== 8) throw bad(`${name} is compressed in a way that is not supported (method ${method})`);
    if (local + 30 > zip.length || view.getUint32(local, true) !== SIG_LOCAL) throw bad(`the header of ${name} is damaged`);
    const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true);
    if (start + packedSize > zip.length) throw bad(`${name} is cut short`);
    const body = zip.subarray(start, start + packedSize);
    let data: Uint8Array;
    try {
      data = method === 8 ? await inflate(body) : body.slice();
    } catch {
      throw bad(`${name} is damaged`);
    }
    if (data.length !== fullSize || crc32(data) !== crc) throw bad(`${name} is damaged (its checksum differs)`);
    entries.push({ name, data });
  }
  return entries;
}
