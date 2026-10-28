// A site across files: the HTML file's own references to files of the same
// workspace, found and put into the document, so that the sandboxed frame
// (which has no origin to fetch from) shows the whole page. Only <link
// rel="stylesheet" href>, <script src> and <img src> are read; every other
// reference stays as written. Strings in, strings out: there is no DOM here.

/** The part of a reference up to its query or fragment, undone from %-escapes, as a workspace file name from the folder of `from`; null for a reference that is not one (absolute, `data:`, protocol-relative, a fragment, above the root). */
export function resolveRef(ref: string, from: string): string | null {
  const raw = ref.trim();
  if (raw === "" || raw.startsWith("#") || raw.startsWith("//") || /^[a-z][a-z0-9+.-]*:/i.test(raw)) return null;
  let path = raw.split(/[?#]/, 1)[0];
  try {
    path = decodeURIComponent(path);
  } catch {
    // An escape that is not one: the name as written.
  }
  const parts = path.startsWith("/") ? [] : from.split("/").slice(0, -1);
  for (const part of path.split("/")) {
    if (part === "" || part === ".") continue;
    if (part === "..") {
      if (parts.length === 0) return null;
      parts.pop();
    } else parts.push(part);
  }
  return parts.length > 0 ? parts.join("/") : null;
}

export type SiteRef = { tag: "stylesheet" | "script" | "image"; ref: string; file: string };

const ATTRS = String.raw`(?:"[^"]*"|'[^']*'|[^'">])*`;
const LINK = new RegExp(String.raw`<link\b(${ATTRS})>`, "gi");
const SCRIPT = new RegExp(String.raw`<script\b(${ATTRS})>([\s\S]*?)</script\s*>`, "gi");
const IMG = new RegExp(String.raw`<img\b(${ATTRS})>`, "gi");
const ATTR = /([^\s"'<>/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;

/** The attributes of a tag's inside, names in lower case, in the order written. */
function attributes(inside: string): { name: string; value: string | undefined; start: number; end: number }[] {
  const out = [];
  for (const m of inside.matchAll(ATTR)) {
    out.push({ name: m[1].toLowerCase(), value: m[2] ?? m[3] ?? m[4], start: m.index, end: m.index + m[0].length });
  }
  return out;
}

const unescapeAttr = (v: string) => v.replace(/&amp;/gi, "&").replace(/&quot;/gi, '"').replace(/&#0*39;|&apos;/gi, "'");

type Hit = { tag: SiteRef["tag"]; start: number; end: number; ref: string; file: string; inside: string; body: string };

function hits(html: string, from: string): Hit[] {
  const out: Hit[] = [];
  const add = (re: RegExp, tag: Hit["tag"], attr: string, want?: (a: ReturnType<typeof attributes>) => boolean) => {
    for (const m of html.matchAll(re)) {
      const attrs = attributes(m[1]);
      if (want && !want(attrs)) continue;
      const value = attrs.find((a) => a.name === attr)?.value;
      if (value === undefined) continue;
      const ref = unescapeAttr(value);
      const file = resolveRef(ref, from);
      if (file) out.push({ tag, start: m.index, end: m.index + m[0].length, ref, file, inside: m[1], body: m[2] ?? "" });
    }
  };
  add(LINK, "stylesheet", "href", (a) => a.some((x) => x.name === "rel" && /(^|\s)stylesheet(\s|$)/i.test(x.value ?? "")));
  add(SCRIPT, "script", "src");
  add(IMG, "image", "src");
  return out.sort((a, b) => a.start - b.start);
}

/** The references of `html` (a file called `from`) that name a file, in document order; the caller keeps those that are workspace files. */
export function siteRefs(html: string, from: string): SiteRef[] {
  return hits(html, from).map(({ tag, ref, file }) => ({ tag, ref, file }));
}

/** What a referenced file puts into the page: a stylesheet's or script's text, a picture's data URL. A file left out stays as written. */
export type SiteFiles = ReadonlyMap<string, string>;

const dropAttrs = (inside: string, names: string[]) => {
  let out = inside;
  for (const a of attributes(inside).reverse()) if (names.includes(a.name)) out = out.slice(0, a.start) + out.slice(a.end);
  return out;
};

const setAttr = (inside: string, name: string, value: string) => {
  const a = attributes(inside).find((x) => x.name === name);
  return a ? `${inside.slice(0, a.start)}${name}="${value}"${inside.slice(a.end)}` : inside;
};

/**
 * `html` with the references whose file is in `files` put into it: a
 * stylesheet link becomes a <style>, a script's src becomes its body, an
 * image's src a data URL. Everything else is left byte for byte.
 */
export function inlineSite(html: string, from: string, files: SiteFiles): string {
  let out = "";
  let at = 0;
  for (const hit of hits(html, from)) {
    const content = files.get(hit.file);
    if (content === undefined || hit.start < at) continue;
    out += html.slice(at, hit.start);
    at = hit.end;
    if (hit.tag === "stylesheet") {
      const media = attributes(hit.inside).find((a) => a.name === "media")?.value;
      out += `<style${media ? ` media="${media.replace(/"/g, "&quot;")}"` : ""}>${content.replace(/<\/style/gi, "<\\/style")}</style>`;
    } else if (hit.tag === "script") {
      const rest = dropAttrs(hit.inside, ["src", "integrity", "crossorigin"]).trim();
      out += `<script${rest ? ` ${rest}` : ""}>${content.replace(/<\/script/gi, "<\\/script").replace(/<!--/g, "<\\!--")}</script>`;
    } else {
      out += `<img${setAttr(hit.inside, "src", content.replace(/"/g, "&quot;"))}>`;
    }
  }
  return out + html.slice(at);
}

/** A picture's bytes as a data URL, which loads inside a frame that has no origin. */
export function dataUrl(type: string, bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return `data:${type || "application/octet-stream"};base64,${btoa(binary)}`;
}

/** An SVG's text as the data URL of an image, which a browser draws without running its scripts. */
export const svgDataUrl = (svg: string) => `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;

export const isMarkdownName = (name: string) => /\.(md|markdown)$/i.test(name);
export const isSvgName = (name: string) => /\.svg$/i.test(name);
export const isCsvName = (name: string) => /\.csv$/i.test(name);

/** CSV text as rows of fields: quoted fields may hold commas, quotes ("") and line breaks; a final line break makes no empty row. */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  let any = false;
  const s = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (quoted) {
      if (c === '"') {
        if (s[i + 1] === '"') {
          field += '"';
          i++;
        } else quoted = false;
      } else field += c;
    } else if (c === '"' && field === "") {
      quoted = true;
      any = true;
    } else if (c === ",") {
      row.push(field);
      field = "";
      any = true;
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && s[i + 1] === "\n") i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      any = false;
    } else {
      field += c;
      any = true;
    }
  }
  if (any || field !== "") {
    row.push(field);
    rows.push(row);
  }
  return rows;
}
