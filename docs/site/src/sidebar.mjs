// The sidebar, read from docs/user/ (ADR 0066). Starlight's `autogenerate`
// groups pages by their path under src/content/docs/, and the guide is loaded
// from where it lives instead, so each section is listed here: every page of
// the section's folder, by its front matter's `sidebar.order`, then its name.

import fs from "node:fs";
import path from "node:path";

/** The front matter fields the sidebar reads: title, sidebar.order, .label, .badge. */
export function sidebarFields(text) {
  const block = text.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  const lines = block ? block[1].split(/\r?\n/) : [];
  const unquote = (value) => value.trim().replace(/^"(.*)"$/, "$1").replace(/\\"/g, '"');
  const fields = { title: undefined, order: undefined, label: undefined, badge: undefined, variant: undefined };
  let section = "";
  for (const line of lines) {
    const top = line.match(/^(\w+):\s*(.*)$/);
    if (top) {
      section = top[1];
      if (section === "title") fields.title = unquote(top[2]);
      continue;
    }
    const nested = line.match(/^(\s+)(\w+):\s*(.*)$/);
    if (!nested || section !== "sidebar") continue;
    const [, indent, key, value] = nested;
    if (indent.length === 2 && key === "order") fields.order = Number(value);
    else if (indent.length === 2 && key === "label") fields.label = unquote(value);
    else if (indent.length === 4 && key === "text") fields.badge = unquote(value);
    else if (indent.length === 4 && key === "variant") fields.variant = unquote(value);
  }
  return fields;
}

/** A section's sidebar items: one per page of `docs/user/<directory>/`. */
export function sectionItems(guide, directory) {
  const dir = path.join(guide, directory);
  const pages = fs
    .readdirSync(dir)
    .filter((name) => /\.mdx?$/.test(name) && !name.startsWith("_"))
    .map((name) => {
      const fields = sidebarFields(fs.readFileSync(path.join(dir, name), "utf8"));
      const stem = name.replace(/\.mdx?$/, "");
      const slug = stem === "index" ? directory : `${directory}/${stem}`;
      const item = { slug, label: fields.label ?? fields.title ?? stem };
      if (fields.badge) item.badge = { text: fields.badge, variant: fields.variant ?? "default" };
      return { item, order: fields.order ?? Number.MAX_SAFE_INTEGER, name };
    });
  pages.sort((a, b) => a.order - b.order || a.name.localeCompare(b.name));
  return pages.map((page) => page.item);
}
