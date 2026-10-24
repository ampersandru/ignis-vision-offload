// The sidebar, read from docs/user/ (ADR 0066). Starlight's `autogenerate`
// groups pages by their path under src/content/docs/, and the guide is loaded
// from where it lives instead, so each section is listed here: every page of
// the section's folder, by its front matter's `sidebar.order`, then its name.

import fs from "node:fs";
import path from "node:path";
import { parse } from "yaml";

/** The front matter fields the sidebar reads: title, sidebar.order, .label, .badge. */
export function sidebarFields(text) {
  const block = text.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  const data = (block && parse(block[1])) || {};
  const sidebar = data.sidebar ?? {};
  const badge = typeof sidebar.badge === "string" ? { text: sidebar.badge } : (sidebar.badge ?? {});
  return { title: data.title, order: sidebar.order, label: sidebar.label, badge: badge.text, variant: badge.variant };
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
