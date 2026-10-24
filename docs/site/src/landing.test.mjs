// The MDX pages (the front page, Why ignis) link their cards, tiles and hero
// actions by route (MDX props, raw HTML and front matter, which the link
// plugin does not see): each one names a page.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const guide = fileURLToPath(new URL("../../user/", import.meta.url));

/** Every .mdx page of the guide, relative to it. */
function mdxPages(dir = guide) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return mdxPages(full);
    return entry.name.endsWith(".mdx") ? [path.relative(guide, full)] : [];
  });
}

// Written by the build, not committed (src/changelog.mjs): the tests run before it.
const GENERATED = [path.join(guide, "changelog")];

/** The page a route names, read from the directory the route is relative to, or null. */
function pageOf(route, from) {
  const stem = path.join(from, route.split("#")[0].replace(/\/$/, ""));
  if (GENERATED.some((dir) => stem === dir || stem.startsWith(dir + path.sep))) return stem;
  const candidates = [`${stem}.md`, `${stem}.mdx`, `${stem}/index.md`, `${stem}/index.mdx`];
  return candidates.find((file) => fs.existsSync(file)) ?? null;
}

describe.each(mdxPages())("%s", (page) => {
  const text = fs.readFileSync(path.join(guide, page), "utf8");
  // A page's route is its directory for an index, else its own name: relative routes start there.
  const from = path.join(guide, path.basename(page, ".mdx") === "index" ? path.dirname(page) : page.replace(/\.mdx$/, ""));
  const routes = [...text.matchAll(/(?:href="|link: )([^"\s]+)/g)]
    .map((m) => m[1])
    .filter((route) => !/^[a-z][a-z0-9+.-]*:/i.test(route) && !route.startsWith("#"));
  it("links some pages", () => {
    expect(routes.length).toBeGreaterThan(5);
  });
  it.each(routes)("%s names a page of the guide", (route) => {
    expect(pageOf(route, from)).not.toBeNull();
  });
});
