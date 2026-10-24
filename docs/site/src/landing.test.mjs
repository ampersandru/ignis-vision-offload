// The front page links its cards and hero actions by route (MDX props and
// front matter, which the link plugin does not see): each one names a page.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const guide = fileURLToPath(new URL("../../user/", import.meta.url));
const landing = fs.readFileSync(path.join(guide, "index.mdx"), "utf8");

/** The page a route relative to the site's root names, or null. */
function pageOf(route) {
  const stem = route.replace(/\/$/, "");
  const candidates = [`${stem}.md`, `${stem}.mdx`, `${stem}/index.md`, `${stem}/index.mdx`];
  return candidates.map((c) => path.join(guide, c)).find((file) => fs.existsSync(file)) ?? null;
}

describe("the front page", () => {
  const routes = [...landing.matchAll(/(?:href="|link: )([^"\s]+)/g)]
    .map((m) => m[1])
    .filter((route) => !/^[a-z][a-z0-9+.-]*:/i.test(route));
  it("links some pages", () => {
    expect(routes.length).toBeGreaterThan(5);
  });
  it.each(routes)("%s names a page of the guide", (route) => {
    expect(pageOf(route)).not.toBeNull();
  });
});
