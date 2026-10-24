import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { afterAll, describe, expect, it } from "vitest";
import relativeLinks, { headingAnchors, pageRoute, rewriteLink } from "./relative-links.mjs";

// A throwaway repository: docs/user is the guide, docs/adr a tree outside it.
const root = fs.mkdtempSync(path.join(os.tmpdir(), "relative-links-"));
const guide = path.join(root, "docs", "user");
const write = (relative, text) => {
  const file = path.join(root, relative);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, text);
  return file;
};
const home = write("docs/user/index.mdx", "---\ntitle: Home\n---\n");
const install = write(
  "docs/user/start/install.md",
  "---\ntitle: Install\n---\n## The container image\n\n### Fetching one: `ignis-server model download`\n\n```sh\n# not a heading\n```\n## The container image\n",
);
const api = write("docs/user/api/index.md", "## Deadlines\n");
write("docs/user/Bad_Name.md", "# x\n");
write("docs/adr/0046-one-declaration-per-config-field.md", "# ADR 0046\n");
write("README.md", "# ignis\n");
const options = { guide, repoRoot: root, repo: "https://github.com/o/r", ref: "v1.2.3", base: "/ignis" };
afterAll(() => fs.rmSync(root, { recursive: true, force: true }));

describe("headingAnchors", () => {
  it("slugs headings as GitHub does, skips code fences and numbers repeats", () => {
    expect([...headingAnchors(fs.readFileSync(install, "utf8"))]).toEqual([
      "the-container-image",
      "fetching-one-ignis-server-model-download",
      "the-container-image-1",
    ]);
  });
});

describe("pageRoute", () => {
  it("drops the extension and a trailing index", () => {
    expect(pageRoute("start/install.md", "/ignis")).toBe("/ignis/start/install/");
    expect(pageRoute("api/index.md", "/ignis")).toBe("/ignis/api/");
    expect(pageRoute("index.mdx", "/ignis")).toBe("/ignis/");
  });
  it("refuses a page name that is not already its route", () => {
    expect(() => pageRoute("Bad_Name.md", "/ignis")).toThrow(/Bad_Name/);
  });
});

describe("rewriteLink", () => {
  it("turns a page link into its route, anchor kept", () => {
    expect(rewriteLink("../api/index.md#deadlines", install, options)).toBe("/ignis/api/#deadlines");
    expect(rewriteLink("start/install.md", home, options)).toBe("/ignis/start/install/");
  });
  it("turns a repository file into its GitHub URL at the ref", () => {
    expect(rewriteLink("../../adr/0046-one-declaration-per-config-field.md", install, options)).toBe(
      "https://github.com/o/r/blob/v1.2.3/docs/adr/0046-one-declaration-per-config-field.md",
    );
    expect(rewriteLink("../../adr/", install, options)).toBe("https://github.com/o/r/tree/v1.2.3/docs/adr");
    expect(rewriteLink("../../../README.md", install, options)).toBe("https://github.com/o/r/blob/v1.2.3/README.md");
  });
  it("leaves absolute URLs, site paths and valid same-page anchors alone", () => {
    expect(rewriteLink("https://example.com/x.md", install, options)).toBeNull();
    expect(rewriteLink("mailto:a@b.c", install, options)).toBeNull();
    expect(rewriteLink("/ignis/api/", install, options)).toBeNull();
    expect(rewriteLink("#the-container-image-1", install, options)).toBeNull();
  });
  it("refuses a dead link: a missing page, a missing anchor, a missing file, a path out of the repo", () => {
    expect(() => rewriteLink("../api/gone.md", install, options)).toThrow(/does not exist/);
    expect(() => rewriteLink("../api/index.md#nothing", install, options)).toThrow(/#nothing/);
    expect(() => rewriteLink("#nothing-here", install, options)).toThrow(/#nothing-here/);
    expect(() => rewriteLink("../../adr/9999-gone.md", install, options)).toThrow(/does not have/);
    expect(() => rewriteLink("../../../../outside.md", install, options)).toThrow(/leaves the repository/);
  });
});

describe("the Sätteri plugin", () => {
  const run = (file, node) => {
    const plugin = relativeLinks(options)({ fileURL: pathToFileURL(file) });
    if (plugin) plugin[node.type](node, { setProperty: (n, key, value) => (n[key] = value) });
    return plugin;
  };
  it("rewrites the links and definitions of a guide page", () => {
    const link = { type: "link", url: "../api/index.md#deadlines", children: [] };
    const definition = { type: "definition", url: "../../adr/" };
    run(install, link);
    run(install, definition);
    expect(link.url).toBe("/ignis/api/#deadlines");
    expect(definition.url).toBe("https://github.com/o/r/tree/v1.2.3/docs/adr");
  });
  it("leaves a link it does not rewrite as written", () => {
    const link = { type: "link", url: "https://example.com/", children: [] };
    run(install, link);
    expect(link.url).toBe("https://example.com/");
  });
  it("stays out of files outside the guide, and of a compile with no file", () => {
    expect(run(path.join(root, "docs", "site", "page.md"), { type: "link", url: "gone.md" })).toBeNull();
    expect(relativeLinks(options)({ fileURL: undefined })).toBeNull();
  });
});
