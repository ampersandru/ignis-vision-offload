// The guide's llms.txt (llmstxt.org): the index an AI agent reads in place of
// the site. Every page of the sidebar is one link, in the sidebar's order, to
// its Markdown source at the ref being published, with its front matter's
// description: an agent reads plain Markdown, not the rendered pages.

import fs from "node:fs";
import path from "node:path";
import { parse } from "yaml";

/** The page a sidebar slug or link names, relative to the guide: `<slug>.md(x)`, or its folder's index. */
export function pageFile(guide, slug) {
  const stem = slug.replace(/^\/+|\/+$/g, "");
  const candidates = [`${stem}.md`, `${stem}.mdx`, `${stem}/index.md`, `${stem}/index.mdx`];
  const file = candidates.find((name) => fs.existsSync(path.join(guide, name)));
  if (!file) throw new Error(`llms.txt: the sidebar names ${slug}, and the guide has no page for it`);
  return file;
}

/** The front matter fields a line reads: title and description. */
function pageFields(text) {
  const block = text.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  const data = (block && parse(block[1])) || {};
  return { title: data.title, description: data.description };
}

/**
 * The text of llms.txt: `# title`, `> summary`, the details paragraph, a `##`
 * section per sidebar group (a lone link is a section of one page), then the
 * `## Optional` links. `source` is the URL the guide's folder is read at as
 * Markdown, ending in a slash.
 */
export function llmsTxt({ guide, title, summary, details, sidebar, source, optional = [] }) {
  const line = (item) => {
    const file = pageFile(guide, item.slug);
    const fields = pageFields(fs.readFileSync(path.join(guide, file), "utf8"));
    const about = fields.description ? `: ${fields.description}` : "";
    return `- [${fields.title ?? item.label}](${source}${file})${about}`;
  };
  const sections = sidebar.map((entry) => {
    const items = entry.items ?? [{ slug: entry.link, label: entry.label }];
    return [`## ${entry.label}`, "", ...items.map(line)].join("\n");
  });
  if (optional.length > 0) {
    sections.push(["## Optional", "", ...optional.map((link) => `- [${link.label}](${link.url}): ${link.description}`)].join("\n"));
  }
  return [`# ${title}`, `> ${summary}`, details, ...sections].join("\n\n") + "\n";
}

/** The integration that writes llms.txt at the root of the built site. */
export function llmsTxtFile(options) {
  return {
    name: "ignis-llms-txt",
    hooks: {
      "astro:build:done": ({ dir }) => {
        fs.writeFileSync(new URL("llms.txt", dir), llmsTxt(options));
      },
    },
  };
}
