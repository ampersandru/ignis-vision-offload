// The guide's Changelog section (#348): one page per ignis release, written
// from the GitHub releases when the site is built, so it always carries the
// notes as they stand on GitHub, edits included. The pages land in
// docs/user/changelog/, which git ignores: GitHub readers have the releases
// page itself. astro.config.mjs writes them before the sidebar is read.

import fs from "node:fs";
import path from "node:path";
import { stringify } from "yaml";

const RELEASE_TAG = /^v(\d+)\.(\d+)\.(\d+)$/;
const FENCE = /^\s*(```+|~~~+)/;

/** The published releases a reader installs: no drafts, no prereleases (`develop`), newest first. */
export function stableReleases(releases) {
  const version = (release) => release.tag_name.match(RELEASE_TAG).slice(1).map(Number);
  return releases
    .filter((release) => !release.draft && !release.prerelease && RELEASE_TAG.test(release.tag_name))
    .sort((a, b) => {
      const [x, y] = [version(a), version(b)];
      return y[0] - x[0] || y[1] - x[1] || y[2] - x[2];
    });
}

/** Each line of `markdown` outside fenced code, through `edit(line)` (null drops it); fenced lines as they are. */
function outsideFences(markdown, edit) {
  let fence = null;
  return markdown
    .split(/\r?\n/)
    .map((line) => {
      const marker = line.match(FENCE);
      if (marker) {
        if (fence === null) fence = marker[1][0];
        else if (marker[1][0] === fence) fence = null;
        return line;
      }
      return fence === null ? edit(line) : line;
    })
    .filter((line) => line !== null)
    .join("\n");
}

/**
 * The notes with what GitHub links by itself linked here too: `#123` to its
 * issue, `ADR 0046` to the ADR's file at `ref`. Code is left as written.
 */
export function linkReferences(markdown, { repo, ref, adrs }) {
  return outsideFences(markdown, (line) =>
    line
      .split(/(`[^`]*`)/)
      .map((part, i) => {
        if (i % 2 === 1) return part;
        return part
          .replace(/(^|[\s(,;\-–])#(\d{1,6})\b/g, (_, before, n) => `${before}[#${n}](${repo}/issues/${n})`)
          .replace(/\bADR (\d{4})\b/g, (text, n) => (adrs.has(n) ? `[ADR ${n}](${repo}/blob/${ref}/docs/adr/${adrs.get(n)})` : text));
      })
      .join(""),
  );
}

/**
 * The notes as a page body: the `## ignis vX` titles dropped (the page has
 * one; notes edited by hand carry a second above the release's file list),
 * the headings raised to start at `##`.
 */
export function noteBody(markdown) {
  const title = /^#{1,6}\s+ignis\s+v\d/i;
  const body = outsideFences(markdown, (line) => (title.test(line) ? null : line))
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  let lowest = 7;
  outsideFences(body, (line) => {
    const heading = line.match(/^(#{1,6})\s/);
    if (heading) lowest = Math.min(lowest, heading[1].length);
    return line;
  });
  if (lowest === 7 || lowest === 2) return body;
  return outsideFences(body, (line) => {
    const heading = line.match(/^(#{1,6})(\s.*)$/);
    return heading ? `${"#".repeat(Math.max(2, heading[1].length - (lowest - 2)))}${heading[2]}` : line;
  });
}

/**
 * One line on what a release changed, for the section's index, read from the
 * notes above the release's file list: their first item's bold lead, else
 * that item's opening clause, else the notes' opening sentence.
 */
export function headline(body) {
  // A lead before `: ` names the change when it is this long ("Config v2" is not); a table cell holds this much.
  const NAMED_LEAD = 12;
  const CELL = 140;
  const plain = (text) =>
    text
      .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
      .replace(/\*\*/g, "")
      .trim();
  const notes = body.split(/^##\s+Container image/m)[0];
  const first = notes.match(/^-\s+(.+)$/m);
  const bold = first?.[1].match(/^\*\*(.+?)\*\*/);
  if (bold) return plain(bold[1]).replace(/[.:]$/, "");
  if (first) {
    const item = plain(first[1]);
    const named = item.split(": ")[0];
    const clause = named !== item && named.length >= NAMED_LEAD ? named : item.split(/[.;] | — | \(/)[0];
    return clause.length > CELL ? `${clause.slice(0, CELL - 1).trimEnd()}…` : clause.replace(/[.:]$/, "");
  }
  const opening = notes.split(/\n\s*\n/).find((paragraph) => /^[A-Z]/.test(paragraph.trim()));
  if (opening) return plain(opening).split(/(?<=\.)\s/)[0].replace(/\.$/, "");
  return "The release's image and archives";
}

const day = (iso) => iso.slice(0, 10);
const frontMatter = (data) => `---\n${stringify(data).trimEnd()}\n---\n`;
/** A release's page name: its tag with dashes, which is its route. */
export const pageName = (tag) => `${tag.replace(/\./g, "-")}.md`;

/** A release's page: its notes, with the date and a link to the release's files. */
export function releasePage(release, { rank, repo, ref, adrs }) {
  const sidebar = { order: rank, label: release.tag_name };
  if (rank === 1) sidebar.badge = { text: "latest", variant: "success" };
  const head = frontMatter({
    title: `ignis ${release.tag_name}`,
    description: `What ignis ${release.tag_name} changed, released ${day(release.published_at)}.`,
    sidebar,
    editUrl: false,
    lastUpdated: false,
  });
  const body = linkReferences(noteBody(release.body ?? ""), { repo, ref, adrs });
  const intro = `Released ${day(release.published_at)}. The container image, the archives and the notes as written: [the release on GitHub](${release.html_url}).`;
  return `${head}\n${intro}\n\n${body}\n`;
}

/** The section's landing page: every release, newest first, with its date and headline. */
export function changelogIndex(releases, { repo }) {
  const cell = (text) => text.replace(/\|/g, "\\|");
  const rows = releases.map(
    (release) => `| [${release.tag_name}](${pageName(release.tag_name)}) | ${day(release.published_at)} | ${cell(headline(noteBody(release.body ?? "")))} |`,
  );
  return `${frontMatter({
    title: "Changelog",
    description: "Every ignis release, newest first, with its notes as published on GitHub.",
    sidebar: { order: 0, label: "All releases" },
    editUrl: false,
    lastUpdated: false,
  })}
:::note[Generated]
These pages are written from the [GitHub releases](${repo}/releases) each time the site is built. Each page links to its release, where the container image and the archives are.
:::

| Release | Date | Headline |
|---|---|---|
${rows.join("\n")}
`;
}

/** Every page of the section, by file name. */
export function changelogPages(releases, options) {
  const stable = stableReleases(releases);
  const pages = new Map([["index.md", changelogIndex(stable, options)]]);
  stable.forEach((release, i) => pages.set(pageName(release.tag_name), releasePage(release, { ...options, rank: i + 1 })));
  return pages;
}

/** Every release of `repo` (owner/name) from the GitHub API, page by page. */
export async function fetchReleases(slug, { token, fetchImpl = fetch } = {}) {
  const headers = { Accept: "application/vnd.github+json", "User-Agent": "ignis-docs" };
  if (token) headers.Authorization = `Bearer ${token}`;
  const all = [];
  for (let page = 1; ; page++) {
    const response = await fetchImpl(`https://api.github.com/repos/${slug}/releases?per_page=100&page=${page}`, { headers });
    if (!response.ok) throw new Error(`GET /repos/${slug}/releases answered ${response.status}`);
    const batch = await response.json();
    all.push(...batch);
    if (batch.length < 100) return all;
  }
}

/** The ADR files by number, for the notes' `ADR 0046` links. */
export function adrFiles(directory) {
  const adrs = new Map();
  for (const name of fs.readdirSync(directory)) {
    const number = name.match(/^(\d{4})-.*\.md$/)?.[1];
    if (number) adrs.set(number, name);
  }
  return adrs;
}

/**
 * Write the section into `directory` (docs/user/changelog/), replacing what
 * an earlier build left. A build that cannot reach GitHub fails in CI, where
 * the site would be published without its changelog; elsewhere it writes a
 * landing page that says so and goes on.
 */
export async function writeChangelog(directory, { slug, repo, ref, adrs, token, ci, fetchImpl }) {
  let pages;
  try {
    pages = changelogPages(await fetchReleases(slug, { token, fetchImpl }), { repo, ref, adrs });
  } catch (error) {
    if (ci) throw new Error(`The changelog could not be read from GitHub: ${error.message}`);
    console.warn(`[changelog] not fetched (${error.message}); the section holds a placeholder`);
    pages = new Map([
      [
        "index.md",
        `${frontMatter({ title: "Changelog", description: "Every ignis release.", sidebar: { order: 0, label: "All releases" }, editUrl: false, lastUpdated: false })}\nThis build could not reach GitHub. The releases are at <${repo}/releases>.\n`,
      ],
    ]);
  }
  fs.rmSync(directory, { recursive: true, force: true });
  fs.mkdirSync(directory, { recursive: true });
  for (const [name, text] of pages) fs.writeFileSync(path.join(directory, name), text);
}
