// One Markdown source, read on GitHub and on the site (ADR 0066): the guide's
// pages link to each other and to the rest of the repository with relative
// paths, which GitHub follows as they are. On the site those paths mean
// nothing, so this remark plugin rewrites each one at build time:
//
//   a page of the guide        -> its route (`../api/index.md#deadlines`
//                                 -> `/ignis/api/#deadlines`)
//   any other file of the repo -> its GitHub URL at the published ref
//                                 (`../../adr/0046-….md` -> `…/blob/v0.7.4/docs/adr/0046-….md`)
//
// and refuses what would be a dead link on either side: a page or a repo
// file that does not exist, an anchor no heading of the target makes, a page
// name that is not already its own route. Images are left to Astro, which
// resolves them relative to the page.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import GithubSlugger from "github-slugger";

const SCHEME = /^[a-z][a-z0-9+.-]*:/i;
const PAGE = /\.mdx?$/;
const SLUG_SEGMENT = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

/** The anchors a Markdown file's headings make, as GitHub and Astro slug them. */
export function headingAnchors(text) {
  const slugger = new GithubSlugger();
  const anchors = new Set();
  let fence = null;
  for (const line of text.split(/\r?\n/)) {
    const marker = line.match(/^\s*(```+|~~~+)/);
    if (marker) {
      if (fence === null) fence = marker[1][0];
      else if (marker[1][0] === fence) fence = null;
      continue;
    }
    if (fence !== null) continue;
    const heading = line.match(/^#{1,6}\s+(.*?)\s*#*\s*$/);
    if (heading) anchors.add(slugger.slug(plainText(heading[1])));
  }
  return anchors;
}

/** A heading's text as it renders: links to their text, emphasis and code marks dropped. */
function plainText(markdown) {
  return markdown
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[*`]/g, "")
    .replace(/<[^>]+>/g, "");
}

/** A guide page's route: its path without the extension, `index` dropped. */
export function pageRoute(relative, base) {
  const segments = relative.replace(PAGE, "").split(/[\\/]/);
  if (segments[segments.length - 1] === "index") segments.pop();
  for (const segment of segments) {
    if (!SLUG_SEGMENT.test(segment)) {
      throw new Error(`\`${relative}\`: a page's path must be its route (lowercase words joined by dashes), \`${segment}\` is not`);
    }
  }
  return `${base}/${segments.map((s) => `${s}/`).join("")}`;
}

const anchorCache = new Map();
function anchorsOf(file) {
  if (!anchorCache.has(file)) anchorCache.set(file, headingAnchors(fs.readFileSync(file, "utf8")));
  return anchorCache.get(file);
}

function checkAnchor(target, hash, from, url) {
  if (hash && !anchorsOf(target).has(hash)) {
    throw new Error(`${from}: \`${url}\` names the anchor #${hash}, which no heading of ${target} makes`);
  }
}

/**
 * The URL a link written in `from` (a page of the guide) has on the site, or
 * null to leave it as written. Throws on a link that would be dead.
 *
 * options: guide (absolute path of docs/user), repoRoot (absolute path of the
 * repository), repo (its GitHub URL), ref (the branch or tag to link at),
 * base (the site's base path, no trailing slash).
 */
export function rewriteLink(url, from, options) {
  if (!url || SCHEME.test(url) || url.startsWith("/")) return null;
  const hashAt = url.indexOf("#");
  const pathPart = hashAt === -1 ? url : url.slice(0, hashAt);
  const hash = hashAt === -1 ? "" : decodeURIComponent(url.slice(hashAt + 1));
  if (pathPart === "") {
    checkAnchor(from, hash, from, url);
    return null;
  }
  const target = path.resolve(path.dirname(from), decodeURI(pathPart));
  const inGuide = path.relative(options.guide, target);
  if (!inGuide.startsWith("..") && !path.isAbsolute(inGuide) && PAGE.test(target)) {
    if (!fs.existsSync(target)) throw new Error(`${from}: \`${url}\` names a page that does not exist (${target})`);
    checkAnchor(target, hash, from, url);
    return pageRoute(inGuide, options.base) + (hash ? `#${hash}` : "");
  }
  const inRepo = path.relative(options.repoRoot, target);
  if (inRepo.startsWith("..") || path.isAbsolute(inRepo)) {
    throw new Error(`${from}: \`${url}\` leaves the repository`);
  }
  if (!fs.existsSync(target)) throw new Error(`${from}: \`${url}\` names a file the repository does not have (${target})`);
  const kind = fs.statSync(target).isDirectory() ? "tree" : "blob";
  const repoPath = inRepo.split(path.sep).join("/");
  return `${options.repo}/${kind}/${options.ref}/${repoPath}${hash ? `#${hash}` : ""}`;
}

/**
 * The Sätteri mdast plugin (Astro 7's Markdown processor): a factory that
 * runs on the guide's pages only and rewrites every link and link definition.
 *
 * Astro's content loader logs an error thrown while it renders a page and
 * builds on, so a dead link is pushed onto `options.dead` instead, when the
 * caller passes one, for `deadLinksFailTheBuild` to report all at once.
 */
export default function relativeLinks(options) {
  return ({ fileURL }) => {
    if (!fileURL) return null;
    const from = fileURLToPath(fileURL);
    const inGuide = path.relative(options.guide, from);
    if (inGuide.startsWith("..") || path.isAbsolute(inGuide)) return null;
    const rewrite = (node, ctx) => {
      let rewritten;
      try {
        rewritten = rewriteLink(node.url, from, options);
      } catch (error) {
        if (!options.dead) throw error;
        options.dead.push(error.message);
        return;
      }
      if (rewritten !== null) ctx.setProperty(node, "url", rewritten);
    };
    return { name: "ignis-relative-links", link: rewrite, definition: rewrite };
  };
}

/** The Astro integration that turns the dead links a build found into its failure. */
export function deadLinksFailTheBuild(dead) {
  return {
    name: "ignis-dead-links",
    hooks: {
      "astro:build:done": () => {
        const unique = [...new Set(dead)];
        if (unique.length > 0) throw new Error(`${unique.length} dead link(s) in the guide:\n  ${unique.join("\n  ")}`);
      },
    },
  };
}
