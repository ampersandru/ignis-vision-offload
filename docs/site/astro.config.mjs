// The user guide's site (ADR 0066): Starlight over the pages in docs/user/,
// published to GitHub Pages by .github/workflows/docs.yml. Nothing here is
// content: a page is added by adding a Markdown file to a section's folder,
// and the sidebar lists it by its `sidebar.order`.
import { fileURLToPath } from "node:url";
import { defineConfig } from "astro/config";
import { satteri } from "@astrojs/markdown-satteri";
import starlight from "@astrojs/starlight";
import relativeLinks, { deadLinksFailTheBuild } from "./src/plugins/relative-links.mjs";

const repo = "https://github.com/gpillon/ignis";
const base = "/ignis";
const guide = fileURLToPath(new URL("../user/", import.meta.url));
const repoRoot = fileURLToPath(new URL("../../", import.meta.url));
// The ref a link out of the guide points at: the tag being published, or main.
const ref = process.env.DOCS_GITHUB_REF || "main";
// Every dead link the pages hold; a build that found one fails, naming them all.
const dead = [];

const section = (label, directory, extra = {}) => ({ label, items: [{ autogenerate: { directory } }], ...extra });

export default defineConfig({
  site: "https://gpillon.github.io",
  base,
  trailingSlash: "always",
  markdown: {
    processor: satteri({ mdastPlugins: [relativeLinks({ guide, repoRoot, repo, ref, base, dead })] }),
  },
  vite: {
    server: { fs: { allow: [repoRoot] } },
    // The guide's MDX pages live outside this package, where no node_modules
    // is an ancestor: the one module they import is resolved from here.
    resolve: {
      alias: [{ find: /^@astrojs\/starlight\/components$/, replacement: fileURLToPath(import.meta.resolve("@astrojs/starlight/components")) }],
    },
  },
  integrations: [
    deadLinksFailTheBuild(dead),
    starlight({
      title: "ignis",
      description: "A CUDA inference engine for Qwen3.8 on RTX 50-series and RTX PRO 6000 cards: install, configure, call and extend it.",
      logo: { src: "../../web/src/brand/wordmark-light.webp", alt: "ignis", replacesTitle: true },
      favicon: "/favicon.png",
      social: [{ icon: "github", label: "GitHub", href: repo }],
      editLink: { baseUrl: `${repo}/edit/main/docs/user/` },
      lastUpdated: true,
      customCss: ["./src/styles/theme.css"],
      markdown: { processedDirs: ["../user"] },
      tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 4 },
      sidebar: [
        section("Start here", "start"),
        section("Reading paths", "paths"),
        section("Guides", "guides"),
        section("Concepts", "concepts"),
        section("Configuration", "configuration"),
        section("Playground", "playground"),
        section("API", "api"),
        section("Reference", "reference"),
        { label: "Troubleshooting", link: "/troubleshooting/" },
        section("Contributing", "contributing", { collapsed: true }),
      ],
    }),
  ],
});
