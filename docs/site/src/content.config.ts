// The site's one collection is the guide itself: every page under docs/user/
// (ADR 0066), loaded where it lives rather than copied into src/content/.
// docs/user/README.md is the folder's index for GitHub readers, not a page.
import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { docsSchema } from "@astrojs/starlight/schema";

export const collections = {
  docs: defineCollection({
    loader: glob({ base: "../user", pattern: ["**/[^_]*.{md,mdx}", "!README.md"] }),
    schema: docsSchema(),
  }),
};
