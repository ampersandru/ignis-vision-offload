import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, describe, expect, it } from "vitest";
import { llmsTxt, pageFile } from "./llms.mjs";
import { sectionItems } from "./sidebar.mjs";

const guide = fs.mkdtempSync(path.join(os.tmpdir(), "llms-"));
const write = (relative, text) => {
  fs.mkdirSync(path.dirname(path.join(guide, relative)), { recursive: true });
  fs.writeFileSync(path.join(guide, relative), text);
};
afterAll(() => fs.rmSync(guide, { recursive: true, force: true }));

const source = "https://raw.githubusercontent.com/o/r/v1.2.3/docs/user/";

describe("pageFile", () => {
  it("finds a page, an MDX page and a section's index", () => {
    write("start/index.md", '---\ntitle: "Start"\n---\n');
    write("start/install.md", '---\ntitle: "Install"\n---\n');
    write("why/index.mdx", '---\ntitle: "Why"\n---\n');
    write("troubleshooting.md", '---\ntitle: "Troubleshooting"\n---\n');
    expect(pageFile(guide, "start")).toBe("start/index.md");
    expect(pageFile(guide, "start/install")).toBe("start/install.md");
    expect(pageFile(guide, "why")).toBe("why/index.mdx");
    expect(pageFile(guide, "/troubleshooting/")).toBe("troubleshooting.md");
  });
  it("fails on a slug with no page, rather than list a dead link", () => {
    expect(() => pageFile(guide, "start/gone")).toThrow(/start\/gone/);
  });
});

describe("llmsTxt", () => {
  it("writes llmstxt.org's shape: title, summary, details, a section per sidebar group, then Optional", () => {
    write("api/index.md", '---\ntitle: "The API"\ndescription: "Every endpoint."\nsidebar:\n  order: 1\n---\n');
    write("api/chat.md", '---\ntitle: "Chat completions"\ndescription: "Fields and streaming."\nsidebar:\n  order: 2\n  label: "Chat"\n---\n');
    write("faq.md", '---\ntitle: "FAQ"\n---\n');
    const text = llmsTxt({
      guide,
      title: "ignis",
      summary: "An engine.",
      details: "Each link is Markdown.",
      sidebar: [
        { label: "API", items: sectionItems(guide, "api") },
        { label: "Questions", link: "/faq/" },
      ],
      source,
      optional: [{ label: "Releases", url: "https://example.com/releases", description: "The notes." }],
    });
    expect(text).toBe(
      [
        "# ignis",
        "",
        "> An engine.",
        "",
        "Each link is Markdown.",
        "",
        "## API",
        "",
        `- [The API](${source}api/index.md): Every endpoint.`,
        `- [Chat completions](${source}api/chat.md): Fields and streaming.`,
        "",
        "## Questions",
        "",
        `- [FAQ](${source}faq.md)`,
        "",
        "## Optional",
        "",
        "- [Releases](https://example.com/releases): The notes.",
        "",
      ].join("\n"),
    );
  });
  it("has no Optional section when there are no optional links", () => {
    const text = llmsTxt({ guide, title: "t", summary: "s", details: "d", sidebar: [{ label: "Q", link: "faq" }], source });
    expect(text).not.toContain("## Optional");
  });
});

describe("the real guide", () => {
  const real = fileURLToPath(new URL("../../user/", import.meta.url));
  it("lists the agent path with its description, as raw Markdown", () => {
    const text = llmsTxt({ guide: real, title: "t", summary: "s", details: "d", sidebar: [{ label: "Reading paths", items: sectionItems(real, "paths") }], source });
    expect(text).toMatch(new RegExp(`^- \\[The agent path\\]\\(${source.replace(/[.]/g, "\\.")}paths/agent\\.md\\): .+$`, "m"));
  });
});
