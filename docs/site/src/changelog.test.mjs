import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { adrFiles, changelogPages, fetchReleases, headline, linkReferences, noteBody, stableReleases, writeChangelog } from "./changelog.mjs";
import { sidebarFields } from "./sidebar.mjs";

const repo = "https://github.com/gpillon/ignis";
const adrs = new Map([["0046", "0046-config-v2.md"]]);
const release = (tag, body = `## ignis ${tag}\n\n### Changes\n\n- **Something new** (#12)\n`, extra = {}) => ({
  tag_name: tag,
  body,
  published_at: "2026-10-22T00:25:32Z",
  html_url: `${repo}/releases/tag/${tag}`,
  draft: false,
  prerelease: false,
  ...extra,
});

describe("stableReleases", () => {
  it("keeps the published v* releases, newest version first", () => {
    const list = [release("v0.9.0"), release("develop", "", { prerelease: true }), release("v0.10.0"), release("v1.0.0", "", { draft: true }), release("v0.9.1")];
    expect(stableReleases(list).map((r) => r.tag_name)).toEqual(["v0.10.0", "v0.9.1", "v0.9.0"]);
  });
});

describe("linkReferences", () => {
  it("links issues and ADRs, and leaves code and unknown ADRs as written", () => {
    const text = "- Fixed (#305; ADR 0046), also #298-#304 and ADR 0099\n- `#12` and `ADR 0046` in code\n```\n#7 ADR 0046\n```";
    expect(linkReferences(text, { repo, ref: "v0.7.3", adrs })).toBe(
      `- Fixed ([#305](${repo}/issues/305); [ADR 0046](${repo}/blob/v0.7.3/docs/adr/0046-config-v2.md)), also [#298](${repo}/issues/298)-[#304](${repo}/issues/304) and ADR 0099\n- \`#12\` and \`ADR 0046\` in code\n\`\`\`\n#7 ADR 0046\n\`\`\``,
    );
  });
  it("leaves an anchor inside a URL alone", () => {
    expect(linkReferences("see https://example.com/page#12", { repo, ref: "main", adrs })).toBe("see https://example.com/page#12");
  });
});

describe("noteBody", () => {
  it("drops the release title and raises the headings to start at ##", () => {
    expect(noteBody("## ignis v0.7.3\r\n\r\n### Changes\r\n\r\n- a\r\n\r\n#### Detail\r\n\r\n```\n### not a heading\n```")).toBe(
      "## Changes\n\n- a\n\n### Detail\n\n```\n### not a heading\n```",
    );
  });
  it("drops a second title above the release's file list", () => {
    expect(noteBody("## ignis v0.7.3\n\n### Changes\n\n- a\n\n## ignis v0.7.3\n\n### Binaries\n\nb")).toBe("## Changes\n\n- a\n\n## Binaries\n\nb");
  });
  it("keeps headings that already start at ##", () => {
    expect(noteBody("## Changes\n\n- a")).toBe("## Changes\n\n- a");
  });
});

describe("headline", () => {
  it("is the first bold change", () => {
    expect(headline("## Changes\n\n- **The prefill reads the codec** (#335). More.\n- **Other**")).toBe("The prefill reads the codec");
  });
  it("is the first item's opening clause when its lead is not bold", () => {
    expect(headline("## Changes\n\n- Config v2: every option declared once\n\n## Upgrade notes\n\n- **Every flag was renamed**")).toBe("Config v2: every option declared once");
    expect(headline("## Changes\n\n- Memory: retained slots on the host — `--retained-host` (default 16)")).toBe("Memory: retained slots on the host");
    expect(headline("## Changes\n\n- Qwen3.8-Flash-Next served by ignis: its own artifact (#298)")).toBe("Qwen3.8-Flash-Next served by ignis");
    expect(headline("## Fixes\n\n- A turn no longer ends inside its reasoning. An end-of-turn token")).toBe("A turn no longer ends inside its reasoning");
  });
  it("reads nothing from the release's file list", () => {
    expect(headline("## Changes\n\n- plain change\n\n## Container image\n\n## Binaries\n\nThe kernel is compiled for **SM120a**")).toBe("plain change");
  });
  it("is the opening sentence of notes with no changes list", () => {
    expect(headline("The first release that can be installed rather than built.\n\n## Container image")).toBe("The first release that can be installed rather than built");
  });
  it("names the files for notes that list only them", () => {
    expect(headline("## Container image (linux/amd64)\n\n```\npodman run\n```\n\n(`docker`: `--gpus all`.)\n\n## Binaries\n\nThe kernel is compiled for **SM120a**")).toBe("The release's image and archives");
  });
});

describe("changelogPages", () => {
  const pages = changelogPages([release("v0.7.2"), release("v0.7.3"), release("develop", "", { prerelease: true })], { repo, ref: "main", adrs });

  it("writes an index and one page per release, named by its route", () => {
    expect([...pages.keys()]).toEqual(["index.md", "v0-7-3.md", "v0-7-2.md"]);
  });
  it("orders the sidebar newest first and badges the latest", () => {
    expect(sidebarFields(pages.get("index.md"))).toMatchObject({ title: "Changelog", order: 0, label: "All releases" });
    expect(sidebarFields(pages.get("v0-7-3.md"))).toMatchObject({ title: "ignis v0.7.3", order: 1, label: "v0.7.3", badge: "latest", variant: "success" });
    expect(sidebarFields(pages.get("v0-7-2.md"))).toMatchObject({ order: 2, badge: undefined });
  });
  it("lists every release on the index with its date and headline", () => {
    expect(pages.get("index.md")).toContain("| [v0.7.3](v0-7-3.md) | 2026-10-22 | Something new |");
  });
  it("gives each page its date, its release link and its linked notes", () => {
    const page = pages.get("v0-7-3.md");
    expect(page).toContain(`[the release on GitHub](${repo}/releases/tag/v0.7.3)`);
    expect(page).toContain(`- **Something new** ([#12](${repo}/issues/12))`);
    expect(page).not.toContain("## ignis v0.7.3");
  });
});

describe("fetchReleases", () => {
  it("reads page after page until one comes back short, with the token when there is one", async () => {
    const asked = [];
    const fetchImpl = async (url, { headers }) => {
      asked.push([new URL(url).searchParams.get("page"), headers.Authorization]);
      const page = Number(new URL(url).searchParams.get("page"));
      const count = page === 1 ? 100 : 7;
      return { ok: true, status: 200, json: async () => Array.from({ length: count }, (_, i) => release(`v0.${page}.${i}`)) };
    };
    const all = await fetchReleases("gpillon/ignis", { token: "t0k", fetchImpl });
    expect(all).toHaveLength(107);
    expect(asked).toEqual([
      ["1", "Bearer t0k"],
      ["2", "Bearer t0k"],
    ]);
  });
});

describe("adrFiles", () => {
  it("maps each ADR's number to its file, and nothing else", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "adr-"));
    for (const name of ["0046-config-v2.md", "0066-the-guide.md", "README.md", "0001-draft.txt"]) fs.writeFileSync(path.join(dir, name), "");
    expect(adrFiles(dir)).toEqual(
      new Map([
        ["0046", "0046-config-v2.md"],
        ["0066", "0066-the-guide.md"],
      ]),
    );
  });
});

describe("writeChangelog", () => {
  const dir = () => fs.mkdtempSync(path.join(os.tmpdir(), "changelog-"));
  const answering = (status, body) => async () => ({ ok: status === 200, status, json: async () => body });

  it("replaces what an earlier build left", async () => {
    const target = path.join(dir(), "changelog");
    fs.mkdirSync(target);
    fs.writeFileSync(path.join(target, "v0-1-0.md"), "stale");
    await writeChangelog(target, { slug: "gpillon/ignis", repo, ref: "main", adrs, fetchImpl: answering(200, [release("v0.7.3")]) });
    expect(fs.readdirSync(target).sort()).toEqual(["index.md", "v0-7-3.md"]);
  });
  it("fails in CI when GitHub does not answer", async () => {
    const target = path.join(dir(), "changelog");
    await expect(writeChangelog(target, { slug: "gpillon/ignis", repo, ref: "main", adrs, ci: true, fetchImpl: answering(503, {}) })).rejects.toThrow(/could not be read from GitHub.*503/);
  });
  it("writes a placeholder elsewhere", async () => {
    const target = path.join(dir(), "changelog");
    await writeChangelog(target, { slug: "gpillon/ignis", repo, ref: "main", adrs, ci: false, fetchImpl: answering(503, {}) });
    expect(fs.readdirSync(target)).toEqual(["index.md"]);
    expect(fs.readFileSync(path.join(target, "index.md"), "utf8")).toContain(`${repo}/releases`);
  });
});
