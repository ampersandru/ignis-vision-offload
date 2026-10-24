import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { sectionItems, sidebarFields } from "./sidebar.mjs";

const guide = fs.mkdtempSync(path.join(os.tmpdir(), "sidebar-"));
const write = (relative, text) => {
  fs.mkdirSync(path.dirname(path.join(guide, relative)), { recursive: true });
  fs.writeFileSync(path.join(guide, relative), text);
};
afterAll(() => fs.rmSync(guide, { recursive: true, force: true }));

describe("sidebarFields", () => {
  it("reads the title, the order, the label and a badge", () => {
    const text = '---\ntitle: "Every \\"config\\" field"\ndescription: "x"\nsidebar:\n  order: 3\n  label: "Fields"\n  badge:\n    text: generated\n    variant: note\neditUrl: false\n---\n\nBody\n';
    expect(sidebarFields(text)).toEqual({ title: 'Every "config" field', order: 3, label: "Fields", badge: "generated", variant: "note" });
  });
  it("leaves out what a page does not say", () => {
    expect(sidebarFields("---\ntitle: Plain\n---\n")).toEqual({ title: "Plain", order: undefined, label: undefined, badge: undefined, variant: undefined });
  });
});

describe("sectionItems", () => {
  it("lists a section by order, then name, its index as the section's own route", () => {
    write("api/index.md", '---\ntitle: "The API"\nsidebar:\n  order: 1\n---\n');
    write("api/zeta.md", '---\ntitle: "Zeta"\nsidebar:\n  order: 2\n---\n');
    write("api/alpha.md", '---\ntitle: "Alpha"\nsidebar:\n  order: 2\n---\n');
    write("api/later.md", '---\ntitle: "Later"\n---\n');
    write("api/_draft.md", '---\ntitle: "Draft"\n---\n');
    write("api/notes.txt", "not a page");
    expect(sectionItems(guide, "api")).toEqual([
      { slug: "api", label: "The API" },
      { slug: "api/alpha", label: "Alpha" },
      { slug: "api/zeta", label: "Zeta" },
      { slug: "api/later", label: "Later" },
    ]);
  });
  it("carries a page's sidebar label and badge", () => {
    write("reference/cli.md", '---\ntitle: "Command line"\nsidebar:\n  order: 6\n  label: "CLI"\n  badge:\n    text: generated\n    variant: note\n---\n');
    expect(sectionItems(guide, "reference")).toEqual([{ slug: "reference/cli", label: "CLI", badge: { text: "generated", variant: "note" } }]);
  });
});
