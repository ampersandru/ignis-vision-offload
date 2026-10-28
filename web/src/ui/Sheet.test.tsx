import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { escapeFree, holdFocus, type PageNode, Sheet, sheetKeyDown } from "./Sheet.tsx";

// The full-screen sheet the HTML preview, a workspace file's text and the diff
// open on. There is no DOM here: its markup is rendered, and what its keys and
// its focus do is driven through the functions the sheet calls, on plain
// objects standing for the page.

/** A key as the sheet's handler sees it: preventDefault marks it taken, as the browser's event does. */
const key = (name: string, taken = false) => {
  const event = { key: name, defaultPrevented: taken, preventDefault: () => (event.defaultPrevented = true) };
  return event;
};

describe("the sheet", () => {
  it("is a modal dialog the focus can go to, with its title, its controls and Close", () => {
    const html = renderToStaticMarkup(
      <Sheet label="Differences in notes.md" title="notes.md" controls={<button type="button">Unified</button>} closeLabel="Close the diff" onClose={() => {}}>
        <p>body</p>
      </Sheet>,
    );
    expect(html).toMatch(/^<div role="dialog" aria-modal="true" aria-label="Differences in notes.md" tabindex="-1"/);
    expect(html).toContain(">notes.md</h2><button type=\"button\">Unified</button>");
    expect(html).toMatch(/<button[^>]*aria-label="Close the diff"[^>]*title="Close \(Esc\)"/);
    expect(html).toMatch(/<div[^>]*overflow-auto[^>]*><p>body<\/p><\/div><\/div>$/);
  });

  it("closes on Esc, taking the key so that the page's own Esc listeners leave it: one Esc closes the sheet and nothing else", () => {
    let closed = 0;
    const other: string[] = [];
    const esc = key("Escape");
    sheetKeyDown(esc, { onClose: () => closed++, onKey: (e) => other.push(e.key) });
    expect(closed).toBe(1);
    expect(esc.defaultPrevented).toBe(true);
    // The chat's drawers close on an Esc no one took (App.tsx).
    expect(escapeFree(esc)).toBe(false);
    expect(escapeFree(key("Escape"))).toBe(true);
    expect(other).toEqual([]);
  });

  it("leaves an Esc something in it took already, and hands any other key to its own handler", () => {
    let closed = 0;
    const other: string[] = [];
    sheetKeyDown(key("Escape", true), { onClose: () => closed++, onKey: (e) => other.push(e.key) });
    const n = key("n");
    sheetKeyDown(n, { onClose: () => closed++, onKey: (e) => other.push(e.key) });
    expect(closed).toBe(0);
    expect(other).toEqual(["n"]);
    expect(n.defaultPrevented).toBe(false);
  });
});

/** A node of the page standing in for an element: its parent, its children, whether it is inert, how often it got the focus. */
type Fake = PageNode & { name: string; focused: number; focus: () => void; children: Fake[] };

function fake(name: string, children: Fake[] = [], inert = false): Fake {
  const node: Fake = { name, inert, parentElement: null, children, focused: 0, focus: () => void node.focused++ };
  for (const child of children) child.parentElement = node;
  return node;
}

describe("the sheet's hold on the page", () => {
  it("moves the focus into the sheet and makes the rest of the page inert, so Tab cannot leave it; closing gives both back", () => {
    const expand = fake("expand");
    const root = fake("root", [fake("panel", [expand])]);
    const asleep = fake("asleep", [], true);
    const sheet = fake("sheet", [fake("close")]);
    const body = fake("body", [root, asleep, sheet]);
    const release = holdFocus(sheet, { body, activeElement: expand });
    expect(sheet.focused).toBe(1);
    expect([root.inert, asleep.inert, sheet.inert]).toEqual([true, true, false]);
    // What is in the sheet stays live.
    expect(sheet.children[0].inert).toBe(false);
    release();
    // The focus goes back to the button that opened the sheet, and what was inert before stays so.
    expect(expand.focused).toBe(1);
    expect([root.inert, asleep.inert]).toEqual([false, true]);
  });

  it("holds the siblings of every element between the sheet and the body", () => {
    const beside = fake("beside");
    const sheet = fake("sheet");
    const outside = fake("outside");
    const body = fake("body", [outside, fake("portal", [beside, sheet])]);
    const release = holdFocus(sheet, { body, activeElement: null });
    expect([outside.inert, beside.inert, body.children[1].inert]).toEqual([true, true, false]);
    release();
    expect([outside.inert, beside.inert]).toEqual([false, false]);
  });
});
