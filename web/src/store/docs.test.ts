import { describe, expect, it } from "vitest";
import { listChanges, mergeLoaded, newKey, restoreOrder } from "./docs.ts";

// The order sessions come back in, and how they join what the page did
// before the store answered: the chat's and the Decide tab's alike.

describe("restoreOrder", () => {
  it("lists the stored sessions newest first, renumbered from the page's next id, oldest lowest", () => {
    const stored = [
      { key: "b", id: 9 },
      { key: "a", id: 2 },
      { key: "c", id: 40 },
    ];
    const { items, nextId } = restoreOrder(stored, 5);
    expect(items).toEqual([
      { key: "c", id: 7 },
      { key: "b", id: 6 },
      { key: "a", id: 5 },
    ]);
    expect(nextId).toBe(8);
  });
});

describe("listChanges", () => {
  it("names the sessions new or changed, by identity, and hands back the ones gone", () => {
    const a = { id: 1 };
    const b = { id: 2 };
    const c = { id: 3 };
    const before = new Map([
      [1, a],
      [2, b],
      [3, c],
    ]);
    const changedB = { id: 2 };
    const added = { id: 4 };
    expect(listChanges(before, [added, changedB, a])).toEqual({ changed: [4, 2], removed: [c] });
    expect(listChanges(before, [a, b, c])).toEqual({ changed: [], removed: [] });
  });
});

describe("mergeLoaded", () => {
  type Item = { id: number; used: boolean };
  const worth = (i: Item) => i.used;
  const loaded: Item[] = [
    { id: 11, used: true },
    { id: 10, used: true },
  ];

  it("replaces an empty fresh session with the stored ones, and reopens the one last open", () => {
    expect(mergeLoaded([{ id: 1, used: false }], 1, loaded, 10, worth)).toEqual({ items: loaded, activeId: 10 });
  });

  it("opens the newest when the one last open is not among them", () => {
    expect(mergeLoaded([{ id: 1, used: false }], 1, loaded, undefined, worth).activeId).toBe(11);
    expect(mergeLoaded([{ id: 1, used: false }], 1, loaded, 99, worth).activeId).toBe(11);
  });

  it("keeps what was done before the store answered on top, and open", () => {
    const early = { id: 1, used: true };
    expect(mergeLoaded([early], 1, loaded, 10, worth)).toEqual({ items: [early, ...loaded], activeId: 1 });
  });

  it("leaves the page as it is when the store holds nothing", () => {
    const fresh = [{ id: 1, used: false }];
    expect(mergeLoaded(fresh, 1, [], undefined, worth)).toEqual({ items: fresh, activeId: 1 });
  });
});

describe("newKey", () => {
  it("hands out a different key each time", () => {
    expect(newKey()).not.toBe(newKey());
  });
});
