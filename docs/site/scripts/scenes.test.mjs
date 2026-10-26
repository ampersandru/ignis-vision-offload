// The screenshot scenes and the pictures they make stay one set: a picture
// the guide shows has a scene that retakes it, and a scene's picture is used.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { pickScenes, scenes, withoutTestRows } from "./scenes.mjs";

const guide = fileURLToPath(new URL("../../user/", import.meta.url));
const pictures = fs
  .readdirSync(path.join(guide, "images/playground"))
  .filter((name) => name.endsWith(".png"))
  .map((name) => name.replace(/\.png$/, ""));

/** Every `images/playground/<name>.png` a page of the guide shows. */
function shown(dir = guide) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return shown(full);
    if (!/\.mdx?$/.test(entry.name)) return [];
    return [...fs.readFileSync(full, "utf8").matchAll(/images\/playground\/([a-z0-9-]+)\.png/g)].map((m) => m[1]);
  });
}

describe("the scenes", () => {
  const names = scenes.map((scene) => scene.name);

  it("have unique names, each a picture's file name", () => {
    expect(new Set(names).size).toBe(names.length);
    for (const name of names) expect(name).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/);
  });
  it("cover every picture in images/playground/, and make no picture that is not there", () => {
    expect([...names].sort()).toEqual([...pictures].sort());
  });
  it("make every picture a page shows", () => {
    for (const name of new Set(shown())) expect(names).toContain(name);
  });
});

describe("pickScenes", () => {
  const all = [{ name: "a" }, { name: "b" }, { name: "c" }];

  it("takes every scene when none is named", () => {
    expect(pickScenes(all, undefined)).toBe(all);
  });
  it("takes the named ones, in the scenes' order", () => {
    expect(pickScenes(all, "c, a").map((s) => s.name)).toEqual(["a", "c"]);
  });
  it("refuses a name no scene has, listing the scenes", () => {
    expect(() => pickScenes(all, "a,z")).toThrow(/no scene named z; the scenes are a, b, c/);
  });
});

describe("withoutTestRows", () => {
  const row = (id) => ({ id, source: "known" });

  it("drops the mock's -dead and -fail rows and keeps the others, in order, with the list's other fields", () => {
    const catalog = { object: "list", data: [row("qwen3.8-27b"), row("qwen3.8-flash-next"), row("qwen3.8-27b-dead"), row("qwen3.8-27b-fail")] };
    expect(withoutTestRows(catalog)).toEqual({ object: "list", data: [row("qwen3.8-27b"), row("qwen3.8-flash-next")] });
  });
  it("is what the server and server-switching scenes ask for", () => {
    for (const name of ["server", "server-switching"]) expect(scenes.find((scene) => scene.name === name).before).toBeTypeOf("function");
  });
});
