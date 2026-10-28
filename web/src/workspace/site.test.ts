import { describe, expect, it } from "vitest";
import { dataUrl, inlineSite, isCsvName, parseCsv, resolveRef, siteRefs, svgDataUrl } from "./site.ts";

// The HTML site's rewriting and the CSV reader: strings in, strings out.

describe("resolving a reference against the HTML file's name", () => {
  it("is relative to the file's folder, with ./ and ../", () => {
    expect(resolveRef("style.css", "index.html")).toBe("style.css");
    expect(resolveRef("css/main.css", "site/index.html")).toBe("site/css/main.css");
    expect(resolveRef("./js/app.js", "site/index.html")).toBe("site/js/app.js");
    expect(resolveRef("../shared/logo.png", "site/pages/about.html")).toBe("site/shared/logo.png");
    expect(resolveRef("/top.css", "site/index.html")).toBe("top.css");
    expect(resolveRef("a%20b.css?v=2#x", "index.html")).toBe("a b.css");
  });

  it("leaves absolute, data:, protocol-relative, fragment and out-of-workspace references", () => {
    for (const ref of ["https://cdn.example/x.js", "data:image/png;base64,AAAA", "//cdn.example/x.js", "#top", "mailto:a@b.c", "", "../x.css"]) {
      expect(resolveRef(ref, "index.html")).toBeNull();
    }
  });
});

const page = `<!doctype html><html><head>
<link rel="stylesheet" href="css/main.css">
<LINK REL='Stylesheet' HREF='print.css' media="print"/>
<link rel="icon" href="favicon.png">
<script src=js/app.js defer></script>
<script src="https://cdn.example/lib.js"></script>
<script>inline()</script>
</head><body><img alt="logo" src='img/logo.png' width=40><img src="data:image/gif;base64,R0lG"><a href="other.html">x</a></body></html>`;

describe("a site across files", () => {
  it("finds the stylesheets, scripts and pictures that name a file, in document order", () => {
    expect(siteRefs(page, "site/index.html").map((r) => [r.tag, r.file])).toEqual([
      ["stylesheet", "site/css/main.css"],
      ["stylesheet", "site/print.css"],
      ["script", "site/js/app.js"],
      ["image", "site/img/logo.png"],
    ]);
  });

  it("puts a three-file site and its picture into one document", () => {
    const files = new Map([
      ["site/css/main.css", "h1{background:url(bg.png)}"],
      ["site/print.css", "body{color:#000}"],
      ["site/js/app.js", 'fetch("data.json");document.write("</script>")'],
      ["site/img/logo.png", "data:image/png;base64,QUJD"],
    ]);
    const out = inlineSite(page, "site/index.html", files);
    expect(out).toContain("<style>h1{background:url(bg.png)}</style>");
    expect(out).toContain('<style media="print">body{color:#000}</style>');
    expect(out).toContain('<script defer>fetch("data.json");document.write("<\\/script>")</script>');
    expect(out).toContain('<img alt="logo" src="data:image/png;base64,QUJD" width=40>');
    expect(out).not.toContain("css/main.css");
    expect(out).not.toContain("js/app.js");
  });

  it("leaves what is not a workspace file, and what nothing resolves, as written", () => {
    const out = inlineSite(page, "site/index.html", new Map([["site/css/main.css", "a{}"]]));
    expect(out).toContain('<link rel="icon" href="favicon.png">');
    expect(out).toContain("<script src=js/app.js defer></script>");
    expect(out).toContain('<script src="https://cdn.example/lib.js"></script>');
    expect(out).toContain("<script>inline()</script>");
    expect(out).toContain('<img src="data:image/gif;base64,R0lG">');
    expect(out).toContain('<a href="other.html">x</a>');
    expect(out).toContain("src='img/logo.png'");
    expect(inlineSite(page, "site/index.html", new Map())).toBe(page);
  });

  it("does not resolve CSS url(), srcset or links", () => {
    const html = '<style>a{background:url(x.png)}</style><img srcset="x.png 2x" alt=""><a href="x.png">x</a>';
    expect(siteRefs(html, "index.html")).toEqual([]);
    expect(inlineSite(html, "index.html", new Map([["x.png", "data:x"]]))).toBe(html);
  });
});

describe("data URLs", () => {
  it("encodes bytes in base64 and an SVG as an image", () => {
    expect(dataUrl("image/png", new Uint8Array([65, 66, 67]))).toBe("data:image/png;base64,QUJD");
    expect(svgDataUrl('<svg a="b"/>')).toBe("data:image/svg+xml;charset=utf-8,%3Csvg%20a%3D%22b%22%2F%3E");
  });
});

describe("reading a CSV file", () => {
  it("reads quoted fields with commas, quotes and line breaks in them", () => {
    expect(parseCsv('name,note\r\n"Rossi, Anna","said ""hi""\nthen left"\nBo,\n')).toEqual([
      ["name", "note"],
      ["Rossi, Anna", 'said "hi"\nthen left'],
      ["Bo", ""],
    ]);
  });

  it("drops a BOM, keeps a last row without a line break and an empty last field", () => {
    expect(parseCsv("﻿a,b\n1,")).toEqual([
      ["a", "b"],
      ["1", ""],
    ]);
    expect(parseCsv("")).toEqual([]);
    expect(isCsvName("DATA.CSV")).toBe(true);
  });
});
