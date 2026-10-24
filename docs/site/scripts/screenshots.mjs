// The guide's Playground screenshots (#348), taken by driving the Playground
// in a headless Chrome: `make docs-screenshots`. Each scene opens a fresh
// browser context (an empty IndexedDB, so no session of an earlier scene
// shows), brings one view to the state a page describes and writes
// docs/user/images/playground/<scene>.png.
//
//   PLAYGROUND_URL   the Playground to drive, e.g. http://127.0.0.1:8000/ui/
//                    for a running ignis-server --server-ui. Unset, the dev
//                    mock (`npm run dev:mock` in web/) is started on a free
//                    port and stopped at the end: no GPU, the same answers
//                    every run.
//   SCENES           a comma-separated subset of the scene names.
//   CHROME           the Chrome to drive; unset, the installed Google Chrome.
//
// Playwright drives the Chrome already installed; it downloads no browser.

import { spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";
import { pickScenes, scenes } from "./scenes.mjs";

const repoRoot = fileURLToPath(new URL("../../../", import.meta.url));
const out = path.join(repoRoot, "docs/user/images/playground");
const viewport = { width: 1440, height: 900 };

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

async function waitFor(url, child, seconds = 60) {
  for (let i = 0; i < seconds * 4; i++) {
    if (child.exitCode !== null) throw new Error(`the dev mock exited with ${child.exitCode}`);
    try {
      if ((await fetch(url)).ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`${url} did not answer in ${seconds} s`);
}

/** The dev mock, on its own port, and how to stop it. */
async function startMock() {
  const port = await freePort();
  const vite = path.join(repoRoot, "web/node_modules/vite/bin/vite.js");
  const child = spawn(process.execPath, [vite, "--mode", "mock", "--port", String(port), "--strictPort", "--host", "127.0.0.1"], {
    cwd: path.join(repoRoot, "web"),
    stdio: ["ignore", "ignore", "inherit"],
  });
  const url = `http://127.0.0.1:${port}/ui/`;
  await waitFor(url, child);
  return { url, stop: () => child.kill() };
}

const chosen = pickScenes(scenes, process.env.SCENES);
const mock = process.env.PLAYGROUND_URL ? null : await startMock();
const url = process.env.PLAYGROUND_URL || mock.url;
const browser = await chromium.launch(process.env.CHROME ? { executablePath: process.env.CHROME } : { channel: "chrome" });
fs.mkdirSync(out, { recursive: true });
let failed = 0;
try {
  for (const scene of chosen) {
    // A scene only one kind of Playground can stage is left to that kind's run.
    if ((mock && scene.mock === false) || (!mock && scene.real === false)) {
      console.log(`${scene.name}: skipped (${mock ? "needs a running server" : "needs the dev mock"})`);
      continue;
    }
    const context = await browser.newContext({ viewport: scene.viewport ?? viewport, colorScheme: "dark", reducedMotion: "reduce" });
    // The Server view copies a download command; headless Chrome asks first.
    await context.grantPermissions(["clipboard-read", "clipboard-write"], { origin: new URL(url).origin });
    const page = await context.newPage();
    page.setDefaultTimeout(20_000);
    try {
      await scene.before?.(page);
      await page.goto(url);
      await scene.run(page, { mock: mock !== null, url });
      // Settle what the scene left moving: a caret, a fading toast, a chart's last tick.
      await page.waitForTimeout(scene.settle ?? 400);
      const file = path.join(out, `${scene.name}.png`);
      await page.screenshot({ path: file, ...(scene.clip ? { clip: await scene.clip(page) } : {}) });
      console.log(`${scene.name}.png`);
    } catch (error) {
      failed++;
      console.error(`${scene.name}: ${error.message.split("\n")[0]}`);
    } finally {
      await context.close();
    }
  }
} finally {
  await browser.close();
  mock?.stop();
}
if (failed) {
  console.error(`${failed} scene(s) failed`);
  process.exitCode = 1;
}
