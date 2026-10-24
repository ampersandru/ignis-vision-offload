// The scenes scripts/screenshots.mjs takes: one per picture of the guide's
// Playground pages, named as the picture (docs/user/images/playground/). A
// scene gets a page already at the Playground's chat and leaves it in the
// state its picture shows; `clip` crops the picture to one region.
//
// Against a running server a prompt is a real request the model answers in
// its own words; against the dev mock (web/mock.ts) it carries the mock's
// trigger word instead ("/agents", "/web", ...), which is all the mock reads.
// `real: false` marks a scene only the mock can stage (a switch held still),
// `mock: false` one only a real server can (a picture made).

const MINUTE = 60_000;

/** Open one of the header's views. */
const view = (page, name) => page.getByRole("navigation", { name: "View" }).getByRole("button", { name, exact: true }).click();
// The chat's prompt box (the Image view has one of the same name).
const prompt = (page) => page.locator("main:not([aria-label]) textarea[name=prompt]");
const region = (page, role, name) => page.getByRole(role, { name, exact: true });

/** The chat is ready when /v1/models has answered and the prompt box takes a message. */
async function chatReady(page) {
  await page.waitForFunction(() => document.querySelector('textarea[name="prompt"]')?.getAttribute("placeholder") === "Ask anything", null, { timeout: 5 * MINUTE });
}

async function send(page, text) {
  await prompt(page).fill(text);
  await page.getByRole("button", { name: "Send", exact: true }).click();
}

/** A turn is over when Stop has come and gone (it shows for the whole turn, agents included). */
async function turnDone(page, minutes = 5) {
  const stop = page.getByRole("button", { name: "Stop", exact: true });
  await stop.waitFor({ timeout: MINUTE });
  await stop.waitFor({ state: "detached", timeout: minutes * MINUTE });
}

/** Pick one option of a Settings radio group by its visible text. */
const pick = (page, group, option) => page.getByRole("group", { name: group, exact: true }).getByText(option, { exact: true }).click();

/** The bounding box of a locator, for `clip`. */
const box = async (locator) => locator.boundingBox();

export const scenes = [
  {
    name: "chat-reply",
    run: async (page, { mock }) => {
      await chatReady(page);
      await send(page, mock ? "Why does a KV cache make a long conversation cheaper to continue?" : "In two short paragraphs: why does a KV cache make a long conversation cheaper to continue? End with a one-line rule of thumb.");
      await turnDone(page);
      await page.getByRole("button", { name: /^Context/ }).hover();
      await page.locator("#context-details").waitFor();
    },
  },
  {
    name: "chat-reasoning",
    run: async (page, { mock }) => {
      await chatReady(page);
      await pick(page, "Thinking", "Low");
      await send(page, mock ? "/budget Is 1,000,003 prime?" : "Is 1,000,003 a prime number? Answer in one sentence.");
      await turnDone(page);
      await page.locator("details.reasoning summary").first().click();
    },
  },
  {
    name: "agents",
    run: async (page, { mock }) => {
      await chatReady(page);
      await send(
        page,
        mock
          ? "/agents 3 explain how ignis serves a request"
          : "Use three agents in parallel: one explains what prefill is, one what a decode round is, one what speculative decoding is, each in three sentences. Then merge their answers into one short table.",
      );
      await region(page, "region", "Agents").waitFor({ timeout: 3 * MINUTE });
      await turnDone(page, 10);
    },
  },
  {
    name: "agent-reader",
    run: async (page, { mock }) => {
      await chatReady(page);
      await send(
        page,
        mock ? "/agents 3 explain how ignis serves a request" : "Use two agents in parallel: one lists three uses of a KV cache, one lists three costs of a long context. Then answer in two lines.",
      );
      await region(page, "region", "Agents").waitFor({ timeout: 3 * MINUTE });
      await turnDone(page, 10);
      await page.locator('section[aria-label="Agents"] button[title^="Read "]').first().click();
      await page.locator('aside[aria-label^="Agent: "]').waitFor();
    },
  },
  {
    name: "session-log",
    run: async (page, { mock }) => {
      await chatReady(page);
      await pick(page, "Thinking", "Off");
      await send(page, mock ? "Name a use of a KV cache." : "Name three uses of a KV cache, one line each.");
      await turnDone(page);
      await pick(page, "Thinking", "Medium");
      await send(page, mock ? "And one cost?" : "And one cost of a long context, in one line?");
      await turnDone(page);
      await page.getByRole("button", { name: "Expand this session's log" }).click();
    },
  },
  {
    name: "ask",
    run: async (page, { mock }) => {
      await chatReady(page);
      await send(
        page,
        mock ? "/ask who won?" : "Before you answer, use your ask tool to ask me which language I write my agents in, with Python, TypeScript and Rust as the options. Then give me one tip for it.",
      );
      await page.getByText("Question for you").waitFor({ timeout: 3 * MINUTE });
    },
  },
  {
    name: "plan-and-code",
    run: async (page, { mock }) => {
      await chatReady(page);
      if (mock) {
        await page.getByRole("tab", { name: /^Tools/ }).click();
        await page.getByRole("switch", { name: "Safety check" }).click();
        await page.getByRole("tab", { name: "General" }).click();
      }
      await send(
        page,
        mock ? "/js sum the squares" : "Keep a plan with your plan tool, then use JavaScript to compute the sum of the squares of the integers from 1 to 1000, and give me the exact result.",
      );
      await turnDone(page, 8);
      const code = page.locator('section[aria-label="Tool calls"] details summary').last();
      if (await code.count()) await code.click();
    },
  },
  {
    name: "html-file",
    run: async (page, { mock }) => {
      await chatReady(page);
      await send(
        page,
        mock ? "/html a tiny page" : "Create an HTML file with your file tool: one small page with a button that counts its clicks, dark background, orange accent. No explanation needed.",
      );
      await turnDone(page, 8);
      await page.locator('iframe[title$=".html"]').first().waitFor({ timeout: MINUTE });
    },
  },
  {
    name: "web",
    run: async (page, { mock }) => {
      await chatReady(page);
      await send(page, mock ? "/web RTX 5090 memory bandwidth" : "Search the web for the RTX 5090's memory bandwidth, read one source, and answer in one line with the link.");
      await turnDone(page, 8);
      const row = page.locator('section[aria-label="Web"] summary').first();
      if (await row.count()) await row.click();
    },
  },
  {
    name: "model-picker",
    run: async (page) => {
      await chatReady(page);
      await page.getByRole("button", { name: /^Model for the next message/ }).click();
      await page.getByRole("menu", { name: "Model for the next message" }).waitFor();
    },
  },
  {
    name: "settings-general",
    run: async (page) => {
      await chatReady(page);
    },
    clip: (page) => box(region(page, "complementary", "Settings")),
  },
  {
    name: "settings-tools",
    run: async (page) => {
      await chatReady(page);
      await page.getByRole("tab", { name: /^Tools/ }).click();
    },
    clip: (page) => box(region(page, "complementary", "Settings")),
  },
  {
    name: "memory",
    run: async (page) => {
      await chatReady(page);
      await page.getByRole("button", { name: "Memory" }).click();
      const sheet = page.getByRole("dialog", { name: "What ignis remembers" });
      for (const [title, text] of [
        ["Writes agents in Rust", "Prefers Rust with tokio for agent code; tests with cargo test."],
        ["Runs ignis on an RTX 5090", "One card, Windows 11, the default rtx5090 profile."],
      ]) {
        await sheet.getByLabel("New note title").fill(title);
        await sheet.getByLabel("New note", { exact: true }).fill(text);
        await sheet.getByRole("button", { name: "Save note" }).click();
      }
    },
  },
  {
    name: "sessions",
    run: async (page, { mock }) => {
      await chatReady(page);
      await pick(page, "Thinking", "Off");
      const prompts = mock
        ? ["What is speculative decoding?", "Write a haiku about GPUs", "/long"]
        : ["What is speculative decoding? Two sentences.", "Write a haiku about a GPU at full load.", "Give me three names for a Rust CLI that tails logs."];
      for (const [i, text] of prompts.entries()) {
        if (i) await page.getByRole("button", { name: "New session" }).click();
        await send(page, text);
        await turnDone(page);
      }
    },
    // The column and the start of the open session beside it.
    clip: async (page) => {
      const sessions = await box(region(page, "complementary", "Sessions"));
      return { x: 0, y: sessions.y, width: 960, height: 420 };
    },
  },
  {
    name: "key-page",
    before: async (page) => {
      await page.route("**/v1/models", (route) => route.fulfill({ status: 401, json: { error: { message: "missing API key", type: "invalid_request_error", code: "invalid_api_key" } } }));
    },
    run: async (page) => {
      await page.getByText("API key required").waitFor();
    },
  },
  {
    name: "decide-text",
    run: async (page) => {
      await chatReady(page);
      await view(page, "Decide");
      await page.getByRole("button", { name: /Triage one message/ }).click();
      await region(page, "complementary", "Questions").getByRole("button", { name: "Decide", exact: true }).click();
      await page.getByText("prompt tokens").waitFor({ timeout: 2 * MINUTE });
      await page.waitForTimeout(1200);
    },
  },
  {
    name: "decide-image",
    run: async (page) => {
      await chatReady(page);
      await view(page, "Decide");
      await page.getByRole("button", { name: /Find it on an image/ }).click();
      await page.locator('aside[aria-label="Questions"] img').first().waitFor();
      await region(page, "complementary", "Questions").getByRole("button", { name: "Decide", exact: true }).click();
      await page.getByText("prompt tokens").waitFor({ timeout: 2 * MINUTE });
      await page.waitForTimeout(1500);
    },
  },
  {
    name: "decide-log",
    run: async (page) => {
      await chatReady(page);
      await view(page, "Decide");
      await page.getByRole("button", { name: /Find the line in a log/ }).click();
      await region(page, "complementary", "Questions").getByRole("button", { name: "Decide", exact: true }).click();
      await page.getByText("prompt tokens").first().waitFor({ timeout: 3 * MINUTE });
      await page.waitForTimeout(1500);
    },
  },
  {
    name: "server",
    run: async (page) => {
      await chatReady(page);
      await view(page, "Server");
      await region(page, "region", "Model").waitFor();
      await page.waitForTimeout(800);
    },
  },
  {
    name: "server-config",
    run: async (page) => {
      await chatReady(page);
      await view(page, "Server");
      // The Model group: the fields a reload follows, with where each value came from.
      await page.getByRole("navigation", { name: "Configuration groups" }).getByRole("button", { name: "Model", exact: true }).click();
      await page.waitForTimeout(1200);
    },
  },
  {
    name: "server-switching",
    real: false,
    run: async (page) => {
      await chatReady(page);
      await view(page, "Server");
      await page.getByRole("button", { name: "Load qwen3.8-flash-next" }).click();
      await page.getByRole("heading", { name: /Switching from/ }).waitFor();
      await page.waitForTimeout(700);
    },
    settle: 0,
    // The header and the band down to the row being loaded: the mock's test-only rows below stay out.
    clip: async (page) => {
      const row = await box(page.getByRole("listitem").filter({ hasText: "qwen3.8-flash-next" }));
      return { x: 0, y: 0, width: 1440, height: row.y + row.height + 12 };
    },
  },
  {
    name: "monitor",
    run: async (page, { mock, url }) => {
      await chatReady(page);
      await page.getByRole("navigation", { name: "View" }).getByRole("button", { name: "Monitor" }).waitFor({ timeout: MINUTE });
      await view(page, "Monitor");
      await page.getByRole("radiogroup", { name: "Every" }).getByRole("radio", { name: "2s" }).click({ force: true });
      if (!mock) await traffic(url, 90_000);
      else await page.waitForTimeout(80_000);
      await page.evaluate(() => window.scrollTo(0, 0));
    },
  },
  {
    name: "monitor-vram",
    run: async (page) => {
      await chatReady(page);
      await page.getByRole("navigation", { name: "View" }).getByRole("button", { name: "Monitor" }).waitFor({ timeout: MINUTE });
      await view(page, "Monitor");
      const plan = region(page, "region", "VRAM plan");
      await plan.scrollIntoViewIfNeeded();
      await page.waitForTimeout(6000);
    },
    clip: (page) => box(region(page, "region", "VRAM plan")),
  },
  {
    name: "image-studio",
    mock: false,
    run: async (page) => {
      await chatReady(page);
      await view(page, "Image");
      await page.getByRole("button", { name: /Lighthouse at dawn/ }).click();
      await page.getByRole("button", { name: /^(Switch & generate|Generate)$/ }).click();
      await page.getByRole("button", { name: "Open #1" }).waitFor({ timeout: 4 * MINUTE });
      await page.getByRole("button", { name: "Stop", exact: true }).waitFor({ state: "detached", timeout: MINUTE });
    },
    settle: 1500,
  },
];

/**
 * Agent-like traffic for the Monitor: waves of concurrent chat requests at
 * the server behind `url` for `ms`, so its charts have something to draw.
 */
async function traffic(url, ms) {
  const origin = new URL(url).origin;
  const ask = (i) =>
    fetch(`${origin}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: [{ role: "user", content: `Write a short Rust function number ${i} that parses a log line into a struct, with a doc comment.` }],
        max_tokens: 400 + (i % 4) * 150,
        reasoning_effort: "none",
        class: i % 3 ? "agent" : "interactive",
      }),
    }).then((r) => r.text(), () => null);
  const end = Date.now() + ms;
  let i = 0;
  while (Date.now() < end) {
    const wave = 2 + (i % 6);
    await Promise.all(Array.from({ length: wave }, () => ask(i++)));
    await new Promise((r) => setTimeout(r, 3000));
  }
}
