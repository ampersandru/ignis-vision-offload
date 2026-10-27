import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { PulseChip } from "../monitor/MonitorPulse.tsx";
import { IDLE, type ListeningHandle, type ListeningState } from "../transcribe/listening.ts";
import { Header } from "./Header.tsx";

/** A listening fixed in `state`. */
const fixed = (state: Partial<ListeningState>): ListeningHandle => {
  const fixedState = { ...IDLE, ...state };
  return { getState: () => fixedState, subscribe: () => () => {} };
};

// The header's view switch (GitHub #165, #247) and the engine pulse.

const base = { model: { state: "reading" } as const, busy: false, onOpen: () => {}, onOpenMemory: () => {} };

describe("Header", () => {
  it("shows the plain Playground label with no switch at all", () => {
    const html = renderToStaticMarkup(<Header {...base} />);
    expect(html).toContain("Playground");
    expect(html).not.toContain('aria-label="View"');
  });

  it("offers Decide without metrics and adds the Monitor only when they answer", () => {
    const plain = renderToStaticMarkup(<Header {...base} view="chat" onView={() => {}} />);
    expect(plain).toContain(">Decide<");
    expect(plain).not.toContain(">Monitor<");
    // The pulse chip reads the metrics too, so it waits for them as well.
    expect(plain).not.toContain('aria-label="Monitor"');

    const withMetrics = renderToStaticMarkup(<Header {...base} view="chat" onView={() => {}} monitorAvailable />);
    expect(withMetrics).toContain(">Monitor<");
  });

  it("offers the Image view, and marks it current with drawers of its own: its sessions and its settings", () => {
    const plain = renderToStaticMarkup(<Header {...base} view="chat" onView={() => {}} />);
    expect(plain).toContain(">Image<");
    const image = renderToStaticMarkup(<Header {...base} view="image" onView={() => {}} />);
    expect(image).toMatch(/aria-current="page"[^>]*>Image</);
    expect(image).toContain('aria-label="Sessions"');
    expect(image).toContain('aria-label="Settings"');
  });

  it("marks the current view and hides the chat drawers outside the chat", () => {
    const chat = renderToStaticMarkup(<Header {...base} view="chat" onView={() => {}} monitorAvailable />);
    expect(chat).toMatch(/aria-current="page"[^>]*>Playground</);
    expect(chat).toContain('aria-label="Sessions"');

    const monitor = renderToStaticMarkup(<Header {...base} view="monitor" onView={() => {}} monitorAvailable />);
    expect(monitor).toMatch(/aria-current="page"[^>]*>Monitor</);
    expect(monitor).not.toContain('aria-label="Sessions"');
    expect(monitor).not.toContain('aria-label="Settings"');

    const decide = renderToStaticMarkup(<Header {...base} view="decide" onView={() => {}} monitorAvailable />);
    expect(decide).toMatch(/aria-current="page"[^>]*>Decide</);
    expect(decide).not.toContain('aria-label="Sessions"');
    expect(decide).not.toContain('aria-label="Settings"');
  });
});

describe("Header on a phone (spec playground/05, AC 10)", () => {
  const select = (html: string) => html.match(/<select[^>]*aria-label="Choose a view"[\s\S]*?<\/select>/)?.[0] ?? "";

  it("folds the views into one menu below sm, over the same views, the current one selected", () => {
    const html = renderToStaticMarkup(<Header {...base} view="server" onView={() => {}} monitorAvailable />);
    const menu = select(html);
    expect(menu).toContain("sm:hidden");
    expect(menu.match(/<option/g)).toHaveLength(6);
    expect(menu).toMatch(/<option value="server" selected="">Server<\/option>/);
    for (const label of ["Playground", "Decide", "Image", "Transcribe", "Monitor"]) expect(menu).toContain(`>${label}</option>`);
    // The tabs are what `sm` and up show.
    expect(html).toMatch(/<nav aria-label="View" class="hidden[^"]*sm:flex/);
  });

  it("offers the Monitor in the menu only when ignis serves metrics, and no menu without a view switch", () => {
    const plain = select(renderToStaticMarkup(<Header {...base} view="chat" onView={() => {}} />));
    expect(plain).toContain(">Decide</option>");
    expect(plain).not.toContain("Monitor");
    expect(select(renderToStaticMarkup(<Header {...base} />))).toBe("");
  });

  it("keeps the model status, Memory and Lock in the bar", () => {
    const html = renderToStaticMarkup(<Header {...base} model={{ state: "ready", id: "qwen3.8-27b", contextLimit: null }} view="chat" onView={() => {}} onForgetKey={() => {}} />);
    expect(html).toContain("ready");
    expect(html).toContain(">Memory");
    expect(html).toContain(">Lock<");
  });
});

describe("Header and the Server view (spec playground/05)", () => {
  it("always offers the Server view, with or without metrics, and gives it no side drawers", () => {
    for (const monitorAvailable of [false, true]) {
      const html = renderToStaticMarkup(<Header {...base} view="chat" onView={() => {}} monitorAvailable={monitorAvailable} />);
      expect(html).toContain(">Server<");
    }
    const server = renderToStaticMarkup(<Header {...base} view="server" onView={() => {}} />);
    expect(server).toMatch(/aria-current="page"[^>]*>Server</);
    expect(server).not.toContain('aria-label="Sessions"');
    expect(server).not.toContain('aria-label="Settings"');
    expect(server).not.toContain('aria-label="Decisions"');
  });

  it("reads \"Switching to <id>…\" while a switch runs, and the reason when the server reports it failed", () => {
    const switching = renderToStaticMarkup(<Header {...base} model={{ state: "switching", from: "qwen3.8-27b", to: "qwen3.8-flash-next", since: 0 }} busy />);
    expect(switching).toContain("Switching to qwen3.8-flash-next…");
    // The flame and the heat bar run through `busy`, as during a reply.
    expect(switching).toMatch(/class="flame[^"]*" data-busy="true"/);
    expect(switching).toMatch(/class="heat" data-busy="true"/);

    const failed = renderToStaticMarkup(<Header {...base} model={{ state: "error", message: "Switch failed: no file", failed: true }} />);
    expect(failed).toContain("Switch failed: no file");
  });
});

// The Transcribe view (spec speech/07, AC 1): a tab of its own, and a listening shown from every view.
describe("Header and the Transcribe view", () => {
  it("offers the Transcribe view with its drawers, the Transcripts and the Speakers", () => {
    expect(renderToStaticMarkup(<Header {...base} view="chat" onView={() => {}} />)).toContain(">Transcribe<");
    const transcribe = renderToStaticMarkup(<Header {...base} view="transcribe" onView={() => {}} />);
    expect(transcribe).toMatch(/aria-current="page"[^>]*>Transcribe</);
    expect(transcribe).toContain('aria-label="Transcripts"');
    expect(transcribe).toContain('aria-label="Notes"');
  });

  it("shows a listening under way from every view, its clock and whether it is paused, and opens the view", () => {
    for (const view of ["chat", "decide", "server", "transcribe"] as const) {
      const html = renderToStaticMarkup(<Header {...base} view={view} onView={() => {}} listening={fixed({ phase: "listening", clockMs: 754_000 })} />);
      expect(html).toMatch(/<button[^>]*title="Open the Transcribe view"[^>]*>.*Listening.*12:34/);
    }
    const paused = renderToStaticMarkup(<Header {...base} view="chat" onView={() => {}} listening={fixed({ phase: "paused", clockMs: 5_000 })} />);
    expect(paused).toContain("Paused");
    expect(paused).not.toContain("dictation-dot");
    expect(renderToStaticMarkup(<Header {...base} view="chat" onView={() => {}} listening={fixed({ phase: "idle", clockMs: 0 })} />)).not.toContain("Open the Transcribe view");
  });
});

describe("PulseChip", () => {
  it("shows the gauges, the queue only when there is one, and a live decoded-token rate", () => {
    const busy = renderToStaticMarkup(<PulseChip running={3} waiting={2} tokensPerSec={96} live paused={false} onOpen={() => {}} />);
    expect(busy).toMatch(/>3<\/b> running/);
    expect(busy).toMatch(/>2<\/b> waiting/);
    expect(busy).toMatch(/>96<\/b> tok\/s</);
    expect(busy).toContain("Tokens decoded per second");
    expect(busy).toContain("pulse-dot");

    const quiet = renderToStaticMarkup(<PulseChip running={0} waiting={0} tokensPerSec={null} live paused onOpen={() => {}} />);
    expect(quiet).not.toContain("waiting");
    expect(quiet).not.toContain("pulse-dot");
  });

  it("says tokens count only at completion on a server without decoded tokens", () => {
    const html = renderToStaticMarkup(<PulseChip running={1} waiting={0} tokensPerSec={40} live={false} paused={false} onOpen={() => {}} />);
    expect(html).toMatch(/>40<\/b> tok\/s done/);
    expect(html).toContain("counted as requests complete");
  });
});
