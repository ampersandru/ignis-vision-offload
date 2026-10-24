import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { DictationMic, DictationStrip, HintText, MIC_TITLE, visibleText } from "./Dictation.tsx";
import { type DictationState, IDLE_STATE, METER_BARS } from "./dictation.ts";
import { fixedDictation } from "./fixture.ts";

// The mic, the strip and the hint line's code spans (spec speech/04, "What
// the user sees"), drawn in each state.

const at = (state: Partial<DictationState>): DictationState => ({ ...IDLE_STATE, ...state });
const mic = (state: Partial<DictationState>, label?: string) => renderToStaticMarkup(<DictationMic dictation={fixedDictation(state)} state={at(state)} label={label} />);
const strip = (state: Partial<DictationState>) => renderToStaticMarkup(<DictationStrip dictation={fixedDictation(state)} state={at(state)} />);

describe("DictationMic", () => {
  it("is an ash icon beside Send, its tooltip naming the shortcut", () => {
    const html = mic({});
    expect(html).toContain('aria-label="Dictate"');
    expect(html).toContain(`title="${MIC_TITLE}"`);
    expect(MIC_TITLE).toBe("Dictate. Or hold Ctrl+Shift+Space while you speak.");
    expect(html).toContain("text-ash");
    expect(html).not.toContain("M2.5 2.5l11 11");
  });

  it("is struck through and dimmed while it cannot start, labelled so", () => {
    const html = mic({ unavailable: { sentence: "x", blocked: false } });
    expect(html).toContain('aria-label="Dictate (unavailable)"');
    expect(html).toContain('aria-disabled="true"');
    expect(html).toContain("opacity-45");
    expect(html).toContain("M2.5 2.5l11 11");
    expect(html).not.toContain("title=");
  });

  it("becomes the ember Done chip with a square stop while listening, and a disabled Finishing after", () => {
    const listening = mic({ phase: "listening" });
    expect(listening).toContain(">Done</button>");
    expect(listening).toContain("bg-[#ff5a1f]");
    expect(listening).toContain('<rect x="3" y="3" width="10" height="10" fill="currentColor">');
    const finishing = mic({ phase: "finishing" });
    expect(finishing).toMatch(/<button[^>]*disabled=""[^>]*>Finishing<\/button>/);
  });

  it("takes a Decide field's label", () => {
    expect(mic({}, "Dictate into Question")).toContain('aria-label="Dictate into Question"');
    expect(mic({ unavailable: { sentence: "x", blocked: true } }, "Dictate into Evidence")).toContain('aria-label="Dictate into Evidence (unavailable)"');
  });
});

describe("DictationStrip", () => {
  it("is a polite status that says Listening… before the first word, with the meter, the timer and Discard Esc", () => {
    const html = strip({ phase: "listening", elapsed: 67 });
    expect(html).toContain('role="status"');
    expect(html).toContain('aria-live="polite"');
    expect(html).toContain("Listening…");
    expect(html).toContain("1:07");
    // The clock is out of the live region: a reader would hear it every second.
    expect(html).toMatch(/<span aria-hidden="true"[^>]*><span class="dictation-dot[^"]*"><\/span>1:07<\/span>/);
    expect(html).toContain("dictation-dot");
    expect(html).toContain("dictation-heat");
    expect(html).toMatch(/Discard <span[^>]*>Esc<\/span>/);
    expect(html.match(/dictation-bar/g)).toHaveLength(METER_BARS);
    expect(html).toMatch(/<div aria-hidden="true"[^>]*>(<div class="dictation-bar)/);
    expect(html).toContain("bg-kiln");
  });

  it("shows the settled words light and the provisional tail grey, the loud bars ember", () => {
    const levels = [...Array(METER_BARS - 1).fill(0), 0.8];
    const html = strip({ phase: "listening", stable: "Apri vram.rs e", tail: " controlla", levels });
    expect(html).toContain('<span class="text-[#eae8e4]">Apri vram.rs e</span><span class="text-[#939ba4]"> controlla</span>');
    expect(html).not.toContain("Listening…");
    expect(html).toContain("background:#ff5a1f");
    expect(html).toContain("background:#4a525c");
  });

  it("says Finishing while the last words are decoded, and drops the timer", () => {
    const html = strip({ phase: "finishing", stable: "ciao" });
    expect(html).toContain(">Finishing<");
    expect(html).not.toContain("dictation-dot");
    expect(html).toContain("dictation-heat");
  });

  it("keeps what was heard after a stop, offering to insert it, with no heat running", () => {
    const html = strip({ phase: "failed", stable: "Apri vram.rs", tail: " e" });
    expect(html).toContain(">Insert what was heard</button>");
    expect(html).toContain("Apri vram.rs");
    expect(html).not.toContain("dictation-heat");
    // Nothing heard, nothing to insert.
    expect(strip({ phase: "failed" })).not.toContain("Insert what was heard");
  });

  it("is not drawn while idle or waiting", () => {
    expect(strip({})).toBe("");
    expect(strip({ phase: "waiting", retryIn: 3 })).toBe("");
  });
});

describe("visibleText", () => {
  it("keeps the newest words when the strip's two lines cannot hold them all", () => {
    expect(visibleText("short", " tail", 140)).toEqual({ stable: "short", tail: " tail" });
    expect(visibleText("abcdefghij", " xyz", 8)).toEqual({ stable: "…ghij", tail: " xyz" });
    expect(visibleText("ab", " uvwxyz", 4)).toEqual({ stable: "…", tail: "wxyz" });
  });
});

describe("HintText", () => {
  it("sets the code spans of a sentence in the mono face", () => {
    expect(renderToStaticMarkup(<HintText text="Speech is turned off on this server (`speech.model = off`)." />)).toBe(
      'Speech is turned off on this server (<code class="font-mono text-[11px]">speech.model = off</code>).',
    );
  });
});

describe("the strip's motion", () => {
  it("stops the heat bar, the dot and the meter under prefers-reduced-motion", () => {
    const css = readFileSync(new URL("../styles.css", import.meta.url), "utf8");
    const reduced = css.match(/@media \(prefers-reduced-motion: reduce\) \{\s*\.dictation-heat,\s*\.dictation-dot \{\s*animation: none;\s*\}\s*\.dictation-bar \{\s*transition: none;\s*\}/);
    expect(reduced).not.toBeNull();
  });
});
