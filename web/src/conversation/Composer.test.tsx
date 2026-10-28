import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { contextUsage } from "../metrics/context.ts";
import { SECURE_CONTEXT } from "../speech/availability.ts";
import { type DictationState, LISTENING } from "../speech/dictation.ts";
import { fixedDictation } from "../speech/fixture.ts";
import type { AttachedChip } from "../workspace/attach.ts";
import { Composer, IDLE_HINT as IDLE } from "./Composer.tsx";

// The chat's prompt box with dictation (spec speech/04): the mic beside Send,
// the strip above the box, Send waiting while a dictation runs, and the hint
// line's sentences.

const none = () => {};

function render(
  state: Partial<DictationState> = {},
  extra: { value?: string; busy?: boolean; canAttach?: boolean; attachments?: AttachedChip[] } = {},
) {
  return renderToStaticMarkup(
    <Composer
      value={extra.value ?? "Prima di toccare il piano:"}
      onChange={none}
      ready
      busy={extra.busy ?? false}
      streamingHere={false}
      usage={contextUsage([], null, null)}
      onSend={none}
      onStop={none}
      canAttach={extra.canAttach ?? false}
      attachments={extra.attachments ?? []}
      attachError={null}
      onAttach={none}
      onDetach={none}
      images={[]}
      imageError={null}
      onAddImages={none}
      onRemoveImage={none}
      dictation={fixedDictation(state)}
    />,
  );
}

/** The Send button's opening tag. */
const send = (html: string) => /<button[^>]*>Send<\/button>/.exec(html)?.[0] ?? "";

describe("the chat's prompt box with dictation", () => {
  it("puts the mic beside Send, after the prompt, and says the shortcut in the idle hint", () => {
    const html = render();
    expect(html.indexOf('aria-label="Dictate"')).toBeGreaterThan(html.indexOf('aria-label="Prompt"'));
    expect(html.indexOf('aria-label="Dictate"')).toBeLessThan(html.indexOf(">Send</button>"));
    expect(html).toContain(IDLE);
    expect(IDLE).toBe("Enter sends, Shift+Enter adds a line. Dictate with the mic, or hold Ctrl+Shift+Space while you speak.");
    expect(send(html)).not.toContain('disabled=""');
  });

  it("holds Send while listening and finishing, so a dictation is never sent by itself", () => {
    for (const phase of ["listening", "finishing", "waiting", "failed"] as const) {
      expect(send(render({ phase }))).toContain('disabled=""');
    }
  });

  it("shows the strip above the box while listening, and runs the heat along the box's foot", () => {
    const html = render({ phase: "listening", stable: "Apri vram.rs" });
    expect(html.indexOf('role="status"')).toBeLessThan(html.indexOf('aria-label="Prompt"'));
    expect(html).toContain("max-w-3xl");
    expect(html.match(/dictation-heat/g)).toHaveLength(2);
    expect(html).toContain(LISTENING);
    expect(html).not.toContain("shadow-[inset_0_-2px_0_var(--line)]");
    expect(render()).toContain("shadow-[inset_0_-2px_0_var(--line)]");
  });

  it("says why the mic cannot start in the hint line, its code in the mono face", () => {
    const html = render({ unavailable: { sentence: SECURE_CONTEXT, blocked: false } });
    expect(html).toContain('aria-label="Dictate (unavailable)"');
    expect(html).toContain('start the server with <code class="font-mono text-[11px]">--server-expose cloudflare-quick</code>.');
    expect(html).not.toContain(IDLE);
  });

  it("colours the busy wait warn and a stopped dictation fault", () => {
    expect(render({ phase: "waiting", retryIn: 2 })).toMatch(/<p class="[^"]*text-warn[^"]*">Speech is busy with another dictation. Trying again in 2 s.<\/p>/);
    const failed = render({ phase: "failed", failure: "Dictation stopped: the connection to the server closed. Insert what was heard, or discard it." });
    expect(failed).toMatch(/<p class="[^"]*text-fault[^"]*">Dictation stopped: the connection to the server closed./);
  });

  it("keeps the other session's streaming note over the idle words", () => {
    expect(render({}, { busy: true })).toContain("A reply is streaming in another session. Send works again once it ends.");
  });
});

describe("the chat's prompt box and the workspace", () => {
  const brief: AttachedChip = { name: "brief.pdf", kind: "pdf", characters: 1200, size: 48_000 };

  it("attaches files into the workspace while the Workspace switch is on, and a file alone is a prompt", () => {
    const html = render({}, { value: "", canAttach: true, attachments: [brief] });
    expect(html).toContain('name="attachments"');
    expect(html).toContain("brief.pdf");
    expect(html).toContain('aria-label="Remove brief.pdf"');
    expect(send(html)).not.toContain('disabled=""');
    expect(send(render({}, { value: "", canAttach: true }))).toContain('disabled=""');
  });

  it("shows an attached PDF by its text and an attached recording by its size", () => {
    const memo: AttachedChip = { name: "standup.m4a", kind: "audio", characters: 0, size: 2_457_600 };
    const html = render({}, { value: "", canAttach: true, attachments: [brief, memo] });
    expect(html).toMatch(/brief\.pdf<\/span><span[^>]*>PDF, 1,200 chars<\/span>/);
    expect(html).toMatch(/standup\.m4a<\/span><span[^>]*>audio, 2\.3 MB<\/span>/);
    expect(html).toContain('aria-label="Remove standup.m4a"');
  });

  it("offers pictures only while the Workspace switch is off", () => {
    const html = render({}, { value: "", canAttach: false, attachments: [brief] });
    expect(html).not.toContain('name="attachments"');
    expect(html).not.toContain("brief.pdf");
    expect(html).toContain('name="images"');
    expect(send(html)).toContain('disabled=""');
  });
});
