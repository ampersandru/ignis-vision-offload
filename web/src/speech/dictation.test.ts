import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LISTENING_ELSEWHERE, MIC_BLOCKED, SECURE_CONTEXT, TURNED_OFF } from "./availability.ts";
import { type Capture, MicBlocked } from "./capture.ts";
import {
  BUSY_RETRY_MAX_SECONDS,
  BUSY_RETRY_SECONDS,
  CONNECTION_LOST,
  DictationController,
  dictationContext,
  type DictationDeps,
  type DictationState,
  dictating,
  dictationHint,
  DISCARDED,
  FINISHING,
  HOLD_MS,
  LISTENING,
  METER_BARS,
  NOTHING_HEARD,
  NO_WEBSOCKET,
  STILL_BUSY,
} from "./dictation.ts";
import { setListening } from "./listening.ts";
import type { AutoStop, Refusal, Transcription, TranscriptionHandlers } from "./realtime.ts";

// One field's dictation (spec speech/04) through the controller the mic, the
// strip and the hint line read: the session and the mic are fakes the test
// drives, the field a string with a cursor.

type FakeSession = { context: string; handlers: TranscriptionHandlers; calls: string[]; appended: number[][]; autoStop: AutoStop; settled: (number | null)[] };
type FakeMic = { chunk: (pcm: number[]) => void; stopped: boolean; flushes: number };

/** Every controller a test made, disposed after it: the page-wide one-at-a-time rule must not leak between tests. */
const made: DictationController[] = [];

function harness(options: { value?: string; at?: number; context?: string; refuseMic?: Error; silence?: number | null } = {}) {
  const field = { value: options.value ?? "", at: options.at ?? (options.value ?? "").length };
  const sessions: FakeSession[] = [];
  const mics: FakeMic[] = [];
  const events: string[] = [];
  const deps: DictationDeps = {
    silence: () => (options.silence === undefined ? 3_000 : options.silence),
    open: (context, handlers, autoStop) => {
      const session: FakeSession = { context, handlers, calls: [], appended: [], autoStop, settled: [] };
      sessions.push(session);
      const transcription: Transcription = {
        append: (pcm) => session.appended.push([...pcm]),
        autoStop: (ms) => session.settled.push(ms),
        commit: () => session.calls.push("commit"),
        discard: () => session.calls.push("discard"),
        close: () => session.calls.push("close"),
      };
      return transcription;
    },
    capture: async (onChunk) => {
      if (options.refuseMic) throw options.refuseMic;
      const mic: FakeMic = { chunk: (pcm) => onChunk(new Int16Array(pcm)), stopped: false, flushes: 0 };
      mics.push(mic);
      const capture: Capture = {
        flush: async () => {
          mic.flushes++;
        },
        stop: () => {
          mic.stopped = true;
        },
      };
      return capture;
    },
    onKeyRequired: () => events.push("key"),
    onUnavailable: () => events.push("unavailable"),
  };
  const controller = new DictationController(
    {
      read: () => ({ value: field.value, at: field.at }),
      write: (value) => {
        field.value = value;
      },
      context: () => options.context ?? "",
    },
    deps,
  );
  made.push(controller);
  const state = (): DictationState => controller.getState();
  return { controller, field, sessions, mics, events, state, session: () => sessions.at(-1)!, mic: () => mics.at(-1)! };
}

/** Lets the mic's promise and the flush settle. */
const settle = () => vi.advanceTimersByTimeAsync(0);

const key = (code: string, init: { key?: string; ctrlKey?: boolean; shiftKey?: boolean; repeat?: boolean } = {}) => {
  const event = { code, key: init.key ?? code, ctrlKey: !!init.ctrlKey, shiftKey: !!init.shiftKey, repeat: !!init.repeat, prevented: false, preventDefault: () => (event.prevented = true) };
  return event;
};
const chord = (repeat = false) => key("Space", { key: " ", ctrlKey: true, shiftKey: true, repeat });

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  setListening(false);
  for (const controller of made.splice(0)) controller.dispose();
  vi.useRealTimers();
});

describe("a dictation", () => {
  it("opens one session at the press, with the field's context, and listens", async () => {
    const h = harness({ context: "The last reply.\n\nThe draft." });
    h.controller.toggle();
    expect(h.state().phase).toBe("listening");
    expect(h.sessions).toHaveLength(1);
    expect(h.session().context).toBe("The last reply.\n\nThe draft.");
    await settle();
    h.mic().chunk([100, -100]);
    expect(h.session().appended).toEqual([[100, -100]]);
  });

  it("shows the level of each chunk, the newest last, and the seconds listened", async () => {
    const h = harness();
    h.controller.toggle();
    await settle();
    expect(h.state().levels).toHaveLength(METER_BARS);
    h.mic().chunk(Array(1600).fill(16000));
    expect(h.state().levels.at(-1)).toBe(1);
    expect(h.state().levels.at(0)).toBe(0);
    await vi.advanceTimersByTimeAsync(2_000);
    expect(h.state().elapsed).toBe(2);
  });

  it("shows the stable text and the provisional tail as they come, at whatever pace they come", async () => {
    const h = harness();
    h.controller.toggle();
    h.session().handlers.onDelta("Apri vram.rs");
    h.session().handlers.onTail(" e controlla");
    expect(h.state()).toMatchObject({ stable: "Apri vram.rs", tail: " e controlla" });
    h.session().handlers.onDelta(" e controlla perché");
    h.session().handlers.onTail(" plan");
    expect(h.state()).toMatchObject({ stable: "Apri vram.rs e controlla perché", tail: " plan" });
  });

  it("Done flushes the mic, lets it go and commits; the transcript lands at the cursor, selected, and is never sent", async () => {
    const h = harness({ value: "Prima di toccare il piano:", at: 26 });
    h.controller.toggle();
    await settle();
    h.mic().chunk([1, 2, 3]);
    h.controller.toggle();
    expect(h.state().phase).toBe("finishing");
    await settle();
    expect(h.mic()).toMatchObject({ flushes: 1, stopped: true });
    expect(h.session().calls).toEqual(["commit"]);
    h.session().handlers.onCompleted("Apri vram.rs e controlla.");
    expect(h.field.value).toBe("Prima di toccare il piano: Apri vram.rs e controlla.");
    expect(h.state()).toMatchObject({ phase: "idle", stable: "", tail: "", selection: { start: 27, end: 52 } });
    expect(h.session().calls).toEqual(["commit", "close"]);
    expect(dictationHint(h.state(), "Send")).toEqual({ text: "Inserted 4 words at the cursor. Nothing is sent until you press Send.", tone: "ash" });
    await vi.advanceTimersByTimeAsync(6_000);
    expect(dictationHint(h.state(), "Send")).toBeNull();
  });

  it("commits nothing when no audio came: there was nothing to hear", async () => {
    const h = harness({ value: "draft" });
    h.controller.toggle();
    await settle();
    h.controller.done();
    await settle();
    expect(h.session().calls).toEqual(["close"]);
    expect(h.field.value).toBe("draft");
    expect(dictationHint(h.state(), "Send")).toEqual({ text: NOTHING_HEARD, tone: "ash" });
  });

  it("Esc clears the buffer, closes the session, lets the mic go and leaves the field as it was", async () => {
    const h = harness({ value: "draft" });
    h.controller.toggle();
    await settle();
    h.mic().chunk([1]);
    h.session().handlers.onDelta("parole");
    const esc = key("Escape");
    h.controller.keyDown(esc);
    expect(esc.prevented).toBe(true);
    expect(h.session().calls).toEqual(["discard"]);
    expect(h.mic().stopped).toBe(true);
    expect(h.field.value).toBe("draft");
    expect(h.state()).toMatchObject({ phase: "idle", stable: "" });
    expect(dictationHint(h.state(), "Send")).toEqual({ text: DISCARDED, tone: "ash" });
    await vi.advanceTimersByTimeAsync(4_000);
    expect(dictationHint(h.state(), "Send")).toBeNull();
    // Esc with nothing under way is the page's, not the dictation's.
    const idle = key("Escape");
    h.controller.keyDown(idle);
    expect(idle.prevented).toBe(false);
  });

  it("leaves an Esc something on the page took already: a full screen's Esc closes it and keeps the dictation", async () => {
    const h = harness({ value: "draft" });
    h.controller.toggle();
    await settle();
    h.session().handlers.onDelta("parole");
    h.controller.keyDown({ ...key("Escape"), defaultPrevented: true });
    expect(h.session().calls).toEqual([]);
    expect(h.state().phase).toBe("listening");
  });

  it("lets the mic go when the page closes it mid-sentence", async () => {
    const h = harness();
    h.controller.toggle();
    await settle();
    h.controller.dispose();
    expect(h.mic().stopped).toBe(true);
    expect(h.session().calls).toEqual(["close"]);
    expect(h.state().phase).toBe("idle");
  });

  it("says each phase in the hint line, Generate waiting in the studio", async () => {
    const h = harness();
    h.controller.toggle();
    expect(dictationHint(h.state(), "Send")).toEqual({ text: LISTENING, tone: "ash" });
    expect(dictationHint(h.state(), "Generate")).toEqual({ text: `${LISTENING} Generate waits until you press it.`, tone: "ash" });
    await settle();
    h.mic().chunk([1]);
    h.controller.done();
    expect(dictationHint(h.state(), "Send")).toEqual({ text: FINISHING, tone: "ash" });
  });
});

describe("Ctrl+Shift+Space", () => {
  it("held past 400 ms is push-to-talk: the release commits", async () => {
    const h = harness();
    const down = chord();
    h.controller.keyDown(down);
    expect(down.prevented).toBe(true);
    expect(h.state().phase).toBe("listening");
    await settle();
    h.mic().chunk([1]);
    h.controller.keyDown(chord(true));
    h.controller.keyDown(chord(true));
    await vi.advanceTimersByTimeAsync(450);
    h.controller.keyUp(key("Space", { key: " " }));
    expect(h.state().phase).toBe("finishing");
    await settle();
    expect(h.session().calls).toEqual(["commit"]);
  });

  it("tapped, toggles: the release leaves it listening, the next tap ends it", async () => {
    const h = harness();
    h.controller.keyDown(chord());
    await vi.advanceTimersByTimeAsync(150);
    h.controller.keyUp(key("Space", { key: " " }));
    expect(h.state().phase).toBe("listening");
    h.mic().chunk([1]);
    h.controller.keyDown(chord());
    expect(h.state().phase).toBe("finishing");
  });

  it("is only the chord: Space alone, or with one modifier, is typing", () => {
    const h = harness();
    const space = key("Space", { key: " ", ctrlKey: true });
    h.controller.keyDown(space);
    expect(space.prevented).toBe(false);
    expect(h.state().phase).toBe("idle");
  });
});

describe("the page's dictation", () => {
  it("is under way from the press to the text landing, in any field, so a send can wait on it (Q4)", async () => {
    const a = harness();
    const b = harness();
    expect(dictating()).toBe(false);
    a.controller.toggle();
    expect(dictating()).toBe(true);
    await settle();
    a.mic().chunk([1]);
    await a.controller.done();
    expect(dictating()).toBe(true);
    a.session().handlers.onCompleted("words");
    expect(dictating()).toBe(false);
    b.controller.toggle();
    expect(dictating()).toBe(true);
    b.controller.discard();
    expect(dictating()).toBe(false);
  });
});

// Q80: a silence ends the dictation by itself, as Done does.

describe("a silence", () => {
  it("opens the session asking for the viewer's silence, read at the press", () => {
    const h = harness({ silence: 5_000 });
    h.controller.toggle();
    expect(h.session().autoStop).toBe(5_000);
  });

  it("is manual commit when the setting is Off", () => {
    const h = harness({ silence: null });
    h.controller.toggle();
    expect(h.session().autoStop).toBeNull();
  });

  it("ends the dictation at the server's first commit as Done does: the mic let go, one item, the text at the cursor", async () => {
    const h = harness({ value: "Hello", at: 5 });
    h.controller.toggle();
    await settle();
    h.mic().chunk([1, 2, 3]);
    h.session().handlers.onDelta("world");
    h.session().handlers.onCommitted();
    expect(h.state().phase).toBe("finishing");
    expect(dictationHint(h.state(), "Send")).toEqual({ text: FINISHING, tone: "ash" });
    expect(h.mic().stopped).toBe(true);
    // The server committed already: nothing more is committed from here.
    expect(h.session().calls).toEqual([]);
    h.session().handlers.onCompleted("world");
    expect(h.field.value).toBe("Hello world");
    expect(h.state().phase).toBe("idle");
    expect(h.session().calls).toEqual(["close"]);
  });

  it("does not commit a second time when Done was already on its way", async () => {
    const h = harness();
    h.controller.toggle();
    await settle();
    h.mic().chunk([1]);
    const done = h.controller.done();
    h.session().handlers.onCommitted();
    await done;
    expect(h.session().calls).toEqual([]);
  });

  it("hears the server's commit that answers Done's own, which changes nothing", async () => {
    const h = harness({ silence: null });
    h.controller.toggle();
    await settle();
    h.mic().chunk([1]);
    await h.controller.done();
    expect(h.session().calls).toEqual(["commit"]);
    h.session().handlers.onCommitted();
    expect(h.state().phase).toBe("finishing");
  });

  it("sends no more audio once the server committed: a chunk after it would open a next item", async () => {
    const h = harness();
    h.controller.toggle();
    await settle();
    h.mic().chunk([1]);
    h.session().handlers.onCommitted();
    h.mic().chunk([2]);
    expect(h.session().appended).toEqual([[1]]);
  });

  it("lets a mic that resolves after the server's commit go at once", async () => {
    const h = harness();
    h.controller.toggle();
    h.session().handlers.onCommitted();
    await settle();
    expect(h.mic().stopped).toBe(true);
  });

  it("lands nothing from a commit that comes after Esc", async () => {
    const h = harness();
    h.controller.toggle();
    await settle();
    h.controller.discard();
    h.session().handlers.onCommitted();
    expect(h.state().phase).toBe("idle");
  });
});

describe("Ctrl+Shift+Space and the silence", () => {
  it("a press waits to know: held it settles to manual commit, so a held key never stops by itself", async () => {
    const h = harness({ silence: 2_000 });
    h.controller.keyDown(chord());
    expect(h.session().autoStop).toBe("pending");
    await vi.advanceTimersByTimeAsync(HOLD_MS + 50);
    expect(h.session().settled).toEqual([null]);
  });

  it("Done pressed before the hold timer settles to manual commit, so its commit is not left waiting", async () => {
    const h = harness({ silence: 2_000 });
    h.controller.keyDown(chord());
    await settle();
    h.mic().chunk([1]);
    await h.controller.done();
    expect(h.session().settled).toEqual([null]);
    expect(h.session().calls).toEqual(["commit"]);
  });

  it("a tap settles to the viewer's silence at the release", async () => {
    const h = harness({ silence: 2_000 });
    h.controller.keyDown(chord());
    await vi.advanceTimersByTimeAsync(150);
    h.controller.keyUp(key("Space", { key: " " }));
    expect(h.session().settled).toEqual([2_000]);
    await vi.advanceTimersByTimeAsync(1_000);
    expect(h.session().settled).toEqual([2_000]);
  });

  it("a tap with the setting Off settles to manual commit", () => {
    const h = harness({ silence: null });
    h.controller.keyDown(chord());
    h.controller.keyUp(key("Space", { key: " " }));
    expect(h.session().settled).toEqual([null]);
  });

  it("the mic's click is decided at once", () => {
    const h = harness({ silence: 2_000 });
    h.controller.toggle();
    expect(h.session().autoStop).toBe(2_000);
  });
});

describe("when the session is refused", () => {
  const refuse = (h: ReturnType<typeof harness>, refusal: Refusal) => h.session().handlers.onRefused(refusal);

  it("busy: lets the mic go, counts down, tries once more, and says so if still busy", async () => {
    const h = harness();
    h.controller.toggle();
    await settle();
    refuse(h, { kind: "busy" });
    expect(h.mic().stopped).toBe(true);
    expect(h.state()).toMatchObject({ phase: "waiting", retryIn: BUSY_RETRY_SECONDS });
    expect(dictationHint(h.state(), "Send")).toEqual({ text: `Speech is busy with another dictation. Trying again in ${BUSY_RETRY_SECONDS} s.`, tone: "warn" });
    await vi.advanceTimersByTimeAsync(1_000);
    expect(h.state().retryIn).toBe(BUSY_RETRY_SECONDS - 1);
    await vi.advanceTimersByTimeAsync((BUSY_RETRY_SECONDS - 1) * 1_000);
    expect(h.sessions).toHaveLength(2);
    expect(h.state().phase).toBe("listening");
    await settle();
    refuse(h, { kind: "busy" });
    expect(h.state().phase).toBe("idle");
    expect(dictationHint(h.state(), "Send")).toEqual({ text: STILL_BUSY, tone: "warn" });
    expect(h.sessions).toHaveLength(2);
  });

  it("busy: counts down the refusal's Retry-After, to a cap", async () => {
    const h = harness();
    h.controller.toggle();
    await settle();
    refuse(h, { kind: "busy", retryAfter: 1 });
    expect(h.state()).toMatchObject({ phase: "waiting", retryIn: 1 });
    await vi.advanceTimersByTimeAsync(1_000);
    expect(h.sessions).toHaveLength(2);
    await settle();
    h.controller.discard();
    h.controller.toggle();
    await settle();
    refuse(h, { kind: "busy", retryAfter: 600 });
    expect(h.state().retryIn).toBe(BUSY_RETRY_MAX_SECONDS);
  });

  it("no_websocket: the server would have opened it, so it is not busy and is not retried", async () => {
    const h = harness();
    h.controller.toggle();
    await settle();
    refuse(h, { kind: "no_websocket" });
    expect(h.state().phase).toBe("idle");
    expect(dictationHint(h.state(), "Send")).toEqual({ text: NO_WEBSOCKET, tone: "fault" });
    await vi.advanceTimersByTimeAsync(10_000);
    expect(h.sessions).toHaveLength(1);
  });

  it("busy, then free on the retry: the dictation goes on", async () => {
    const h = harness();
    h.controller.toggle();
    refuse(h, { kind: "busy" });
    await vi.advanceTimersByTimeAsync(BUSY_RETRY_SECONDS * 1_000);
    await settle();
    h.mic().chunk([1]);
    expect(h.session().appended).toEqual([[1]]);
    expect(h.state().phase).toBe("listening");
  });

  it("unavailable: the mic is struck with the server's reason in the spec's words, and the page reads the models again", () => {
    const h = harness();
    h.controller.toggle();
    refuse(h, { kind: "unavailable", reason: "the speech model is turned off (`speech.model` is `off`)" });
    expect(h.state()).toMatchObject({ phase: "idle", unavailable: { sentence: TURNED_OFF, blocked: false } });
    expect(h.events).toEqual(["unavailable"]);
    // A fresh read of the models is the word from then on.
    h.controller.setAvailability({ secure: true, speech: { available: true } });
    expect(h.state().unavailable).toBeNull();
  });

  it("wants the key: the page asks for it", () => {
    const h = harness();
    h.controller.toggle();
    refuse(h, { kind: "key", message: "401" });
    expect(h.state().phase).toBe("idle");
    expect(h.events).toEqual(["key"]);
  });

  it("unreachable: says the dictation could not start", () => {
    const h = harness();
    h.controller.toggle();
    refuse(h, { kind: "unreachable", message: "Could not reach ignis: TypeError: fetch failed" });
    expect(dictationHint(h.state(), "Send")).toEqual({ text: "Dictation could not start: Could not reach ignis: TypeError: fetch failed.", tone: "fault" });
  });
});

describe("a dictation that stops under the person", () => {
  it("keeps what was heard — the stable text and the last tail — with the fault sentence, to insert or discard", async () => {
    const h = harness({ value: "Nota:" });
    h.controller.toggle();
    await settle();
    h.session().handlers.onDelta("Apri vram.rs e");
    h.session().handlers.onTail(" controlla");
    h.session().handlers.onClosed();
    expect(h.mic().stopped).toBe(true);
    expect(h.state()).toMatchObject({ phase: "failed", stable: "Apri vram.rs e", tail: " controlla" });
    expect(dictationHint(h.state(), "Send")).toEqual({ text: CONNECTION_LOST, tone: "fault" });
    h.controller.insertHeard();
    expect(h.field.value).toBe("Nota: Apri vram.rs e controlla");
    expect(h.state().phase).toBe("idle");
    expect(dictationHint(h.state(), "Send")?.text).toBe("Inserted 4 words at the cursor. Nothing is sent until you press Send.");
  });

  it("names a failed item with the server's message, and discards on Esc", async () => {
    const h = harness({ value: "x" });
    h.controller.toggle();
    await settle();
    h.mic().chunk([1]);
    h.controller.done();
    await settle();
    h.session().handlers.onFailed("the speech model failed this item's transcription");
    expect(dictationHint(h.state(), "Send")).toEqual({
      text: "Dictation stopped: the speech model failed this item's transcription. Insert what was heard, or discard it.",
      tone: "fault",
    });
    h.controller.keyDown(key("Escape"));
    expect(h.field.value).toBe("x");
    expect(dictationHint(h.state(), "Send")?.text).toBe(DISCARDED);
  });
});

describe("a mic that cannot start", () => {
  it("outside https and localhost: never opens a session, and a press says why", () => {
    const h = harness();
    h.controller.setAvailability({ secure: false, speech: { available: true } });
    expect(h.state().unavailable).toEqual({ sentence: SECURE_CONTEXT, blocked: false });
    h.controller.toggle();
    expect(h.sessions).toHaveLength(0);
    expect(dictationHint(h.state(), "Decide")).toEqual({ text: SECURE_CONTEXT, tone: "ash" });
  });

  it("blocked by the browser: closes the session, strikes the mic, and a press asks again", async () => {
    const h = harness({ refuseMic: new MicBlocked("Permission denied") });
    h.controller.toggle();
    await settle();
    expect(h.session().calls).toEqual(["close"]);
    expect(h.state()).toMatchObject({ phase: "idle", unavailable: { sentence: MIC_BLOCKED, blocked: true } });
    // Said at once too: a Decide field's line shows no idle reason.
    expect(dictationHint(h.state(), "Decide")).toEqual({ text: MIC_BLOCKED, tone: "ash" });
    h.controller.toggle();
    expect(h.sessions).toHaveLength(2);
  });
});

describe("one dictation at a time", () => {
  it("starting another field's mic ends the current dictation, whose text still lands in its own field", async () => {
    const question = harness({ value: "" });
    const evidence = harness({ value: "" });
    question.controller.toggle();
    await settle();
    question.mic().chunk([1]);
    evidence.controller.toggle();
    expect(question.state().phase).toBe("finishing");
    expect(evidence.state().phase).toBe("listening");
    await settle();
    expect(question.session().calls).toEqual(["commit"]);
    question.session().handlers.onCompleted("Il merge è pronto?");
    expect(question.field.value).toBe("Il merge è pronto?");
    expect(evidence.field.value).toBe("");
    evidence.controller.dispose();
  });

  it("a stopped dictation inserts what was heard when another starts", async () => {
    const first = harness({ value: "" });
    const second = harness();
    first.controller.toggle();
    first.session().handlers.onDelta("già sentito");
    first.session().handlers.onClosed();
    second.controller.toggle();
    expect(first.field.value).toBe("già sentito");
    second.controller.dispose();
  });
});

// The Transcribe view's listening holds the page's speech (spec speech/07, Q85).
describe("while the Transcribe view listens", () => {
  it("a dictation under way ends as when another mic starts, its text landing in its own field", async () => {
    const h = harness({ value: "" });
    h.controller.toggle();
    await settle();
    h.mic().chunk([1]);
    setListening(true);
    expect(h.state().phase).toBe("finishing");
    await settle();
    expect(h.session().calls).toEqual(["commit"]);
    h.session().handlers.onCompleted("Prima della riunione.");
    expect(h.field.value).toBe("Prima della riunione.");
  });

  it("strikes the mic with Listening in Transcribe: a press says it and opens no session", () => {
    const h = harness();
    h.controller.setAvailability({ secure: true, listening: true });
    expect(h.state().unavailable).toEqual({ sentence: LISTENING_ELSEWHERE, blocked: false });
    expect(LISTENING_ELSEWHERE).toBe("Listening in Transcribe.");
    h.controller.toggle();
    expect(h.sessions).toHaveLength(0);
    expect(h.state().notice).toEqual({ kind: "said", text: LISTENING_ELSEWHERE, tone: "ash" });
    h.controller.setAvailability({ secure: true, listening: false });
    expect(h.state().unavailable).toBeNull();
  });
});

describe("dictationContext", () => {
  it("is the last reply first and the field's text last, whole, skipping what is empty", () => {
    expect(dictationContext(["Parti da plan_vram.", "  ", "Prima di toccare il piano: "])).toBe("Parti da plan_vram.\n\nPrima di toccare il piano:");
    expect(dictationContext(["", ""])).toBe("");
  });
});
