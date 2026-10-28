import type { SpeechStatus } from "../api/model.ts";
import { MIC_BLOCKED, unavailable } from "./availability.ts";
import { type Capture, MicBlocked } from "./capture.ts";
import { insertAt, wordCount } from "./insert.ts";
import { levelOf } from "./pcm.ts";
import type { AutoStop, Refusal, Transcription, TranscriptionHandlers } from "./realtime.ts";
import { dictationSilence } from "./silence.ts";

// One field's dictation (spec speech/04): the state the mic, the strip and the
// hint line read, and what drives it. It knows the field only as text with a
// cursor, the session and the mic only through what they hand it, so it runs
// in a test with fakes for all three (dictation.test.ts).
//
// - A press opens the session and asks for the mic at once; the audio waits
//   in the session until the server has taken the configuration.
// - Done flushes the mic, lets it go and commits; the transcript lands at the
//   cursor, selected. So does a silence (Q80): the server's `server_vad` commits
//   the item after the viewer's silence (3 s unless set otherwise), and that
//   first commit ends the dictation as Done does. A held Ctrl+Shift+Space
//   (push-to-talk) never stops by itself: a press by the key waits out the
//   hold before the session is told which it is. Esc clears the session's
//   buffer and closes it. Nothing a dictation does sends the prompt (Q4).
// - The whole page dictates into one field at a time: starting another ends
//   the current one, whose text still lands in its own field (Q17).
// - Partials are best effort: the text shows as it comes, at no assumed pace.

/** Bars in the level meter. */
export const METER_BARS = 28;

/** Ctrl+Shift+Space held longer than this is push-to-talk; shorter, a toggle (owner, 2026-10-24). */
export const HOLD_MS = 400;

/**
 * The wait before the one retry of a session refused for every one being taken, when the refusal gave no
 * `Retry-After`: a session frees when a person ends a dictation.
 */
export const BUSY_RETRY_SECONDS = 3;

/** The longest the page waits on a `Retry-After` before its one retry. */
export const BUSY_RETRY_MAX_SECONDS = 15;

const INSERTED_MS = 6_000;
const DISCARDED_MS = 4_000;
const NOTICE_MS = 6_000;

// The hint line's sentences (spec speech/04, "The hint line"): the shipped copy.
export const LISTENING = "Listening. Done puts the text at the cursor; Esc discards it.";
export const FINISHING = "Finishing the last words.";
export const DISCARDED = "Dictation discarded.";
export const CONNECTION_LOST = "Dictation stopped: the connection to the server closed. Insert what was heard, or discard it.";
const busy = (seconds: number) => `Speech is busy with another dictation. Trying again in ${seconds} s.`;
// Beyond the spec's tables: the retry refused too, a commit with no audio, a start that failed, an item that failed.
export const STILL_BUSY = "The dictation still could not open: speech is busy. Press the mic to try again.";
export const NO_WEBSOCKET = "Dictation could not start: the server is up, but this address does not carry WebSockets.";
export const NOTHING_HEARD = "Nothing was heard.";
const cannotStart = (why: string) => `Dictation could not start: ${why.replace(/\.$/, "")}.`;
const stopped = (why: string) => `Dictation stopped: ${why.replace(/\.$/, "")}. Insert what was heard, or discard it.`;

export type Tone = "ash" | "warn" | "fault";

/** What a field's dictation shows. */
export type DictationState = {
  phase: "idle" | "listening" | "finishing" | "waiting" | "failed";
  /** The settled words, as the deltas brought them. */
  stable: string;
  /** The provisional tail, which the next partial may revise. */
  tail: string;
  /** The level of the last METER_BARS chunks, 0 to 1, the newest last. */
  levels: number[];
  /** Seconds since the press. */
  elapsed: number;
  /** Seconds left before the retry of a busy refusal. */
  retryIn: number;
  /** Why a failed dictation stopped. */
  failure: string;
  /** A passing word for the hint line: what was inserted, a discard, a refusal. */
  notice: { kind: "inserted"; words: number } | { kind: "said"; text: string; tone: Tone } | null;
  /** Why the mic cannot start, when it cannot. */
  unavailable: { sentence: string; blocked: boolean } | null;
  /** The range the last insertion holds, for the field to select; `seq` tells one insertion from the next. */
  selection: { start: number; end: number; seq: number } | null;
};

/** The field a dictation writes into, as text with a cursor. */
export type DictationTarget = {
  /** The text, and where its cursor or selection ends: a second dictation lands after the first one's selected text. */
  read: () => { value: string; at: number };
  write: (value: string) => void;
  /** The context the session is opened with (spec speech/04, AC 7). */
  context: () => string;
};

export type DictationDeps = {
  open: (context: string, handlers: TranscriptionHandlers, autoStop: AutoStop) => Transcription;
  /** The silence that ends a dictation, read at the press; `null` is Off. */
  silence?: () => number | null;
  capture: (onChunk: (pcm: Int16Array) => void) => Promise<Capture>;
  now?: () => number;
  onKeyRequired?: () => void;
  /** The server said the speech model cannot serve: the page reads GET /v1/models again. */
  onUnavailable?: () => void;
};

/** The part of a keyboard event the shortcut reads; `defaultPrevented` once something on the page took it, as a full screen takes its Esc. */
export type KeyLike = { key: string; code: string; ctrlKey: boolean; shiftKey: boolean; repeat: boolean; defaultPrevented?: boolean; preventDefault: () => void };

const silenceOf = (deps: DictationDeps) => (deps.silence ?? dictationSilence)();

const quiet = (): number[] => Array(METER_BARS).fill(0);

/** A field's dictation before any press. */
export const IDLE_STATE: DictationState = {
  phase: "idle",
  stable: "",
  tail: "",
  levels: quiet(),
  elapsed: 0,
  retryIn: 0,
  failure: "",
  notice: null,
  unavailable: null,
  selection: null,
};

/**
 * The context a session is opened with (spec speech/04, Q15): the parts that have text, in the order given — the
 * last reply first, the field's own text last (owner, 2026-10-24) — whole. The server keeps the last
 * `speech.context_max_tokens` of it, so a long reply contributes its tail.
 */
export function dictationContext(parts: string[]): string {
  return parts
    .map((part) => part.trim())
    .filter((part) => part !== "")
    .join("\n\n");
}

/** The dictation under way on this page, if any: one field at a time. */
let holder: DictationController | null = null;

/** Every field whose dictation is not idle, for what waits on any of them: Decide's run (Q4). */
const underWay = new Set<DictationController>();
const pageListeners = new Set<() => void>();

/** Whether any field on the page is being dictated into: a listening, finishing, waiting or stopped dictation. */
export const dictating = (): boolean => underWay.size > 0;

export function subscribeDictating(listener: () => void): () => void {
  pageListeners.add(listener);
  return () => pageListeners.delete(listener);
}

export class DictationController {
  private state: DictationState = IDLE_STATE;
  private readonly listeners = new Set<() => void>();
  private readonly now: () => number;
  /** Bumped by every start and every end, so a late answer from an earlier one is dropped. */
  private attempt = 0;
  private session: Transcription | null = null;
  /** The mic once it has started, and the promise of it while the browser is asked. */
  private capture: Capture | null = null;
  private mic: Promise<Capture | null> = Promise.resolve(null);
  private samples = 0;
  private startedAt = 0;
  private clock: ReturnType<typeof setInterval> | undefined;
  private noticeTimer: ReturnType<typeof setTimeout> | undefined;
  /** This press was refused busy once already. */
  private retried = false;
  private holdAt = 0;
  private chordPress = false;
  private holdTimer: ReturnType<typeof setTimeout> | undefined;
  /** The server committed this dictation's item: Done must not commit again. */
  private serverCommitted = false;
  private seq = 0;
  private secure = true;
  private speech: SpeechStatus | undefined;
  /** The Transcribe view is listening (spec speech/07, Q85). */
  private listening = false;
  /** What the last refused upgrade said, until GET /v1/models is read again. */
  private refusedReason: string | null = null;
  private blocked = false;

  constructor(
    private readonly target: DictationTarget,
    private readonly deps: DictationDeps,
  ) {
    this.now = deps.now ?? (() => Date.now());
  }

  getState = (): DictationState => this.state;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  /** Whether a dictation is under way: Send waits until it ends (Q4). */
  get active(): boolean {
    return this.state.phase !== "idle";
  }

  /** The page's secure context, the server's word on the speech model, and whether the Transcribe view listens. */
  setAvailability({ secure, speech, listening = false }: { secure: boolean; speech?: SpeechStatus; listening?: boolean }) {
    this.secure = secure;
    this.speech = speech;
    this.listening = listening;
    this.refusedReason = null;
    this.update({});
  }

  /** The mic's click, and a tap of the shortcut. */
  toggle(options: { chord?: boolean } = {}) {
    if (this.state.phase === "listening") return void this.done();
    if (this.state.phase !== "idle") return;
    const cause = this.state.unavailable;
    if (cause && !cause.blocked) return this.say(cause.sentence, "ash", NOTICE_MS);
    this.retried = false;
    this.chordPress = options.chord === true;
    this.start();
  }

  /** Done: the rest of the audio goes, and the commit after it. */
  async done() {
    if (this.state.phase !== "listening") return;
    const attempt = this.attempt;
    this.stopClock();
    this.update({ phase: "finishing" });
    // A release that came before the hold timer: the commit must not wait on a configuration still pending.
    this.settleAutoStop(null);
    const mic = await this.mic;
    if (attempt !== this.attempt) return;
    if (mic) {
      await mic.flush();
      mic.stop();
    }
    if (attempt !== this.attempt) return;
    if (this.samples === 0) {
      this.session?.close();
      this.session = null;
      this.end();
      return this.say(NOTHING_HEARD, "ash", NOTICE_MS, { phase: "idle" });
    }
    // The server may have committed the item already (a silence): a second commit would find nothing.
    if (!this.serverCommitted) this.session?.commit();
  }

  /** The server committed the item after a silence: the dictation ends as Done does, the mic let go. */
  private serverCommit() {
    this.serverCommitted = true;
    if (this.state.phase !== "listening") return;
    clearTimeout(this.holdTimer);
    this.stopClock();
    this.update({ phase: "finishing" });
    this.capture?.stop();
    this.capture = null;
    this.mic = Promise.resolve(null);
  }

  /** Esc, or the strip's Discard: the field stays as it was. */
  discard() {
    if (this.state.phase === "idle") return;
    this.session?.discard();
    this.session = null;
    this.end();
    this.say(DISCARDED, "ash", DISCARDED_MS, { phase: "idle", stable: "", tail: "" });
  }

  /** A stopped dictation's "Insert what was heard": the stable text and the last tail. */
  insertHeard() {
    if (this.state.phase !== "failed") return;
    this.land(this.state.stable + this.state.tail);
  }

  keyDown(event: KeyLike) {
    if (event.key === "Escape" && !event.defaultPrevented && this.state.phase !== "idle") {
      event.preventDefault();
      return this.discard();
    }
    if (!(event.ctrlKey && event.shiftKey && event.code === "Space")) return;
    event.preventDefault();
    if (event.repeat) return;
    if (this.state.phase === "idle") {
      this.holdAt = this.now();
      this.toggle({ chord: true });
    } else if (this.state.phase === "listening") {
      void this.done();
    }
  }

  keyUp(event: KeyLike) {
    if (event.code !== "Space" || this.holdAt === 0) return;
    const held = this.now() - this.holdAt;
    this.holdAt = 0;
    if (held > HOLD_MS) void this.done();
    else this.settleAutoStop(silenceOf(this.deps));
  }

  /** A chord press told which it is: a tap follows the setting, a hold is manual. */
  private settleAutoStop(silenceMs: number | null) {
    clearTimeout(this.holdTimer);
    this.holdTimer = undefined;
    this.session?.autoStop(silenceMs);
  }

  /** The field is going away: whatever is under way stops, and the mic is let go. */
  dispose() {
    if (this.state.phase !== "idle") {
      this.session?.close();
      this.session = null;
      this.end();
      this.update({ phase: "idle", stable: "", tail: "" });
    }
    clearTimeout(this.noticeTimer);
    if (holder === this) holder = null;
  }

  /** Another field's mic was pressed, or a listening started: this dictation ends, and what it heard lands here. */
  giveWay() {
    if (this.state.phase === "listening") void this.done();
    else if (this.state.phase === "failed") this.insertHeard();
    else if (this.state.phase === "waiting") {
      this.end();
      this.update({ phase: "idle" });
    }
  }

  private start() {
    if (holder && holder !== this) holder.giveWay();
    holder = this;
    const attempt = ++this.attempt;
    this.samples = 0;
    this.serverCommitted = false;
    this.blocked = false;
    this.startedAt = this.now();
    clearTimeout(this.noticeTimer);
    this.update({ phase: "listening", stable: "", tail: "", levels: quiet(), elapsed: 0, failure: "", notice: null });
    this.stopClock();
    this.clock = setInterval(() => this.update({ elapsed: this.elapsed() }), 1_000);

    const live = () => attempt === this.attempt;
    const silence = silenceOf(this.deps);
    // A key press may turn out push-to-talk, which never stops by itself: until it is known, the session waits.
    const pending = this.chordPress && this.holdAt !== 0;
    clearTimeout(this.holdTimer);
    if (pending) this.holdTimer = setTimeout(() => this.settleAutoStop(null), HOLD_MS + 1);
    this.session = this.deps.open(this.target.context(), {
      onCommitted: () => live() && this.serverCommit(),
      onDelta: (text) => live() && this.update({ stable: this.state.stable + text }),
      onTail: (text) => live() && this.update({ tail: text }),
      onCompleted: (transcript) => live() && this.land(transcript, "completed"),
      onFailed: (message) => live() && this.fail(stopped(message)),
      onClosed: () => live() && this.fail(CONNECTION_LOST),
      onRefused: (refusal) => live() && this.refused(refusal),
    }, pending ? "pending" : silence);
    this.mic = this.deps
      .capture((pcm) => {
        // After the server's commit the item is closed: more audio would open a next one.
        if (!live() || this.serverCommitted) return;
        this.samples += pcm.length;
        this.session?.append(pcm);
        this.update({ levels: [...this.state.levels.slice(1), levelOf(pcm)], elapsed: this.elapsed() });
      })
      .then((capture) => {
        if (live() && !this.serverCommitted) return (this.capture = capture);
        capture.stop();
        return null;
      })
      .catch((err: unknown) => {
        if (!live()) return null;
        this.session?.close();
        this.session = null;
        this.end();
        if (err instanceof MicBlocked) {
          // Struck from now on; said at once too, for a Decide field, whose line shows no idle reason.
          this.blocked = true;
          this.say(MIC_BLOCKED, "ash", NOTICE_MS, { phase: "idle" });
        } else {
          this.say(cannotStart(err instanceof Error ? err.message : String(err)), "fault", NOTICE_MS, { phase: "idle" });
        }
        return null;
      });
  }

  private refused(refusal: Refusal) {
    this.session = null;
    this.end();
    switch (refusal.kind) {
      case "busy":
        if (this.retried) return this.say(STILL_BUSY, "warn", NOTICE_MS, { phase: "idle" });
        this.retried = true;
        return this.countDown(Math.min(BUSY_RETRY_MAX_SECONDS, Math.ceil(refusal.retryAfter ?? BUSY_RETRY_SECONDS)));
      case "unavailable":
        this.refusedReason = refusal.reason;
        this.update({ phase: "idle" });
        return this.deps.onUnavailable?.();
      case "key":
        this.update({ phase: "idle" });
        return this.deps.onKeyRequired?.();
      case "unreachable":
        return this.say(cannotStart(refusal.message), "fault", NOTICE_MS, { phase: "idle" });
      case "no_websocket":
        return this.say(NO_WEBSOCKET, "fault", NOTICE_MS, { phase: "idle" });
    }
  }

  /** The busy wait: one second at a time, then the retry. */
  private countDown(seconds: number) {
    holder = this;
    const attempt = this.attempt;
    this.update({ phase: "waiting", retryIn: seconds });
    this.clock = setInterval(() => {
      if (attempt !== this.attempt) return this.stopClock();
      const left = this.state.retryIn - 1;
      if (left > 0) return this.update({ retryIn: left });
      this.stopClock();
      this.start();
    }, 1_000);
  }

  private fail(sentence: string) {
    this.session?.close();
    this.session = null;
    this.end();
    // Still the page's dictation until its text is inserted or discarded.
    holder = this;
    this.update({ phase: "failed", failure: sentence });
  }

  /** Puts `text` at the field's cursor, selected, and ends the dictation. */
  private land(text: string, how: "completed" | "heard" = "heard") {
    if (how === "completed") this.session?.close();
    this.session = null;
    this.end();
    const words = text.trim();
    if (words === "") return this.say(NOTHING_HEARD, "ash", NOTICE_MS, { phase: "idle", stable: "", tail: "" });
    const { value, at } = this.target.read();
    const placed = insertAt(value, at, words);
    this.target.write(placed.value);
    this.notice({ kind: "inserted", words: wordCount(words) }, INSERTED_MS, {
      phase: "idle",
      stable: "",
      tail: "",
      selection: { start: placed.start, end: placed.end, seq: ++this.seq },
    });
  }

  /** Stops the mic and the clocks, and lets later answers from this attempt go unheard. */
  private end() {
    this.attempt++;
    clearTimeout(this.holdTimer);
    this.stopClock();
    this.capture?.stop();
    this.capture = null;
    this.mic = Promise.resolve(null);
    this.levelsQuiet();
    if (holder === this) holder = null;
  }

  private levelsQuiet() {
    this.state = { ...this.state, levels: quiet() };
  }

  private elapsed() {
    return Math.floor((this.now() - this.startedAt) / 1_000);
  }

  private stopClock() {
    clearInterval(this.clock);
    this.clock = undefined;
  }

  private say(text: string, tone: Tone, ms: number, also: Partial<DictationState> = {}) {
    this.notice({ kind: "said", text, tone }, ms, also);
  }

  private notice(notice: DictationState["notice"], ms: number, also: Partial<DictationState>) {
    clearTimeout(this.noticeTimer);
    this.update({ ...also, notice });
    this.noticeTimer = setTimeout(() => this.update({ notice: null }), ms);
  }

  private update(patch: Partial<DictationState>) {
    const next = { ...this.state, ...patch };
    const cause = unavailable({
      secure: this.secure,
      speech: this.refusedReason !== null ? { available: false, reason: this.refusedReason } : this.speech,
      blocked: this.blocked,
      listening: this.listening,
    });
    const same = cause?.sentence === next.unavailable?.sentence && cause?.blocked === next.unavailable?.blocked;
    this.state = { ...next, unavailable: same ? next.unavailable : cause };
    const was = underWay.size > 0;
    if (next.phase === "idle") underWay.delete(this);
    else underWay.add(this);
    if (was !== underWay.size > 0) for (const listener of pageListeners) listener();
    for (const listener of this.listeners) listener();
  }
}

/**
 * A listening in the Transcribe view starts (spec speech/07, Q85): the dictation under way ends as it does when
 * another mic starts, its text landing in its own field.
 */
export function endDictation() {
  holder?.giveWay();
}

/** What the mic, the strip and the hint line read and press: the controller, or a fixed one in a test. */
export type DictationHandle = Pick<DictationController, "getState" | "subscribe" | "toggle" | "done" | "discard" | "insertHeard" | "keyDown" | "keyUp" | "active">;

/**
 * The hint line's sentence for what a dictation is doing or has just done, or `null` for the field's own words.
 * `action` is what sends the field: Send in the chat, Generate in the image studio, Decide in Decide. An
 * unavailable mic's reason is the field's to place: a composer's line says it in place of its idle words, and a
 * Decide field says it when its mic is pressed, as a notice.
 */
export function dictationHint(state: DictationState, action: "Send" | "Generate" | "Decide"): { text: string; tone: Tone } | null {
  switch (state.phase) {
    case "listening":
      return { text: action === "Generate" ? `${LISTENING} Generate waits until you press it.` : LISTENING, tone: "ash" };
    case "finishing":
      return { text: FINISHING, tone: "ash" };
    case "waiting":
      return { text: busy(state.retryIn), tone: "warn" };
    case "failed":
      return { text: state.failure, tone: "fault" };
  }
  const { notice } = state;
  if (notice?.kind === "inserted") {
    const words = notice.words === 1 ? "1 word" : `${notice.words} words`;
    return { text: `Inserted ${words} at the cursor. Nothing is sent until you press ${action}.`, tone: "ash" };
  }
  if (notice) return { text: notice.text, tone: notice.tone };
  return null;
}
