import { authHeaders, getAuth } from "../api/auth.ts";
import { apiErrorMessage } from "../api/errors.ts";
import type { SocketLike } from "../api/socket.ts";
import { pcmBase64, TARGET_RATE } from "./pcm.ts";

// One dictation's session on GET /v1/realtime?intent=transcription (spec
// speech/04; the wire is spec speech/03): opened at the press, configured,
// fed the mic's audio, committed by Done or cleared by Esc, and closed.
//
// - **Configured** once the server has greeted it: PCM16 at 16 kHz, the
//   context as the prompt, the provisional tail (Q21, Q43), and the turn
//   detection: the server's `server_vad` with the viewer's silence, which ends
//   the item by itself (Q80), or manual commit when that is Off (Q33). No
//   language: the model detects it (Q51). Audio that arrives before the
//   server has answered the configuration waits here, so none of it is read
//   at the default 24 kHz.
// - **A chord press** does not know yet whether it is a tap or push-to-talk,
//   and a switch between `null` and `server_vad` while an item holds audio
//   holds only from the next item (spec speech/05). So the configuration waits
//   for `autoStop()`, the audio with it, and a held key never auto-stops.
// - **A refused upgrade** is a close before the open, whatever the server
//   answered: a page cannot read a failed upgrade's status or `Retry-After`.
//   A plain GET of the same route is answered what the upgrade would have been
//   (spec speech/03), and the page reads that: a 401 wants the key; `503
//   speech_unavailable` is `unavailable` with its reason; `503 engine_full` is
//   `busy`, with the answer's `Retry-After` (a same-origin `fetch` can read
//   it); `400 upgrade_required` says the server would have opened it, so the
//   upgrade was lost on the way (a proxy that does not carry WebSockets).
// - The socket, its URL and the probe's `fetch` are injectable, so all of it
//   runs in a test without a server.

export const REALTIME_PATH = "/v1/realtime?intent=transcription";

/** Why the session never opened. */
export type Refusal =
  | { kind: "busy"; /** The answer's `Retry-After`, in seconds, when it gave one. */ retryAfter?: number }
  | { kind: "unavailable"; reason: string }
  | { kind: "key"; message: string }
  | { kind: "unreachable"; message: string }
  /** The server would have opened it: the upgrade did not get through. */
  | { kind: "no_websocket" };

/**
 * The silence that ends the item by itself, in ms; `null` for manual commit (Done); `"pending"` while the
 * dictation does not know yet (a chord press that may be push-to-talk): the configuration waits for `autoStop()`.
 */
export type AutoStop = number | null | "pending";

/** What the session tells the dictation. Nothing is called once the page has closed it. */
export type TranscriptionHandlers = {
  /** Stable text, appended to what came before (spec speech/03: deltas only ever append). */
  onDelta: (text: string) => void;
  /** The whole provisional tail, which replaces the last; `""` once the item ends. */
  onTail: (text: string) => void;
  /** The server committed the item itself, after a silence (`server_vad`): the first commit ends the dictation. */
  onCommitted: () => void;
  /** The committed item's transcript: the deltas joined. */
  onCompleted: (transcript: string) => void;
  /** The item failed, or the server refused an event: its message. */
  onFailed: (message: string) => void;
  onRefused: (refusal: Refusal) => void;
  /** The connection closed under an open session before its transcript came. */
  onClosed: () => void;
};

export type Transcription = {
  append: (pcm: Int16Array) => void;
  /** Settles a `"pending"` auto-stop: the silence in ms, or `null` for manual commit. */
  autoStop: (silenceMs: number | null) => void;
  /** Ends the item: its transcript follows. */
  commit: () => void;
  /** Esc: clears the server's buffer and closes the session. */
  discard: () => void;
  close: () => void;
};

export type RealtimeDeps = {
  /** Opens a socket; may throw, as `new WebSocket` does for a key that is not a valid subprotocol. */
  open?: (url: string, protocols: string[]) => SocketLike;
  url?: () => string;
  /** For the probe after a refused upgrade. */
  fetch?: typeof fetch;
};

/** The route on the page's own origin, `wss` when the page is served over TLS. */
function pageUrl(): string {
  const { protocol, host } = globalThis.location;
  return `${protocol === "https:" ? "wss" : "ws"}://${host}${REALTIME_PATH}`;
}

/** `realtime`, which the server selects back, and the credential beside it: a page cannot set headers on a socket. */
function protocols(): string[] {
  const key = getAuth().key;
  return key ? ["realtime", `openai-insecure-api-key.${key}`] : ["realtime"];
}

/** The session's configuration (spec speech/04, AC 6-8). */
function configuration(context: string, silenceMs: number | null) {
  return {
    type: "session.update",
    session: {
      type: "transcription",
      audio: {
        input: {
          format: { type: "audio/pcm", rate: TARGET_RATE },
          transcription: context === "" ? {} : { prompt: context },
          // The server's defaults for the rest (spec speech/05): threshold, prefix padding.
          turn_detection: silenceMs === null ? null : { type: "server_vad", silence_duration_ms: silenceMs },
        },
      },
      // The length guard off (spec speech/08): the dictation's first server commit ends it (Q80),
      // so a long one with short pauses must not be cut; `speech.max_clip_seconds` still bounds it.
      ignis: { provisional_tail: true, max_utterance_seconds: 0 },
    },
  };
}

/** The error a commit gets when the buffer holds nothing: spec speech/03. */
const EMPTY_COMMIT = "input_audio_buffer_commit_empty";

/** The reason in `the speech model is unavailable: <reason>`, the server's sentence. */
const UNAVAILABLE_PREFIX = "the speech model is unavailable: ";

/** Why the upgrade was refused, asked of a plain GET of the route. */
async function probe(doFetch: typeof fetch): Promise<Refusal> {
  try {
    const res = await doFetch(REALTIME_PATH, { headers: authHeaders() });
    const text = await res.text();
    if (res.status === 401) return { kind: "key", message: apiErrorMessage(401, text) };
    let error: { code?: string; message?: string } = {};
    try {
      error = (JSON.parse(text) as { error?: typeof error }).error ?? {};
    } catch {
      // Not JSON: no code to read.
    }
    if (res.status === 503 && error.code === "engine_full") {
      const seconds = Number(res.headers.get("Retry-After"));
      return Number.isFinite(seconds) && seconds > 0 ? { kind: "busy", retryAfter: seconds } : { kind: "busy" };
    }
    if (res.status === 503 && error.code === "speech_unavailable") {
      const message = error.message ?? "no reason given";
      return { kind: "unavailable", reason: message.startsWith(UNAVAILABLE_PREFIX) ? message.slice(UNAVAILABLE_PREFIX.length) : message };
    }
    if (res.status === 400 && error.code === "upgrade_required") return { kind: "no_websocket" };
    // A server older than speech has no such route.
    if (res.status === 404) return { kind: "unavailable", reason: "this server has no speech model" };
    return { kind: "unreachable", message: apiErrorMessage(res.status, text) };
  } catch (err) {
    return { kind: "unreachable", message: `Could not reach ignis: ${String(err)}` };
  }
}

type ServerEvent = {
  type?: string;
  item_id?: string;
  delta?: string;
  text?: string;
  transcript?: string;
  audio_start_ms?: number;
  audio_end_ms?: number;
  speaker?: number;
  language?: string;
  error?: { message?: string; code?: string; event_id?: string | null; param?: string | null };
};

export function openTranscription(
  context: string,
  handlers: TranscriptionHandlers,
  deps: RealtimeDeps = {},
  autoStop: AutoStop = null,
): Transcription {
  const open = deps.open ?? ((url, offered) => new WebSocket(url, offered) as unknown as SocketLike);
  const doFetch = deps.fetch ?? ((input, init) => fetch(input, init));
  /** The page closed it: nothing more is said, a refusal still being asked about included. */
  let closedByPage = false;
  /** The connection is over, by the page or under it. */
  let ended = false;
  let opened = false;
  let configured = false;
  let completed = false;
  let commitAsked = false;
  let greeted = false;
  /** The item the first commit made: a later item's events are not this dictation's. */
  let item: string | undefined;
  const waiting: string[] = [];
  let socket: SocketLike | null = null;

  const send = (event: object) => socket?.send(JSON.stringify(event));
  const sendAppend = (audio: string) => send({ type: "input_audio_buffer.append", audio });

  function onMessage(event: ServerEvent) {
    if (item !== undefined && event.item_id !== undefined && event.item_id !== item) return;
    switch (event.type) {
      case "session.created":
        greeted = true;
        if (autoStop !== "pending") send(configuration(context, autoStop));
        return;
      case "input_audio_buffer.committed":
        if (item !== undefined) return;
        item = event.item_id;
        handlers.onCommitted();
        return;
      case "session.updated":
        if (configured) return;
        configured = true;
        for (const audio of waiting.splice(0)) sendAppend(audio);
        if (commitAsked) send({ type: "input_audio_buffer.commit" });
        return;
      case "conversation.item.input_audio_transcription.delta":
        handlers.onDelta(event.delta ?? "");
        return;
      case "ignis.input_audio_transcription.provisional":
        handlers.onTail(event.text ?? "");
        return;
      case "conversation.item.input_audio_transcription.completed":
        completed = true;
        handlers.onCompleted(event.transcript ?? "");
        return;
      case "error":
        // A commit of ours that crossed the server's own: the item is committed, the empty buffer is no fault.
        if (item !== undefined && event.error?.code === EMPTY_COMMIT) return;
        handlers.onFailed(event.error?.message ?? "the server refused the dictation");
        return;
      case "conversation.item.input_audio_transcription.failed":
        handlers.onFailed(event.error?.message ?? "the server refused the dictation");
        return;
    }
  }

  try {
    socket = open(deps.url?.() ?? pageUrl(), protocols());
  } catch {
    // A key that is not a valid subprotocol token throws: a refused upgrade like any other.
    ended = true;
    void probe(doFetch).then((refusal) => closedByPage || handlers.onRefused(refusal));
  }
  if (socket) {
    socket.onopen = () => {
      opened = true;
    };
    socket.onerror = () => {};
    socket.onmessage = (message) => {
      if (ended) return;
      try {
        onMessage(JSON.parse(String(message.data)) as ServerEvent);
      } catch {
        // Not JSON: nothing this session reads.
      }
    };
    socket.onclose = () => {
      if (ended || completed) return;
      ended = true;
      if (opened) handlers.onClosed();
      else void probe(doFetch).then((refusal) => closedByPage || handlers.onRefused(refusal));
    };
  }

  const close = () => {
    closedByPage = true;
    ended = true;
    socket?.close();
  };

  return {
    append(pcm) {
      if (ended) return;
      const audio = pcmBase64(pcm);
      if (configured) sendAppend(audio);
      else waiting.push(audio);
    },
    commit() {
      if (ended) return;
      commitAsked = true;
      if (configured) send({ type: "input_audio_buffer.commit" });
    },
    autoStop(silenceMs) {
      if (autoStop !== "pending" || ended) return;
      autoStop = silenceMs;
      if (greeted) send(configuration(context, silenceMs));
    },
    discard() {
      if (!ended && configured) send({ type: "input_audio_buffer.clear" });
      close();
    },
    close,
  };
}

/** The speakers' geometry a listening asks for (spec speech/08): `"live"` for audio at real time, `"file"` for a file. */
export type Diarization = "live" | "file";

/**
 * What a listening's transcription is told (spec speech/07, Q128, Q130): the context as its `prompt`, the terms as
 * its `keywords`, and a `language` to force; `""`, none and `null` send nothing.
 */
export type Told = { prompt: string; keywords: string[]; language: string | null };

export const NOTHING_TOLD: Told = { prompt: "", keywords: [], language: null };

/** The configuration's `transcription`: only what is told. */
const transcriptionOf = ({ prompt, keywords, language }: Told) => ({
  ...(prompt === "" ? {} : { prompt }),
  ...(keywords.length === 0 ? {} : { keywords }),
  ...(language === null ? {} : { language }),
});

/** An update's `transcription`: every field, `null` putting back what is no longer told. */
const transcriptionUpdate = ({ prompt, keywords, language }: Told) => ({
  prompt: prompt === "" ? null : prompt,
  keywords: keywords.length === 0 ? null : keywords,
  language,
});

const sameTold = (a: Told, b: Told) => a.prompt === b.prompt && a.language === b.language && a.keywords.join("\n") === b.keywords.join("\n");

export type ListeningOptions = {
  context: Told;
  /** `null`: no speakers (Speakers off, or a call's microphone, which is "Me"). */
  diarization: Diarization | null;
  /** The silence that ends an utterance, in ms (Q122); absent or `null`, the server's `speech.vad_silence_duration_ms`. */
  silenceMs?: number | null;
};

/** What a listening's session hears, each item by its id: items overlap, an item's `completed` coming after the next one starts. */
export type ListeningEvent =
  | { type: "started"; item: string; atMs: number }
  | { type: "stopped"; item: string; atMs: number }
  | { type: "committed"; item: string }
  | { type: "delta"; item: string; text: string }
  | { type: "tail"; item: string; text: string }
  | { type: "speaker"; item: string; speaker: number }
  /** The language the model heard the item in, by ISO code (`session.ignis.item_language`, Q131). */
  | { type: "language"; item: string; language: string }
  | { type: "completed"; item: string; transcript: string }
  | { type: "item_failed"; item: string; message: string }
  /** The server refused an event of ours; `eventId` is the one the page gave it. The session goes on. */
  | { type: "error"; code: string | null; message: string; eventId: string | null }
  | { type: "refused"; refusal: Refusal }
  /** The server has read every event sent before the page's last `flush()`, its appends included. */
  | { type: "flushed" }
  /** The connection closed under the session. */
  | { type: "closed" };

export type Listening = {
  /** `eventId` comes back on the `error` that refuses this append (spec speech/08's queue limit). */
  append: (pcm: Int16Array, eventId?: string) => void;
  /** What the transcription is told, from the next utterance on (Q87, Q128). */
  setContext: (context: Told) => void;
  /** The silence that ends an utterance, at once (`server_vad`'s fields hold at once: spec speech/05). */
  setSilence: (silenceMs: number) => void;
  /** Ends the utterance under way: its transcript follows. */
  commit: () => void;
  /**
   * Asks for a `flushed` once the server has read everything sent before: an empty `session.update`, answered in
   * order. Before it, the VAD's events for the last audio sent may still be on their way.
   */
  flush: () => void;
  close: () => void;
};

/** The event id of a listening's configuration: the `error` that refuses it names it. */
export const CONFIGURATION_EVENT = "listening_configuration";

/** The event id of a flush's empty update. */
export const FLUSH_EVENT = "listening_flush";

/** The code of an update refused for the speakers it asks for: the session goes on without them (spec speech/08). */
export const DIARIZER_UNAVAILABLE = "ignis_diarizer_unavailable";

/** A diarizer that failed mid-session: the session goes on without speakers (spec speech/08). */
export const DIARIZER_FAILED = "ignis_diarizer_failed";

/** An append past the audio a session may hold not computed yet (spec speech/08): it is refused, not read. */
export const QUEUE_FULL = "ignis_audio_queue_full";

/** The `param` a server older than the items' language refuses the configuration with: asked again without it. */
const ITEM_LANGUAGE = "session.ignis.item_language";

/** `server_vad` with the view's silence, the server's threshold and padding. */
const vad = (silenceMs: number | null | undefined) => (silenceMs == null ? { type: "server_vad" } : { type: "server_vad", silence_duration_ms: silenceMs });

/** A listening's configuration (spec speech/07, "The session"); `languages`, each item's language asked for (Q131). */
function listeningConfiguration({ context, diarization, silenceMs }: ListeningOptions, languages: boolean) {
  return {
    type: "session.update",
    event_id: CONFIGURATION_EVENT,
    session: {
      type: "transcription",
      audio: {
        input: {
          format: { type: "audio/pcm", rate: TARGET_RATE },
          transcription: transcriptionOf(context),
          // The server's `speech.vad_*` fields, but for the silence the view sets (Q122), never the dictation's (Q80).
          turn_detection: vad(silenceMs),
        },
      },
      // No `max_utterance_seconds`: the server's length guard holds (spec speech/08).
      ignis: { provisional_tail: true, ...(diarization === null ? {} : { diarization }), ...(languages ? { item_language: true } : {}) },
    },
  };
}

/**
 * One listening's session on the Realtime route (spec speech/07): a sibling of
 * a dictation's, not one. It asks for the server's turn detection at the
 * server's lengths and hears every commit as an utterance of its own, never
 * ending at the first; the speakers come when the options ask for them. The
 * audio waits, as a dictation's does, until the server has taken the
 * configuration. A refused upgrade is asked about as a dictation's is.
 */
export function openListening(asked: ListeningOptions, onEvent: (event: ListeningEvent) => void, deps: RealtimeDeps = {}): Listening {
  let options = asked;
  let silenceMs = asked.silenceMs ?? null;
  const open = deps.open ?? ((url, offered) => new WebSocket(url, offered) as unknown as SocketLike);
  const doFetch = deps.fetch ?? ((input, init) => fetch(input, init));
  let context = options.context;
  /** Each item's language is asked for, until a server older than it refuses the key. */
  let languages = true;
  /** The page closed it: nothing more is said, a refusal still being asked about included. */
  let closedByPage = false;
  /** The connection is over, by the page or under it. */
  let ended = false;
  let opened = false;
  let greeted = false;
  let configured = false;
  const waiting: object[] = [];
  /** The page's updates after the configuration, in the order sent: each is answered by one `session.updated`. */
  const updates: ("setting" | "flush")[] = [];
  let socket: SocketLike | null = null;
  const send = (event: object) => socket?.send(JSON.stringify(event));
  const say = (event: ListeningEvent) => closedByPage || onEvent(event);

  function onMessage(event: ServerEvent) {
    const item = event.item_id ?? "";
    switch (event.type) {
      case "session.created":
        greeted = true;
        send(listeningConfiguration({ ...options, context, silenceMs }, languages));
        return;
      case "session.updated":
        if (configured) {
          if (updates.shift() === "flush") say({ type: "flushed" });
          return;
        }
        configured = true;
        for (const queued of waiting.splice(0)) send(queued);
        return;
      case "input_audio_buffer.speech_started":
        return say({ type: "started", item, atMs: event.audio_start_ms ?? 0 });
      case "input_audio_buffer.speech_stopped":
        return say({ type: "stopped", item, atMs: event.audio_end_ms ?? 0 });
      case "input_audio_buffer.committed":
        return say({ type: "committed", item });
      case "conversation.item.input_audio_transcription.delta":
        return say({ type: "delta", item, text: event.delta ?? "" });
      case "ignis.input_audio_transcription.provisional":
        return say({ type: "tail", item, text: event.text ?? "" });
      case "ignis.input_audio_transcription.speaker":
        if (typeof event.speaker === "number") say({ type: "speaker", item, speaker: event.speaker });
        return;
      case "ignis.input_audio_transcription.language":
        if (typeof event.language === "string" && event.language !== "") say({ type: "language", item, language: event.language });
        return;
      case "conversation.item.input_audio_transcription.completed":
        return say({ type: "completed", item, transcript: event.transcript ?? "" });
      case "conversation.item.input_audio_transcription.failed":
        return say({ type: "item_failed", item, message: event.error?.message ?? "the server could not transcribe it" });
      case "error":
        // A server older than the items' language refuses the key, and the whole update with it: configured again
        // without it, in silence, the session goes on with no language suggested.
        if (!configured && event.error?.event_id === CONFIGURATION_EVENT && event.error.param === ITEM_LANGUAGE && languages) {
          languages = false;
          send(listeningConfiguration({ ...options, context, silenceMs }, languages));
          return;
        }
        // The diarizer refuses the whole update: configured again without speakers, the session goes on.
        if (!configured && event.error?.event_id === CONFIGURATION_EVENT && event.error.code === DIARIZER_UNAVAILABLE && options.diarization !== null) {
          options = { ...options, diarization: null };
          send(listeningConfiguration({ ...options, context, silenceMs }, languages));
        }
        return say({
          type: "error",
          code: event.error?.code ?? null,
          message: event.error?.message ?? "the server refused an event",
          eventId: event.error?.event_id ?? null,
        });
    }
  }

  /** Sent once the configuration is taken; held until then, in order. */
  const sendConfigured = (event: object) => (configured ? send(event) : waiting.push(event));
  const refused = () => void probe(doFetch).then((refusal) => say({ type: "refused", refusal }));

  try {
    socket = open(deps.url?.() ?? pageUrl(), protocols());
  } catch {
    // A key that is not a valid subprotocol token throws: a refused upgrade like any other.
    ended = true;
    refused();
  }
  if (socket) {
    socket.onopen = () => {
      opened = true;
    };
    socket.onerror = () => {};
    socket.onmessage = (message) => {
      if (ended) return;
      try {
        onMessage(JSON.parse(String(message.data)) as ServerEvent);
      } catch {
        // Not JSON: nothing this session reads.
      }
    };
    socket.onclose = () => {
      if (ended) return;
      ended = true;
      if (opened) say({ type: "closed" });
      else refused();
    };
  }

  return {
    append(pcm, eventId) {
      if (ended) return;
      sendConfigured({ type: "input_audio_buffer.append", audio: pcmBase64(pcm), ...(eventId === undefined ? {} : { event_id: eventId }) });
    },
    setContext(next) {
      if (ended || sameTold(next, context)) return;
      context = next;
      // Not greeted yet: the configuration carries it.
      if (!greeted) return;
      updates.push("setting");
      sendConfigured({ type: "session.update", session: { type: "transcription", audio: { input: { transcription: transcriptionUpdate(next) } } } });
    },
    setSilence(next) {
      if (ended || next === silenceMs) return;
      silenceMs = next;
      // Not greeted yet: the configuration carries it.
      if (!greeted) return;
      updates.push("setting");
      sendConfigured({ type: "session.update", session: { type: "transcription", audio: { input: { turn_detection: vad(next) } } } });
    },
    flush() {
      if (ended) return;
      updates.push("flush");
      sendConfigured({ type: "session.update", event_id: FLUSH_EVENT, session: { type: "transcription" } });
    },
    commit() {
      if (!ended) sendConfigured({ type: "input_audio_buffer.commit" });
    },
    close() {
      closedByPage = true;
      ended = true;
      socket?.close();
    },
  };
}
