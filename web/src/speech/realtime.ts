import { authHeaders, getAuth } from "../api/auth.ts";
import { apiErrorMessage } from "../api/errors.ts";
import type { SocketLike } from "../api/socket.ts";
import { pcmBase64, TARGET_RATE } from "./pcm.ts";

// One dictation's session on GET /v1/realtime?intent=transcription (spec
// speech/04; the wire is spec speech/03): opened at the press, configured,
// fed the mic's audio, committed by Done or cleared by Esc, and closed.
//
// - **Configured** once the server has greeted it: manual commit, PCM16 at
//   16 kHz, the context as the prompt, the provisional tail (Q21, Q33, Q43).
//   No language: the model detects it (Q51). Audio that arrives before the
//   server has answered the configuration waits here, so none of it is read
//   at the default 24 kHz.
// - **A refused upgrade** is a close before the open, whatever the server
//   answered: a page cannot read a failed upgrade's status or `Retry-After`.
//   A plain GET of the same route is answered what the upgrade would have been
//   (spec speech/03), and the page reads that: a 401 wants the key; `503
//   speech_unavailable` is `unavailable` with its reason; `503 engine_full` is
//   `busy`; `400 upgrade_required` says the server would have opened it, so the
//   upgrade was lost on the way (a proxy that does not carry WebSockets).
// - The socket, its URL and the probe's `fetch` are injectable, so all of it
//   runs in a test without a server.

export const REALTIME_PATH = "/v1/realtime?intent=transcription";

/** Why the session never opened. */
export type Refusal =
  | { kind: "busy" }
  | { kind: "unavailable"; reason: string }
  | { kind: "key"; message: string }
  | { kind: "unreachable"; message: string }
  /** The server would have opened it: the upgrade did not get through. */
  | { kind: "no_websocket" };

/** What the session tells the dictation. Nothing is called once the page has closed it. */
export type TranscriptionHandlers = {
  /** Stable text, appended to what came before (spec speech/03: deltas only ever append). */
  onDelta: (text: string) => void;
  /** The whole provisional tail, which replaces the last; `""` once the item ends. */
  onTail: (text: string) => void;
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
function configuration(context: string) {
  return {
    type: "session.update",
    session: {
      type: "transcription",
      audio: {
        input: {
          format: { type: "audio/pcm", rate: TARGET_RATE },
          transcription: context === "" ? {} : { prompt: context },
          turn_detection: null,
        },
      },
      ignis: { provisional_tail: true },
    },
  };
}

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
    if (res.status === 503 && error.code === "engine_full") return { kind: "busy" };
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
  delta?: string;
  text?: string;
  transcript?: string;
  error?: { message?: string };
};

export function openTranscription(context: string, handlers: TranscriptionHandlers, deps: RealtimeDeps = {}): Transcription {
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
  const waiting: string[] = [];
  let socket: SocketLike | null = null;

  const send = (event: object) => socket?.send(JSON.stringify(event));
  const sendAppend = (audio: string) => send({ type: "input_audio_buffer.append", audio });

  function onMessage(event: ServerEvent) {
    switch (event.type) {
      case "session.created":
        send(configuration(context));
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
      case "conversation.item.input_audio_transcription.failed":
      case "error":
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
    discard() {
      if (!ended && configured) send({ type: "input_audio_buffer.clear" });
      close();
    },
    close,
  };
}
