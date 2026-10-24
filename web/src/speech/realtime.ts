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
//   GET /v1/models tells the causes apart, as the Responses socket's probe
//   does (api/socket.ts): a 401 wants the key; a speech model that cannot
//   serve is `unavailable` with its reason; one that can was refused for
//   every session being taken (`503 engine_full`): `busy`.
// - The socket, its URL and the probe's `fetch` are injectable, so all of it
//   runs in a test without a server.

export const REALTIME_PATH = "/v1/realtime?intent=transcription";

/** Why the session never opened. */
export type Refusal =
  | { kind: "busy" }
  | { kind: "unavailable"; reason: string }
  | { kind: "key"; message: string }
  | { kind: "unreachable"; message: string };

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

/** Why the upgrade was refused, asked of GET /v1/models. */
async function probe(doFetch: typeof fetch): Promise<Refusal> {
  try {
    const res = await doFetch("/v1/models", { headers: authHeaders() });
    if (res.status === 401) return { kind: "key", message: apiErrorMessage(401, await res.text()) };
    if (!res.ok) return { kind: "unreachable", message: apiErrorMessage(res.status, await res.text()) };
    const body = (await res.json()) as { speech?: { available?: boolean; reason?: string } };
    if (!body.speech) return { kind: "unavailable", reason: "this server has no speech model" };
    if (body.speech.available !== true) return { kind: "unavailable", reason: body.speech.reason ?? "no reason given" };
    return { kind: "busy" };
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
