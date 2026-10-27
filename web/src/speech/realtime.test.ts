import { beforeEach, describe, expect, it } from "vitest";
import { forgetKey, saveKey } from "../api/auth.ts";
import type { SocketLike } from "../api/socket.ts";
import { type ListeningEvent, openListening, openTranscription, type Refusal, type Told, type TranscriptionHandlers } from "./realtime.ts";

// One dictation's Realtime transcription session (spec speech/04, spec
// speech/03 for the wire) through its public contract: a fake WebSocket plays
// the server's side with the event shapes #353 sends.

/** A socket the test accepts, refuses, drops and speaks through. */
class FakeSocket implements SocketLike {
  sent: Record<string, unknown>[] = [];
  closedByPage = false;
  onopen: SocketLike["onopen"] = null;
  onmessage: SocketLike["onmessage"] = null;
  onclose: SocketLike["onclose"] = null;
  onerror: SocketLike["onerror"] = null;
  constructor(
    readonly url: string,
    readonly protocols: string[],
  ) {}
  send(data: string) {
    this.sent.push(JSON.parse(data) as Record<string, unknown>);
  }
  close() {
    this.closedByPage = true;
    this.onclose?.({ code: 1000 });
  }
  /** The upgrade succeeds and the server greets the session. */
  accept() {
    this.onopen?.({});
    this.emit({ type: "session.created", event_id: "event_1", session: { type: "transcription", audio: { input: { format: { type: "audio/pcm", rate: 24000 }, turn_detection: null } }, ignis: { provisional_tail: false } } });
  }
  /** What the browser shows of a refused upgrade, whatever its status: a close before the open, 1006. */
  refuse() {
    this.onerror?.({});
    this.onclose?.({ code: 1006 });
  }
  drop() {
    this.onclose?.({ code: 1006 });
  }
  emit(event: Record<string, unknown>) {
    this.onmessage?.({ data: JSON.stringify(event) });
  }
  types() {
    return this.sent.map((e) => e.type);
  }
}

/** Every handler call, in order, as `[name, argument]`. */
function recorder() {
  const calls: [string, unknown][] = [];
  const handlers: TranscriptionHandlers = {
    onDelta: (text) => calls.push(["delta", text]),
    onTail: (text) => calls.push(["tail", text]),
    onCommitted: () => calls.push(["committed", null]),
    onCompleted: (text) => calls.push(["completed", text]),
    onFailed: (message) => calls.push(["failed", message]),
    onRefused: (refusal) => calls.push(["refused", refusal]),
    onClosed: () => calls.push(["closed", null]),
  };
  return { calls, handlers };
}

const answering = (status: number, body: unknown) => (async () => new Response(JSON.stringify(body), { status })) as typeof fetch;

const upgradeRequired = answering(400, { error: { code: "upgrade_required", message: "GET /v1/realtime takes a WebSocket upgrade" } });

function setup(fetcher: typeof fetch = upgradeRequired) {
  const sockets: FakeSocket[] = [];
  const { calls, handlers } = recorder();
  const deps = {
    open: (url: string, protocols: string[]) => {
      const socket = new FakeSocket(url, protocols);
      sockets.push(socket);
      return socket;
    },
    url: () => "ws://ignis.test/v1/realtime?intent=transcription",
    fetch: fetcher,
  };
  return { sockets, calls, handlers, deps };
}

/** Lets the probe after a refused upgrade answer. */
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

beforeEach(() => forgetKey());

describe("openTranscription", () => {
  it("opens /v1/realtime?intent=transcription with the key as a subprotocol beside `realtime`", () => {
    saveKey("sk-test");
    const { sockets, handlers, deps } = setup();
    openTranscription("", handlers, deps);
    expect(sockets[0].url).toBe("ws://ignis.test/v1/realtime?intent=transcription");
    expect(sockets[0].protocols).toEqual(["realtime", "openai-insecure-api-key.sk-test"]);
    forgetKey();
    openTranscription("", handlers, deps);
    expect(sockets[1].protocols).toEqual(["realtime"]);
  });

  it("configures the session as spec 04 says: manual commit when the auto-stop is Off, PCM16 at 16 kHz, the context, the provisional tail, no language, the length guard off (spec 08)", () => {
    const { sockets, handlers, deps } = setup();
    openTranscription("The last reply.\n\nThe composer's text.", handlers, deps, null);
    sockets[0].accept();
    expect(sockets[0].sent).toEqual([
      {
        type: "session.update",
        session: {
          type: "transcription",
          audio: {
            input: {
              format: { type: "audio/pcm", rate: 16000 },
              transcription: { prompt: "The last reply.\n\nThe composer's text." },
              turn_detection: null,
            },
          },
          ignis: { provisional_tail: true, max_utterance_seconds: 0 },
        },
      },
    ]);
    expect(JSON.stringify(sockets[0].sent)).not.toContain("language");
  });

  it("asks for server_vad with the viewer's silence and the server's defaults for the rest (Q80)", () => {
    const { sockets, handlers, deps } = setup();
    openTranscription("", handlers, deps, 3_000);
    sockets[0].accept();
    const detection = (sockets[0].sent[0] as { session: { audio: { input: { turn_detection: object } } } }).session.audio.input.turn_detection;
    expect(detection).toEqual({ type: "server_vad", silence_duration_ms: 3_000 });
    // Its first server commit ends the dictation: the length guard must not cut it (spec 08).
    expect(sockets[0].sent[0]).toMatchObject({ session: { ignis: { max_utterance_seconds: 0 } } });
  });

  it("waits to configure while the auto-stop is pending, audio included, then sends it as settled", () => {
    const { sockets, handlers, deps } = setup();
    const session = openTranscription("", handlers, deps, "pending");
    sockets[0].accept();
    session.append(new Int16Array([1]));
    expect(sockets[0].sent).toEqual([]);
    session.autoStop(null);
    expect(sockets[0].sent).toHaveLength(1);
    expect(sockets[0].sent[0]).toMatchObject({ type: "session.update", session: { audio: { input: { turn_detection: null } } } });
    session.autoStop(2_000);
    expect(sockets[0].sent).toHaveLength(1);
    sockets[0].emit({ type: "session.updated", event_id: "event_2", session: {} });
    expect(sockets[0].types()).toEqual(["session.update", "input_audio_buffer.append"]);
  });

  it("settled before the greeting, configures at the greeting", () => {
    const { sockets, handlers, deps } = setup();
    const session = openTranscription("", handlers, deps, "pending");
    session.autoStop(5_000);
    expect(sockets[0].sent).toEqual([]);
    sockets[0].accept();
    expect(sockets[0].sent[0]).toMatchObject({ session: { audio: { input: { turn_detection: { type: "server_vad", silence_duration_ms: 5_000 } } } } });
  });

  it("tells the first server commit, and hears nothing of a later item", () => {
    const { sockets, calls, handlers, deps } = setup();
    openTranscription("", handlers, deps, 3_000);
    sockets[0].accept();
    sockets[0].emit({ type: "session.updated", event_id: "e2", session: {} });
    sockets[0].emit({ type: "input_audio_buffer.speech_stopped", audio_end_ms: 900, item_id: "item_1" });
    sockets[0].emit({ type: "input_audio_buffer.committed", item_id: "item_1", previous_item_id: null });
    sockets[0].emit({ type: "conversation.item.input_audio_transcription.delta", item_id: "item_1", delta: "hello" });
    sockets[0].emit({ type: "input_audio_buffer.committed", item_id: "item_2", previous_item_id: "item_1" });
    sockets[0].emit({ type: "conversation.item.input_audio_transcription.delta", item_id: "item_2", delta: "again" });
    sockets[0].emit({ type: "conversation.item.input_audio_transcription.completed", item_id: "item_2", transcript: "again" });
    sockets[0].emit({ type: "conversation.item.input_audio_transcription.completed", item_id: "item_1", transcript: "hello" });
    expect(calls).toEqual([["committed", null], ["delta", "hello"], ["completed", "hello"]]);
  });

  it("ignores the empty-commit error of a commit that crossed the server's own, and reports any other", () => {
    const { sockets, calls, handlers, deps } = setup();
    openTranscription("", handlers, deps, 3_000);
    sockets[0].accept();
    const empty = { type: "error", error: { code: "input_audio_buffer_commit_empty", message: "nothing to commit" } };
    sockets[0].emit(empty);
    expect(calls).toEqual([["failed", "nothing to commit"]]);
    calls.length = 0;
    sockets[0].emit({ type: "input_audio_buffer.committed", item_id: "item_1", previous_item_id: null });
    sockets[0].emit(empty);
    sockets[0].emit({ type: "error", error: { code: "other", message: "boom" } });
    expect(calls).toEqual([["committed", null], ["failed", "boom"]]);
  });

  it("sends no prompt when there is no context", () => {
    const { sockets, handlers, deps } = setup();
    openTranscription("", handlers, deps);
    sockets[0].accept();
    expect(sockets[0].sent[0]).toMatchObject({ session: { audio: { input: { transcription: {} } } } });
    expect(JSON.stringify(sockets[0].sent[0])).not.toContain("prompt");
  });

  it("holds the audio until the session is configured, then sends it in order, and the commit after it", () => {
    const { sockets, handlers, deps } = setup();
    const session = openTranscription("", handlers, deps);
    session.append(new Int16Array([1, -2, 256]));
    session.append(new Int16Array([3]));
    session.commit();
    sockets[0].accept();
    expect(sockets[0].types()).toEqual(["session.update"]);
    sockets[0].emit({ type: "session.updated", event_id: "event_2", session: {} });
    expect(sockets[0].sent.slice(1)).toEqual([
      { type: "input_audio_buffer.append", audio: "AQD+/wAB" },
      { type: "input_audio_buffer.append", audio: "AwA=" },
      { type: "input_audio_buffer.commit" },
    ]);
    session.append(new Int16Array([1]));
    expect(sockets[0].types().at(-1)).toBe("input_audio_buffer.append");
  });

  it("hands over the stable text, the provisional tail and the transcript, and closes on nothing after", () => {
    const { sockets, calls, handlers, deps } = setup();
    openTranscription("", handlers, deps);
    const socket = sockets[0];
    socket.accept();
    socket.emit({ type: "session.updated", session: {} });
    socket.emit({ type: "conversation.item.input_audio_transcription.delta", event_id: "e3", item_id: "item_1", content_index: 0, delta: "Apri vram.rs" });
    socket.emit({ type: "ignis.input_audio_transcription.provisional", event_id: "e4", item_id: "item_1", content_index: 0, text: " e controlla" });
    socket.emit({ type: "input_audio_buffer.committed", event_id: "e5", item_id: "item_1", previous_item_id: null });
    socket.emit({ type: "ignis.input_audio_transcription.provisional", item_id: "item_1", content_index: 0, text: "" });
    socket.emit({ type: "conversation.item.input_audio_transcription.completed", item_id: "item_1", content_index: 0, transcript: "Apri vram.rs e controlla.", usage: { type: "duration", seconds: 2.1 } });
    socket.drop();
    expect(calls).toEqual([
      ["delta", "Apri vram.rs"],
      ["tail", " e controlla"],
      ["committed", null],
      ["tail", ""],
      ["completed", "Apri vram.rs e controlla."],
    ]);
  });

  it("reports a failed item, and a refused event, with the server's message", () => {
    const { sockets, calls, handlers, deps } = setup();
    openTranscription("", handlers, deps);
    sockets[0].accept();
    sockets[0].emit({
      type: "conversation.item.input_audio_transcription.failed",
      item_id: "item_1",
      content_index: 0,
      error: { type: "server_error", code: "engine_error", message: "the speech model failed this item's transcription", param: null },
    });
    sockets[0].emit({ type: "error", event_id: "e9", error: { type: "invalid_request_error", code: "unsupported_value", message: "server_vad is not served yet", param: "session.audio.input.turn_detection", event_id: null } });
    expect(calls).toEqual([
      ["failed", "the speech model failed this item's transcription"],
      ["failed", "server_vad is not served yet"],
    ]);
  });

  it("says when the connection closes under the session, and not when the page closed it", () => {
    const dropped = setup();
    openTranscription("", dropped.handlers, dropped.deps);
    dropped.sockets[0].accept();
    dropped.sockets[0].drop();
    expect(dropped.calls).toEqual([["closed", null]]);

    const closed = setup();
    const session = openTranscription("", closed.handlers, closed.deps);
    closed.sockets[0].accept();
    session.close();
    closed.sockets[0].emit({ type: "conversation.item.input_audio_transcription.delta", delta: "late" });
    expect(closed.sockets[0].closedByPage).toBe(true);
    expect(closed.calls).toEqual([]);
  });

  it("discards: clears the buffer, then closes the session", () => {
    const { sockets, calls, handlers, deps } = setup();
    const session = openTranscription("", handlers, deps);
    sockets[0].accept();
    sockets[0].emit({ type: "session.updated", session: {} });
    session.append(new Int16Array([1]));
    session.discard();
    expect(sockets[0].types()).toEqual(["session.update", "input_audio_buffer.append", "input_audio_buffer.clear"]);
    expect(sockets[0].closedByPage).toBe(true);
    expect(calls).toEqual([]);
  });

  describe("a refused upgrade, which the browser shows without its status", () => {
    const refusedWith = async (fetcher: typeof fetch): Promise<Refusal> => {
      const { sockets, calls, handlers, deps } = setup(fetcher);
      openTranscription("", handlers, deps);
      sockets[0].refuse();
      await settle();
      expect(calls).toHaveLength(1);
      expect(calls[0][0]).toBe("refused");
      return calls[0][1] as Refusal;
    };

    it("asks the route itself, a plain GET with the key, not the model list", async () => {
      const asked: { url: string; headers: unknown }[] = [];
      saveKey("sk-test");
      const fetcher = (async (url: string, init?: RequestInit) => {
        asked.push({ url, headers: init?.headers });
        return upgradeRequired(url, init);
      }) as typeof fetch;
      await refusedWith(fetcher);
      expect(asked).toEqual([{ url: "/v1/realtime?intent=transcription", headers: { Authorization: "Bearer sk-test" } }]);
    });

    it("is busy on 503 engine_full: every session is taken", async () => {
      expect(await refusedWith(answering(503, { error: { code: "engine_full", message: "the speech model has 4 Realtime sessions open" } }))).toEqual({ kind: "busy" });
    });

    it("reads the busy answer's Retry-After", async () => {
      const busy = (async () =>
        new Response(JSON.stringify({ error: { code: "engine_full", message: "full" } }), { status: 503, headers: { "Retry-After": "7" } })) as typeof fetch;
      expect(await refusedWith(busy)).toEqual({ kind: "busy", retryAfter: 7 });
      const odd = (async () =>
        new Response(JSON.stringify({ error: { code: "engine_full", message: "full" } }), { status: 503, headers: { "Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT" } })) as typeof fetch;
      expect(await refusedWith(odd)).toEqual({ kind: "busy" });
    });

    it("is unavailable, with the server's reason, on 503 speech_unavailable", async () => {
      const reason = "the speech model is turned off (`speech.model` is `off`)";
      const body = { error: { code: "speech_unavailable", message: `the speech model is unavailable: ${reason}` } };
      expect(await refusedWith(answering(503, body))).toEqual({ kind: "unavailable", reason });
      // A server older than speech has no such route.
      expect(await refusedWith(answering(404, { error: { message: "not found" } }))).toEqual({ kind: "unavailable", reason: "this server has no speech model" });
    });

    it("is not busy when the server would have opened it: the upgrade was lost on the way", async () => {
      expect(await refusedWith(upgradeRequired)).toEqual({ kind: "no_websocket" });
    });

    it("wants the key on a 401, and says the server is unreachable when nothing answers", async () => {
      expect(await refusedWith(answering(401, { error: { message: "missing API key", code: "invalid_api_key" } }))).toMatchObject({ kind: "key" });
      expect(await refusedWith((async () => Promise.reject(new TypeError("fetch failed"))) as typeof fetch)).toEqual({
        kind: "unreachable",
        message: "Could not reach ignis: TypeError: fetch failed",
      });
      expect(await refusedWith(answering(502, { error: { message: "bad gateway" } }))).toMatchObject({ kind: "unreachable" });
    });

    it("says nothing once the page has closed the session, even while it was asking why", async () => {
      const before = setup();
      const first = openTranscription("", before.handlers, before.deps);
      first.close();
      before.sockets[0].refuse();
      await settle();
      expect(before.calls).toEqual([]);

      const during = setup();
      const second = openTranscription("", during.handlers, during.deps);
      during.sockets[0].refuse();
      second.close();
      await settle();
      expect(during.calls).toEqual([]);
    });
  });
});

// A listening's session (spec speech/07, AC 2): not a dictation's. Every commit is an utterance, the server's
// turn detection holds at its own lengths, and the speakers are asked for as the source says (spec speech/08).
describe("openListening", () => {
  function listen(options: Parameters<typeof openListening>[0], fetcher: typeof fetch = upgradeRequired) {
    const { sockets, deps } = setup(fetcher);
    const heard: ListeningEvent[] = [];
    const session = openListening(options, (event) => heard.push(event), deps);
    return { socket: () => sockets[0], heard, session };
  }

  /** A context of a prompt alone. */
  const said = (prompt: string): Told => ({ prompt, keywords: [], language: null });

  it("asks for server_vad with the view's silence and the server's other lengths, PCM16 at 16 kHz, the context, the provisional tail and the live speakers; no language, no length guard of its own", () => {
    const { socket } = listen({ context: said("Standup: Ada, Linus, the vram planner."), diarization: "live", silenceMs: 1_500 });
    socket().accept();
    expect(socket().sent).toEqual([
      {
        type: "session.update",
        event_id: "listening_configuration",
        session: {
          type: "transcription",
          audio: {
            input: {
              format: { type: "audio/pcm", rate: 16000 },
              transcription: { prompt: "Standup: Ada, Linus, the vram planner." },
              // The silence that ends an utterance is the view's (Q122); the threshold and the padding are the server's.
              turn_detection: { type: "server_vad", silence_duration_ms: 1_500 },
            },
          },
          ignis: { provisional_tail: true, diarization: "live" },
        },
      },
    ]);
    expect(JSON.stringify(socket().sent)).not.toContain("language");
    expect(JSON.stringify(socket().sent)).not.toContain("max_utterance_seconds");
  });

  it("leaves the silence to the server when the view names none", () => {
    const { socket } = listen({ context: said(""), diarization: null, silenceMs: null });
    socket().accept();
    expect(socket().sent[0]).toMatchObject({ session: { audio: { input: { turn_detection: { type: "server_vad" } } } } });
    expect(JSON.stringify(socket().sent[0])).not.toContain("silence_duration_ms");
  });

  it("sends a changed silence in a session.update of the turn detection alone, which holds at once", () => {
    const { socket, heard, session } = listen({ context: said(""), diarization: null, silenceMs: 1_500 });
    socket().accept();
    socket().emit({ type: "session.updated", session: {} });
    session.setSilence(2_000);
    session.setSilence(2_000);
    session.flush();
    expect(socket().sent.slice(1)).toEqual([
      { type: "session.update", session: { type: "transcription", audio: { input: { turn_detection: { type: "server_vad", silence_duration_ms: 2_000 } } } } },
      { type: "session.update", event_id: "listening_flush", session: { type: "transcription" } },
    ]);
    socket().emit({ type: "session.updated", session: {} });
    socket().emit({ type: "session.updated", session: {} });
    expect(heard).toEqual([{ type: "flushed" }]);
  });

  it("asks for no speakers with Speakers off or on a call's microphone, and for the file geometry on a file", () => {
    const off = listen({ context: said(""), diarization: null });
    off.socket().accept();
    expect(off.socket().sent[0]).toMatchObject({ session: { ignis: { provisional_tail: true }, audio: { input: { transcription: {} } } } });
    expect(JSON.stringify(off.socket().sent[0])).not.toContain("diarization");
    const file = listen({ context: said(""), diarization: "file" });
    file.socket().accept();
    expect(file.socket().sent[0]).toMatchObject({ session: { ignis: { diarization: "file" } } });
  });

  it("hears every commit as an utterance of its own, each item's events by its id, the next starting before the last completes", () => {
    const { socket, heard } = listen({ context: said(""), diarization: "live" });
    socket().accept();
    socket().emit({ type: "session.updated", session: {} });
    const s = socket();
    s.emit({ type: "input_audio_buffer.speech_started", audio_start_ms: 480, item_id: "item_1" });
    s.emit({ type: "ignis.input_audio_transcription.provisional", item_id: "item_1", content_index: 0, text: "Buongiorno a" });
    s.emit({ type: "input_audio_buffer.speech_stopped", audio_end_ms: 3_040, item_id: "item_1" });
    s.emit({ type: "input_audio_buffer.committed", item_id: "item_1", previous_item_id: null });
    s.emit({ type: "input_audio_buffer.speech_started", audio_start_ms: 3_040, item_id: "item_2" });
    s.emit({ type: "conversation.item.input_audio_transcription.delta", item_id: "item_1", content_index: 0, delta: "Buongiorno a tutti." });
    s.emit({ type: "ignis.input_audio_transcription.provisional", item_id: "item_1", content_index: 0, text: "" });
    s.emit({ type: "ignis.input_audio_transcription.speaker", item_id: "item_1", content_index: 0, speaker: 0 });
    s.emit({ type: "conversation.item.input_audio_transcription.completed", item_id: "item_1", content_index: 0, transcript: "Buongiorno a tutti." });
    s.emit({ type: "input_audio_buffer.speech_stopped", audio_end_ms: 5_120, item_id: "item_2" });
    s.emit({ type: "input_audio_buffer.committed", item_id: "item_2", previous_item_id: "item_1" });
    s.emit({ type: "ignis.input_audio_transcription.speaker", item_id: "item_2", content_index: 0, speaker: 1 });
    s.emit({ type: "conversation.item.input_audio_transcription.completed", item_id: "item_2", content_index: 0, transcript: "Grazie." });
    expect(heard).toEqual([
      { type: "started", item: "item_1", atMs: 480 },
      { type: "tail", item: "item_1", text: "Buongiorno a" },
      { type: "stopped", item: "item_1", atMs: 3_040 },
      { type: "committed", item: "item_1" },
      { type: "started", item: "item_2", atMs: 3_040 },
      { type: "delta", item: "item_1", text: "Buongiorno a tutti." },
      { type: "tail", item: "item_1", text: "" },
      { type: "speaker", item: "item_1", speaker: 0 },
      { type: "completed", item: "item_1", transcript: "Buongiorno a tutti." },
      { type: "stopped", item: "item_2", atMs: 5_120 },
      { type: "committed", item: "item_2" },
      { type: "speaker", item: "item_2", speaker: 1 },
      { type: "completed", item: "item_2", transcript: "Grazie." },
    ]);
  });

  it("reports a failed item by its id, and a refused event with its code and the event id the page gave it", () => {
    const { socket, heard } = listen({ context: said(""), diarization: "live" });
    socket().accept();
    socket().emit({ type: "conversation.item.input_audio_transcription.failed", item_id: "item_4", content_index: 0, error: { code: "engine_error", message: "the speech model failed" } });
    socket().emit({ type: "error", error: { type: "invalid_request_error", code: "ignis_audio_queue_full", message: "the session holds 120.0 s", param: "audio", event_id: "audio_37" } });
    socket().emit({ type: "error", error: { type: "invalid_request_error", code: "ignis_diarizer_unavailable", message: "the diarizer is turned off", param: null, event_id: null } });
    expect(heard).toEqual([
      { type: "item_failed", item: "item_4", message: "the speech model failed" },
      { type: "error", code: "ignis_audio_queue_full", message: "the session holds 120.0 s", eventId: "audio_37" },
      { type: "error", code: "ignis_diarizer_unavailable", message: "the diarizer is turned off", eventId: null },
    ]);
  });

  it("configures again without speakers when the diarizer is unavailable, which refuses the whole update, and says why", () => {
    const { socket, heard, session } = listen({ context: said(""), diarization: "live" });
    socket().accept();
    session.append(new Int16Array([1]));
    expect(socket().sent[0]).toMatchObject({ event_id: "listening_configuration" });
    socket().emit({ type: "error", error: { code: "ignis_diarizer_unavailable", message: "the diarizer is turned off (`diarizer.model` is `off`)", event_id: "listening_configuration" } });
    expect(socket().sent[1]).toMatchObject({ type: "session.update", event_id: "listening_configuration", session: { ignis: { provisional_tail: true } } });
    expect(JSON.stringify(socket().sent[1])).not.toContain("diarization");
    socket().emit({ type: "session.updated", session: {} });
    expect(socket().types()).toEqual(["session.update", "session.update", "input_audio_buffer.append"]);
    expect(heard).toEqual([
      { type: "error", code: "ignis_diarizer_unavailable", message: "the diarizer is turned off (`diarizer.model` is `off`)", eventId: "listening_configuration" },
    ]);
  });

  it("holds the audio until the configuration is taken, then sends it in order, each append with the page's event id", () => {
    const { socket, session } = listen({ context: said(""), diarization: null });
    session.append(new Int16Array([1, -2, 256]), "audio_1");
    socket().accept();
    session.append(new Int16Array([3]), "audio_2");
    expect(socket().types()).toEqual(["session.update"]);
    socket().emit({ type: "session.updated", session: {} });
    session.append(new Int16Array([4]));
    session.commit();
    expect(socket().sent.slice(1)).toEqual([
      { type: "input_audio_buffer.append", audio: "AQD+/wAB", event_id: "audio_1" },
      { type: "input_audio_buffer.append", audio: "AwA=", event_id: "audio_2" },
      { type: "input_audio_buffer.append", audio: "BAA=" },
      { type: "input_audio_buffer.commit" },
    ]);
  });

  it("sends what is told in a session.update of the transcription, null for what is no longer told; changed before the greeting, it is in the configuration", () => {
    const { socket, session } = listen({ context: said("first"), diarization: "live" });
    session.setContext({ prompt: "second", keywords: ["AnyDesk"], language: "it" });
    socket().accept();
    expect(socket().sent[0]).toMatchObject({ session: { audio: { input: { transcription: { prompt: "second", keywords: ["AnyDesk"], language: "it" } } } } });
    socket().emit({ type: "session.updated", session: {} });
    session.setContext({ prompt: "Names: Ada.", keywords: ["Ada"], language: null });
    session.setContext({ prompt: "Names: Ada.", keywords: ["Ada"], language: null });
    session.setContext(said(""));
    expect(socket().sent.slice(1)).toEqual([
      { type: "session.update", session: { type: "transcription", audio: { input: { transcription: { prompt: "Names: Ada.", keywords: ["Ada"], language: null } } } } },
      { type: "session.update", session: { type: "transcription", audio: { input: { transcription: { prompt: null, keywords: null, language: null } } } } },
    ]);
    // The answer to a context's update is not the configuration's: nothing waits on it.
    socket().emit({ type: "session.updated", session: {} });
    expect(socket().sent).toHaveLength(3);
  });

  it("flushes: an empty session.update whose answer comes once the server has read every append before it, a context's answer not counted", () => {
    const { socket, heard, session } = listen({ context: said(""), diarization: null });
    socket().accept();
    socket().emit({ type: "session.updated", session: {} });
    session.append(new Int16Array([1]));
    session.setContext(said("Names."));
    session.flush();
    expect(socket().sent.slice(1)).toEqual([
      { type: "input_audio_buffer.append", audio: "AQA=" },
      { type: "session.update", session: { type: "transcription", audio: { input: { transcription: { prompt: "Names.", keywords: null, language: null } } } } },
      { type: "session.update", event_id: "listening_flush", session: { type: "transcription" } },
    ]);
    socket().emit({ type: "session.updated", session: {} });
    expect(heard).toEqual([]);
    socket().emit({ type: "session.updated", session: {} });
    expect(heard).toEqual([{ type: "flushed" }]);
  });

  it("says when the connection closes under it, and nothing once the page has closed it", () => {
    const dropped = listen({ context: said(""), diarization: null });
    dropped.socket().accept();
    dropped.socket().drop();
    expect(dropped.heard).toEqual([{ type: "closed" }]);

    const closed = listen({ context: said(""), diarization: null });
    closed.socket().accept();
    closed.session.close();
    closed.socket().emit({ type: "conversation.item.input_audio_transcription.completed", item_id: "item_1", transcript: "late" });
    closed.session.append(new Int16Array([1]));
    expect(closed.socket().closedByPage).toBe(true);
    expect(closed.heard).toEqual([]);
    expect(closed.socket().types()).toEqual(["session.update"]);
  });

  it("asks the route why a refused upgrade was refused, as a dictation does", async () => {
    const { socket, heard } = listen({ context: said(""), diarization: null }, answering(503, { error: { code: "engine_full", message: "full" } }));
    socket().refuse();
    await settle();
    expect(heard).toEqual([{ type: "refused", refusal: { kind: "busy" } }]);
  });
});
