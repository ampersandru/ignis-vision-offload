import { beforeEach, describe, expect, it } from "vitest";
import { forgetKey, saveKey } from "../api/auth.ts";
import type { SocketLike } from "../api/socket.ts";
import { openTranscription, type Refusal, type TranscriptionHandlers } from "./realtime.ts";

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
    onCompleted: (text) => calls.push(["completed", text]),
    onFailed: (message) => calls.push(["failed", message]),
    onRefused: (refusal) => calls.push(["refused", refusal]),
    onClosed: () => calls.push(["closed", null]),
  };
  return { calls, handlers };
}

const answering = (status: number, body: unknown) => (async () => new Response(JSON.stringify(body), { status })) as typeof fetch;

function setup(fetcher: typeof fetch = answering(200, { data: [], speech: { available: true } })) {
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

  it("configures the session as spec 04 says: manual commit, PCM16 at 16 kHz, the context, the provisional tail, no language", () => {
    const { sockets, handlers, deps } = setup();
    openTranscription("The last reply.\n\nThe composer's text.", handlers, deps);
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
          ignis: { provisional_tail: true },
        },
      },
    ]);
    expect(JSON.stringify(sockets[0].sent)).not.toContain("language");
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

    it("is busy when GET /v1/models says the speech model can serve: every session is taken (503 engine_full)", async () => {
      expect(await refusedWith(answering(200, { data: [], speech: { available: true } }))).toEqual({ kind: "busy" });
    });

    it("is unavailable, with the server's reason, when it cannot serve", async () => {
      const reason = "the speech model is turned off (`speech.model` is `off`)";
      expect(await refusedWith(answering(200, { data: [], speech: { available: false, reason } }))).toEqual({ kind: "unavailable", reason });
      // A server older than speech has neither the object nor the route.
      expect(await refusedWith(answering(200, { data: [] }))).toEqual({ kind: "unavailable", reason: "this server has no speech model" });
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
