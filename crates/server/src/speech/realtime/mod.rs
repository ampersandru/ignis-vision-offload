//! `GET /v1/realtime?intent=transcription` (spec speech/03): OpenAI's
//! Realtime GA transcription session over a WebSocket, on the speech model.
//!
//! - **The socket.** The upgrade is keyed like the Responses socket's (a
//!   bearer token, or the `openai-insecure-api-key.<key>` subprotocol),
//!   sits outside the serving gate and the text model's idle clock, and is
//!   refused before it opens: a conversation session (`400` naming
//!   `intent`), a speech model that cannot serve (`503
//!   speech_unavailable`), `speech.max_sessions` open (`503` with
//!   `Retry-After`).
//! - **Items.** Audio appended at 24 kHz (16 kHz, the declared extension)
//!   goes to the current item, which a commit ends and a clear discards.
//!   Each item has its own session on the executor (spec speech/01 §4),
//!   opened before its first append, so the context is prefilled when the
//!   socket opens or a `session.update` changes it, never by a commit
//!   (Q50).
//! - **Turn detection** (spec speech/05). Under `server_vad`, the default,
//!   the VAD reads the 16 kHz audio on this socket's task and decides what
//!   reaches an item: speech starts one (`speech_started`, the prefix padding
//!   before it), the silence after it commits it (`speech_stopped`), and with
//!   an idle timeout so much audio without speech is an item of its own. The
//!   model reads each such item after `speech.vad_leading_silence_ms` of
//!   digital silence, which no event and no `usage` counts (Q77). With
//!   `turn_detection: null` every sample goes to the current item and the
//!   client commits.
//! - **Units.** Every call to the executor is one unit on the speech worker,
//!   in this socket's lane: a window pushed, a partial, a commit, an open.
//!   The sessions take turns, unit by unit, ahead of the batch windows (Q44).
//!   The executor's session lives in a slot the units share; they run one at
//!   a time on the worker's one thread, in the order queued.
//! - **Refreshes are best effort** (Q74). The live transcript is refreshed
//!   at most every `speech.partial_interval_ms`, and a refresh is skipped,
//!   never queued, while any unit of the socket's is still in the worker: a
//!   CPU that cannot keep up drops refreshes, never windows or the commit.
//!   The next refresh also waits `speech.partial_cost_factor` times the last
//!   one's compute time (Q76), so the refreshes keep to about that share of
//!   the worker as the item grows and each costs more.
//! - **Text.** `delta` carries stable text only, so the deltas joined are the
//!   transcript `completed` carries (Q43). A session that opted in with
//!   `session.ignis.provisional_tail` also gets the tail, the text still
//!   inside the rollback zone, as `ignis.input_audio_transcription.provisional`.
//! - **Limits.** An item past `speech.max_clip_seconds`, past
//!   `server.stream_timeout` from its first append, or whose transcription
//!   fails, gets `failed`: at its commit, after an `error` saying why at
//!   once, or at once if it is already committed. A socket idle past `server.socket_idle_close` is closed, by
//!   the Responses sockets' rule.
//!
//! Neither the audio nor the transcript is logged: each committed item writes
//! one `ignis.speech.transcription` line (Q13, Q45).

pub(crate) mod config;
pub(crate) mod pcm;

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use axum::extract::ws::rejection::WebSocketUpgradeRejection;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use ignis_speech::vad::{TurnEvent, Turns, Vad};
use serde_json::{json, Map, Value};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::time::Instant;

use super::audio::TARGET_RATE;
use super::executor::{LoadSettings, Partial, SessionOpen, SpeechExecutor, SpeechFiles, SpeechSession, Transcript, WINDOW_SAMPLES};
use super::route::{self, Ending};
use super::worker::Full;
use super::SessionGuard;
use crate::api::{self, bad_request_param, ApiError};
use crate::deadline::Timeout;
use crate::metrics::SpeechOutcome;
use crate::responses::ids::random_suffix;
use crate::responses::socket::{close_idle, offered, CREDENTIAL_PREFIX};
use crate::Server;
use config::{Refusal, SessionConfig, VadDefaults};
use pcm::Pcm;

/// The most base64 `audio` one append carries: OpenAI's limit, 15 MiB.
pub(crate) const MAX_APPEND_BYTES: usize = 15 << 20;

/// `GET /v1/realtime` — a Realtime transcription session.
#[utoipa::path(
    get,
    path = "/v1/realtime",
    tag = "audio",
    operation_id = "realtime_transcription",
    summary = "A Realtime transcription session",
    description = "Upgrades to a WebSocket carrying OpenAI's Realtime GA transcription session (`session.type: \"transcription\"`) on the speech model (Qwen3-ASR, ADR 0066). Open it with `?intent=transcription` and no `model` in the URL; a conversation session (no `intent=transcription`, or a `model` in the URL) is refused `400` naming `intent`. Authenticate with `Authorization: Bearer <key>` on the upgrade or, from a browser, with a `Sec-WebSocket-Protocol` entry `openai-insecure-api-key.<key>`; the server selects the first offered entry that is not a credential (`realtime`, for OpenAI's browser clients). The socket serves through a model switch and with the text model unloaded, and opening it starts loading a text model the idle clock unloaded, while the session is served on the CPU. At most `speech.max_sessions` (4) sessions are open at once.

On open the server sends `session.created` with the effective configuration: `audio.input.format` `{type: \"audio/pcm\", rate: 24000}`, `transcription.model` the speech model's served id, `turn_detection` `{type: \"server_vad\", threshold, prefix_padding_ms, silence_duration_ms, idle_timeout_ms: null}` at the server's `speech.vad_threshold`, `speech.vad_prefix_padding_ms` and `speech.vad_silence_duration_ms` (0.5, 300, 500, OpenAI's), `noise_reduction: null`, `include: []` and `ignis.provisional_tail: false`. The speech model is loaded if it is not, and the context prefilled.

Turn detection. Under `server_vad` (the default) the server finds the speech itself, with Silero VAD on the CPU, one speech probability per 32 ms of audio: the first 32 ms at or over `threshold` start an item, `input_audio_buffer.speech_started` `{audio_start_ms, item_id}`, and the item keeps `prefix_padding_ms` of the audio before it (silence, even the silence that ended the item before, never an earlier item's speech); once the speech has been silent (under `threshold`) for `silence_duration_ms`, `input_audio_buffer.speech_stopped` `{audio_end_ms, item_id}`, and the server commits the item, that silence included: `committed`, then its transcription. Audio outside an item goes nowhere. With `idle_timeout_ms`, that much audio with no speech, since the session opened, the last item or the field's change, is committed as an item of its own after `input_audio_buffer.timeout_triggered` `{audio_start_ms, audio_end_ms, item_id}`. Every time is in milliseconds of the audio the session has received. The speech model reads each of these items after `speech.vad_leading_silence_ms` of digital silence, since its transcript depends on the audio before the speech; no time and no `usage` counts it. With `turn_detection: null` the client commits (manual commit), as OpenAI's Realtime clients that set it, the Playground among them, do.

Client events, each a text frame of one JSON object:
- `session.update` with `session.type: \"transcription\"`, answered `session.updated` with the effective configuration. It names only what it changes; `null` puts a field back to its default; one refused field refuses the whole update. `audio.input.format.rate` is 24000 (resampled to 16 kHz) or 16000 (the model's own, an ignis extension); `audio/pcmu` and `audio/pcma` (G.711) are refused. `transcription.model` is absent, the served id or one of `speech.compat_names`; a text or image model's id is an `error` with `model_kind_mismatch`, any other `model_not_found`. `transcription.prompt` and `transcription.keywords` are the context, prefilled when the update arrives (a change while an item holds audio applies from the next item); `transcription.language` and `transcription.languages` force or detect the language as `POST /v1/audio/transcriptions` does. `transcription.delay` is inert: the live transcript refreshes at most every `speech.partial_interval_ms`. `turn_detection` takes `null` or `{type: \"server_vad\"}` with any of `threshold` (0 to 1), `prefix_padding_ms`, `silence_duration_ms` and `idle_timeout_ms` (each at most `speech.max_clip_seconds`): a field left out keeps its value, `null` is the server's default (no idle timeout); `create_response` and `interrupt_response` are inert, since they belong to conversation sessions; `semantic_vad` is refused. A switch between `null` and `server_vad` while an item holds audio holds from the next item; `server_vad`'s fields hold at once. `noise_reduction` takes `null`; `near_field` and `far_field` are refused. `include: [\"item.input_audio_transcription.logprobs\"]` is refused. `session.ignis.provisional_tail: true` opts in to the provisional tail.
- `input_audio_buffer.append` `{audio}`: base64 16-bit little-endian mono PCM at the session's rate, up to 15 MiB of base64 an event. No reply.
- `input_audio_buffer.commit`: ends the item. `input_audio_buffer.committed` `{item_id, previous_item_id}`, then the last stable text as a `delta`, then `completed`. An empty buffer is an `error` (`input_audio_buffer_commit_empty`); under `server_vad` the buffer holds the speech the VAD has heard since the last item, and a commit during it ends that item at once, with no `speech_stopped`.
- `input_audio_buffer.clear`: discards the buffer and the item's partial state (under `server_vad`, the speech and the padding the VAD holds); answered `input_audio_buffer.cleared`.
- `transcription_session.update` (the retired beta) is an `error` naming `session.update`; conversation events (`response.*`, `conversation.*`, `output_audio_buffer.*`) are an `error`.

Server events, each with an `event_id`: under `server_vad`, `input_audio_buffer.speech_started`, `input_audio_buffer.speech_stopped` and `input_audio_buffer.timeout_triggered`, as above; `conversation.item.input_audio_transcription.delta` `{item_id, content_index: 0, delta}` carries stable text only, while the audio arrives (the live transcript is refreshed at most every `speech.partial_interval_ms`, best effort: a refresh is skipped while the session's audio is still being computed, and the next waits at least `speech.partial_cost_factor` times the last one's compute time), so the deltas of an item joined equal its `completed.transcript`; `conversation.item.input_audio_transcription.completed` `{item_id, content_index, transcript, usage: {type: \"duration\", seconds}}`; `conversation.item.input_audio_transcription.failed` `{item_id, content_index, error}` for an item over `speech.max_clip_seconds`, one past `--server-stream-timeout` from its first append (`request_timeout`), or one the speech model failed; `error` `{error: {type, code, message, param, event_id}}` for every refused event or field, which never closes the socket. With `session.ignis.provisional_tail: true` the session also gets `ignis.input_audio_transcription.provisional` `{item_id, content_index, text}`: the whole current tail, the text the next refresh may still revise, which replaces the previous one, and `\"\"` when the item ends. A session that does not opt in never receives it.

A socket idle for `server.socket_idle_close` seconds (default 300; 0 never) -- no item committed and still being transcribed, and no text or binary frame from the client -- is closed with a normal close frame (1000) whose reason says so.",
    params(("intent" = String, Query, description = "`transcription`: the only session this route serves.")),
    responses(
        (status = 101, description = "Switching protocols: the session is open."),
        (status = 400, description = "Not a transcription session (`param: \"intent\"`): `intent` is not `transcription`, or the URL names a `model`; or a plain `GET`, not a WebSocket upgrade, to a route that could open (`upgrade_required`).", body = ApiError),
        (status = 401, description = "The server was started with `--server-api-key` and the upgrade carried no matching bearer token or `openai-insecure-api-key.<key>` subprotocol.", body = ApiError),
        (status = 503, description = "The speech model is unavailable (`speech_unavailable`, with the reason, and no `Retry-After`), or `speech.max_sessions` sessions are open (`engine_full`, with `Retry-After`). A plain `GET` without the upgrade is answered the same, so a page can ask why its upgrade was refused.", body = ApiError),
    ),
)]
pub(crate) async fn connect(
    State(server): State<Arc<Server>>,
    Query(query): Query<HashMap<String, String>>,
    headers: HeaderMap,
    upgrade: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
) -> Response {
    if query.get("intent").map(String::as_str) != Some("transcription") || query.contains_key("model") {
        return bad_request_param(
            "this server serves Realtime transcription sessions only: open /v1/realtime?intent=transcription with no model in the URL; conversation sessions are not served",
            "intent",
        );
    }
    let live = server.live();
    let group = &live.speech;
    let files = match server.speech.resolve(group) {
        Ok(files) => files,
        Err(why) => return route::unavailable(why),
    };
    let refused = |full: Full| {
        route::engine_full(
            format!("the speech model has {} Realtime sessions open (speech.max_sessions); retry in {} s", group.max_sessions, full.retry_after_secs),
            full,
        )
    };
    // A plain GET is answered what an upgrade would be, so a page that cannot
    // read a refused upgrade's status can still ask (spec speech/04, AC 11):
    // it claims no place.
    let Ok(upgrade) = upgrade else {
        return match server.speech.session_room(group) {
            Err(why) => refused(why),
            Ok(()) => api::error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "upgrade_required",
                "GET /v1/realtime is the Realtime transcription session and takes a WebSocket upgrade, with ?intent=transcription; the speech model can serve now",
            ),
        };
    };
    let guard = match server.speech.open_session(group) {
        Ok(guard) => guard,
        Err(why) => return refused(why),
    };
    // The client's own names, never the credential.
    let protocols: Vec<String> = offered(&headers).filter(|protocol| !protocol.starts_with(CREDENTIAL_PREFIX)).map(str::to_owned).collect();
    upgrade
        .protocols(protocols)
        .max_message_size(api::TEXT_REQUEST_BODY_LIMIT)
        .max_frame_size(api::TEXT_REQUEST_BODY_LIMIT)
        .on_upgrade(move |socket| Connection::new(server, socket, guard, files).run())
}

/// The executor's session the units of one socket take turns at, and what
/// the item's units have computed so far.
#[derive(Default)]
struct Slot {
    session: Option<Box<dyn SpeechSession>>,
    computed: Duration,
}

/// One call to the executor, a window or less of work.
enum Unit {
    /// Load the model if it is not, and open the next item's session,
    /// prefilling its context.
    Open(SessionOpen, LoadSettings),
    /// Push audio: a whole window, as soon as one is in.
    Push(Vec<f32>),
    /// Push what is in, then refresh the live transcript.
    Partial(Vec<f32>),
    /// Push the rest, then decode to the end.
    Commit(Vec<f32>),
}

/// What a unit tells the socket.
enum Outcome {
    /// A refresh, and its compute time on the worker.
    Partial(Partial, Duration),
    Transcript(Transcript, Duration),
    Failed(Failure),
}

/// A unit is out of the worker: what it says, if anything.
struct Done {
    item: u64,
    outcome: Option<Outcome>,
}

/// Why an item failed.
#[derive(Debug, Clone)]
enum Failure {
    /// The model would not load.
    Load(String),
    /// A call to the executor failed.
    Executor(String),
    /// Longer than `speech.max_clip_seconds` (the limit), counting the
    /// milliseconds of leading silence a `server_vad` item is fed after.
    TooLong(u32, u32),
    /// Past `server.stream_timeout` from its first append.
    TimedOut(Timeout),
}

impl Failure {
    /// The `failed` event's `error`.
    fn error(&self) -> Value {
        let (kind, code, message) = match self {
            Self::Load(why) => ("server_error", "speech_unavailable", format!("the speech model could not be loaded: {why}")),
            Self::Executor(_) => ("server_error", "engine_error", "the speech model failed this item's transcription".to_owned()),
            Self::TooLong(limit, 0) => (
                "invalid_request_error",
                "invalid_request_error",
                format!("the item holds more than speech.max_clip_seconds ({limit} s) of audio; nothing is split automatically, commit sooner"),
            ),
            Self::TooLong(limit, lead_ms) => (
                "invalid_request_error",
                "invalid_request_error",
                format!(
                    "the item holds more than speech.max_clip_seconds ({limit} s) of audio, counting the {lead_ms} ms of silence (speech.vad_leading_silence_ms) the speech model reads before it; nothing is split automatically, commit sooner"
                ),
            ),
            Self::TimedOut(timeout) => ("request_timeout", "request_timeout", timeout.message()),
        };
        json!({ "type": kind, "code": code, "message": message, "param": null })
    }

    fn outcome(&self) -> SpeechOutcome {
        match self {
            Self::TimedOut(_) => SpeechOutcome::TimedOut,
            _ => SpeechOutcome::Failed,
        }
    }
}

/// One item: the audio between two commits (or clears).
struct Item {
    seq: u64,
    /// Given at its first append.
    id: Option<String>,
    /// Set when the item is cleared, failed or the socket closes: its units
    /// still queued do nothing.
    cancel: Arc<AtomicBool>,
    /// What its executor session was opened with.
    opened_with: SessionOpen,
    /// Samples appended, at whatever rate, and the seconds they make.
    samples_in: u64,
    seconds: f64,
    /// The milliseconds of digital silence the model reads before its audio,
    /// under `server_vad`: counted against `speech.max_clip_seconds`, as the
    /// model counts them, and nowhere else.
    lead_ms: u32,
    timeout: Timeout,
    deadline: Option<Instant>,
    /// The stable text sent as deltas, and the tail last sent.
    emitted: String,
    tail: String,
    failure: Option<Failure>,
    /// Set at its commit: the line and the count it ends with.
    ending: Option<Ending>,
    /// Audio was queued since the last partial.
    fresh_audio: bool,
}

struct Connection {
    server: Arc<Server>,
    socket: WebSocket,
    guard: SessionGuard,
    files: SpeechFiles,
    session_id: String,
    /// What this session's event ids begin with: a piece of its id.
    event_prefix: String,
    config: SessionConfig,
    pcm: Pcm,
    slot: Arc<Mutex<Slot>>,
    results_tx: UnboundedSender<Done>,
    results: UnboundedReceiver<Done>,
    /// The committed items still being transcribed, in order, then the
    /// current one, last: there is always one.
    items: VecDeque<Item>,
    next_seq: u64,
    /// The current item's 16 kHz samples not yet queued.
    pending: Vec<f32>,
    previous_item_id: Option<String>,
    /// When the current item's live transcript may next be refreshed.
    next_partial: Option<Instant>,
    /// When the session's last refresh was queued, and the earliest the next
    /// may be: `speech.partial_cost_factor` times its compute time later.
    refresh_queued: Instant,
    refresh_hold: Instant,
    /// This socket's units queued on the worker and not yet out of it.
    in_worker: usize,
    outbox: Vec<Value>,
    next_event: u64,
    /// The client's latest text or binary frame, or the end of the latest
    /// item: where the idle time counts from.
    quiet_since: Instant,
    /// The 16 kHz samples the session has heard: the VAD's clock.
    heard: u64,
    /// Under `server_vad`, the turns the VAD finds in the audio; `None` for
    /// manual commit (spec speech/05).
    turns: Option<Turns<Vad>>,
}

impl Connection {
    fn new(server: Arc<Server>, socket: WebSocket, guard: SessionGuard, files: SpeechFiles) -> Self {
        let (results_tx, results) = unbounded_channel();
        let config = SessionConfig::new(files.id.clone(), VadDefaults::of(&server.live().speech));
        let suffix = random_suffix();
        Self {
            server,
            socket,
            guard,
            files,
            event_prefix: format!("event_{}", &suffix[..16]),
            session_id: format!("sess_{suffix}"),
            pcm: Pcm::new(config.rate),
            config,
            slot: Arc::default(),
            results_tx,
            results,
            items: VecDeque::new(),
            next_seq: 0,
            pending: Vec::new(),
            previous_item_id: None,
            next_partial: None,
            refresh_queued: Instant::now(),
            refresh_hold: Instant::now(),
            in_worker: 0,
            outbox: Vec::new(),
            next_event: 0,
            quiet_since: Instant::now(),
            heard: 0,
            turns: None,
        }
    }

    async fn run(mut self) {
        // Q40: a person dictating a prompt sends it next, so a text model the
        // idle clock unloaded starts loading now; the session is served on
        // the CPU meanwhile.
        let server = Arc::clone(&self.server);
        tokio::spawn(async move {
            let _ = crate::model_switch::ensure_loaded(&server, None).await;
        });
        let session = self.config.effective(&self.session_id);
        self.send(json!({ "type": "session.created", "session": session }));
        self.open_next();
        let look = tokio::time::sleep_until(self.next_look());
        tokio::pin!(look);
        loop {
            if !self.flush().await {
                break;
            }
            look.as_mut().reset(self.next_look());
            // Biased, the client's frames first, as on the Responses socket:
            // a frame already read when a deadline falls due wins it.
            tokio::select! {
                biased;
                frame = self.socket.recv() => match frame {
                    Some(Ok(Message::Text(text))) => {
                        self.quiet_since = Instant::now();
                        self.receive(text.as_str());
                    }
                    Some(Ok(Message::Binary(_))) => {
                        self.quiet_since = Instant::now();
                        self.refuse(Refusal::invalid("a frame must be a text frame carrying one JSON event", "type"), None);
                    }
                    Some(Ok(Message::Ping(_) | Message::Pong(_))) => {}
                    Some(Ok(Message::Close(_))) => {
                        while let Some(Ok(_)) = self.socket.recv().await {}
                        break;
                    }
                    Some(Err(_)) | None => break,
                },
                Some(done) = self.results.recv() => self.done(done),
                () = &mut look => {
                    if let Some(after) = self.idle_past() {
                        close_idle(&mut self.socket, after, "realtime").await;
                        break;
                    }
                    self.tick();
                }
            }
        }
        // What is still queued does nothing; a committed item not transcribed
        // counts as cancelled (its `Ending`, dropped).
        for item in &self.items {
            item.cancel.store(true, Ordering::SeqCst);
        }
    }

    /// Queue a server event, with its `event_id`.
    fn send(&mut self, mut event: Value) {
        self.next_event += 1;
        event["event_id"] = json!(format!("{}{:08x}", self.event_prefix, self.next_event));
        self.outbox.push(event);
    }

    async fn flush(&mut self) -> bool {
        for event in std::mem::take(&mut self.outbox) {
            if self.socket.send(Message::Text(event.to_string().into())).await.is_err() {
                return false;
            }
        }
        true
    }

    /// An `error` event refusing a client event or a field.
    fn refuse(&mut self, refusal: Refusal, event_id: Option<String>) {
        self.send(json!({
            "type": "error",
            "error": { "type": "invalid_request_error", "code": refusal.code, "message": refusal.message, "param": refusal.param, "event_id": event_id },
        }));
    }

    fn current(&mut self) -> &mut Item {
        self.items.back_mut().expect("a socket always has a current item")
    }

    /// The `speech.context_max_tokens` and load settings in effect now.
    fn open_settings(&self) -> (SessionOpen, LoadSettings) {
        let live = self.server.live();
        let group = &live.speech;
        (self.config.open(group), LoadSettings { threads: group.threads, pinned_weights: group.pinned_weights })
    }

    /// Start the next item: its executor session opens now, so its context
    /// is prefilled before its first append (Q50).
    fn open_next(&mut self) {
        let (open, load) = self.open_settings();
        let item = Item {
            seq: self.next_seq,
            id: None,
            cancel: Arc::new(AtomicBool::new(false)),
            opened_with: open.clone(),
            samples_in: 0,
            seconds: 0.0,
            lead_ms: 0,
            timeout: Timeout::NONE,
            deadline: None,
            emitted: String::new(),
            tail: String::new(),
            failure: None,
            ending: None,
            fresh_audio: false,
        };
        self.next_seq += 1;
        self.items.push_back(item);
        self.queue(Unit::Open(open, load));
        self.pending.clear();
        self.next_partial = None;
        self.sync_turns();
    }

    /// The turn detection `session.update` set, from now while the current
    /// item holds no audio; a switch between manual commit and `server_vad`
    /// otherwise waits for the item to end, as a changed context does.
    /// `server_vad`'s fields hold from the VAD's next frame either way.
    fn sync_turns(&mut self) {
        let holds_audio = self.current().samples_in > 0;
        match (&self.config.turn_detection, self.turns.as_mut()) {
            (Some(vad), Some(turns)) => turns.set_settings(vad.settings()),
            (Some(vad), None) if !holds_audio => self.turns = Some(Turns::new(Vad::new(), vad.settings(), self.heard)),
            (None, Some(_)) if !holds_audio => self.turns = None,
            _ => {}
        }
    }

    /// Queue `unit` of the current item on the worker, in this socket's lane.
    /// Every unit answers when it is out of the worker, so the socket knows
    /// what it still has in it.
    fn queue(&mut self, unit: Unit) {
        let item = self.items.back().expect("a current item");
        let (seq, cancel) = (item.seq, Arc::clone(&item.cancel));
        let slot = Arc::clone(&self.slot);
        let executor = Arc::clone(self.server.speech.executor());
        let files = self.files.clone();
        let results = self.results_tx.clone();
        let task = Box::new(move || {
            let outcome = match cancel.load(Ordering::SeqCst) {
                true => None,
                false => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(unit, &slot, executor.as_ref(), &files))).unwrap_or_else(|_| {
                    // A unit that panicked leaves no session to go on with.
                    slot.lock().unwrap_or_else(PoisonError::into_inner).session = None;
                    Some(Outcome::Failed(Failure::Executor("the speech executor panicked".to_owned())))
                }),
            };
            let _ = results.send(Done { item: seq, outcome });
        });
        self.in_worker += 1;
        self.server.speech.submit_interactive(&self.guard, task, &self.server.live().speech);
    }

    /// One client frame.
    fn receive(&mut self, text: &str) {
        let Ok(Value::Object(event)) = serde_json::from_str::<Value>(text) else {
            return self.refuse(Refusal::invalid("a frame must carry one JSON object event", "type"), None);
        };
        let event_id = event.get("event_id").and_then(Value::as_str).map(str::to_owned);
        match event.get("type").and_then(Value::as_str) {
            Some("session.update") => self.update(&event, event_id),
            Some("input_audio_buffer.append") => self.append(&event, event_id),
            Some("input_audio_buffer.commit") => self.commit(event_id),
            Some("input_audio_buffer.clear") => self.clear(),
            Some("transcription_session.update") => self.refuse(
                Refusal::unsupported(
                    "transcription_session.update is the retired beta's event: send session.update with session.type `transcription`",
                    "type",
                ),
                event_id,
            ),
            Some(kind) if ["response.", "conversation.", "output_audio_buffer."].iter().any(|prefix| kind.starts_with(prefix)) => self.refuse(
                Refusal::unsupported(
                    format!("`{kind}` belongs to a Realtime conversation session, which this server does not serve: this session transcribes"),
                    "type",
                ),
                event_id,
            ),
            Some(kind) => self.refuse(
                Refusal::invalid(
                    format!("unknown event type `{kind}`: a transcription session takes session.update and input_audio_buffer.append, .commit and .clear"),
                    "type",
                ),
                event_id,
            ),
            None => self.refuse(Refusal::invalid("an event needs a string type", "type"), event_id),
        }
    }

    fn update(&mut self, event: &Map<String, Value>, event_id: Option<String>) {
        let live = self.server.live();
        let served = self.files.id.clone();
        let model = |named: &str| {
            route::model_named(&self.server, &live.speech, Some(named), "session.audio.input.transcription.model").map(|()| served.clone())
        };
        let next = match self.config.updated(event.get("session").unwrap_or(&Value::Null), model) {
            Ok(next) => next,
            Err(refusal) => return self.refuse(refusal, event_id),
        };
        if next.rate != self.pcm.rate() {
            // What the item holds at the old rate goes in first.
            let mut rest = Vec::new();
            self.pcm.set_rate(next.rate, &mut rest);
            self.stream(rest);
        }
        self.config = next;
        // A changed context is prefilled now, while the item holds no audio;
        // with audio, it applies from the next item.
        let (open, _) = self.open_settings();
        let current = self.current();
        if current.samples_in == 0 && current.opened_with != open {
            current.cancel.store(true, Ordering::SeqCst);
            self.items.pop_back();
            self.open_next();
        }
        self.sync_turns();
        let session = self.config.effective(&self.session_id);
        self.send(json!({ "type": "session.updated", "session": session }));
    }

    fn append(&mut self, event: &Map<String, Value>, event_id: Option<String>) {
        let Some(audio) = event.get("audio").and_then(Value::as_str) else {
            return self.refuse(Refusal::invalid("input_audio_buffer.append needs `audio`: base64 16-bit mono PCM", "audio"), event_id);
        };
        if audio.len() > MAX_APPEND_BYTES {
            return self.refuse(Refusal::invalid("`audio` carries more than 15 MiB of base64; send it over several appends", "audio"), event_id);
        }
        let rate = self.pcm.rate();
        let mut samples = Vec::new();
        let Ok(carried) = self.pcm.append(audio, &mut samples) else {
            return self.refuse(Refusal::invalid("`audio` is not base64", "audio"), event_id);
        };
        if self.turns.is_some() {
            // Under `server_vad` the VAD decides what reaches an item.
            return self.stream(samples);
        }
        self.begin_item();
        if self.admit(carried, rate) {
            self.stream(samples);
        }
    }

    /// The current item takes `count` samples at `rate`: its length grows,
    /// and whether they go on to the executor, which they do not once the
    /// item has failed or once it holds more than `speech.max_clip_seconds`.
    fn admit(&mut self, count: usize, rate: u32) -> bool {
        let limit = self.server.live().speech.max_clip_seconds;
        let item = self.current();
        item.samples_in += count as u64;
        item.seconds += count as f64 / f64::from(rate);
        if item.failure.is_some() {
            return false;
        }
        if item.seconds + f64::from(item.lead_ms) / 1000.0 > f64::from(limit) {
            let lead_ms = item.lead_ms;
            self.fail_current(Failure::TooLong(limit, lead_ms));
            return false;
        }
        true
    }

    /// The current item's first audio, at its first append or when the VAD
    /// heard speech: its id, its deadline (`server.stream_timeout`, per item
    /// from its first audio, ADR 0058), and when its first refresh falls due:
    /// an interval on, and not before the session's last refresh allows.
    fn begin_item(&mut self) {
        let live = self.server.live();
        let interval = Duration::from_millis(u64::from(live.speech.partial_interval_ms));
        let hold = self.refresh_hold;
        let item = self.current();
        if item.id.is_none() {
            item.id = Some(format!("item_{}", random_suffix()));
            item.timeout = Timeout::of(&live, true);
            item.deadline = Some(item.timeout.from_now());
            self.next_partial = Some((Instant::now() + interval).max(hold));
        }
    }

    /// A `server_vad` item's leading silence, before its first sample:
    /// `speech.vad_leading_silence_ms` of zeros the model reads first (Q77).
    /// Its times, its `usage` and the samples it counts are the client's
    /// audio alone.
    fn lead_in(&mut self) {
        let lead_ms = self.server.live().speech.vad_leading_silence_ms;
        self.current().lead_ms = lead_ms;
        self.take_audio(ignis_speech::vad::leading_silence(lead_ms));
    }

    /// The session's next 16 kHz samples: the current item's under manual
    /// commit, the VAD's under `server_vad`, which gives an item the turns it
    /// finds (spec speech/05). The VAD reads every sample where it arrives, on
    /// this socket's task: a 32 ms frame never waits behind a window.
    fn stream(&mut self, samples: Vec<f32>) {
        self.heard += samples.len() as u64;
        let Some(turns) = self.turns.as_mut() else {
            return self.take_audio(samples);
        };
        let mut events = Vec::new();
        turns.push(&samples, &mut events);
        for event in events {
            self.turn(event);
        }
    }

    /// What the VAD found, as the Realtime events and the item's audio: an
    /// item begins at `speech_started`, its padded audio follows, and it is
    /// committed at `speech_stopped`; an idle timeout commits the silence.
    fn turn(&mut self, event: TurnEvent) {
        let millis = |sample: u64| sample * 1000 / u64::from(TARGET_RATE);
        match event {
            TurnEvent::Started { start } => {
                self.begin_item();
                let id = self.current().id.clone();
                self.send(json!({ "type": "input_audio_buffer.speech_started", "audio_start_ms": millis(start), "item_id": id }));
                self.lead_in();
            }
            TurnEvent::Audio(samples) => self.give(samples),
            TurnEvent::Stopped { end } => {
                let id = self.current().id.clone();
                self.send(json!({ "type": "input_audio_buffer.speech_stopped", "audio_end_ms": millis(end), "item_id": id }));
                self.end_item();
            }
            TurnEvent::TimedOut { start, end, audio } => {
                self.begin_item();
                let id = self.current().id.clone();
                self.send(json!({
                    "type": "input_audio_buffer.timeout_triggered",
                    "audio_start_ms": millis(start),
                    "audio_end_ms": millis(end),
                    "item_id": id,
                }));
                self.lead_in();
                self.give(audio);
                self.end_item();
            }
        }
    }

    /// Samples the VAD gave the current item, counted at 16 kHz.
    fn give(&mut self, samples: Vec<f32>) {
        if self.admit(samples.len(), TARGET_RATE) {
            self.take_audio(samples);
        }
    }

    /// The current item's new 16 kHz samples: each whole window is queued at
    /// once, the rest waits for the window to fill, a partial or the commit.
    fn take_audio(&mut self, samples: Vec<f32>) {
        if self.current().failure.is_some() {
            return;
        }
        self.pending.extend(samples);
        while self.pending.len() >= WINDOW_SAMPLES {
            let window: Vec<f32> = self.pending.drain(..WINDOW_SAMPLES).collect();
            self.queue(Unit::Push(window));
            self.current().fresh_audio = true;
        }
        if !self.pending.is_empty() {
            self.current().fresh_audio = true;
        }
    }

    fn commit(&mut self, event_id: Option<String>) {
        if let Some(turns) = self.turns.as_mut() {
            // Under `server_vad` a commit ends the turn the VAD is in; the
            // stream goes on, so the resampler keeps its state.
            if !turns.speaking() {
                let empty = "the input audio buffer holds no speech: under server_vad it holds what the VAD heard as speech since the last item".to_owned();
                return self.refuse(Refusal { code: "input_audio_buffer_commit_empty", message: empty, param: None }, event_id);
            }
            let rest = turns.end_turn();
            self.give(rest);
            return self.end_item();
        }
        if self.current().samples_in == 0 {
            let empty = "the input audio buffer is empty: append audio before committing".to_owned();
            return self.refuse(Refusal { code: "input_audio_buffer_commit_empty", message: empty, param: None }, event_id);
        }
        let mut rest = Vec::new();
        self.pcm.finish(&mut rest);
        self.stream(rest);
        self.end_item();
    }

    /// The current item ends: `committed`, its transcription queued (or its
    /// `failed`), and the next item opened.
    fn end_item(&mut self) {
        let previous = self.previous_item_id.take();
        let ending = Ending::new(
            Arc::clone(&self.server),
            Arc::clone(&self.current().cancel),
            self.server.speech.executor().name(),
            self.files.id.clone(),
            self.current().seconds,
            std::time::Instant::now(),
        );
        let item = self.current();
        item.ending = Some(ending);
        let id = item.id.clone().expect("an item with audio has its id");
        self.previous_item_id = Some(id.clone());
        self.send(json!({ "type": "input_audio_buffer.committed", "item_id": id, "previous_item_id": previous }));
        if self.current().failure.is_some() {
            let last = self.items.len() - 1;
            self.end_failed(last);
        } else {
            let rest = std::mem::take(&mut self.pending);
            self.queue(Unit::Commit(rest));
        }
        self.open_next();
    }

    fn clear(&mut self) {
        let vad = self.turns.is_some();
        if let Some(turns) = self.turns.as_mut() {
            // The padding the VAD keeps, and a turn on, go as well.
            turns.clear();
        }
        let current = self.current();
        if current.samples_in > 0 || current.id.is_some() {
            current.cancel.store(true, Ordering::SeqCst);
            let mut cleared = self.items.pop_back().expect("a current item");
            self.end_tail(&mut cleared);
            if !vad {
                // Under `server_vad` the stream goes on, and the VAD's clock
                // with it: the resampler keeps its state.
                self.pcm.reset();
            }
            self.open_next();
        }
        self.send(json!({ "type": "input_audio_buffer.cleared" }));
    }

    /// The current item fails before its commit: its audio goes nowhere from
    /// now, an `error` says why at once, and its commit answers `failed`.
    fn fail_current(&mut self, failure: Failure) {
        let mut error = failure.error();
        error["event_id"] = Value::Null;
        let item = self.current();
        item.failure = Some(failure);
        item.cancel.store(true, Ordering::SeqCst);
        self.pending.clear();
        self.next_partial = None;
        self.send(json!({ "type": "error", "error": error }));
    }

    /// The committed item at `index` fails: `failed` now.
    fn fail_committed(&mut self, index: usize, failure: Failure) {
        let item = &mut self.items[index];
        item.failure = Some(failure);
        item.cancel.store(true, Ordering::SeqCst);
        self.end_failed(index);
    }

    /// Send `failed` for the committed item at `index`, end it and drop it.
    fn end_failed(&mut self, index: usize) {
        let mut item = self.items.remove(index).expect("an item at the index");
        let failure = item.failure.clone().expect("a failed item");
        if let Failure::Load(why) | Failure::Executor(why) = &failure {
            tracing::warn!(name: "ignis.speech.failed", variant = %self.files.id, reason = %why, "a transcription failed");
        }
        self.end_tail(&mut item);
        self.send(json!({
            "type": "conversation.item.input_audio_transcription.failed",
            "item_id": item.id,
            "content_index": 0,
            "error": failure.error(),
        }));
        if let Some(mut ending) = item.ending.take() {
            ending.end(failure.outcome(), None, None);
        }
        self.quiet_since = Instant::now();
    }

    /// An item ends: a tail it showed goes, for a session that opted in.
    fn end_tail(&mut self, item: &mut Item) {
        if self.config.provisional_tail && !item.tail.is_empty() {
            item.tail.clear();
            let id = item.id.clone();
            self.send(json!({ "type": "ignis.input_audio_transcription.provisional", "item_id": id, "content_index": 0, "text": "" }));
        }
    }

    /// A unit is out of the worker.
    fn done(&mut self, done: Done) {
        self.in_worker -= 1;
        let Some(outcome) = done.outcome else { return };
        if let Outcome::Partial(_, cost) = &outcome {
            self.hold_refreshes(*cost);
        }
        // An item cleared or ended since: nobody is waiting for it.
        let Some(index) = self.items.iter().position(|item| item.seq == done.item) else { return };
        match outcome {
            Outcome::Partial(partial, _) => {
                let opted_in = self.config.provisional_tail;
                let item = &mut self.items[index];
                if item.failure.is_some() {
                    return;
                }
                let delta = extension(&item.emitted, &partial.stable);
                let id = item.id.clone();
                if let Some(delta) = delta {
                    item.emitted = partial.stable.clone();
                    self.send(json!({ "type": "conversation.item.input_audio_transcription.delta", "item_id": id, "content_index": 0, "delta": delta }));
                }
                let item = &mut self.items[index];
                if opted_in && item.tail != partial.tail {
                    item.tail = partial.tail.clone();
                    self.send(json!({ "type": "ignis.input_audio_transcription.provisional", "item_id": id, "content_index": 0, "text": partial.tail }));
                }
            }
            Outcome::Transcript(transcript, computed) => {
                let mut item = self.items.remove(index).expect("an item at the index");
                if item.failure.is_some() {
                    return;
                }
                match extension(&item.emitted, &transcript.text) {
                    Some(delta) => self.send(json!({
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": item.id,
                        "content_index": 0,
                        "delta": delta,
                    })),
                    None if !transcript.text.starts_with(item.emitted.as_str()) => {
                        // The executor revised stable text, which spec 01 §4
                        // says it never does: the transcript is still its own.
                        tracing::warn!(
                            name: "ignis.speech.stable_text_revised",
                            variant = %self.files.id,
                            "a transcript does not begin with the stable text its deltas sent, so they do not join to it"
                        );
                    }
                    None => {}
                }
                self.end_tail(&mut item);
                self.send(json!({
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item.id,
                    "content_index": 0,
                    "transcript": transcript.text,
                    "usage": { "type": "duration", "seconds": (item.seconds * 1000.0).round() / 1000.0 },
                }));
                if let Some(mut ending) = item.ending.take() {
                    ending.end(SpeechOutcome::Completed, Some(&transcript), Some(computed));
                }
                self.quiet_since = Instant::now();
            }
            Outcome::Failed(failure) => {
                if self.items[index].failure.is_some() {
                    return;
                }
                if self.items[index].ending.is_some() {
                    self.fail_committed(index, failure);
                } else {
                    self.fail_current(failure);
                }
            }
        }
    }

    /// A refresh of the session's took `cost` on the worker: the next waits
    /// at least `speech.partial_cost_factor` times that from when this one was
    /// queued, never less than `speech.partial_interval_ms` (Q76), so the
    /// refreshes take at most about one part in that many of the worker
    /// however much each costs. Read live, at each refresh.
    fn hold_refreshes(&mut self, cost: Duration) {
        self.refresh_hold = self.refresh_queued + cost * self.server.live().speech.partial_cost_factor;
        if let Some(at) = self.next_partial.as_mut() {
            *at = (*at).max(self.refresh_hold);
        }
    }

    /// The timers: the live transcript's refresh, and the items' deadlines.
    fn tick(&mut self) {
        let now = Instant::now();
        if self.next_partial.is_some_and(|at| at <= now) {
            let live = self.server.live();
            self.next_partial = Some(now + Duration::from_millis(u64::from(live.speech.partial_interval_ms)));
            // Best effort (Q74): with anything of this socket's still in the
            // worker (a window, the last refresh, a commit) the refresh is
            // skipped, never queued behind it.
            let item = self.items.back().expect("a current item");
            if self.in_worker == 0 && item.failure.is_none() && item.fresh_audio {
                let rest = std::mem::take(&mut self.pending);
                self.queue(Unit::Partial(rest));
                self.refresh_queued = now;
                self.current().fresh_audio = false;
            }
        }
        let mut index = 0;
        while index < self.items.len() {
            let item = &self.items[index];
            if item.failure.is_none() && item.deadline.is_some_and(|deadline| deadline <= now) {
                let failure = Failure::TimedOut(item.timeout);
                if item.ending.is_some() {
                    self.fail_committed(index, failure);
                    continue;
                }
                self.fail_current(failure);
            }
            index += 1;
        }
    }

    /// The live `server.socket_idle_close`, when the socket has been idle at
    /// least that long: no committed item being transcribed, and nothing
    /// from the client since.
    fn idle_past(&self) -> Option<Duration> {
        let after = self.server.live().socket_idle_close;
        let idle = self.items.len() == 1;
        (idle && !after.is_zero() && self.quiet_since.elapsed() >= after).then_some(after)
    }

    /// When to look at the timers next: the next refresh, the nearest
    /// deadline, the idle time, and at least once an
    /// [`IDLE_TICK`](crate::idle::IDLE_TICK), so a live change of a knob
    /// reaches an open socket within one.
    fn next_look(&self) -> Instant {
        let mut at = Instant::now() + crate::idle::IDLE_TICK;
        if let Some(partial) = self.next_partial {
            at = at.min(partial);
        }
        for item in &self.items {
            if let (None, Some(deadline)) = (&item.failure, item.deadline) {
                at = at.min(deadline);
            }
        }
        let after = self.server.live().socket_idle_close;
        if !after.is_zero() && self.items.len() == 1 {
            at = at.min(self.quiet_since + after);
        }
        at
    }
}

/// What `text` adds to `sent`, when it extends it: the only text a delta may
/// carry, since a delta only ever appends.
fn extension<'a>(sent: &str, text: &'a str) -> Option<&'a str> {
    (text.len() > sent.len() && text.starts_with(sent)).then(|| &text[sent.len()..])
}

/// Run one unit on the worker thread. `None` when it has nothing to say: an
/// open or a push that went through, or a unit after a failure that left no
/// session (the failure already answered).
fn run(unit: Unit, slot: &Mutex<Slot>, executor: &dyn SpeechExecutor, files: &SpeechFiles) -> Option<Outcome> {
    let mut slot = slot.lock().unwrap_or_else(PoisonError::into_inner);
    let failed = |slot: &mut Slot, why: String| {
        slot.session = None;
        Some(Outcome::Failed(Failure::Executor(why)))
    };
    match unit {
        Unit::Open(open, load) => {
            slot.session = None;
            slot.computed = Duration::ZERO;
            if !executor.is_loaded() {
                if let Err(error) = executor.load(files, load) {
                    return Some(Outcome::Failed(Failure::Load(error.0)));
                }
            }
            match executor.open(open) {
                Ok(session) => {
                    slot.session = Some(session);
                    None
                }
                Err(error) => failed(&mut *slot, error.0),
            }
        }
        Unit::Push(samples) => {
            let began = std::time::Instant::now();
            let pushed = slot.session.as_mut()?.push(&samples);
            slot.computed += began.elapsed();
            pushed.err().and_then(|error| failed(&mut *slot, error.0))
        }
        Unit::Partial(samples) => {
            let began = std::time::Instant::now();
            let session = slot.session.as_mut()?;
            let partial = push(session.as_mut(), &samples).and_then(|()| session.partial());
            let cost = began.elapsed();
            slot.computed += cost;
            match partial {
                Ok(partial) => Some(Outcome::Partial(partial, cost)),
                Err(error) => failed(&mut *slot, error.0),
            }
        }
        Unit::Commit(samples) => {
            let began = std::time::Instant::now();
            let mut session = slot.session.take()?;
            let transcript = push(session.as_mut(), &samples).and_then(|()| session.commit());
            let computed = slot.computed + began.elapsed();
            match transcript {
                Ok(transcript) => Some(Outcome::Transcript(transcript, computed)),
                Err(error) => failed(&mut *slot, error.0),
            }
        }
    }
}

/// Push `samples` when there are any: a partial or a commit with nothing new
/// pushes nothing.
fn push(session: &mut dyn SpeechSession, samples: &[f32]) -> Result<(), super::executor::SpeechError> {
    if samples.is_empty() {
        return Ok(());
    }
    session.push(samples)
}
