//! `GET /v1/realtime?intent=transcription` (spec speech/03, GitHub #353)
//! over a live listener, on a scripted executor ([`FakeExecutor`]): the
//! upgrade and its refusals, the session's configuration, the audio, the
//! items and their events, the stable deltas and the opt-in provisional
//! tail, the context prefilled when the session opens, the limits, the
//! sessions taking turns on the worker, and the socket's place outside the
//! serving gate.
//!
//! The fake's transcript is the words it was scripted with: one more is
//! stable per 8 s window pushed (128,000 samples at 16 kHz), the last only
//! at the commit. The tests send 16 kHz, the model's rate, so their samples
//! are the executor's, except where the 24 kHz resampling is the point.
//!
//! The log line of an item is `speech_realtime_log.rs`'s, a binary of its
//! own.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use ignis_server::config::schema::SpeechGroup;
use ignis_server::model_switch::Switcher;
use ignis_server::speech::executor::FakeExecutor;
use ignis_server::speech::Speech;
use ignis_server::{ModelStatus, Server};
use serde_json::{json, Value};

#[path = "support/mod.rs"]
mod support;
#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;
use support::realtime::{self, append, at_16_khz, commit, live, next_event, of, open, send, session, until, until_true, update, Socket, WINDOW};
use support::speech::{self, SPEECH_MODEL, TEXT_MODEL};

const DELTA: &str = "conversation.item.input_audio_transcription.delta";
const COMPLETED: &str = "conversation.item.input_audio_transcription.completed";
const FAILED: &str = "conversation.item.input_audio_transcription.failed";
const PROVISIONAL: &str = "ignis.input_audio_transcription.provisional";

fn saying(words: &[&str]) -> Arc<FakeExecutor> {
    Arc::new(FakeExecutor::saying(words))
}

/// A server over `executor`, refreshing the live transcript every 100 ms.
async fn served(executor: &Arc<FakeExecutor>, edit: impl FnOnce(&mut SpeechGroup)) -> (realtime::Live, Server, std::path::PathBuf) {
    let (server, dir) = speech::server(Arc::clone(executor), |group| {
        group.partial_interval_ms = 100;
        edit(group);
    });
    (live(server.clone()).await, server, dir)
}

/// A session at 16 kHz, its `session.updated` read.
async fn session_16k(live: &realtime::Live) -> Socket {
    let (mut socket, _) = session(live).await;
    send(&mut socket, at_16_khz()).await;
    assert_eq!(next_event(&mut socket).await["type"], "session.updated");
    socket
}

fn error_code(event: &Value) -> (&str, &str) {
    assert_eq!(event["type"], "error", "{event}");
    (event["error"]["code"].as_str().unwrap(), event["error"]["param"].as_str().unwrap_or_default())
}

// ── AC 1: the upgrade ───────────────────────────────────────────────────────

/// AC 1: the upgrade takes the server's key as a bearer token or as the
/// `openai-insecure-api-key.<key>` subprotocol, which is never selected
/// back; without it the upgrade is `401`.
#[tokio::test]
async fn the_upgrade_is_keyed_by_bearer_or_by_subprotocol() {
    let (server, _dir) = speech::server(saying(&["hi"]), |_| {});
    let live = live(server.with_api_key(ignis_server::config::ApiKey::new("sk-rt"))).await;
    assert_eq!(realtime::refusal(open(&live, "?intent=transcription", &[]).await.unwrap_err()).0, 401);
    let (mut by_header, _) = open(&live, "?intent=transcription", &[("Authorization", "Bearer sk-rt")]).await.expect("bearer opens");
    assert_eq!(next_event(&mut by_header).await["type"], "session.created");
    let (mut by_protocol, handshake) =
        open(&live, "?intent=transcription", &[("Sec-WebSocket-Protocol", "realtime, openai-insecure-api-key.sk-rt")]).await.expect("the subprotocol opens");
    assert_eq!(handshake.headers()["sec-websocket-protocol"], "realtime", "the credential is never selected");
    assert_eq!(next_event(&mut by_protocol).await["type"], "session.created");
}

/// AC 1: a conversation session (no `intent=transcription`, or a `model` in
/// the URL, which is how OpenAI opens one) is refused `400` naming `intent`.
#[tokio::test]
async fn a_conversation_session_is_refused_400_naming_intent() {
    let (server, _dir) = speech::server(saying(&["hi"]), |_| {});
    let live = live(server).await;
    for query in ["", "?intent=conversation", "?model=gpt-realtime", "?intent=transcription&model=gpt-realtime"] {
        let (status, _, body) = realtime::refusal(open(&live, query, &[]).await.unwrap_err());
        assert_eq!(status, 400, "{query}");
        assert_eq!(body["error"]["param"], "intent", "{query}: {body}");
    }
}

// ── AC 2, 3; spec 05's AC 4: the session's configuration ────────────────────

/// AC 2, 3; spec 05's AC 4: `session.created` carries the effective
/// configuration: 24 kHz, the speech model, `server_vad` at the server's
/// `speech.vad_*`; `session.update` is answered `session.updated` with the
/// configuration now in effect.
#[tokio::test]
async fn the_session_declares_its_configuration_and_an_update_is_answered_with_it() {
    let executor = saying(&["hi"]);
    let (live, _server, _dir) = served(&executor, |group| group.vad_silence_duration_ms = 700).await;
    let (mut socket, created) = session(&live).await;
    let input = &created["session"]["audio"]["input"];
    assert_eq!(created["session"]["type"], "transcription");
    assert_eq!(input["format"], json!({ "type": "audio/pcm", "rate": 24000 }));
    assert_eq!(input["transcription"]["model"], SPEECH_MODEL);
    assert_eq!(
        input["turn_detection"],
        json!({ "type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 700, "idle_timeout_ms": null }),
        "a session that sends no turn_detection runs server_vad, at the server's settings"
    );
    assert!(created["event_id"].as_str().unwrap().starts_with("event_"));

    send(&mut socket, update(json!({ "audio": { "input": { "format": { "type": "audio/pcm", "rate": 16000 }, "transcription": { "prompt": "GPUs." } } } }))).await;
    let updated = next_event(&mut socket).await;
    assert_eq!(updated["type"], "session.updated");
    let input = &updated["session"]["audio"]["input"];
    assert_eq!(input["format"]["rate"], 16000);
    assert_eq!(input["transcription"]["prompt"], "GPUs.");
    assert_eq!(input["turn_detection"]["type"], "server_vad", "an update that names none keeps it");
    assert_eq!(updated["session"]["id"], created["session"]["id"]);
    send(&mut socket, realtime::manual()).await;
    assert!(next_event(&mut socket).await["session"]["audio"]["input"]["turn_detection"].is_null());
}

/// AC 3: 24 kHz is resampled to 16 kHz before the executor sees it; G.711 is
/// refused, naming the field, and the session goes on.
#[tokio::test]
async fn twenty_four_khz_reaches_the_executor_at_sixteen_and_g711_is_refused() {
    let executor = saying(&["hi"]);
    let (live, _server, _dir) = served(&executor, |_| {}).await;
    let (mut socket, _) = session(&live).await;
    send(&mut socket, realtime::manual()).await;
    assert_eq!(next_event(&mut socket).await["type"], "session.updated");
    send(&mut socket, update(json!({ "audio": { "input": { "format": { "type": "audio/pcmu" } } } }))).await;
    assert_eq!(error_code(&next_event(&mut socket).await), ("unsupported_value", "session.audio.input.format.type"));
    // Three seconds at 24 kHz.
    send(&mut socket, append(72_000)).await;
    send(&mut socket, commit()).await;
    let events = until(&mut socket, COMPLETED).await;
    assert_eq!(of(&events, COMPLETED)[0]["usage"], json!({ "type": "duration", "seconds": 3.0 }));
    let pushed = executor.samples_pushed() as i64;
    assert!((pushed - 48_000).abs() <= 64, "{pushed} samples at 16 kHz");
}

/// AC 3: at 16 kHz, the declared extension, the samples reach the executor
/// as they are.
#[tokio::test]
async fn sixteen_khz_reaches_the_executor_as_sent() {
    let executor = saying(&["hi"]);
    let (live, _server, _dir) = served(&executor, |_| {}).await;
    let mut socket = session_16k(&live).await;
    send(&mut socket, append(16_000)).await;
    send(&mut socket, append(8_000)).await;
    send(&mut socket, commit()).await;
    let events = until(&mut socket, COMPLETED).await;
    assert_eq!(of(&events, COMPLETED)[0]["usage"]["seconds"], 1.5);
    assert_eq!(executor.samples_pushed(), 24_000);
}

// ── AC 4: append, commit, clear ─────────────────────────────────────────────

/// AC 4: a commit sends `committed` with the item and the one before it,
/// then the transcript; an empty commit is an `error`; a clear sends
/// `cleared` and its audio is never transcribed.
#[tokio::test]
async fn commit_and_clear_answer_as_the_table_says() {
    let executor = saying(&["hello", "there"]);
    let (live, _server, _dir) = served(&executor, |_| {}).await;
    let mut socket = session_16k(&live).await;

    send(&mut socket, commit()).await;
    assert_eq!(error_code(&next_event(&mut socket).await).0, "input_audio_buffer_commit_empty");

    send(&mut socket, append(16_000)).await;
    send(&mut socket, commit()).await;
    let first = until(&mut socket, COMPLETED).await;
    let committed = of(&first, "input_audio_buffer.committed")[0];
    let id = committed["item_id"].as_str().unwrap().to_owned();
    assert!(id.starts_with("item_"), "{committed}");
    assert!(committed["previous_item_id"].is_null());
    assert_eq!(of(&first, COMPLETED)[0]["item_id"], id.as_str());
    assert_eq!(of(&first, COMPLETED)[0]["transcript"], "hello there");
    assert_eq!(first[0]["type"], "input_audio_buffer.committed", "committed comes first: {first:?}");
    until_true("the next item's session opened", || executor.opened.lock().unwrap().len() == 2).await;

    // Audio cleared goes nowhere.
    send(&mut socket, append(32_000)).await;
    send(&mut socket, json!({ "type": "input_audio_buffer.clear" })).await;
    assert_eq!(next_event(&mut socket).await["type"], "input_audio_buffer.cleared");
    send(&mut socket, commit()).await;
    assert_eq!(error_code(&next_event(&mut socket).await).0, "input_audio_buffer_commit_empty", "the buffer was cleared");

    send(&mut socket, append(8_000)).await;
    send(&mut socket, commit()).await;
    let second = until(&mut socket, COMPLETED).await;
    let committed = of(&second, "input_audio_buffer.committed")[0];
    assert_eq!(committed["previous_item_id"], id.as_str());
    assert_ne!(committed["item_id"], id.as_str());
    // The cleared item's executor session was dropped, never committed.
    let commits: Vec<String> = executor.calls().into_iter().filter(|call| call.starts_with("commit")).collect();
    assert_eq!(commits, ["commit 0", "commit 2"], "{:?}", executor.calls());
}

// ── AC 5, 6: stable deltas and the provisional tail ─────────────────────────

/// A dictation of three windows and a bit, a partial after each window: the
/// events of its item, to `completed`.
async fn dictate(socket: &mut Socket, provisional: bool) -> Vec<Value> {
    let mut events = Vec::new();
    for _ in 0..3 {
        send(socket, append(WINDOW)).await;
        // Wait for the partial that window brings.
        loop {
            let event = next_event(socket).await;
            let delta = event["type"] == DELTA;
            events.push(event);
            if delta {
                break;
            }
        }
        if provisional {
            // The tail follows its delta.
            events.push(next_event(socket).await);
        }
    }
    send(socket, append(WINDOW / 4)).await;
    send(socket, commit()).await;
    events.extend(until(socket, COMPLETED).await);
    events
}

/// AC 5: the deltas are stable text only, each extending what came before,
/// and joined they are `completed.transcript`.
#[tokio::test]
async fn the_deltas_are_stable_text_and_join_to_the_transcript() {
    let executor = saying(&["uno", "due", "tre", "quattro", "cinque"]);
    let (live, _server, _dir) = served(&executor, |_| {}).await;
    let mut socket = session_16k(&live).await;
    let events = dictate(&mut socket, false).await;
    let deltas: Vec<&str> = of(&events, DELTA).iter().map(|event| event["delta"].as_str().unwrap()).collect();
    // One stable word per window: the tail word never goes out as a delta
    // before it is stable.
    assert_eq!(deltas, ["uno", " due", " tre", " quattro cinque"], "{events:?}");
    let completed = of(&events, COMPLETED)[0];
    assert_eq!(deltas.concat(), completed["transcript"].as_str().unwrap());
    assert_eq!(completed["content_index"], 0);
    assert!(of(&events, DELTA).iter().all(|delta| delta["item_id"] == completed["item_id"] && delta["content_index"] == 0));
    assert!(of(&events, PROVISIONAL).is_empty(), "a session that did not opt in never gets the tail");
}

/// AC 6: a session that opted in gets the whole current tail after each
/// partial, and an empty one at completion; one that did not, never.
#[tokio::test]
async fn the_provisional_tail_reaches_only_a_session_that_opted_in() {
    let executor = saying(&["uno", "due", "tre", "quattro", "cinque"]);
    let (live, _server, _dir) = served(&executor, |_| {}).await;
    let mut socket = session_16k(&live).await;
    send(&mut socket, update(json!({ "ignis": { "provisional_tail": true } }))).await;
    assert_eq!(next_event(&mut socket).await["session"]["ignis"]["provisional_tail"], true);
    let events = dictate(&mut socket, true).await;
    let tails: Vec<&str> = of(&events, PROVISIONAL).iter().map(|event| event["text"].as_str().unwrap()).collect();
    assert_eq!(tails, [" due", " tre", " quattro", ""], "{events:?}");
    let completed = of(&events, COMPLETED)[0];
    assert!(of(&events, PROVISIONAL).iter().all(|tail| tail["item_id"] == completed["item_id"]));
    let deltas: String = of(&events, DELTA).iter().map(|event| event["delta"].as_str().unwrap()).collect();
    assert_eq!(deltas, completed["transcript"].as_str().unwrap(), "the tail is never part of the deltas");
}

/// AC 18 (Q74): refreshes are best effort. With the worker behind, none is
/// queued while the socket's windows are in it, so the windows run back to
/// back; once the worker is free the next refresh runs; the commit is never
/// held behind one, and the deltas still join to the transcript.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refreshes_are_skipped_while_the_worker_is_behind() {
    let executor = saying(&["uno", "due", "tre", "quattro"]);
    executor.set_step_delay(Duration::from_millis(250));
    let (live, _server, _dir) = served(&executor, |_| {}).await;
    let mut socket = session_16k(&live).await;
    for _ in 0..3 {
        send(&mut socket, append(WINDOW)).await;
    }
    // Three windows of 250 ms each: some seven refreshes fall due meanwhile.
    let refreshed = until(&mut socket, DELTA).await;
    assert_eq!(of(&refreshed, DELTA)[0]["delta"], "uno due tre", "the one refresh ran after the windows");
    // No audio since that refresh: none more falls due before the commit.
    send(&mut socket, commit()).await;
    let events = until(&mut socket, COMPLETED).await;
    let calls: Vec<String> = executor.calls().into_iter().filter(|call| !call.starts_with("open")).collect();
    assert_eq!(calls, ["push 0 128000", "push 0 128000", "push 0 128000", "partial 0", "commit 0"]);
    let deltas: String = of(&refreshed, DELTA).iter().chain(&of(&events, DELTA)).map(|event| event["delta"].as_str().unwrap()).collect();
    assert_eq!(deltas, of(&events, COMPLETED)[0]["transcript"].as_str().unwrap());
}

/// Q76: after a refresh the next waits `speech.partial_cost_factor` times its
/// compute time, from when it was queued, beyond `speech.partial_interval_ms`.
/// Audio arrives all along and never fills a window, so every unit is a
/// refresh, and each takes at least 200 ms on the worker (a 100 ms push of
/// the audio so far, a 100 ms partial). At factor 3 every refresh begins at
/// least 600 ms after the one before, though the 100 ms interval and the free
/// worker would allow one as soon as the last ends; at factor 1, today's
/// rule, one follows as soon as the last ends.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refresh_waits_the_factor_times_the_last_ones_compute_time() {
    let mut gaps = Vec::new();
    for factor in [3, 1] {
        let executor = saying(&["uno", "due"]);
        executor.set_step_delay(Duration::from_millis(100));
        let (live, _server, _dir) = served(&executor, |group| group.partial_cost_factor = factor).await;
        let mut socket = session_16k(&live).await;
        // Three seconds, 100 ms at a time.
        for _ in 0..30 {
            send(&mut socket, append(1_600)).await;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let began = executor.partials_at();
        assert!(began.len() >= 3, "factor {factor}: {} refreshes", began.len());
        gaps.push(began.windows(2).map(|pair| pair[1] - pair[0]).collect::<Vec<_>>());
    }
    let (held, today) = (&gaps[0], &gaps[1]);
    // 3 × 200 ms, less the scheduling slack between two partials' starts.
    let bound = Duration::from_millis(550);
    assert!(held.iter().all(|gap| *gap >= bound), "a refresh began sooner than 3 times the last one's compute time: {held:?}");
    assert!(today.iter().any(|gap| *gap < bound), "at factor 1 a refresh follows the last at once: {today:?}");
}

// ── AC 7: usage and failures ────────────────────────────────────────────────

/// AC 7: an item over `speech.max_clip_seconds` gets an `error` naming the
/// limit at once and `failed` at its commit; one whose transcription fails
/// gets `failed` with the error; the session goes on serving the next.
#[tokio::test]
async fn an_item_over_the_limit_or_failing_gets_failed() {
    let executor = saying(&["hi"]);
    let (live, _server, _dir) = served(&executor, |group| group.max_clip_seconds = 2).await;
    let mut socket = session_16k(&live).await;
    send(&mut socket, append(48_000)).await;
    let at_once = next_event(&mut socket).await;
    assert_eq!(at_once["type"], "error", "{at_once}");
    assert!(at_once["error"]["message"].as_str().unwrap().contains("speech.max_clip_seconds"), "{at_once}");
    send(&mut socket, commit()).await;
    let events = until(&mut socket, FAILED).await;
    let committed = of(&events, "input_audio_buffer.committed")[0];
    let failed = of(&events, FAILED)[0];
    assert_eq!(failed["item_id"], committed["item_id"]);
    let message = failed["error"]["message"].as_str().unwrap();
    assert!(message.contains("speech.max_clip_seconds") && message.contains("2 s"), "{message}");
    assert!(executor.samples_pushed() < 48_000, "an item over the limit is not transcribed on");

    // Read at the commit, so the item's session already open fails too.
    executor.fail_commit.store(true, Ordering::SeqCst);
    send(&mut socket, append(8_000)).await;
    send(&mut socket, commit()).await;
    let events = until(&mut socket, FAILED).await;
    assert_eq!(of(&events, FAILED)[0]["error"]["code"], "engine_error");

    executor.fail_commit.store(false, Ordering::SeqCst);
    send(&mut socket, append(8_000)).await;
    send(&mut socket, commit()).await;
    let events = until(&mut socket, COMPLETED).await;
    assert_eq!(of(&events, COMPLETED)[0]["usage"], json!({ "type": "duration", "seconds": 0.5 }));
}

/// `server.stream_timeout` holds per item, from its first append (ADR 0058):
/// past it the item fails `request_timeout`.
#[tokio::test]
async fn an_item_past_the_stream_timeout_fails() {
    let executor = saying(&["hi"]);
    let (server, _dir) = speech::server(Arc::clone(&executor), |group| group.partial_interval_ms = 100);
    let server = server.with_stream_timeout(Duration::from_secs(1));
    let live = live(server).await;
    let mut socket = session_16k(&live).await;
    send(&mut socket, append(8_000)).await;
    tokio::time::sleep(Duration::from_millis(1_300)).await;
    send(&mut socket, commit()).await;
    let events = until(&mut socket, FAILED).await;
    assert_eq!(events[0]["type"], "error", "the deadline said so when it passed: {events:?}");
    assert_eq!(events[0]["error"]["code"], "request_timeout");
    assert_eq!(of(&events, FAILED)[0]["error"]["code"], "request_timeout");
    assert!(of(&events, FAILED)[0]["error"]["message"].as_str().unwrap().contains("--server-stream-timeout"));
}

// ── AC 8, 10, 11: every other field and event ───────────────────────────────

/// AC 8; spec 05's AC 5: `semantic_vad` is refused, the retired beta's
/// event names `session.update`, conversation events are refused, and none
/// of it closes the socket.
#[tokio::test]
async fn semantic_vad_the_beta_and_conversation_events_are_refused() {
    let executor = saying(&["hi"]);
    let (live, _server, _dir) = served(&executor, |_| {}).await;
    let mut socket = session_16k(&live).await;
    send(&mut socket, update(json!({ "audio": { "input": { "turn_detection": { "type": "semantic_vad" } } } }))).await;
    assert_eq!(error_code(&next_event(&mut socket).await), ("unsupported_value", "session.audio.input.turn_detection"));
    send(&mut socket, json!({ "type": "transcription_session.update", "event_id": "evt_7", "session": {} })).await;
    let refused = next_event(&mut socket).await;
    assert!(refused["error"]["message"].as_str().unwrap().contains("session.update"), "{refused}");
    assert_eq!(refused["error"]["event_id"], "evt_7", "the error names the event it refuses");
    for kind in ["response.create", "conversation.item.create", "output_audio_buffer.clear"] {
        send(&mut socket, json!({ "type": kind })).await;
        let refused = next_event(&mut socket).await;
        assert!(refused["error"]["message"].as_str().unwrap().contains("conversation"), "{kind}: {refused}");
    }
    send(&mut socket, json!({ "type": "session.update", "session": { "include": ["item.input_audio_transcription.logprobs"] } })).await;
    assert_eq!(error_code(&next_event(&mut socket).await), ("unsupported_value", "session.include"));
    send(&mut socket, update(json!({ "audio": { "input": { "noise_reduction": { "type": "near_field" } } } }))).await;
    assert_eq!(error_code(&next_event(&mut socket).await).1, "session.audio.input.noise_reduction");
    // A refused update changed nothing, and the socket still serves.
    send(&mut socket, append(16_000)).await;
    send(&mut socket, commit()).await;
    assert_eq!(of(&until(&mut socket, COMPLETED).await, COMPLETED)[0]["usage"]["seconds"], 1.0);
}

/// AC 10, 11: `transcription.model` follows spec 02's resolution, and
/// `language` forces the language only when asked.
#[tokio::test]
async fn the_model_and_the_language_follow_spec_02() {
    let executor = saying(&["hi"]);
    let (live, _server, _dir) = served(&executor, |_| {}).await;
    let (mut socket, _) = session(&live).await;
    until_true("the session opened on the executor", || executor.opened.lock().unwrap().len() == 1).await;
    send(&mut socket, update(json!({ "audio": { "input": { "transcription": { "model": TEXT_MODEL } } } }))).await;
    let refused = next_event(&mut socket).await;
    assert_eq!(error_code(&refused), ("model_kind_mismatch", "session.audio.input.transcription.model"));
    assert!(refused["error"]["message"].as_str().unwrap().contains("a text model"), "{refused}");
    send(&mut socket, update(json!({ "audio": { "input": { "transcription": { "model": "no-such-model" } } } }))).await;
    assert_eq!(error_code(&next_event(&mut socket).await).0, "model_not_found");
    send(&mut socket, update(json!({ "audio": { "input": { "transcription": { "model": "whisper-1", "language": "it" } } } }))).await;
    let updated = next_event(&mut socket).await;
    assert_eq!(updated["session"]["audio"]["input"]["transcription"]["model"], SPEECH_MODEL);
    until_true("the update reopened the item's session", || executor.opened.lock().unwrap().len() == 2).await;
    let opened = executor.opened.lock().unwrap().clone();
    assert_eq!(opened[0].language, None, "nothing is forced unless asked");
    assert_eq!(opened[1].language.as_deref(), Some("Italian"));
}

/// `speech.rollback_tokens` and `speech.unfixed_partials` reach the executor
/// when a session opens: a session opened after a live change of the config
/// opens with the new values, and the one open before keeps its own.
#[tokio::test]
async fn a_session_opened_after_a_live_change_opens_with_the_new_rollback_and_unfixed_partials() {
    let executor = saying(&["hi"]);
    let (live, server, _dir) = served(&executor, |_| {}).await;
    let _first = session_16k(&live).await;
    until_true("the first session opened", || executor.opened.lock().unwrap().len() == 1).await;
    server.update_live(|live| {
        live.speech.rollback_tokens = 3;
        live.speech.unfixed_partials = 0;
    });
    let mut second = session_16k(&live).await;
    until_true("the second session opened", || executor.opened.lock().unwrap().len() == 2).await;
    // An update reopens the item's session, which reads the config again.
    server.update_live(|live| live.speech.rollback_tokens = 7);
    send(&mut second, update(json!({ "audio": { "input": { "transcription": { "prompt": "GPUs." } } } }))).await;
    assert_eq!(next_event(&mut second).await["type"], "session.updated");
    until_true("the update reopened the item's session", || executor.opened.lock().unwrap().len() == 3).await;
    let opened = executor.opened.lock().unwrap().clone();
    let tuning = |i: usize| (opened[i].rollback_tokens, opened[i].unfixed_partials);
    assert_eq!(tuning(0), (10, 2), "the defaults, before the change");
    assert_eq!(tuning(1), (3, 0), "the values in force when the session opened");
    assert_eq!(tuning(2), (7, 0), "an update reopens with the values in force then");
}

// ── AC 9: the context, prefilled when the session opens ─────────────────────

/// AC 9: the context is prefilled when the session opens and when an update
/// changes it; a commit's work opens nothing, and the next item opens on the
/// same context, the prefix cache's entry.
#[tokio::test]
async fn the_context_is_prefilled_at_open_and_at_update_never_by_a_commit() {
    let executor = saying(&["hi"]);
    let (live, _server, _dir) = served(&executor, |group| group.partial_interval_ms = 60_000).await;
    let mut socket = session_16k(&live).await;
    until_true("the session opened on the executor", || executor.calls().len() == 1).await;
    assert_eq!(executor.calls(), ["open 0 "], "opened with no context before any audio");
    send(&mut socket, update(json!({ "audio": { "input": { "transcription": { "prompt": "Talk.", "keywords": ["ignis"] } } } }))).await;
    assert_eq!(next_event(&mut socket).await["type"], "session.updated");
    until_true("the update prefilled the context", || executor.calls().len() == 2).await;
    assert_eq!(executor.calls()[1], "open 1 Talk.\nignis", "prefilled when the update arrived, before any audio");

    send(&mut socket, append(16_000)).await;
    send(&mut socket, commit()).await;
    until(&mut socket, COMPLETED).await;
    until_true("the next item opened", || executor.calls().len() == 5).await;
    // The commit pushed what was left and decoded, opening nothing; the next
    // item opens after it, on the same context.
    assert_eq!(executor.calls()[2..], ["push 1 16000", "commit 1", "open 2 Talk.\nignis"]);
}

// ── AC 12: loading ──────────────────────────────────────────────────────────

/// AC 12: opening a session loads the speech model if it is not loaded,
/// before any audio.
#[tokio::test]
async fn opening_a_session_loads_the_speech_model() {
    let executor = saying(&["hi"]);
    let (live, _server, _dir) = served(&executor, |_| {}).await;
    assert!(!executor.loaded.load(Ordering::SeqCst));
    let (_socket, _) = session(&live).await;
    until_true("the speech model loaded", || executor.loaded.load(Ordering::SeqCst)).await;
    assert_eq!(executor.loads.load(Ordering::SeqCst), 1);
    let (_second, _) = session(&live).await;
    until_true("the second session opened", || executor.opened.lock().unwrap().len() == 2).await;
    assert_eq!(executor.loads.load(Ordering::SeqCst), 1, "loaded once");
}

/// Wait (bounded) until `server` reports `wanted`.
async fn status(server: &Server, wanted: &str) {
    until_true(&format!("the status reads {wanted}"), || server.status().as_str() == wanted).await;
}

/// AC 12 (Q40): opening a session starts loading a text model the idle
/// clock unloaded, in the background, and the session is served on the CPU
/// while that load runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn opening_a_session_starts_loading_an_unloaded_text_model_and_serves_meanwhile() {
    let loader = MockLoader::new();
    let dir = speech::temp_dir("realtime-load");
    speech::put_files(&dir, SPEECH_MODEL);
    let executor = saying(&["hi"]);
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, Duration::from_secs(30)))
        .with_idle_unload(Duration::from_secs(1))
        .with_speech(Arc::new(Speech::new(executor.clone(), Some(speech::models(&dir)))))
        .with_speech_group(SpeechGroup { partial_interval_ms: 100, ..SpeechGroup::default() });
    server.usage.touch();
    // The serve loop runs the server's own idle clock.
    let live = live(server.clone()).await;
    for _ in 0..3 {
        if server.status().as_str() == "unloaded" {
            break;
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    status(&server, "unloaded").await;
    server.update_live(|live| live.idle_unload = Duration::ZERO);

    let release = loader.hold_next_load();
    let mut socket = session_16k(&live).await;
    status(&server, "loading").await;
    send(&mut socket, append(16_000)).await;
    send(&mut socket, commit()).await;
    let events = until(&mut socket, COMPLETED).await;
    assert_eq!(of(&events, COMPLETED)[0]["transcript"], "hi");
    assert_eq!(server.status().as_str(), "loading", "served while the text model loads");
    release.send(()).expect("the held load is waiting");
    status(&server, "serving").await;
    assert_eq!(loader.loads(), ["mock-a"]);
}

/// The metrics a session moves: the open sessions' gauge, and a completed
/// item counted with its wait from the commit.
#[tokio::test(flavor = "multi_thread")]
async fn a_session_and_its_items_are_counted() {
    let executor = saying(&["hi"]);
    let (server, _dir) = speech::server(Arc::clone(&executor), |_| {});
    let server = server.with_metrics();
    let live = live(server.clone()).await;
    let mut socket = session_16k(&live).await;
    let scrape = || async {
        use tower::ServiceExt;
        let request = axum::http::Request::builder().uri("/metrics").body(axum::body::Body::empty()).unwrap();
        String::from_utf8(speech::bytes(server.metrics_app().unwrap().oneshot(request).await.unwrap()).await).unwrap()
    };
    assert!(scrape().await.contains("ignis_speech_sessions_open 1"));
    send(&mut socket, append(16_000)).await;
    send(&mut socket, commit()).await;
    until(&mut socket, COMPLETED).await;
    let text = scrape().await;
    assert!(text.contains("ignis_speech_transcriptions_total{executor=\"cpu\",outcome=\"completed\"} 1"), "{text}");
    assert!(text.contains("ignis_speech_commit_wait_seconds_count 1"), "{text}");
    assert!(text.contains("ignis_speech_audio_seconds_total 1\n"), "{text}");
    drop(socket);
    until_true("the closed session left the gauge", || server.speech.sessions_open() == 0).await;
}

/// An open session keeps the speech model loaded past `speech.idle_unload`,
/// however quiet; the model is unloaded once the session ends.
#[tokio::test]
async fn an_open_session_keeps_the_speech_model_loaded() {
    let executor = saying(&["hi"]);
    let (live, _server, _dir) = served(&executor, |group| group.idle_unload = 1).await;
    let (socket, _) = session(&live).await;
    until_true("the speech model loaded", || executor.loaded.load(Ordering::SeqCst)).await;
    tokio::time::sleep(Duration::from_millis(1_800)).await;
    assert!(executor.loaded.load(Ordering::SeqCst), "the session holds it");
    drop(socket);
    until_true("unloaded after the session ended", || !executor.loaded.load(Ordering::SeqCst)).await;
}

// ── AC 14: sessions, turns, idle close ──────────────────────────────────────

/// AC 14: up to `speech.max_sessions` sessions are open; one more is refused
/// at the upgrade `503` with a `Retry-After` of at most 15 s; a closed one
/// gives its place back.
#[tokio::test]
async fn past_max_sessions_the_upgrade_is_refused_503_with_retry_after() {
    let executor = saying(&["hi"]);
    let (live, server, _dir) = served(&executor, |group| group.max_sessions = 2).await;
    let (first, _) = session(&live).await;
    let (_second, _) = session(&live).await;
    let (status, headers, body) = realtime::refusal(open(&live, "?intent=transcription", &[]).await.unwrap_err());
    assert_eq!(status, 503, "{body}");
    assert_eq!(body["error"]["code"], "engine_full");
    let retry: u64 = headers["retry-after"].to_str().unwrap().parse().unwrap();
    assert!((1..=15).contains(&retry), "{retry}");
    drop(first);
    until_true("the closed session gave its place back", || server.speech.sessions_open() == 1).await;
    let (_third, _) = session(&live).await;
}

/// AC 14: two sessions with windows to push take turns on the worker, window
/// by window.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sessions_take_turns_window_by_window() {
    let executor = saying(&["a", "b", "c", "d"]);
    // Long enough that the second session's frames are queued while the
    // first's opening windows run, on a loaded machine too: the worker's own
    // test proves the turns deterministically.
    executor.set_step_delay(Duration::from_millis(150));
    let (live, _server, _dir) = served(&executor, |group| group.partial_interval_ms = 60_000).await;
    let mut first = session_16k(&live).await;
    let mut second = session_16k(&live).await;
    until_true("both sessions opened", || executor.opened.lock().unwrap().len() >= 2).await;
    for _ in 0..3 {
        send(&mut first, append(WINDOW)).await;
    }
    for _ in 0..3 {
        send(&mut second, append(WINDOW)).await;
    }
    until_true("six windows pushed", || executor.calls().iter().filter(|call| call.starts_with("push")).count() == 6).await;
    let sessions: Vec<String> =
        executor.calls().iter().filter_map(|call| call.strip_prefix("push ")).map(|rest| rest.split(' ').next().unwrap().to_owned()).collect();
    // Each session pushes on its latest-opened executor session.
    let (a, b) = (sessions[0].clone(), sessions.iter().find(|s| **s != sessions[0]).unwrap().clone());
    let turns: Vec<bool> = sessions.iter().map(|s| *s == a).collect();
    let alternations = turns.windows(2).filter(|pair| pair[0] != pair[1]).count();
    assert!(alternations >= 3, "the sessions took turns: {sessions:?} ({a}, {b})");
}

/// AC 14: a session idle past `server.socket_idle_close` is closed with a
/// normal close frame saying why.
#[tokio::test]
async fn an_idle_session_is_closed() {
    let executor = saying(&["hi"]);
    let (server, _dir) = speech::server(Arc::clone(&executor), |_| {});
    let live = live(server.with_socket_idle_close(Duration::from_secs(1))).await;
    let (mut socket, _) = session(&live).await;
    use futures_util::StreamExt;
    let frame = tokio::time::timeout(Duration::from_secs(10), socket.next()).await.expect("closed within 10 s");
    match frame {
        Some(Ok(tokio_tungstenite::tungstenite::Message::Close(Some(close)))) => {
            assert_eq!(u16::from(close.code), 1000);
            assert!(close.reason.contains("socket_idle_close"), "{}", close.reason);
        }
        other => panic!("expected a close frame, got {other:?}"),
    }
}

// ── the serving gate ────────────────────────────────────────────────────────

/// Q40: the socket opens and serves through a switch, a warm-up and an
/// unloaded text model, and an open session is no use of the text model.
#[tokio::test]
async fn sessions_are_served_through_a_switch_a_warm_up_and_an_unload() {
    for status in [
        ModelStatus::Switching { from: TEXT_MODEL.to_owned(), to: "another".to_owned() },
        ModelStatus::WarmingUp,
        ModelStatus::Unloaded,
    ] {
        let executor = saying(&["hi"]);
        let (server, _dir) = speech::server(Arc::clone(&executor), |_| {});
        server.status.store(Arc::new(status.clone()));
        let live = live(server.clone()).await;
        let mut socket = session_16k(&live).await;
        send(&mut socket, append(16_000)).await;
        send(&mut socket, commit()).await;
        let events = until(&mut socket, COMPLETED).await;
        assert_eq!(of(&events, COMPLETED)[0]["transcript"], "hi", "{status:?}");
        assert_eq!(server.usage.in_flight(), 0, "{status:?}: a session is no use of the text model");
    }
}

/// An unavailable speech model refuses the upgrade `503 speech_unavailable`
/// with its reason and no `Retry-After`.
#[tokio::test]
async fn an_unavailable_speech_model_refuses_the_upgrade() {
    let (server, _dir) = speech::server(saying(&["hi"]), |group| group.model = "off".to_owned());
    let live = live(server).await;
    let (status, headers, body) = realtime::refusal(open(&live, "?intent=transcription", &[]).await.unwrap_err());
    assert_eq!(status, 503);
    assert_eq!(body["error"]["code"], "speech_unavailable");
    assert!(body["error"]["message"].as_str().unwrap().contains("turned off"), "{body}");
    assert!(!headers.contains_key("retry-after"));
}

/// A plain `GET /v1/realtime{query}`, no upgrade: the status, headers and JSON of the answer.
async fn plain_get(server: &Server, query: &str) -> (u16, axum::http::HeaderMap, Value) {
    use tower::ServiceExt;
    let request = axum::http::Request::builder().uri(format!("/v1/realtime{query}")).body(axum::body::Body::empty()).unwrap();
    let response = server.app().oneshot(request).await.unwrap();
    let (status, headers) = (response.status().as_u16(), response.headers().clone());
    let bytes = speech::bytes(response).await;
    (status, headers, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// Spec speech/04 AC 11: a plain `GET` is answered what an upgrade would be, so a
/// page that cannot read a refused upgrade's status can ask why: `503
/// speech_unavailable` with the reason and no `Retry-After`; `503 engine_full`
/// with `Retry-After` past `speech.max_sessions`, claiming no place; and, when
/// the route could open, `400 upgrade_required`. A wrong `intent` is still
/// `400` naming it.
#[tokio::test]
async fn a_plain_get_is_answered_what_an_upgrade_would_be() {
    let executor = saying(&["hi"]);
    let (live, server, _dir) = served(&executor, |group| group.max_sessions = 1).await;
    let (status, headers, body) = plain_get(&server, "?intent=transcription").await;
    assert_eq!((status, body["error"]["code"].as_str()), (400, Some("upgrade_required")), "{body}");
    assert!(!headers.contains_key("retry-after"));
    assert_eq!(plain_get(&server, "").await.2["error"]["param"], "intent");
    assert_eq!(server.speech.sessions_open(), 0, "asking claims no place");

    let (_first, _) = session(&live).await;
    let (status, headers, body) = plain_get(&server, "?intent=transcription").await;
    assert_eq!((status, body["error"]["code"].as_str()), (503, Some("engine_full")), "{body}");
    let retry: u64 = headers["retry-after"].to_str().unwrap().parse().unwrap();
    assert!((1..=15).contains(&retry), "{retry}");
    assert_eq!(server.speech.sessions_open(), 1, "asking claims no place");

    server.update_live(|live| live.speech.model = "off".to_owned());
    let (status, headers, body) = plain_get(&server, "?intent=transcription").await;
    assert_eq!((status, body["error"]["code"].as_str()), (503, Some("speech_unavailable")), "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("turned off"), "{body}");
    assert!(!headers.contains_key("retry-after"), "the model's absence is checked before the sessions");
}
