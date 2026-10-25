//! A Realtime item's log line (spec speech/03, spec speech/02 §Privacy, Q13,
//! Q45, GitHub #353): each committed item writes one
//! `ignis.speech.transcription` line with its numbers, the wait counted from
//! its commit; a failed one says so; the run's log holds neither the audio
//! nor the transcript nor the context.
//!
//! Its own binary, like `speech_log.rs`: tracing caches a callsite's interest
//! process-wide, and the socket logs from tasks of its own, so the capture is
//! the process-wide default.

use std::sync::Arc;
use std::time::Duration;

use ignis_logging::MemorySink;
use ignis_server::speech::executor::FakeExecutor;
use serde_json::{json, Value};
use tracing_subscriber::layer::SubscriberExt;

#[path = "support/mod.rs"]
mod support;
use support::realtime::{append, at_16_khz, commit, live, next_event, send, session, until, update, WINDOW};
use support::speech::{self, SPEECH_MODEL};

/// Words nothing else in a log would say.
const WORDS: [&str; 3] = ["zanzibar", "quokka", "marmalade"];

fn events(sink: &MemorySink, name: &str) -> Vec<Value> {
    sink.lines()
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| event["event_name"] == name)
        .map(|event| event["attributes"].clone())
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn each_committed_item_writes_one_line_of_numbers_and_no_text_or_audio() {
    let sink = Arc::new(MemorySink::new());
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())))
        .expect("no other subscriber in this binary");
    let executor = Arc::new(FakeExecutor::saying(&WORDS));
    executor.set_step_delay(Duration::from_millis(20));
    let (server, _dir) = speech::server(Arc::clone(&executor), |group| {
        group.partial_interval_ms = 100;
        group.max_clip_seconds = 20;
    });
    let live = live(server).await;
    let (mut socket, _) = session(&live).await;
    send(&mut socket, at_16_khz()).await;
    send(&mut socket, update(json!({ "audio": { "input": { "transcription": { "prompt": "A talk about quokkas.", "keywords": ["Zanzibar"] } } } }))).await;
    send(&mut socket, json!({ "type": "session.update", "session": { "ignis": { "provisional_tail": true } } })).await;
    for _ in 0..3 {
        assert_eq!(next_event(&mut socket).await["type"], "session.updated");
    }
    // A completed item of a window and a half.
    send(&mut socket, append(WINDOW)).await;
    send(&mut socket, append(WINDOW / 2)).await;
    send(&mut socket, commit()).await;
    let completed = until(&mut socket, "conversation.item.input_audio_transcription.completed").await;
    let transcript = completed.last().unwrap()["transcript"].as_str().unwrap().to_owned();
    assert!(WORDS.iter().all(|word| transcript.contains(word)), "the session carries the text: {transcript}");
    // A failed one, over `speech.max_clip_seconds`.
    send(&mut socket, append(WINDOW * 3)).await;
    send(&mut socket, commit()).await;
    until(&mut socket, "conversation.item.input_audio_transcription.failed").await;
    // One the speech model fails: its line, at WARN, carries the executor's words.
    executor.fail_commit.store(true, std::sync::atomic::Ordering::SeqCst);
    send(&mut socket, append(WINDOW / 2)).await;
    send(&mut socket, commit()).await;
    until(&mut socket, "conversation.item.input_audio_transcription.failed").await;
    drop(socket);

    let lines = events(&sink, "ignis.speech.transcription");
    assert_eq!(lines.len(), 3, "one line per committed item: {lines:?}");
    assert_eq!((lines[2]["outcome"].as_str(), lines[2]["reason"].as_str()), (Some("failed"), Some("the fake executor was told to fail")), "{}", lines[2]);
    assert!(lines[1]["reason"].is_null(), "an item over the limit is the client's, with no executor reason: {}", lines[1]);
    assert!(events(&sink, "ignis.speech.failed").is_empty(), "a second line for a failure");
    let done = &lines[0];
    assert!((done["audio_seconds"].as_f64().unwrap() - 12.0).abs() < 0.01, "{done}");
    assert_eq!(done["executor"], "cpu");
    assert_eq!(done["variant"], SPEECH_MODEL);
    assert!(done["x_real_time"].as_f64().unwrap() > 0.0, "{done}");
    // From the commit: the commit's own push and decode, not the dictation.
    assert!(done["wait_ms"].as_u64().unwrap() >= 20 && done["wait_ms"].as_u64().unwrap() < 12_000, "{done}");
    assert_eq!(done["language"], "English");
    assert_eq!(done["outcome"], "completed");
    assert_eq!(lines[1]["outcome"], "failed");
    let everything = sink.lines().join("\n").to_lowercase();
    for word in WORDS.iter().chain(&["quokkas"]) {
        assert!(!everything.contains(word), "`{word}` is in the log:\n{everything}");
    }
    // Nor the appends' base64.
    let audio = append(64)["audio"].as_str().unwrap().to_lowercase();
    assert!(!everything.contains(&audio[..24]), "audio is in the log");
}
