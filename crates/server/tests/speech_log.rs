//! A transcription's log line (spec speech/02 AC 14, 15, GitHub
//! #352): one `ignis.speech.transcription` line with its numbers, no audio
//! and no transcript text in the whole run's log. The metrics are in
//! `speech_http.rs`.
//!
//! Its own binary, like `media_request_log.rs`: tracing caches a callsite's
//! interest process-wide, so a test running beside it without a subscriber
//! can switch the events off before this one captures them.

use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
use ignis_logging::MemorySink;
use ignis_server::speech::executor::FakeExecutor;
use serde_json::Value;
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

#[path = "support/mod.rs"]
mod support;
use support::speech::{self, Part, SPEECH_MODEL};

/// Words nothing else in a log would say.
const WORDS: [&str; 3] = ["zanzibar", "quokka", "marmalade"];

/// A streamed transcription logs from a task of its own, so the capture is
/// the process-wide default: this binary holds one test.
fn captured() -> Arc<MemorySink> {
    let sink = Arc::new(MemorySink::new());
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())))
        .expect("no other subscriber in this binary");
    sink
}

fn events(sink: &MemorySink, name: &str) -> Vec<Value> {
    sink.lines()
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| event["event_name"] == name)
        .map(|event| event["attributes"].clone())
        .collect()
}


/// AC 14, 15: the run's log holds no audio byte and none of the transcript's
/// words nor the prompt's, and each transcription wrote one line with the
/// audio seconds, the executor, the variant, x real time, the wait, the
/// language and the outcome, and a failed one its reason.
#[tokio::test(flavor = "multi_thread")]
async fn a_transcription_writes_one_line_of_numbers_and_no_text_or_audio() {
    let sink = captured();
    let executor = Arc::new(FakeExecutor::saying(&WORDS));
    executor.set_step_delay(Duration::from_millis(20));
    let failing = Arc::clone(&executor);
    let (server, _dir) = speech::server(executor, |_| {});
    let app = server.app();
    for stream in [false, true] {
        let mut parts = vec![
            Part::File("file", speech::WAV),
            Part::Text("prompt", "A talk about quokkas and their marmalade."),
            Part::Text("keywords[]", "Zanzibar"),
        ];
        if stream {
            parts.push(Part::Text("stream", "true"));
        }
        let response = app.clone().oneshot(speech::transcription(&parts)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8(speech::bytes(response).await).unwrap();
        assert!(WORDS.iter().all(|word| body.contains(word)), "the answer carries the text: {body}");
    }
    // A refused one logs nothing of its text either.
    let refused = app.clone().oneshot(speech::transcription(&[Part::File("file", speech::WAV), Part::Text("prompt", "quokka"), Part::Text("temperature", "0.9")])).await.unwrap();
    assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
    // One the executor fails writes its one line too, at WARN, with the
    // executor's own words as its reason.
    failing.fail_commit.store(true, std::sync::atomic::Ordering::SeqCst);
    let failed = app.clone().oneshot(speech::transcription(&[Part::File("file", speech::WAV)])).await.unwrap();
    assert_eq!(failed.status(), StatusCode::INTERNAL_SERVER_ERROR);

    let mut lines = events(&sink, "ignis.speech.transcription");
    assert_eq!(lines.len(), 3, "one line per transcription: {lines:?}");
    let failed = lines.pop().unwrap();
    assert_eq!((failed["outcome"].as_str(), failed["reason"].as_str()), (Some("failed"), Some("the fake executor was told to fail")), "{failed}");
    assert!(events(&sink, "ignis.speech.failed").is_empty(), "a second line for a failure");
    for line in &lines {
        assert!((line["audio_seconds"].as_f64().unwrap() - 1.0).abs() < 0.01, "{line}");
        assert_eq!(line["executor"], "cpu");
        assert_eq!(line["variant"], SPEECH_MODEL);
        assert!(line["x_real_time"].as_f64().unwrap() > 0.0, "{line}");
        assert!(line["wait_ms"].as_u64().unwrap() >= 20, "{line}");
        assert_eq!(line["language"], "English");
        assert_eq!(line["outcome"], "completed");
    }
    let everything = sink.lines().join("\n");
    for word in WORDS.iter().chain(&["quokkas", "RIFF"]) {
        assert!(!everything.to_lowercase().contains(&word.to_lowercase()), "`{word}` is in the log:\n{everything}");
    }
    use base64_free::contains_audio;
    assert!(!contains_audio(&everything), "audio bytes are in the log");
}

/// What "audio in the log" means without a base64 crate: neither a run of the
/// WAV's raw header bytes as text, nor its base64.
mod base64_free {
    pub fn contains_audio(log: &str) -> bool {
        // "RIFF....WAVE" as base64 starts "UklGR"; as hex, "52494646".
        log.contains("UklGR") || log.contains("52494646") || log.contains("WAVEfmt")
    }
}

