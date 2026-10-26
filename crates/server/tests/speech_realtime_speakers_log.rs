//! A Realtime session with speakers writes one `info` line when it ends
//! (spec speech/08, Out of Scope: no new metric families): how many speakers
//! its items were given and what a diarizer step cost on its own thread, mean and
//! max, and nothing of the audio or the text.
//!
//! Its own binary, like `speech_realtime_log.rs`: tracing caches a callsite's
//! interest process-wide, so the capture is the process-wide default.

use std::sync::Arc;
use std::time::Duration;

use ignis_logging::MemorySink;
use ignis_server::speech::executor::{FakeDiarizer, FakeExecutor};
use serde_json::{json, Value};
use tracing_subscriber::layer::SubscriberExt;

#[path = "support/mod.rs"]
mod support;
use support::realtime::{base64, live, next_event, send, session, until_true, update};
use support::speech;

const COMPLETED: &str = "conversation.item.input_audio_transcription.completed";

/// The clip's PCM16 bytes (`speech_realtime_speakers.rs`'s FLEURS fixture, 9.06 s of Italian).
fn clip() -> &'static [u8] {
    const WAV: &[u8] = include_bytes!("../../speech/tests/fixtures/oracle/it_it-1703.wav");
    let mut at = 12;
    loop {
        let len = u32::from_le_bytes(WAV[at + 4..at + 8].try_into().unwrap()) as usize;
        if &WAV[at..at + 4] == b"data" {
            return &WAV[at + 8..at + 8 + len];
        }
        at += 8 + len + (len & 1);
    }
}

fn lines(sink: &MemorySink, name: &str) -> Vec<Value> {
    sink.lines()
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| event["event_name"] == name)
        .map(|event| event["attributes"].clone())
        .collect()
}

/// Two voices over the clip, a change at 4,000 ms, four steps of 20 ms and a
/// tail: the line names two speakers, four steps and their mean and max, one
/// tail, and appears when the session ends, not before. A session without
/// speakers writes none.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_session_with_speakers_writes_one_line_of_numbers_when_it_ends() {
    let sink = Arc::new(MemorySink::new());
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())))
        .expect("no other subscriber in this binary");
    let executor = Arc::new(FakeExecutor::saying(&["zanzibar", "quokka"]));
    let diarizer = Arc::new(FakeDiarizer::hearing(|frame| (15..100).contains(&frame).then(|| usize::from(frame >= 50))));
    diarizer.set_step_delay(Duration::from_millis(20));
    let (server, _dir) = speech::server_with_diarizer(Arc::clone(&executor), Arc::clone(&diarizer), |group| group.partial_interval_ms = 60_000, |group| group.min_speaker_seconds = 0);
    let live = live(server).await;

    // A session that asks for no speakers: no line.
    let (mut plain, _) = session(&live).await;
    send(&mut plain, update(json!({ "audio": { "input": { "format": { "type": "audio/pcm", "rate": 16000 } } } }))).await;
    assert_eq!(next_event(&mut plain).await["type"], "session.updated");
    drop(plain);

    let (mut socket, _) = session(&live).await;
    send(&mut socket, update(json!({ "audio": { "input": { "format": { "type": "audio/pcm", "rate": 16000 } } }, "ignis": { "diarization": "live" } }))).await;
    assert_eq!(next_event(&mut socket).await["type"], "session.updated");
    // Up to 6.875 s, the third step's frames heard: the change ends the first item.
    let send_clip = |from: usize, to: usize| clip()[2 * from..2 * to].chunks(3_200).map(|piece| json!({ "type": "input_audio_buffer.append", "audio": base64(piece) })).collect::<Vec<_>>();
    for event in send_clip(0, 110_000) {
        send(&mut socket, event).await;
    }
    let mut started = 0;
    while started < 2 {
        started += usize::from(next_event(&mut socket).await["type"] == "input_audio_buffer.speech_started");
    }
    for event in send_clip(110_000, 144_960) {
        send(&mut socket, event).await;
    }
    let mut completed = 0;
    while completed < 2 {
        completed += usize::from(next_event(&mut socket).await["type"] == COMPLETED);
    }
    assert!(lines(&sink, "ignis.speech.diarization").is_empty(), "the line is the session's end");
    drop(socket);
    until_true("the session's line", || !lines(&sink, "ignis.speech.diarization").is_empty()).await;
    let lines = lines(&sink, "ignis.speech.diarization");
    assert_eq!(lines.len(), 1, "{lines:?}");
    let line = &lines[0];
    assert_eq!((line["speakers"].as_u64(), line["steps"].as_u64()), (Some(2), Some(4)), "{line}");
    // The second item's end, past the steps, came from a tail once the stream stalled.
    assert_eq!(line["tails"].as_u64(), Some(1), "{line}");
    let (mean, max) = (line["step_mean_ms"].as_f64().unwrap(), line["step_max_ms"].as_f64().unwrap());
    assert!(mean >= 20.0 && max >= mean, "a step takes its 20 ms delay: {line}");
    // Both items had their speaker; a frame's probabilities wait at least for their step.
    assert_eq!(line["items"].as_u64(), Some(2), "{line}");
    assert!(line["lag_max_ms"].as_f64().unwrap() >= 20.0, "a frame waits for its step: {line}");
    let all = sink.lines().join("\n");
    assert!(!all.contains("zanzibar") && !all.contains("quokka"), "no text in the log");
}
