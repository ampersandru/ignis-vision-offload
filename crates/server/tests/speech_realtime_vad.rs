//! `server_vad` on `GET /v1/realtime?intent=transcription` (spec speech/05,
//! GitHub #355) over a live listener: the real VAD (Silero's model, ported)
//! on a real clip, before a scripted executor ([`FakeExecutor`]).
//!
//! The clip is the speech oracle's FLEURS fixture, `it_it-1703.wav` (9.06 s
//! of Italian at 16 kHz, 144,960 samples). Silero's own model, recorded in
//! `crates/speech/tests/fixtures/vad/`, hears speech in its 512-sample frames
//! 37 to 249 at the threshold 0.5 (1,184 ms to 8,000 ms), every one at least
//! 0.087 from it, and in frames 38 to 248 at 0.9; the port is within 4e-5 of
//! it, so the turns below are exact. The sessions take 16 kHz, so the samples
//! sent are the VAD's.
//!
//! The executor is fed each `server_vad` item after
//! `speech.vad_leading_silence_ms` of digital silence ([`lead`]), which no
//! event and no `usage` shows: the samples it is pushed count it.

use std::sync::Arc;

use ignis_server::config::schema::SpeechGroup;
use ignis_server::speech::executor::FakeExecutor;
use serde_json::{json, Value};

#[path = "support/mod.rs"]
mod support;
use support::realtime::{self, base64, commit, live, next_event, of, send, session, until, update, Socket};
use support::speech;

const DELTA: &str = "conversation.item.input_audio_transcription.delta";
const COMPLETED: &str = "conversation.item.input_audio_transcription.completed";
const STARTED: &str = "input_audio_buffer.speech_started";
const STOPPED: &str = "input_audio_buffer.speech_stopped";
const COMMITTED: &str = "input_audio_buffer.committed";
const FAILED: &str = "conversation.item.input_audio_transcription.failed";

/// The clip's PCM16 bytes: its `data` chunk.
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

/// The clip's samples before its first that is not zero.
fn clip_silent_start() -> usize {
    clip().chunks_exact(2).take_while(|sample| sample == &[0, 0]).count()
}

/// The zero samples the executor is fed before each `server_vad` item at the
/// default `speech.vad_leading_silence_ms`.
fn lead() -> usize {
    SpeechGroup::default().vad_leading_silence_ms as usize * 16
}

/// The events' types, in order.
fn kinds(events: &[Value]) -> Vec<&str> {
    events.iter().map(|event| event["type"].as_str().unwrap()).collect()
}

/// `samples` 16-bit samples of the clip from `from`, sent 100 ms at a time.
async fn send_clip(socket: &mut Socket, from: usize, samples: usize) {
    for piece in clip()[2 * from..2 * (from + samples)].chunks(3_200) {
        send(socket, json!({ "type": "input_audio_buffer.append", "audio": base64(piece) })).await;
    }
}

/// A session at 16 kHz with `turn_detection` as given (`None`: left as it
/// opens), its `session.updated` read, and the executor behind it. No
/// refresh falls due within a test, so its events are the turns'.
async fn opened(turn_detection: Option<Value>) -> (Socket, Arc<FakeExecutor>, realtime::Live, std::path::PathBuf) {
    opened_with(turn_detection, |_| {}).await
}

/// [`opened`], on a server whose `speech` group `edit` changed.
async fn opened_with(turn_detection: Option<Value>, edit: impl FnOnce(&mut SpeechGroup)) -> (Socket, Arc<FakeExecutor>, realtime::Live, std::path::PathBuf) {
    let executor = Arc::new(FakeExecutor::saying(&["ciao"]));
    let (server, dir) = speech::server(Arc::clone(&executor), |group| {
        group.partial_interval_ms = 60_000;
        edit(group);
    });
    let live = live(server).await;
    let (mut socket, _) = session(&live).await;
    let mut input = json!({ "format": { "type": "audio/pcm", "rate": 16000 } });
    if let Some(detection) = turn_detection {
        input["turn_detection"] = detection;
    }
    send(&mut socket, update(json!({ "audio": { "input": input } }))).await;
    let updated = next_event(&mut socket).await;
    assert_eq!(updated["type"], "session.updated", "{updated}");
    (socket, executor, live, dir)
}

fn vad(fields: Value) -> Option<Value> {
    let mut detection = fields;
    detection["type"] = json!("server_vad");
    Some(detection)
}

/// AC 2, 3, 4: a session that sends no `turn_detection` runs `server_vad` at
/// OpenAI's defaults. The first speech frame starts an item, 300 ms of audio
/// before it kept: `speech_started` at 884 ms (frame 37's 1,184 less 300).
/// Once 500 ms of silence follow the last (frame 249, ending at 8,000 ms), at
/// the end of the frame that makes them 512: `speech_stopped` at 8,512 ms, and
/// the server commits the item itself, holding exactly that audio.
#[tokio::test]
async fn server_vad_hears_the_speech_and_commits_after_its_silence() {
    let (mut socket, executor, _live, _dir) = opened(None).await;
    send_clip(&mut socket, 0, 144_960).await;
    let events = until(&mut socket, COMPLETED).await;
    // The commit's stable text goes out as a delta before `completed` (spec 03).
    assert_eq!(kinds(&events), [STARTED, STOPPED, COMMITTED, DELTA, COMPLETED], "{events:?}");
    let id = events[0]["item_id"].as_str().unwrap();
    assert!(id.starts_with("item_"));
    assert_eq!(events[0]["audio_start_ms"], 884);
    assert_eq!((events[1]["audio_end_ms"].as_u64(), events[1]["item_id"].as_str()), (Some(8_512), Some(id)));
    assert_eq!((events[2]["item_id"].as_str(), &events[2]["previous_item_id"]), (Some(id), &Value::Null));
    assert_eq!(events[4]["item_id"], id);
    assert_eq!(events[4]["transcript"], "ciao");
    // 884 ms to 8,512 ms: samples 14,144 to 136,192.
    assert_eq!(events[4]["usage"]["seconds"], 7.628);
    assert_eq!(executor.samples_pushed(), lead() + 136_192 - 14_144);
}

/// Q77: the model reads each `server_vad` item after
/// `speech.vad_leading_silence_ms` of digital silence, and the client never
/// sees it. At 750 ms the item's session is pushed 12,000 zero samples before
/// the item's first (the clip is not silent at the padded start, sample
/// 14,144), while `audio_start_ms`, `audio_end_ms` and `usage` stay the
/// audio's; with no prefix padding the silence still comes first; an idle
/// timeout's item gets it too; at 0 none is added; a manual commit's item
/// never gets it.
#[tokio::test]
async fn each_server_vad_item_is_fed_after_the_leading_silence() {
    let (mut socket, executor, _live, _dir) = opened_with(None, |group| group.vad_leading_silence_ms = 750).await;
    send_clip(&mut socket, 0, 144_960).await;
    let events = until(&mut socket, COMPLETED).await;
    assert_eq!(kinds(&events), [STARTED, STOPPED, COMMITTED, DELTA, COMPLETED], "{events:?}");
    assert_eq!((events[0]["audio_start_ms"].as_u64(), events[1]["audio_end_ms"].as_u64()), (Some(884), Some(8_512)));
    assert_eq!(events[4]["usage"]["seconds"], 7.628);
    assert_eq!(executor.samples_pushed(), 12_000 + 136_192 - 14_144);
    assert_eq!(executor.silent_lead()[0], 12_000, "the silence comes first, then the item's audio");

    let (mut socket, executor, _live, _dir) = opened_with(vad(json!({ "prefix_padding_ms": 0 })), |group| group.vad_leading_silence_ms = 750).await;
    send_clip(&mut socket, 0, 144_960).await;
    let events = until(&mut socket, COMPLETED).await;
    assert_eq!(of(&events, STARTED)[0]["audio_start_ms"], 1_184);
    assert_eq!(executor.samples_pushed(), 12_000 + 136_192 - 18_944);
    assert_eq!(executor.silent_lead()[0], 12_000);

    let (mut socket, executor, _live, _dir) = opened_with(vad(json!({ "idle_timeout_ms": 1_000 })), |group| group.vad_leading_silence_ms = 750).await;
    send(&mut socket, json!({ "type": "input_audio_buffer.append", "audio": base64(&[0u8; 48_000]) })).await;
    let events = until(&mut socket, COMPLETED).await;
    assert_eq!((events[0]["audio_start_ms"].as_u64(), events[0]["audio_end_ms"].as_u64()), (Some(0), Some(1_024)));
    assert_eq!(executor.samples_pushed(), 12_000 + 16_384);

    let (mut socket, executor, _live, _dir) = opened_with(None, |group| group.vad_leading_silence_ms = 0).await;
    send_clip(&mut socket, 0, 144_960).await;
    until(&mut socket, COMPLETED).await;
    assert_eq!(executor.samples_pushed(), 136_192 - 14_144);
    assert_eq!(executor.silent_lead()[0], 0);

    let (mut socket, executor, _live, _dir) = opened_with(Some(Value::Null), |group| group.vad_leading_silence_ms = 750).await;
    send_clip(&mut socket, 0, 144_960).await;
    send(&mut socket, commit()).await;
    let events = until(&mut socket, COMPLETED).await;
    assert_eq!(events[2]["usage"]["seconds"], 9.06);
    assert_eq!(executor.samples_pushed(), 144_960);
    assert_eq!(executor.silent_lead()[0], clip_silent_start(), "a manual commit's item is fed as sent");
}

/// AC 2: `threshold`, `prefix_padding_ms` and `silence_duration_ms` are each
/// honoured: at 0.9 the speech runs from frame 38 to 248; with no padding the
/// item starts at the speech; with 2 s of silence asked the clip never ends
/// its turn, and a commit then ends it at once, with no `speech_stopped`; at
/// 1.0 nothing is speech, and a commit finds the buffer empty.
#[tokio::test]
async fn the_threshold_the_padding_and_the_silence_are_honoured() {
    for (fields, start, end) in [
        // 38 × 32 - 300; frame 248 ends at 7,968 ms, and 512 ms more.
        (json!({ "threshold": 0.9 }), 916, 8_480),
        (json!({ "prefix_padding_ms": 0 }), 1_184, 8_512),
    ] {
        let (mut socket, _executor, _live, _dir) = opened(vad(fields.clone())).await;
        send_clip(&mut socket, 0, 144_960).await;
        let events = until(&mut socket, COMPLETED).await;
        assert_eq!(of(&events, STARTED)[0]["audio_start_ms"], start, "{fields}");
        assert_eq!(of(&events, STOPPED)[0]["audio_end_ms"], end, "{fields}");
    }

    let (mut socket, executor, _live, _dir) = opened(vad(json!({ "silence_duration_ms": 2_000 }))).await;
    send_clip(&mut socket, 0, 144_960).await;
    send(&mut socket, commit()).await;
    let events = until(&mut socket, COMPLETED).await;
    assert_eq!(kinds(&events), [STARTED, COMMITTED, DELTA, COMPLETED], "{events:?}");
    assert_eq!(executor.samples_pushed(), lead() + 144_960 - 14_144, "the item holds the clip from its padded start to its last sample");

    let (mut socket, executor, _live, _dir) = opened(vad(json!({ "threshold": 1.0 }))).await;
    send_clip(&mut socket, 0, 144_960).await;
    send(&mut socket, commit()).await;
    let refused = next_event(&mut socket).await;
    assert_eq!((refused["type"].as_str(), refused["error"]["code"].as_str()), (Some("error"), Some("input_audio_buffer_commit_empty")), "{refused}");
    assert_eq!(executor.samples_pushed(), 0);
}

/// Q77: the leading silence counts toward `speech.max_clip_seconds`, as the
/// model reads it. The clip's item holds 7.628 s of audio: under a limit of
/// 8 s it is transcribed with no leading silence, and fails with 500 ms of it,
/// the error naming the silence; its times and `usage` stay the audio's.
#[tokio::test]
async fn the_leading_silence_counts_toward_the_clip_limit() {
    let (mut socket, _executor, _live, _dir) = opened_with(None, |group| {
        group.max_clip_seconds = 8;
        group.vad_leading_silence_ms = 0;
    })
    .await;
    send_clip(&mut socket, 0, 144_960).await;
    let events = until(&mut socket, COMPLETED).await;
    assert_eq!(of(&events, COMPLETED)[0]["usage"]["seconds"], 7.628);

    let (mut socket, _executor, _live, _dir) = opened_with(None, |group| {
        group.max_clip_seconds = 8;
        group.vad_leading_silence_ms = 500;
    })
    .await;
    send_clip(&mut socket, 0, 144_960).await;
    let events = until(&mut socket, FAILED).await;
    assert_eq!(kinds(&events), [STARTED, "error", STOPPED, COMMITTED, FAILED], "{events:?}");
    let message = of(&events, FAILED)[0]["error"]["message"].as_str().unwrap();
    assert!(message.contains("speech.max_clip_seconds (8 s)") && message.contains("500 ms of silence"), "{message}");
    assert_eq!((events[0]["audio_start_ms"].as_u64(), events[2]["audio_end_ms"].as_u64()), (Some(884), Some(8_512)));
}

/// A commit during a turn ends its item at once, with no `speech_stopped`; the
/// speech going on starts the next item where the last ended. A clear drops
/// the turn and the padding: the speech after it starts an item there.
#[tokio::test]
async fn a_commit_ends_the_turn_at_once_and_a_clear_drops_it() {
    let (mut socket, executor, _live, _dir) = opened(None).await;
    // Four seconds: 125 frames, so nothing is left waiting for a frame.
    send_clip(&mut socket, 0, 64_000).await;
    send(&mut socket, commit()).await;
    let first = until(&mut socket, COMPLETED).await;
    assert_eq!(kinds(&first), [STARTED, COMMITTED, DELTA, COMPLETED], "{first:?}");
    send_clip(&mut socket, 64_000, 80_960).await;
    let second = until(&mut socket, COMPLETED).await;
    assert_eq!(of(&second, STARTED)[0]["audio_start_ms"], 4_000, "no padding from the item before");
    assert_eq!(of(&second, STOPPED)[0]["audio_end_ms"], 8_512);
    assert_eq!(of(&second, COMMITTED)[0]["previous_item_id"], first[0]["item_id"]);
    assert_eq!(executor.samples_pushed(), 2 * lead() + 136_192 - 14_144, "every sample from the padded start to the stop, once");

    let (mut socket, _executor, _live, _dir) = opened(None).await;
    send_clip(&mut socket, 0, 64_000).await;
    send(&mut socket, json!({ "type": "input_audio_buffer.clear" })).await;
    let cleared = until(&mut socket, "input_audio_buffer.cleared").await;
    assert_eq!(cleared[0]["type"], STARTED, "{cleared:?}");
    send_clip(&mut socket, 64_000, 80_960).await;
    let after = until(&mut socket, COMPLETED).await;
    assert_eq!(of(&after, STARTED)[0]["audio_start_ms"], 4_000);
    assert!(of(&after, COMMITTED)[0]["previous_item_id"].is_null(), "the cleared item was never committed");
}

/// `idle_timeout_ms`: that much audio with no speech is committed as an item
/// of its own after `timeout_triggered`, at the end of the frame that makes it
/// (1,000 ms asked: 32 frames, 1,024 ms); the clock starts again after it.
#[tokio::test]
async fn an_idle_timeout_commits_the_silence() {
    let (mut socket, executor, _live, _dir) = opened(vad(json!({ "idle_timeout_ms": 1_000 }))).await;
    // A second and a half of silence.
    send(&mut socket, json!({ "type": "input_audio_buffer.append", "audio": base64(&[0u8; 48_000]) })).await;
    let events = until(&mut socket, COMPLETED).await;
    assert_eq!(kinds(&events), ["input_audio_buffer.timeout_triggered", COMMITTED, DELTA, COMPLETED], "{events:?}");
    assert_eq!((events[0]["audio_start_ms"].as_u64(), events[0]["audio_end_ms"].as_u64()), (Some(0), Some(1_024)));
    assert_eq!(events[0]["item_id"], events[1]["item_id"]);
    assert_eq!(executor.samples_pushed(), lead() + 16_384);
    assert!(realtime::quiet_for(&mut socket, std::time::Duration::from_millis(300)).await.is_none(), "the next idle second has not passed");
}

/// AC 6: the Playground's session, `turn_detection: null`, hears no VAD: no
/// speech event, every sample in the item it commits.
#[tokio::test]
async fn with_null_the_client_commits_and_every_sample_is_in_the_item() {
    let (mut socket, executor, _live, _dir) = opened(Some(Value::Null)).await;
    send_clip(&mut socket, 0, 144_960).await;
    send(&mut socket, commit()).await;
    let events = until(&mut socket, COMPLETED).await;
    assert_eq!(kinds(&events), [COMMITTED, DELTA, COMPLETED], "{events:?}");
    assert_eq!(events[2]["usage"]["seconds"], 9.06);
    assert_eq!(executor.samples_pushed(), 144_960);
}
