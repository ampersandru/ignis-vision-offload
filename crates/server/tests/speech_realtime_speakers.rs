//! Speakers, the length guard and the queue limit on `GET
//! /v1/realtime?intent=transcription` (spec speech/08, GitHub #361) over a
//! live listener: the real VAD on a real clip, before a scripted executor
//! ([`FakeExecutor`]) and a diarizer scripted frame by frame
//! ([`FakeDiarizer`]).
//!
//! The clip is the speech oracle's FLEURS fixture, `it_it-1703.wav` (9.06 s
//! of Italian at 16 kHz, 144,960 samples): at the VAD's defaults one turn,
//! from 884 ms (its 300 ms of padding before the speech at 1,184 ms) to
//! 8,512 ms, samples 14,144 to 136,192. Inside it Silero hears one 32 ms
//! frame under 0.6, its frame 192, which ends at 6,176 ms (sample 98,816);
//! at 0.5 none. The diarizer's frames are 80 ms, 1,280 samples, from the
//! session's first; at the live geometry its first step scores frames 0 to
//! 24 once 29 frames are heard, and each later one the next 25.

use std::sync::Arc;
use std::time::Duration;

use ignis_server::config::schema::SpeechGroup;
use ignis_server::speech::executor::{FakeDiarizer, FakeExecutor};
use serde_json::{json, Value};

#[path = "support/mod.rs"]
mod support;
use support::realtime::{self, append, at_16_khz, base64, commit, live, next_event, of, send, session, until, until_true, update, Socket, WINDOW};
use support::speech;

const DELTA: &str = "conversation.item.input_audio_transcription.delta";
const COMPLETED: &str = "conversation.item.input_audio_transcription.completed";
const STARTED: &str = "input_audio_buffer.speech_started";
const STOPPED: &str = "input_audio_buffer.speech_stopped";
const COMMITTED: &str = "input_audio_buffer.committed";
const SPEAKER: &str = "ignis.input_audio_transcription.speaker";
const PROVISIONAL: &str = "ignis.input_audio_transcription.provisional";

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

/// The clip's samples `from..to`, sent 100 ms at a time.
async fn send_clip(socket: &mut Socket, from: usize, to: usize) {
    for piece in clip()[2 * from..2 * to].chunks(3_200) {
        send(socket, json!({ "type": "input_audio_buffer.append", "audio": base64(piece) })).await;
    }
}

/// The zero samples the executor is fed before each `server_vad` item.
fn lead() -> usize {
    SpeechGroup::default().vad_leading_silence_ms as usize * 16
}

/// The events' types, in order.
fn kinds(events: &[&Value]) -> Vec<String> {
    events.iter().map(|event| event["type"].as_str().unwrap().to_owned()).collect()
}

/// The events of item `id`, in order.
fn of_item<'a>(events: &'a [Value], id: &str) -> Vec<&'a Value> {
    events.iter().filter(|event| event["item_id"] == id).collect()
}

/// The events up to the `n`th of type `kind`, after `before`.
async fn until_nth(socket: &mut Socket, kind: &str, n: usize, mut before: Vec<Value>) -> Vec<Value> {
    while of(&before, kind).len() < n {
        before.push(next_event(socket).await);
    }
    before
}

/// A diarizer that hears `speaker` in each frame of the clip's speech, frame
/// 15 (1,200 ms) to 99, and no one else.
fn hearing(speaker: impl Fn(usize) -> usize + Send + Sync + 'static) -> Arc<FakeDiarizer> {
    Arc::new(FakeDiarizer::hearing(move |frame| (15..100).contains(&frame).then(|| speaker(frame))))
}

/// A served session at 16 kHz with `session` (`ignis` and the input's
/// fields) applied, its `session.updated` read. No refresh falls due unless
/// `edit` asks for one, and the ghost rule is off: every voice is a speaker
/// at its first frame.
async fn opened(
    executor: &Arc<FakeExecutor>,
    diarizer: &Arc<FakeDiarizer>,
    session_fields: Value,
    edit: impl FnOnce(&mut SpeechGroup),
) -> (Socket, Value, realtime::Live, std::path::PathBuf) {
    let (server, dir) = speech::server_with_diarizer(
        Arc::clone(executor),
        Arc::clone(diarizer),
        |group| {
            group.partial_interval_ms = 60_000;
            edit(group);
        },
        |group| group.min_speaker_seconds = 0,
    );
    let live = live(server).await;
    let (mut socket, _) = session(&live).await;
    let mut fields = session_fields;
    fields["audio"]["input"]["format"] = json!({ "type": "audio/pcm", "rate": 16000 });
    send(&mut socket, update(fields)).await;
    let updated = next_event(&mut socket).await;
    assert_eq!(updated["type"], "session.updated", "{updated}");
    (socket, updated, live, dir)
}

/// The samples the executor's session `index` was pushed.
fn pushed_to(executor: &FakeExecutor, index: usize) -> usize {
    let prefix = format!("push {index} ");
    executor.calls().iter().filter_map(|call| call.strip_prefix(&prefix)).map(|n| n.parse::<usize>().unwrap()).sum()
}

/// AC 6, 7, 8: speaker 0 speaks until frame 50 (4,000 ms), speaker 1 from
/// it. Once the diarizer's frames show the change, the item ends there, as
/// the VAD ends one at a silence: `speech_stopped` at 4,000 ms, `committed`,
/// then `speech_started` at 4,000 ms for the next item, which has no prefix
/// padding. The first item's session is cut at its audio up to the change
/// (the leading silence, then 49,856 samples) and decoded again from it; the
/// next is fed the leading silence, then the audio from the change on, which
/// was heard already. Each item's one delta comes at its end, then its
/// speaker, then `completed`; the second's frames over its end come with no
/// more audio, from the stream's copy finished.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_speaker_change_ends_the_item_at_the_change() {
    let executor = Arc::new(FakeExecutor::saying(&["ciao"]));
    let diarizer = hearing(|frame| usize::from(frame >= 50));
    let (mut socket, updated, _live, _dir) = opened(&executor, &diarizer, json!({ "ignis": { "diarization": "live" } }), |_| {}).await;
    assert_eq!(updated["session"]["ignis"]["diarization"], "live");
    // Up to 6.875 s: the third step, frames 50 to 74, needs 79 frames heard.
    send_clip(&mut socket, 0, 110_000).await;
    let events = until_nth(&mut socket, STARTED, 2, Vec::new()).await;
    send_clip(&mut socket, 110_000, 144_960).await;
    let events = until_nth(&mut socket, COMPLETED, 2, events).await;
    let started = of(&events, STARTED);
    let (first, second) = (started[0]["item_id"].as_str().unwrap(), started[1]["item_id"].as_str().unwrap());
    let first_events = of_item(&events, first);
    assert_eq!(kinds(&first_events), [STARTED, STOPPED, COMMITTED, DELTA, SPEAKER, COMPLETED], "{events:?}");
    assert_eq!((first_events[0]["audio_start_ms"].as_u64(), first_events[1]["audio_end_ms"].as_u64()), (Some(884), Some(4_000)));
    let second_events = of_item(&events, second);
    assert_eq!(kinds(&second_events), [STARTED, STOPPED, COMMITTED, DELTA, SPEAKER, COMPLETED], "{events:?}");
    assert_eq!((second_events[0]["audio_start_ms"].as_u64(), second_events[1]["audio_end_ms"].as_u64()), (Some(4_000), Some(8_512)));
    // The speaker event's fields, after the item's last delta and before its end.
    assert_eq!(first_events[4], &json!({ "type": SPEAKER, "event_id": first_events[4]["event_id"], "item_id": first, "content_index": 0, "speaker": 0 }));
    assert_eq!(second_events[4]["speaker"], 1);
    for item in [&first_events, &second_events] {
        assert_eq!(item[3]["delta"], item[5]["transcript"], "one delta, the whole transcript");
    }
    // 884 ms to 4,000 ms, then 4,000 ms to 8,512 ms.
    assert_eq!((first_events[5]["usage"]["seconds"].as_f64(), second_events[5]["usage"]["seconds"].as_f64()), (Some(3.116), Some(4.512)));
    // The first item's session is the one cut; the next opened after it.
    let cut = executor.calls().iter().find_map(|call| call.strip_prefix("cut ").map(str::to_owned)).expect("a cut");
    let (index, at) = cut.split_once(' ').unwrap();
    let index: usize = index.parse().unwrap();
    assert_eq!(at, (lead() + 64_000 - 14_144).to_string());
    assert_eq!(pushed_to(&executor, index), lead() + 64_000 - 14_144, "the first item's audio is up to the change");
    assert_eq!(pushed_to(&executor, index + 1), lead() + 136_192 - 64_000, "the next item's is from the change, after the silence");
    assert_eq!(executor.silent_lead()[index + 1], lead(), "no padding: the speech follows the silence at once");
    // Four steps over 113 frames heard, then the copy finished for the rest.
    let calls = diarizer.calls();
    assert_eq!(calls, ["open 25", "step 0 25", "step 25 25", "step 50 25", "step 75 25", "tail 100 14"]);
}

/// AC 7 (spec 03 AC 5): with speakers, a refresh fixes no text: the tail a
/// session opted in to carries all of the item's text, the stable words
/// with it, and the deltas say nothing until the item ends. Its end sends
/// the whole transcript as one delta, then the speaker, then `completed`.
/// The clip's turn is held open (2 s of silence asked), the client commits
/// it, and the frames over its end come from the stream's copy finished.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn with_speakers_no_delta_goes_before_the_item_ends() {
    let executor = Arc::new(FakeExecutor::saying(&["uno", "due", "tre"]));
    let diarizer = hearing(|_| 0);
    let fields = json!({ "ignis": { "diarization": "live", "provisional_tail": true }, "audio": { "input": { "turn_detection": { "type": "server_vad", "silence_duration_ms": 2_000 } } } });
    let (mut socket, _, _live, _dir) = opened(&executor, &diarizer, fields, |group| group.partial_interval_ms = 100).await;
    send_clip(&mut socket, 0, 144_960).await;
    // The item holds a window (the silence and 122,048 samples): a refresh
    // after it fixes "uno" and leaves " due" provisional.
    let mut events = Vec::new();
    while !of(&events, PROVISIONAL).iter().any(|tail| tail["text"] == "uno due") {
        events.push(next_event(&mut socket).await);
    }
    send(&mut socket, commit()).await;
    let events = until_nth(&mut socket, COMPLETED, 1, events).await;
    let kinds: Vec<String> = events.iter().map(|event| event["type"].as_str().unwrap().to_owned()).filter(|kind| kind != PROVISIONAL).collect();
    assert_eq!(kinds, [STARTED, COMMITTED, DELTA, SPEAKER, COMPLETED], "{events:?}");
    let delta = of(&events, DELTA)[0];
    assert_eq!(delta["delta"], "uno due tre");
    assert_eq!(delta["delta"], of(&events, COMPLETED)[0]["transcript"]);
    assert_eq!(of(&events, PROVISIONAL).last().unwrap()["text"], "", "the tail goes at the end");
    assert_eq!(of(&events, SPEAKER)[0]["speaker"], 0);
    assert!(diarizer.calls().last().unwrap().starts_with("tail"), "{:?}", diarizer.calls());
}

/// A stream that flows needs no tail (spec speech/08): fed at four times real
/// time, 100 ms of audio every 25 ms, the clip and 2 s of silence after it,
/// the item's end (frame 107) waits for the step whose look-ahead ends at
/// 165,120 samples, and the audio brings it: the item has its speaker from
/// the steps, and no tail runs. A tail spends a forward pass and scores frames
/// with no look-ahead.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_flowing_stream_gives_the_speaker_from_its_steps_and_runs_no_tail() {
    let executor = Arc::new(FakeExecutor::saying(&["ciao"]));
    let diarizer = hearing(|_| 0);
    let (mut socket, _, _live, _dir) = opened(&executor, &diarizer, json!({ "ignis": { "diarization": "live" } }), |_| {}).await;
    let mut audio = clip().to_vec();
    audio.resize(2 * 176_960, 0);
    for piece in audio.chunks(3_200) {
        send(&mut socket, json!({ "type": "input_audio_buffer.append", "audio": base64(piece) })).await;
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let events = until(&mut socket, COMPLETED).await;
    assert_eq!(of(&events, SPEAKER)[0]["speaker"], 0, "{events:?}");
    assert_eq!(diarizer.calls(), ["open 25", "step 0 25", "step 25 25", "step 50 25", "step 75 25", "step 100 25"], "no tail");
}

/// A stream that stalls gets the tail (spec speech/08): the clip sent at once
/// and nothing after it, the item's end (frame 107) waits for a step 20,160
/// samples away, which a client at real time would have sent within 1.26 s
/// and its next piece. Once that long has passed with no audio, a copy of the
/// stream finished scores the 14 frames past the last step, and the item has
/// its speaker. A commit after it, with no more audio, makes an item of the
/// silence after the turn (frames 107 to 114), which the tail's frames
/// already cover: it completes, with no speaker since none of them is speech.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stalled_stream_gives_the_speaker_from_a_tail_once_the_next_step_would_have_come() {
    let executor = Arc::new(FakeExecutor::saying(&["ciao"]));
    let diarizer = hearing(|_| 0);
    let (mut socket, _, _live, _dir) = opened(&executor, &diarizer, json!({ "ignis": { "diarization": "live" } }), |_| {}).await;
    send_clip(&mut socket, 0, 144_960).await;
    let stalled = std::time::Instant::now();
    let events = until(&mut socket, COMPLETED).await;
    let waited = stalled.elapsed();
    assert_eq!(of(&events, SPEAKER)[0]["speaker"], 0, "{events:?}");
    assert_eq!(diarizer.calls().last().map(String::as_str), Some("tail 100 14"));
    assert!(waited >= Duration::from_millis(1_200), "the tail waited for the stall: {waited:?}");
    send(&mut socket, commit()).await;
    let events = until(&mut socket, COMPLETED).await;
    assert_eq!(kinds(&events.iter().collect::<Vec<_>>()), [COMMITTED, DELTA, COMPLETED], "{events:?}");
    assert_eq!(diarizer.calls().iter().filter(|call| call.starts_with("tail")).count(), 1, "{:?}", diarizer.calls());
}

/// AC 8: a session that does not opt in gets no speaker, and its deltas are
/// as before: the stable text while the audio arrives.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_session_without_speakers_gets_none() {
    let executor = Arc::new(FakeExecutor::saying(&["ciao"]));
    let diarizer = hearing(|frame| usize::from(frame >= 50));
    let (mut socket, updated, _live, _dir) = opened(&executor, &diarizer, json!({}), |_| {}).await;
    assert_eq!(updated["session"]["ignis"]["diarization"], Value::Null);
    send_clip(&mut socket, 0, 144_960).await;
    let events = until(&mut socket, COMPLETED).await;
    let kinds: Vec<&str> = events.iter().map(|event| event["type"].as_str().unwrap()).collect();
    assert_eq!(kinds, [STARTED, STOPPED, COMMITTED, DELTA, COMPLETED], "{events:?}");
    assert!(diarizer.calls().is_empty(), "the diarizer heard nothing: {:?}", diarizer.calls());
}

/// AC 9: `session.ignis.diarization` is refused on a manual-commit session,
/// naming `turn_detection`; with the diarizer unavailable it is refused with
/// the reason; either way the session stays open, without speakers, and its
/// next update is answered.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn diarization_is_refused_on_manual_commit_and_without_a_diarizer() {
    let executor = Arc::new(FakeExecutor::saying(&["ciao"]));
    let speakers = json!({ "type": "transcription", "ignis": { "diarization": "file" } });
    let refused = |error: &Value, code: &str| {
        assert_eq!((error["type"].as_str(), error["error"]["code"].as_str()), (Some("error"), Some(code)), "{error}");
        assert_eq!(error["error"]["param"], "session.ignis.diarization");
        error["error"]["message"].as_str().unwrap().to_owned()
    };
    // No diarizer on the server.
    let (server, _dir) = speech::server(Arc::clone(&executor), |_| {});
    let served = live(server).await;
    let (mut socket, _) = session(&served).await;
    send(&mut socket, json!({ "type": "session.update", "session": speakers })).await;
    let message = refused(&next_event(&mut socket).await, "ignis_diarizer_unavailable");
    assert!(message.contains("no diarizer is configured"), "{message}");
    send(&mut socket, update(json!({ "ignis": { "provisional_tail": true } }))).await;
    let updated = next_event(&mut socket).await;
    assert_eq!((updated["type"].as_str(), &updated["session"]["ignis"]["diarization"]), (Some("session.updated"), &Value::Null));

    // A diarizer the host cannot run.
    let diarizer = hearing(|_| 0);
    diarizer.set_unavailable(Some("the CPU lacks AVX2, which the diarizer needs"));
    let (mut socket, _, _live, _dir) = opened(&executor, &diarizer, json!({}), |_| {}).await;
    send(&mut socket, json!({ "type": "session.update", "session": speakers })).await;
    assert!(refused(&next_event(&mut socket).await, "ignis_diarizer_unavailable").contains("AVX2"));

    // Manual commit.
    diarizer.set_unavailable(None);
    let (mut socket, _, _live, _dir) = opened(&executor, &diarizer, json!({}), |_| {}).await;
    send(&mut socket, at_16_khz()).await;
    assert_eq!(next_event(&mut socket).await["type"], "session.updated");
    send(&mut socket, json!({ "type": "session.update", "session": speakers })).await;
    assert!(refused(&next_event(&mut socket).await, "invalid_request_error").contains("turn_detection"));
    send(&mut socket, append(1_600)).await;
    send(&mut socket, commit()).await;
    let events = until(&mut socket, COMPLETED).await;
    assert!(of(&events, SPEAKER).is_empty(), "no speakers: {events:?}");
    assert!(diarizer.calls().is_empty());
}

/// AC 12: the length guard. An item past `max_utterance_seconds` ends at the
/// VAD's next frame under its threshold, a pause too short to end it: at 0.6
/// and 3 s, frame 192, at 6,176 ms, the search (5 s) reaching past it; the
/// speech goes on in a new item from there, padded with nothing. With a
/// search of 500 ms and no pause (at 0.5), it ends where the audio is 3.5 s
/// on, the end of the VAD's frame then: 4,384 ms, then 7,904 ms. A session's
/// own `0` turns it off.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_length_guard_ends_a_long_item_at_a_pause_or_past_the_search() {
    let diarizer = hearing(|_| 0);
    for (threshold, search_ms, starts, stops) in [(0.6, 5_000, vec![884, 6_176], vec![6_176, 8_512]), (0.5, 500, vec![884, 4_384, 7_904], vec![4_384, 7_904, 8_512])] {
        let executor = Arc::new(FakeExecutor::saying(&["ciao"]));
        let fields = json!({ "audio": { "input": { "turn_detection": { "type": "server_vad", "threshold": threshold } } } });
        let (mut socket, _, _live, _dir) = opened(&executor, &diarizer, fields, |group| {
            group.max_utterance_seconds = 3;
            group.utterance_cut_search_ms = search_ms;
        })
        .await;
        send_clip(&mut socket, 0, 144_960).await;
        let events = until_nth(&mut socket, COMPLETED, starts.len(), Vec::new()).await;
        let at = |kind: &str, field: &str| of(&events, kind).iter().map(|event| event[field].as_u64().unwrap()).collect::<Vec<_>>();
        assert_eq!((at(STARTED, "audio_start_ms"), at(STOPPED, "audio_end_ms")), (starts.clone(), stops.clone()), "{threshold}: {events:?}");
        assert_eq!(executor.silent_lead()[..starts.len()], vec![lead(); starts.len()], "each item after the leading silence alone");
        let pushed: usize = (0..starts.len()).map(|index| pushed_to(&executor, index)).sum();
        assert_eq!(pushed, starts.len() * lead() + 136_192 - 14_144, "the turn's audio, each sample in one item");
        assert!(!executor.calls().iter().any(|call| call.starts_with("cut")), "a guard's cut takes nothing back");
    }
    let executor = Arc::new(FakeExecutor::saying(&["ciao"]));
    let fields = json!({ "ignis": { "max_utterance_seconds": 0 } });
    let (mut socket, updated, _live, _dir) = opened(&executor, &diarizer, fields, |group| group.max_utterance_seconds = 3).await;
    assert_eq!(updated["session"]["ignis"]["max_utterance_seconds"], 0);
    send_clip(&mut socket, 0, 144_960).await;
    let events = until(&mut socket, COMPLETED).await;
    assert_eq!(of(&events, STOPPED)[0]["audio_end_ms"], 8_512, "the session turned the guard off");
}

/// AC 13: past `speech.max_queued_audio_seconds` of audio not computed yet,
/// an append is refused with `ignis_audio_queue_full`, and the session stays
/// open. A window the slow executor still holds and the second after it are
/// 9 s of the 10 allowed: 2 s more are refused; once the window is computed,
/// the same append is taken, and the item commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_append_past_the_queued_audio_is_refused_and_the_session_stays_open() {
    let executor = Arc::new(FakeExecutor::saying(&["ciao"]));
    executor.set_step_delay(Duration::from_millis(500));
    let diarizer = hearing(|_| 0);
    let (mut socket, _, _live, _dir) = opened(&executor, &diarizer, json!({}), |group| group.max_queued_audio_seconds = 10).await;
    send(&mut socket, at_16_khz()).await;
    assert_eq!(next_event(&mut socket).await["type"], "session.updated");
    send(&mut socket, append(WINDOW + 16_000)).await;
    send(&mut socket, append(32_000)).await;
    let error = next_event(&mut socket).await;
    assert_eq!((error["type"].as_str(), error["error"]["code"].as_str()), (Some("error"), Some("ignis_audio_queue_full")), "{error}");
    assert_eq!(error["error"]["param"], "audio");
    let message = error["error"]["message"].as_str().unwrap();
    assert!(message.contains("speech.max_queued_audio_seconds") && message.contains("9.0 s"), "{message}");
    until_true("the window computed", || executor.samples_pushed() >= WINDOW).await;
    // The socket hears the window is out of the worker.
    tokio::time::sleep(Duration::from_millis(100)).await;
    send(&mut socket, append(32_000)).await;
    send(&mut socket, commit()).await;
    let events = until(&mut socket, COMPLETED).await;
    assert!(of(&events, "error").is_empty(), "{events:?}");
    assert_eq!(executor.samples_pushed(), WINDOW + 16_000 + 32_000, "the refused append went nowhere");
}

/// The queue limit counts each sample once, however many consumers hold it
/// (spec speech/08, Q99): the audio the session holds not computed yet is the
/// larger of the speech model's and the diarizer's. At a file geometry whose
/// first step needs 83 s, the diarizer computes none of the clip, so the
/// session holds its 9.06 s (the speech model holds less: the turn and its
/// leading silence). Under a 10 s limit the whole clip and 0.9 s more are
/// taken; 0.1 s past them is refused, naming the 10.0 s held.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_diarized_session_counts_its_audio_once_against_the_queue_limit() {
    let executor = Arc::new(FakeExecutor::saying(&["ciao"]));
    let diarizer = hearing(|_| 0);
    let (server, _dir) = speech::server_with_diarizer(
        Arc::clone(&executor),
        Arc::clone(&diarizer),
        |group| {
            group.partial_interval_ms = 60_000;
            group.max_queued_audio_seconds = 10;
        },
        |group| group.file_chunk_frames = 1_000,
    );
    let live = live(server).await;
    let (mut socket, _) = session(&live).await;
    send(&mut socket, update(json!({ "audio": { "input": { "format": { "type": "audio/pcm", "rate": 16000 } } }, "ignis": { "diarization": "file" } }))).await;
    assert_eq!(next_event(&mut socket).await["type"], "session.updated");
    send_clip(&mut socket, 0, 144_960).await;
    let silence = |samples: usize, id: &str| json!({ "type": "input_audio_buffer.append", "event_id": id, "audio": base64(&vec![0; 2 * samples]) });
    send(&mut socket, silence(14_400, "fits")).await;
    send(&mut socket, silence(1_600, "past")).await;
    let mut events = Vec::new();
    while of(&events, "error").is_empty() {
        events.push(next_event(&mut socket).await);
    }
    let error = of(&events, "error")[0];
    assert_eq!((error["error"]["code"].as_str(), error["error"]["event_id"].as_str()), (Some("ignis_audio_queue_full"), Some("past")), "{error}");
    let message = error["error"]["message"].as_str().unwrap();
    assert!(message.contains("holds 10.0 s"), "{message}");
}

/// The clip at twice real time in a session with speakers `live`, refreshes
/// due every 100 ms and held back no longer than they take, to its item's
/// `completed`, with what every such run serves: the item's one speaker, every
/// step (frames 0 to 99), the tail over the item's end (frame 107) from what was
/// heard then, and the commit.
async fn a_diarized_clip(executor: &Arc<FakeExecutor>, diarizer: &Arc<FakeDiarizer>) {
    let fields = json!({ "ignis": { "diarization": "live" } });
    let (mut socket, _, _live, _dir) = opened(executor, diarizer, fields, |group| {
        group.partial_interval_ms = 100;
        group.partial_cost_factor = 1;
    })
    .await;
    for piece in clip().chunks(3_200) {
        send(&mut socket, json!({ "type": "input_audio_buffer.append", "audio": base64(piece) })).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let events = until(&mut socket, COMPLETED).await;
    assert_eq!(of(&events, SPEAKER).len(), 1, "{events:?}");
    let calls = diarizer.calls();
    assert_eq!(calls[..5], ["open 25", "step 0 25", "step 25 25", "step 50 25", "step 75 25"], "every step served");
    let tail: usize = calls[5].strip_prefix("tail 100 ").expect("then the tail").parse().unwrap();
    assert!((7..=14).contains(&tail) && calls.len() == 6, "{calls:?}");
    assert!(executor.calls().iter().any(|call| call.starts_with("commit")));
}

/// Spans as milliseconds from `base`, for a failure's message.
fn millis(base: std::time::Instant, spans: &[(std::time::Instant, std::time::Instant)]) -> Vec<(u128, u128)> {
    spans.iter().map(|(from, to)| ((*from - base).as_millis(), (*to - base).as_millis())).collect()
}

/// AC 10 (Q113, amended 2026-10-26): the diarizer computes on a thread of its
/// own, so its steps never wait for the speech model's units and never stop
/// them. Beside refreshes of a second each, a step of 100 ms runs whole inside
/// one, and no refresh is stopped but by the commit; every step, the tail and
/// the commit are served.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_diarizer_step_neither_waits_for_a_refresh_nor_stops_it() {
    let executor = Arc::new(FakeExecutor::saying(&["uno", "due"]));
    executor.set_step_delay(Duration::from_millis(20));
    executor.set_partial_delay(Duration::from_secs(1));
    let diarizer = hearing(|_| 0);
    diarizer.set_step_delay(Duration::from_millis(100));
    a_diarized_clip(&executor, &diarizer).await;
    let (steps, refreshes) = (diarizer.steps_at(), executor.partial_spans());
    let inside = steps.iter().any(|(began, ended)| refreshes.iter().any(|(from, to)| from <= began && ended <= to));
    let base = refreshes[0].0;
    assert!(inside, "no step ran inside a refresh: steps {:?}, refreshes {:?} (ms)", millis(base, &steps), millis(base, &refreshes));
    let stopped = executor.stopped.load(std::sync::atomic::Ordering::SeqCst);
    assert!(stopped <= 1, "{stopped} refreshes stopped: only the commit stops one");
}

/// Spec speech/08, spec speech/03 (Q74): a refresh is skipped while a unit of
/// its session's speech is in the speech worker, never for a diarizer step,
/// which on its own thread is pending almost always while audio flows. Beside
/// steps of a second each, the refreshes due every 100 ms go on: several begin
/// while a step runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pending_diarizer_step_skips_no_refresh() {
    let executor = Arc::new(FakeExecutor::saying(&["uno", "due"]));
    executor.set_step_delay(Duration::from_millis(20));
    let diarizer = hearing(|_| 0);
    diarizer.set_step_delay(Duration::from_secs(1));
    a_diarized_clip(&executor, &diarizer).await;
    let (steps, refreshes) = (diarizer.steps_at(), executor.partial_spans());
    let during = refreshes.iter().filter(|(from, _)| steps.iter().any(|(began, ended)| began <= from && from < ended)).count();
    let base = refreshes[0].0;
    assert!(during >= 5, "{during} refreshes began while a step ran: steps {:?}, refreshes {:?} (ms)", millis(base, &steps), millis(base, &refreshes));
}

/// A diarizer that fails (here, its open) is an `error` saying so, and the
/// session goes on without speakers: its items complete with no speaker.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_diarizer_that_fails_leaves_the_session_without_speakers() {
    let executor = Arc::new(FakeExecutor::saying(&["ciao"]));
    let diarizer = hearing(|_| 0);
    diarizer.fail_open.store(true, std::sync::atomic::Ordering::SeqCst);
    let (mut socket, updated, _live, _dir) = opened(&executor, &diarizer, json!({ "ignis": { "diarization": "live" } }), |_| {}).await;
    assert_eq!(updated["session"]["ignis"]["diarization"], "live");
    let error = next_event(&mut socket).await;
    assert_eq!((error["type"].as_str(), error["error"]["code"].as_str()), (Some("error"), Some("ignis_diarizer_failed")), "{error}");
    assert!(error["error"]["message"].as_str().unwrap().contains("without speakers"), "{error}");
    send_clip(&mut socket, 0, 144_960).await;
    let events = until(&mut socket, COMPLETED).await;
    let kinds: Vec<&str> = events.iter().map(|event| event["type"].as_str().unwrap()).collect();
    assert_eq!(kinds, [STARTED, STOPPED, COMMITTED, DELTA, COMPLETED], "{events:?}");
}
