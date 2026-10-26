//! Acceptance 11 of spec speech/08 on the real CPU executor and the real diarizer: a 5-minute
//! conversation of the finding's twenty (`IGNIS_DIARIZER_CONV`, default `conv07`, three speakers)
//! fed through `GET /v1/realtime?intent=transcription` at real-time pace, 100 ms of 16 kHz PCM every
//! 100 ms, with `server_vad` and `session.ignis.diarization: "live"`, Qwen3-ASR `qwen3-asr-1.7b`
//! transcribing every item in the same session. The diarizer keeps up when every frame's speaker
//! probabilities come within the live geometry's latency plus 2 s of the frame's first sample
//! arriving, the speech model computing beside it on its own worker: the session's log line's
//! `lag_max_ms`. (An item's speaker can come later than that by design: a cut at a change reaches
//! back to where the new speaker's run began, up to `diarizer.min_speaker_seconds` before.) It
//! prints each item's time from its `committed` to its `completed`, as the client heard them, with
//! their median and max: what the diarizer costs the commits. `--ignored`: it reads the study disk (`IGNIS_SPEECH_STUDY`,
//! default `H:/speech-study`): the converted Qwen3-ASR pair and the diarizer's GGUF
//! (`IGNIS_DIARIZER_GGUF`, default the F16 one), and fails without them.
//!
//! It is a timing run: nothing else may be using the CPU. Its own binary, since it captures the
//! process's log. Run it alone:
//!
//! ```text
//! cargo test --release -p ignis-server --test speech_diarizer_study -- --ignored --nocapture the_diarizer_keeps_up
//! ```
//!
//! Acceptance 7 is the binary's second leg, on the same models: every conversation of the twenty
//! (`IGNIS_DIARIZER_CONVS`, names separated by commas, to run fewer), each in a session of its own,
//! fed `IGNIS_DIARIZER_PACE` (default 2) times faster than real time with speakers `live`. In such a
//! session no delta comes before its item's `committed`, the deltas of each item joined are its
//! `completed.transcript`, cut items included, and its speaker comes after its last delta and
//! before `completed`. It prints each conversation's items, cuts and speakers. About 95 minutes on
//! the i9 in the test profile, where the worker falls behind a 2x feed: an item's change found
//! only after the item ended splits nothing, so a worker that keeps up should cut more items.
//!
//! ```text
//! cargo test -p ignis-server --test speech_diarizer_study -- --ignored --nocapture every_item_of_the_conversations
//! ```

#[path = "../../speech/tests/common/mod.rs"]
mod common;
#[path = "support/mod.rs"]
mod support;

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::{read_wav, study};
use futures_util::{SinkExt, StreamExt};
use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_logging::MemorySink;
use ignis_server::config::schema::{DiarizerGroup, SpeechGroup};
use ignis_server::download::catalog::Catalog;
use ignis_server::engine::Engine;
use ignis_server::speech::cpu::{CpuDiarizer, CpuExecutor};
use ignis_server::speech::{Speech, SpeechModels};
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::Server;
use serde_json::{json, Value};
use support::realtime::{base64, live, next_event, send, session, until_true, update};
use tracing_subscriber::layer::SubscriberExt;

/// The live geometry's latency (25 + 4 frames of 80 ms) and acceptance 11's 2 s on top of it.
const BOUND_MS: f64 = 2_320.0 + 2_000.0;

fn diarizer_gguf() -> PathBuf {
    let p = std::env::var("IGNIS_DIARIZER_GGUF").map(PathBuf::from).unwrap_or_else(|_| study().join("models").join("diarizer").join("Nemotron-3-Diarization-F16.gguf"));
    assert!(p.is_file(), "{} missing (tools/speech-diarizer/convert.py)", p.display());
    p
}

fn conversation() -> (String, Vec<f32>) {
    let name = std::env::var("IGNIS_DIARIZER_CONV").unwrap_or_else(|_| "conv07".into());
    let path = study().join("nemotron-spike").join("data").join("conv").join(format!("{name}.wav"));
    assert!(path.is_file(), "{} missing", path.display());
    (name, read_wav(&path))
}

/// `speech.*` for the run: the defaults, or `IGNIS_SPEECH_PARTIAL_MS` and
/// `IGNIS_SPEECH_PARTIAL_FACTOR` for `speech.partial_interval_ms` and `speech.partial_cost_factor`,
/// to see what the refreshes cost the diarizer on a shared worker.
fn speech_group() -> SpeechGroup {
    let env = |name: &str, default: u32| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
    let d = SpeechGroup::default();
    SpeechGroup { partial_interval_ms: env("IGNIS_SPEECH_PARTIAL_MS", d.partial_interval_ms), partial_cost_factor: env("IGNIS_SPEECH_PARTIAL_FACTOR", d.partial_cost_factor), ..d }
}

fn pcm16(samples: &[f32]) -> String {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| ((s * 32768.0).round().clamp(-32768.0, 32767.0) as i16).to_le_bytes()).collect();
    base64(&bytes)
}

fn lines(sink: &MemorySink, name: &str) -> Vec<Value> {
    sink.lines()
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| event["event_name"] == name)
        .map(|event| event["attributes"].clone())
        .collect()
}

/// The GPU profile's serialized pass runs every `--ignored` test it does not skip; this leg is the
/// CPU's, on the study disk, so `scripts/gpu-profile.ps1` skips it by name.
#[test]
fn the_gpu_profile_skips_this_leg() {
    let script = include_str!("../../../scripts/gpu-profile.ps1");
    let listed = script.split("$NotGpuLegs = @(").nth(1).and_then(|rest| rest.split(')').next()).expect("gpu-profile.ps1 lists $NotGpuLegs");
    assert!(listed.contains("\"the_diarizer_keeps_up_with_a_conversation_at_real_time_pace\""), "{listed}");
    assert!(listed.contains("\"every_item_of_the_conversations_sends_its_text_whole_at_its_end\""), "{listed}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "a timing run on the study disk's models and conversations"]
async fn the_diarizer_keeps_up_with_a_conversation_at_real_time_pace() {
    let sink = Arc::new(MemorySink::new());
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())))
        .expect("no other subscriber in this binary");
    let (name, audio) = conversation();
    let executor = Arc::new(CpuExecutor::new());
    let dir = study().join("models").join("ignis-conv");
    let scheduler = ConcreteScheduler::with_config(SchedulerConfig { model: "test-model".into(), ..SchedulerConfig::default() }, Arc::new(MockCompute::new()));
    let models = SpeechModels { catalog: Arc::new(Catalog::built_in().clone()), dir };
    let speech = Speech::new(executor.clone(), Some(models)).with_diarizer(Arc::new(CpuDiarizer::new(diarizer_gguf(), false, executor)));
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
        .with_speech(Arc::new(speech))
        .with_speech_group(speech_group())
        .with_diarizer_group(DiarizerGroup::default());
    let live = live(server).await;

    let (mut socket, _) = session(&live).await;
    send(&mut socket, update(json!({ "audio": { "input": { "format": { "type": "audio/pcm", "rate": 16000 }, "turn_detection": { "type": "server_vad" } } }, "ignis": { "diarization": "live" } }))).await;
    assert_eq!(next_event(&mut socket).await["type"], "session.updated");
    let (mut out, mut events) = socket.split();
    let started = Instant::now();
    // Each item's `committed` and `completed` as the client heard them, in the order committed.
    let ends = Arc::new(Mutex::new(Vec::<(String, Instant, Option<Instant>)>::new()));
    let heard = Arc::clone(&ends);
    // Every event is read, so the socket never backs up; a refusal fails the run.
    let reader = tokio::spawn(async move {
        while let Some(Ok(frame)) = events.next().await {
            let tokio_tungstenite::tungstenite::Message::Text(text) = frame else { continue };
            let event: Value = serde_json::from_str(text.as_str()).unwrap();
            let id = event["item_id"].as_str().unwrap_or_default().to_owned();
            match event["type"].as_str() {
                Some("error" | "conversation.item.input_audio_transcription.failed") => panic!("the session refused: {event}"),
                Some("input_audio_buffer.committed") => heard.lock().unwrap().push((id, Instant::now(), None)),
                Some("conversation.item.input_audio_transcription.completed") => {
                    if let Some(item) = heard.lock().unwrap().iter_mut().find(|item| item.0 == id) {
                        item.2 = Some(Instant::now());
                    }
                }
                _ => {}
            }
        }
    });
    let step = 1600;
    for (i, chunk) in audio.chunks(step).enumerate() {
        tokio::time::sleep_until((started + Duration::from_millis(100 * i as u64)).into()).await;
        let event = json!({ "type": "input_audio_buffer.append", "audio": pcm16(chunk) });
        out.send(tokio_tungstenite::tungstenite::Message::Text(event.to_string().into())).await.unwrap();
    }
    // Two seconds of silence after the last word, so the VAD ends the last item, then time for
    // the last transcripts.
    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let event = json!({ "type": "input_audio_buffer.append", "audio": pcm16(&[0.0; 1600]) });
        out.send(tokio_tungstenite::tungstenite::Message::Text(event.to_string().into())).await.unwrap();
    }
    tokio::time::sleep(Duration::from_secs(20)).await;
    drop(out);
    assert!(!reader.is_finished(), "the reader stopped early");
    reader.abort();
    until_true("the session's line", || !lines(&sink, "ignis.speech.diarization").is_empty()).await;
    let line = lines(&sink, "ignis.speech.diarization").remove(0);
    let lag = line["lag_max_ms"].as_f64().unwrap();
    let group = speech_group();
    println!("{name}: {:.0} s of audio, partials every {} ms, cost factor {}, {line}", audio.len() as f64 / 16_000.0, group.partial_interval_ms, group.partial_cost_factor);
    // From each item's `committed` to its `completed` (its transcript and its speaker), as spec
    // speech/03's leg times a dictation's commit: what the diarizer beside it costs the commits.
    let ends = std::mem::take(&mut *ends.lock().unwrap());
    let mut waits: Vec<Duration> = Vec::new();
    for (i, (id, committed, completed)) in ends.iter().enumerate() {
        match completed {
            Some(completed) => {
                waits.push(*completed - *committed);
                println!("{name}: item {i} ({id}): committed to completed {} ms", (*completed - *committed).as_millis());
            }
            None => println!("{name}: item {i} ({id}): committed, never completed"),
        }
    }
    waits.sort();
    if let (Some(median), Some(max)) = (waits.get(waits.len() / 2), waits.last()) {
        println!("{name}: committed to completed over {} items: median {} ms, max {} ms", waits.len(), median.as_millis(), max.as_millis());
    }
    assert!(line["items"].as_u64().unwrap() >= 10, "a conversation's items had their speakers: {line}");
    assert!(lag <= BOUND_MS, "{name}: a frame's probabilities came {lag:.0} ms after its first sample, over {BOUND_MS} ms");
}

/// The conversations acceptance 7 reads: `IGNIS_DIARIZER_CONVS`, else every one on the study disk.
fn conversations() -> Vec<(String, Vec<f32>)> {
    let dir = study().join("nemotron-spike").join("data").join("conv");
    let names: Vec<String> = match std::env::var("IGNIS_DIARIZER_CONVS") {
        Ok(names) => names.split(',').map(|name| name.trim().to_owned()).filter(|name| !name.is_empty()).collect(),
        Err(_) => {
            let mut names: Vec<String> = std::fs::read_dir(&dir)
                .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
                .filter_map(|entry| entry.ok()?.file_name().to_str()?.strip_suffix(".wav").map(str::to_owned))
                .collect();
            names.sort();
            names
        }
    };
    assert!(!names.is_empty(), "no conversation in {}", dir.display());
    names.into_iter().map(|name| {
        let path = dir.join(format!("{name}.wav"));
        assert!(path.is_file(), "{} missing", path.display());
        (name, read_wav(&path))
    }).collect()
}

/// What acceptance 7 checks of one session's events, in the order they came: per item, no delta
/// before its `committed`, the deltas joined equal to `completed.transcript`, the speaker after
/// the last delta and before `completed`. Returns the items, the cuts (a `speech_started` at the
/// very `audio_end_ms` of the `speech_stopped` before it), those of them the length guard made
/// (after an item of `guard_ms`), the speakers heard and the items with none.
fn check_items(name: &str, events: &[Value], guard_ms: u64) -> (usize, usize, usize, BTreeSet<u64>, usize) {
    let at = |kind: &str, id: &str| events.iter().position(|event| event["type"] == kind && event["item_id"] == id);
    let completed: Vec<&Value> = events.iter().filter(|event| event["type"] == "conversation.item.input_audio_transcription.completed").collect();
    let mut speakers = BTreeSet::new();
    let mut unspoken = 0;
    for done in &completed {
        let id = done["item_id"].as_str().unwrap();
        let committed = at("input_audio_buffer.committed", id).unwrap_or_else(|| panic!("{name}: {id} completed with no committed"));
        let end = at("conversation.item.input_audio_transcription.completed", id).unwrap();
        let deltas: Vec<usize> = (0..events.len()).filter(|&i| events[i]["type"] == "conversation.item.input_audio_transcription.delta" && events[i]["item_id"] == id).collect();
        assert!(deltas.iter().all(|&i| i > committed), "{name}: {id} sent a delta before its committed");
        let joined: String = deltas.iter().map(|&i| events[i]["delta"].as_str().unwrap()).collect();
        assert_eq!(joined, done["transcript"].as_str().unwrap(), "{name}: {id}'s deltas joined are not its transcript");
        match at("ignis.input_audio_transcription.speaker", id) {
            Some(speaker) => {
                assert!(deltas.iter().all(|&i| i < speaker) && speaker < end, "{name}: {id}'s speaker is not between its last delta and completed");
                speakers.insert(events[speaker]["speaker"].as_u64().unwrap());
            }
            None => unspoken += 1,
        }
    }
    let (mut cuts, mut guarded, mut start, mut stopped) = (0, 0, None, None);
    for event in events {
        match event["type"].as_str() {
            Some("input_audio_buffer.speech_started") => {
                let at = event["audio_start_ms"].as_u64();
                if at.is_some() && at == stopped {
                    cuts += 1;
                    guarded += usize::from(stopped.unwrap() - start.unwrap_or(0) >= guard_ms);
                }
                start = at;
            }
            Some("input_audio_buffer.speech_stopped") => stopped = event["audio_end_ms"].as_u64(),
            _ => {}
        }
    }
    (completed.len(), cuts, guarded, speakers, unspoken)
}

/// Acceptance 7 of spec speech/08 (see the header): every item of the twenty conversations, cut
/// items included, sends no delta before it ends, and its deltas joined are its transcript.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "the twenty conversations on the study disk's models"]
async fn every_item_of_the_conversations_sends_its_text_whole_at_its_end() {
    let conversations = conversations();
    let pace: f64 = std::env::var("IGNIS_DIARIZER_PACE").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0);
    let executor = Arc::new(CpuExecutor::new());
    let dir = study().join("models").join("ignis-conv");
    let scheduler = ConcreteScheduler::with_config(SchedulerConfig { model: "test-model".into(), ..SchedulerConfig::default() }, Arc::new(MockCompute::new()));
    let models = SpeechModels { catalog: Arc::new(Catalog::built_in().clone()), dir };
    let speech = Speech::new(executor.clone(), Some(models)).with_diarizer(Arc::new(CpuDiarizer::new(diarizer_gguf(), false, executor)));
    // Fed faster than real time, a session may hold audio the worker has not reached yet: never
    // more than its whole conversation.
    let longest = conversations.iter().map(|(_, audio)| audio.len().div_ceil(16_000)).max().unwrap() as u32;
    let group = SpeechGroup { max_queued_audio_seconds: longest, ..speech_group() };
    let guard_ms = u64::from(group.max_utterance_seconds) * 1000;
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
        .with_speech(Arc::new(speech))
        .with_speech_group(group)
        .with_diarizer_group(DiarizerGroup::default());
    let live = live(server).await;
    let mut totals = (0, 0, 0);
    for (name, audio) in &conversations {
        let (mut socket, _) = session(&live).await;
        send(&mut socket, update(json!({ "audio": { "input": { "format": { "type": "audio/pcm", "rate": 16000 }, "turn_detection": { "type": "server_vad" } } }, "ignis": { "diarization": "live" } }))).await;
        assert_eq!(next_event(&mut socket).await["type"], "session.updated");
        let (mut out, mut incoming) = socket.split();
        let events = Arc::new(Mutex::new(Vec::<Value>::new()));
        let heard = Arc::clone(&events);
        let reader = tokio::spawn(async move {
            while let Some(Ok(frame)) = incoming.next().await {
                let tokio_tungstenite::tungstenite::Message::Text(text) = frame else { continue };
                heard.lock().unwrap().push(serde_json::from_str(text.as_str()).unwrap());
            }
        });
        let started = Instant::now();
        let piece = Duration::from_secs_f64(0.1 / pace);
        // The conversation, then 2 s of silence, so the VAD ends the last item.
        let silence = vec![0.0; 32_000];
        for (i, chunk) in audio.chunks(1600).chain(silence.chunks(1600)).enumerate() {
            tokio::time::sleep_until((started + piece * i as u32).into()).await;
            let event = json!({ "type": "input_audio_buffer.append", "audio": pcm16(chunk) });
            out.send(tokio_tungstenite::tungstenite::Message::Text(event.to_string().into())).await.unwrap();
        }
        let fed = started.elapsed();
        // Every item the VAD began has ended and completed; an error or a failed item ends the run.
        let count = |events: &[Value], kind: &str| events.iter().filter(|event| event["type"] == kind).count();
        let waited = Instant::now();
        loop {
            {
                let events = events.lock().unwrap();
                if let Some(bad) = events.iter().find(|event| matches!(event["type"].as_str(), Some("error" | "conversation.item.input_audio_transcription.failed"))) {
                    panic!("{name}: the session refused: {bad}");
                }
                let (began, ended) = (count(&events, "input_audio_buffer.speech_started"), count(&events, "input_audio_buffer.committed"));
                if began > 0 && began == ended && ended == count(&events, "conversation.item.input_audio_transcription.completed") {
                    break;
                }
                assert!(waited.elapsed() < Duration::from_secs(600), "{name}: {began} items began, {ended} ended, not all completed 600 s after the audio");
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        drop(out);
        reader.abort();
        let events = std::mem::take(&mut *events.lock().unwrap());
        let (items, cuts, guarded, speakers, unspoken) = check_items(name, &events, guard_ms);
        let deltas: HashMap<&str, usize> = events.iter().filter(|event| event["type"] == "conversation.item.input_audio_transcription.delta").fold(HashMap::new(), |mut n, event| {
            *n.entry(event["item_id"].as_str().unwrap()).or_default() += 1;
            n
        });
        println!(
            "{name}: {:.0} s of audio fed in {:.0} s ({pace}x), all completed {:.1} s later: {items} items, {cuts} cuts ({guarded} by the length guard, the rest at a change of speaker), speakers {speakers:?}, {unspoken} items with no speaker, at most {} delta an item",
            audio.len() as f64 / 16_000.0,
            fed.as_secs_f64(),
            waited.elapsed().as_secs_f64(),
            deltas.values().max().copied().unwrap_or(0)
        );
        totals = (totals.0 + items, totals.1 + cuts, totals.2 + guarded);
    }
    println!("{} conversations: {} items, {} cuts ({} by the length guard), every item's deltas its transcript", conversations.len(), totals.0, totals.1, totals.2);
}
