//! The live text of a long monologue with speakers on (spec speech/08): one voice for minutes, fed
//! through `GET /v1/realtime?intent=transcription` at real-time pace with the provisional tail,
//! once with `session.ignis.diarization: "live"` and once without, on the real executor and the
//! real diarizer. The diarizer computes on a thread of its own, so its steps should leave the
//! refreshes about as they are without speakers. In the first design (decision D6) a step that
//! fell due stopped the refresh under way; on a long item a refresh took longer than the steps
//! left it, and the live text stood still until `completed`. Per item it prints the refreshes the
//! client saw and the longest wait between two, with speakers and without. `--ignored`: it reads the study disk
//! (`IGNIS_SPEECH_STUDY`): the converted Qwen3-ASR pair, the diarizer's F16 GGUF and a long item of
//! `tools/speech-diarizer/long_items.py` (`IGNIS_LONG_ITEM`, default `125192-240`). A timing run:
//! nothing else may be using the CPU.
//!
//! ```text
//! cargo test --release -p ignis-server --test speech_diarizer_refreshes -- --ignored --nocapture
//! ```

#[path = "../../speech/tests/common/mod.rs"]
mod common;
#[path = "support/mod.rs"]
mod support;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{read_wav, study};
use futures_util::{SinkExt, StreamExt};
use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::config::schema::{DiarizerGroup, SpeechGroup};
use ignis_server::download::catalog::Catalog;
use ignis_server::engine::Engine;
use ignis_server::speech::cpu::{CpuDiarizer, CpuExecutor};
use ignis_server::speech::{Speech, SpeechModels};
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::Server;
use serde_json::{json, Value};
use support::realtime::{base64, live, next_event, send, session, update};

const PROVISIONAL: &str = "ignis.input_audio_transcription.provisional";
const COMPLETED: &str = "conversation.item.input_audio_transcription.completed";

fn pcm16(samples: &[f32]) -> String {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| ((s * 32768.0).round().clamp(-32768.0, 32767.0) as i16).to_le_bytes()).collect();
    base64(&bytes)
}

/// An item's live text as the client saw it: when it started, its refreshes, when it completed.
#[derive(Default, Debug)]
struct Seen {
    started: Option<Instant>,
    refreshes: Vec<Instant>,
    completed: Option<Instant>,
}

impl Seen {
    /// The longest the live text stood still: between the item's start, its refreshes and its end.
    fn longest_wait(&self) -> Duration {
        let mut marks: Vec<Instant> = self.started.into_iter().chain(self.refreshes.iter().copied()).chain(self.completed).collect();
        marks.sort();
        marks.windows(2).map(|w| w[1] - w[0]).max().unwrap_or_default()
    }
}

async fn monologue(audio: &[f32], speakers: bool) -> Vec<Seen> {
    let executor = Arc::new(CpuExecutor::new());
    let dir = study().join("models").join("ignis-conv");
    let gguf = study().join("models").join("diarizer").join("Nemotron-3-Diarization-F16.gguf");
    let scheduler = ConcreteScheduler::with_config(SchedulerConfig { model: "test-model".into(), ..SchedulerConfig::default() }, Arc::new(MockCompute::new()));
    let models = SpeechModels { catalog: Arc::new(Catalog::built_in().clone()), dir };
    let speech = Speech::new(executor.clone(), Some(models)).with_diarizer(Arc::new(CpuDiarizer::new(gguf, false, executor)));
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
        .with_speech(Arc::new(speech))
        .with_speech_group(SpeechGroup::default())
        .with_diarizer_group(DiarizerGroup::default());
    let live = live(server).await;
    let (mut socket, _) = session(&live).await;
    let diarization = if speakers { json!("live") } else { Value::Null };
    send(&mut socket, update(json!({ "audio": { "input": { "format": { "type": "audio/pcm", "rate": 16000 }, "turn_detection": { "type": "server_vad" } } }, "ignis": { "provisional_tail": true, "diarization": diarization } }))).await;
    assert_eq!(next_event(&mut socket).await["type"], "session.updated");
    let (mut out, mut events) = socket.split();
    let reader = tokio::spawn(async move {
        let (mut items, mut order): (HashMap<String, Seen>, Vec<String>) = (HashMap::new(), Vec::new());
        while let Some(Ok(frame)) = events.next().await {
            let tokio_tungstenite::tungstenite::Message::Text(text) = frame else { continue };
            let event: Value = serde_json::from_str(text.as_str()).unwrap();
            let Some(id) = event["item_id"].as_str().map(str::to_owned) else { continue };
            if !items.contains_key(&id) {
                order.push(id.clone());
            }
            let seen = items.entry(id).or_default();
            match event["type"].as_str().unwrap() {
                "input_audio_buffer.speech_started" => seen.started = Some(Instant::now()),
                PROVISIONAL if !event["text"].as_str().unwrap_or_default().is_empty() => seen.refreshes.push(Instant::now()),
                COMPLETED => seen.completed = Some(Instant::now()),
                "error" | "conversation.item.input_audio_transcription.failed" => panic!("the session refused: {event}"),
                _ => {}
            }
        }
        order.into_iter().filter_map(|id| items.remove(&id)).collect::<Vec<_>>()
    });
    let started = Instant::now();
    for (i, chunk) in audio.chunks(1600).enumerate() {
        tokio::time::sleep_until((started + Duration::from_millis(100 * i as u64)).into()).await;
        out.send(tokio_tungstenite::tungstenite::Message::Text(json!({ "type": "input_audio_buffer.append", "audio": pcm16(chunk) }).to_string().into())).await.unwrap();
    }
    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        out.send(tokio_tungstenite::tungstenite::Message::Text(json!({ "type": "input_audio_buffer.append", "audio": pcm16(&[0.0; 1600]) }).to_string().into())).await.unwrap();
    }
    tokio::time::sleep(Duration::from_secs(40)).await;
    // A close frame: the server closes the socket, and the reader's stream ends.
    out.close().await.unwrap();
    tokio::time::timeout(Duration::from_secs(30), reader).await.expect("the socket closes").unwrap()
}

/// The GPU profile's serialized pass runs every `--ignored` test it does not skip; this leg is the
/// CPU's, on the study disk, so `scripts/gpu-profile.ps1` skips it by name.
#[test]
fn the_gpu_profile_skips_this_leg() {
    let script = include_str!("../../../scripts/gpu-profile.ps1");
    let listed = script.split("$NotGpuLegs = @(").nth(1).and_then(|rest| rest.split(')').next()).expect("gpu-profile.ps1 lists $NotGpuLegs");
    assert!(listed.contains("\"a_long_monologue_keeps_its_live_text_with_speakers_on\""), "{listed}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "a timing run on the study disk's models and a long item"]
async fn a_long_monologue_keeps_its_live_text_with_speakers_on() {
    let name = std::env::var("IGNIS_LONG_ITEM").unwrap_or_else(|_| "125192-240".into());
    let audio = read_wav(&study().join("length-guard-361").join(format!("{name}.wav")));
    for speakers in [false, true] {
        let items = monologue(&audio, speakers).await;
        for (i, seen) in items.iter().enumerate().filter(|(_, seen)| seen.started.is_some()) {
            let span = seen.completed.zip(seen.started).map(|(c, s)| (c - s).as_secs_f64()).unwrap_or(f64::NAN);
            println!(
                "{name}, speakers {speakers}: item {i}: {span:.1} s from start to completed, {} refreshes, the live text stood still at most {:.1} s",
                seen.refreshes.len(),
                seen.longest_wait().as_secs_f64()
            );
        }
    }
}
