//! Acceptance 11 of spec speech/08 on the real CPU executor and the real diarizer: a 5-minute
//! conversation of the finding's twenty (`IGNIS_DIARIZER_CONV`, default `conv07`, three speakers)
//! fed through `GET /v1/realtime?intent=transcription` at real-time pace, 100 ms of 16 kHz PCM every
//! 100 ms, with `server_vad` and `session.ignis.diarization: "live"`, Qwen3-ASR `qwen3-asr-1.7b`
//! transcribing every item in the same session. The diarizer keeps up when every frame's speaker
//! probabilities come within the live geometry's latency plus 2 s of the frame's first sample
//! arriving, the speech model's units beside it on the one worker: the session's log line's
//! `lag_max_ms`. (An item's speaker can come later than that by design: a cut at a change reaches
//! back to where the new speaker's run began, up to `diarizer.min_speaker_seconds` before.) `--ignored`: it reads the study disk (`IGNIS_SPEECH_STUDY`,
//! default `H:/speech-study`): the converted Qwen3-ASR pair and the diarizer's GGUF
//! (`IGNIS_DIARIZER_GGUF`, default the F16 one), and fails without them.
//!
//! It is a timing run: nothing else may be using the CPU. Its own binary, since it captures the
//! process's log. Run it alone:
//!
//! ```text
//! cargo test --release -p ignis-server --test speech_diarizer_study -- --ignored --nocapture
//! ```

#[path = "../../speech/tests/common/mod.rs"]
mod common;
#[path = "support/mod.rs"]
mod support;

use std::path::PathBuf;
use std::sync::Arc;
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
    // Every event is read, so the socket never backs up; a refusal fails the run.
    let reader = tokio::spawn(async move {
        while let Some(Ok(frame)) = events.next().await {
            let tokio_tungstenite::tungstenite::Message::Text(text) = frame else { continue };
            let event: Value = serde_json::from_str(text.as_str()).unwrap();
            if matches!(event["type"].as_str(), Some("error" | "conversation.item.input_audio_transcription.failed")) {
                panic!("the session refused: {event}");
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
    assert!(line["items"].as_u64().unwrap() >= 10, "a conversation's items had their speakers: {line}");
    assert!(lag <= BOUND_MS, "{name}: a frame's probabilities came {lag:.0} ms after its first sample, over {BOUND_MS} ms");
}
