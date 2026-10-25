//! The Realtime route on the real CPU executor, over the owner's 11 dictations (spec speech/03,
//! GitHub #353, acceptance 5 and 13). `--ignored`: it reads the study disk
//! (`IGNIS_SPEECH_STUDY`, default `H:/speech-study`) and fails without it, as the executor's
//! `oracle.rs` does; a leg that skips proves nothing (docs/agents/testing.md).
//!
//! Each dictation is fed through `GET /v1/realtime?intent=transcription` at real-time pace (100 ms
//! of 16 kHz PCM every 100 ms), the way the Playground feeds it, then committed. Per dictation the
//! deltas joined must equal `completed.transcript` (AC 5); the pooled WER of the transcripts is
//! only reported against the batch path (the same executor, 8 s pushes, as the transcriptions route
//! drives it); and the time from the commit to `completed` is printed, per dictation and as the
//! median and the slowest of the set (AC 13: at most 2 s for a dictation of at least 60 s,
//! `IGNIS_SPEECH_COMMIT_MS` to override the bound where the figure is only measured, as in the
//! 4-core container). `IGNIS_SPEECH_PARTIAL_MS` and `IGNIS_SPEECH_PARTIAL_FACTOR` set
//! `speech.partial_interval_ms` and `speech.partial_cost_factor` for the run (Q76).
//!
//! It is a timing run: nothing else may be using the CPU. Run it alone:
//!
//! ```text
//! cargo test --release -p ignis-server --test speech_realtime_study -- --ignored --nocapture
//! ```

#[path = "../../speech/tests/common/mod.rs"]
mod common;
#[path = "support/mod.rs"]
mod support;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{norm_dictation, pooled_wer, read_wav, study};
use futures_util::{SinkExt, StreamExt};
use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::config::schema::SpeechGroup;
use ignis_server::download::catalog::Catalog;
use ignis_server::engine::Engine;
use ignis_server::speech::cpu::CpuExecutor;
use ignis_server::speech::executor::{LoadSettings, Partial, SessionOpen, SpeechError, SpeechExecutor, SpeechFiles, SpeechSession, Transcript};
use ignis_server::speech::{Speech, SpeechModels};
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::Server;
use serde_json::{json, Value};
use support::realtime::{at_16_khz, base64, live, next_event, send, session};

const DELTA: &str = "conversation.item.input_audio_transcription.delta";
const COMPLETED: &str = "conversation.item.input_audio_transcription.completed";
/// The default variant, `speech.model: auto`.
const MODEL: &str = "qwen3-asr-1.7b";

struct Dictation {
    id: String,
    reference: String,
    audio: Vec<f32>,
}

/// All 11, or the ones named in `IGNIS_SPEECH_ONLY` (`d10`, comma-separated): the timing legs
/// need a quiet machine for minutes, and AC 13 is about the one dictation over 60 s.
fn dictations() -> Vec<Dictation> {
    let dir = study().join("dictations");
    let only = std::env::var("IGNIS_SPEECH_ONLY").unwrap_or_default();
    (1..=11)
        .filter(|i| only.is_empty() || only.split(',').any(|id| id == format!("d{i:02}")))
        .map(|i| {
            let id = format!("d{i:02}");
            Dictation {
                reference: std::fs::read_to_string(dir.join(format!("{id}.txt"))).unwrap(),
                audio: read_wav(&dir.join("16k").join(format!("{id}.wav"))),
                id,
            }
        })
        .collect()
}


/// The real executor with every call to a session timed, so a slow commit can be told apart: the
/// executor's own work, or the wait for the worker.
struct Timed {
    inner: Arc<CpuExecutor>,
    calls: Arc<std::sync::Mutex<Vec<(&'static str, Duration)>>>,
}

struct TimedSession {
    inner: Box<dyn SpeechSession>,
    calls: Arc<std::sync::Mutex<Vec<(&'static str, Duration)>>>,
}

impl SpeechSession for TimedSession {
    fn push(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        let began = Instant::now();
        let out = self.inner.push(samples);
        self.calls.lock().unwrap().push(("push", began.elapsed()));
        out
    }
    fn partial(&mut self, stop: &dyn Fn() -> bool) -> Result<Option<Partial>, SpeechError> {
        let began = Instant::now();
        let out = self.inner.partial(stop);
        // A refresh a commit stopped (Q79) is a `stop`, timed apart from the ones that ran.
        let name = if matches!(out, Ok(None)) { "stop" } else { "partial" };
        self.calls.lock().unwrap().push((name, began.elapsed()));
        out
    }
    fn commit(self: Box<Self>) -> Result<Transcript, SpeechError> {
        let calls = Arc::clone(&self.calls);
        let began = Instant::now();
        let out = self.inner.commit();
        calls.lock().unwrap().push(("commit", began.elapsed()));
        out
    }
}

impl SpeechExecutor for Timed {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn availability(&self) -> Result<(), String> {
        self.inner.availability()
    }
    fn load(&self, files: &SpeechFiles, settings: LoadSettings) -> Result<(), SpeechError> {
        self.inner.load(files, settings)
    }
    fn unload(&self) {
        self.inner.unload()
    }
    fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }
    fn open(&self, open: SessionOpen) -> Result<Box<dyn SpeechSession>, SpeechError> {
        Ok(Box::new(TimedSession { inner: self.inner.open(open)?, calls: Arc::clone(&self.calls) }))
    }
}

/// The number in the environment variable `name`, or `default`.
fn env_or(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// `speech.partial_interval_ms` for the run: its default, or `IGNIS_SPEECH_PARTIAL_MS` to see what
/// a slower cadence does to the commit.
fn interval_ms() -> u32 {
    env_or("IGNIS_SPEECH_PARTIAL_MS", SpeechGroup::default().partial_interval_ms)
}

/// `speech.partial_cost_factor` for the run: its default, or `IGNIS_SPEECH_PARTIAL_FACTOR`.
fn cost_factor() -> u32 {
    env_or("IGNIS_SPEECH_PARTIAL_FACTOR", SpeechGroup::default().partial_cost_factor)
}

/// A server over the real CPU executor and the built-in catalog, the files under the study disk's
/// converted models.
fn real_server(executor: Arc<dyn SpeechExecutor>) -> (Server, PathBuf) {
    let dir = study().join("models").join("ignis-conv");
    let scheduler = ConcreteScheduler::with_config(SchedulerConfig { model: "test-model".into(), ..SchedulerConfig::default() }, Arc::new(MockCompute::new()));
    let models = SpeechModels { catalog: Arc::new(Catalog::built_in().clone()), dir: dir.clone() };
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
        .with_speech(Arc::new(Speech::new(executor, Some(models))))
        .with_speech_group(SpeechGroup { partial_interval_ms: interval_ms(), partial_cost_factor: cost_factor(), ..SpeechGroup::default() });
    (server, dir)
}

/// The PCM16 little-endian bytes of `samples`, base64.
fn pcm16(samples: &[f32]) -> String {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| ((s * 32768.0).round().clamp(-32768.0, 32767.0) as i16).to_le_bytes()).collect();
    base64(&bytes)
}

struct Heard {
    deltas: String,
    transcript: String,
    /// From sending the commit to receiving `completed`.
    commit_to_completed: Duration,
}

/// One dictation through a session: appended at real-time pace, then committed.
async fn dictate(live: &support::realtime::Live, audio: &[f32]) -> Heard {
    let (mut socket, _) = session(live).await;
    send(&mut socket, at_16_khz()).await;
    assert_eq!(next_event(&mut socket).await["type"], "session.updated");
    let (mut sink, mut stream) = socket.split();
    let reader = tokio::spawn(async move {
        let mut deltas = String::new();
        loop {
            let frame = stream.next().await.expect("the socket stays open").expect("a frame");
            let tokio_tungstenite::tungstenite::Message::Text(text) = frame else { continue };
            let event: Value = serde_json::from_str(text.as_str()).unwrap();
            match event["type"].as_str().unwrap() {
                DELTA => deltas.push_str(event["delta"].as_str().unwrap()),
                COMPLETED => return (deltas, event["transcript"].as_str().unwrap().to_owned(), Instant::now()),
                "error" | "conversation.item.input_audio_transcription.failed" => panic!("the session refused: {event}"),
                _ => {}
            }
        }
    });
    let (start, step) = (Instant::now(), 1600);
    for (i, chunk) in audio.chunks(step).enumerate() {
        tokio::time::sleep_until((start + Duration::from_millis(100 * i as u64)).into()).await;
        let event = json!({ "type": "input_audio_buffer.append", "audio": pcm16(chunk) });
        sink.send(tokio_tungstenite::tungstenite::Message::Text(event.to_string().into())).await.unwrap();
    }
    // The last chunk lasts its 100 ms: the commit is sent when the audio has all been spoken.
    tokio::time::sleep_until((start + Duration::from_millis(100 * audio.chunks(step).count() as u64)).into()).await;
    let committed = Instant::now();
    sink.send(tokio_tungstenite::tungstenite::Message::Text(json!({ "type": "input_audio_buffer.commit" }).to_string().into())).await.unwrap();
    let (deltas, transcript, done) = reader.await.unwrap();
    Heard { deltas, transcript, commit_to_completed: done - committed }
}

/// The batch path as the transcriptions route drives the executor: 8 s pushed at a time, no
/// partial, one commit.
fn batch(executor: &CpuExecutor, audio: &[f32]) -> String {
    let mut s = executor.open(SessionOpen { rollback_tokens: 10, unfixed_partials: 2, context_max_tokens: 512, ..SessionOpen::default() }).unwrap();
    for chunk in audio.chunks(128_000) {
        s.push(chunk).unwrap();
    }
    s.commit().unwrap().text
}

/// AC 5 and 13 on the real executor: the 11 dictations through the route at real-time pace.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn the_eleven_dictations_through_the_real_route_at_real_time_pace() {
    let bound = Duration::from_millis(std::env::var("IGNIS_SPEECH_COMMIT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000));
    let executor = Arc::new(CpuExecutor::new());
    executor.availability().expect("the real executor needs AVX2");
    let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
    let (server, dir) = real_server(Arc::new(Timed { inner: Arc::clone(&executor), calls: Arc::clone(&calls) }));
    let files = server.speech.resolve(&server.live().speech).unwrap_or_else(|why| panic!("{}: {why:?}", dir.display()));
    assert_eq!(files.id, MODEL);
    let live = live(server).await;
    let dictations = dictations();
    println!("speech.partial_interval_ms {}, speech.partial_cost_factor {}", interval_ms(), cost_factor());

    // The batch reference first, on the same loaded model, and a warm one for the route.
    executor.load(&files, LoadSettings::default()).unwrap();
    let batch_pairs: Vec<_> = dictations.iter().map(|d| (norm_dictation(&d.reference), norm_dictation(&batch(&executor, &d.audio)))).collect();

    let mut pairs = Vec::new();
    let mut slowest = (Duration::ZERO, String::new());
    let mut commits = Vec::new();
    for d in &dictations {
        calls.lock().unwrap().clear();
        let heard = dictate(&live, &d.audio).await;
        assert_eq!(heard.deltas, heard.transcript, "{}: the deltas joined are not completed.transcript", d.id);
        let seconds = d.audio.len() as f64 / 16_000.0;
        println!("{} {seconds:6.1} s: commit to completed {} ms, {} words", d.id, heard.commit_to_completed.as_millis(), heard.transcript.split_whitespace().count());
        let summary = |kind: &str| {
            let all = calls.lock().unwrap();
            let of: Vec<_> = all.iter().filter(|(k, _)| *k == kind).map(|(_, d)| d.as_millis()).collect();
            format!("{} {kind}s {} ms", of.len(), of.iter().sum::<u128>())
        };
        println!("      in the executor: {}, {}, {}, {}", summary("push"), summary("partial"), summary("stop"), summary("commit"));
        commits.push(heard.commit_to_completed);
        if seconds >= 60.0 && heard.commit_to_completed > slowest.0 {
            slowest = (heard.commit_to_completed, d.id.clone());
        }
        pairs.push((norm_dictation(&d.reference), norm_dictation(&heard.transcript)));
    }
    let (route, batch) = (pooled_wer(&pairs), pooled_wer(&batch_pairs));
    println!("pooled WER on the 11 dictations: route {:.2}%, batch {:.2}% ({:+.2} pt)", 100.0 * route, 100.0 * batch, 100.0 * (route - batch));
    commits.sort();
    println!(
        "commit to completed over the {} dictations: median {} ms, slowest {} ms",
        commits.len(),
        commits[commits.len() / 2].as_millis(),
        commits[commits.len() - 1].as_millis()
    );
    println!("slowest commit to completed on a dictation of at least 60 s: {} ms ({})", slowest.0.as_millis(), slowest.1);
    assert!(slowest.0 <= bound, "{} took {} ms from the commit to completed, over {} ms", slowest.1, slowest.0.as_millis(), bound.as_millis());
}
