//! The speech model in the server (spec speech/02, ADR 0066): Qwen3-ASR,
//! loaded beside the active model, on the CPU, at first use.
//!
//! It is a slot of its own on [`crate::Server`], not a [`crate::Runner`]: it
//! is never switched to, a switch does not tear it down, and the text
//! model's idle clock never sees it. Its state is computed where it is read,
//! from the live `speech.*` settings, the catalog and what is on disk:
//!
//! - **off**: `speech.model` is `off`;
//! - **unavailable, with a reason**: the catalog has no such entry, the host
//!   cannot run the executor (no AVX2), or its files are not under
//!   `download.path` (the reason names the command that fetches them);
//! - **available**: it can serve, whether or not it is loaded right now. The
//!   first request loads it (Q53), and the worker unloads it when idle.
//!
//! The executor is behind [`executor::SpeechExecutor`]; the route
//! ([`route`]), this slot and the worker ([`worker`]) are tested against
//! [`executor::FakeExecutor`].

pub mod audio;
pub mod executor;
pub mod languages;
pub mod route;
pub mod worker;

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock};

use crate::config::schema::SpeechGroup;
use crate::download::Catalog;
use executor::{NoExecutor, SpeechExecutor, SpeechFiles};
use worker::{Full, Gauges, Job, Worker};

/// What marks a catalog entry as a speech model rather than one a switch can
/// load: the Qwen3-ASR entries' ids (`qwen3-asr-1.7b`, `-1.7b-q8`, `-0.6b`).
/// The catalog has no `kind` key, so the id is the discriminator; the
/// listing, the switch and the kind check read it here.
pub const SPEECH_ID_PREFIX: &str = "qwen3-asr-";

/// Whether `id` names a speech model (a catalog entry never switched to).
pub fn is_speech_id(id: &str) -> bool {
    id.starts_with(SPEECH_ID_PREFIX)
}

/// The catalog and the directory the speech model's files are looked for in:
/// `download.path`.
#[derive(Debug, Clone)]
pub struct SpeechModels {
    pub catalog: Arc<Catalog>,
    pub dir: PathBuf,
}

/// What `GET /v1/models` and the route say of the speech model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Availability {
    /// The id it is served under, when `speech.model` names one.
    pub id: Option<String>,
    /// Why it cannot serve; `None` when it can.
    pub reason: Option<String>,
}

impl Availability {
    pub fn available(&self) -> bool {
        self.reason.is_none()
    }
}

/// The speech model's slot.
pub struct Speech {
    executor: Arc<dyn SpeechExecutor>,
    models: Option<SpeechModels>,
    gauges: Arc<Gauges>,
    /// Started by the first request: a server that is never asked for a
    /// transcription never has the thread.
    worker: OnceLock<Worker>,
}

impl Default for Speech {
    /// A server built without a speech model: no executor, no catalog.
    fn default() -> Self {
        Self::new(Arc::new(NoExecutor), None)
    }
}

impl Speech {
    pub fn new(executor: Arc<dyn SpeechExecutor>, models: Option<SpeechModels>) -> Self {
        Self { executor, models, gauges: Arc::default(), worker: OnceLock::new() }
    }

    pub fn executor(&self) -> &Arc<dyn SpeechExecutor> {
        &self.executor
    }

    /// Publish the worker's gauges to `metrics` (`Server::with_metrics`).
    pub fn attach_metrics(&self, metrics: Arc<crate::metrics::Metrics>) {
        self.gauges.attach(metrics);
    }

    fn worker(&self) -> &Worker {
        self.worker.get_or_init(|| Worker::start(Arc::clone(&self.executor), Arc::clone(&self.gauges)))
    }

    /// The id `speech.model` names: `auto` is the 1.7B Q4_K_M entry, `off`
    /// is none.
    pub fn served_id(group: &SpeechGroup) -> Option<String> {
        match group.model.as_str() {
            "off" => None,
            "auto" => Some(crate::config::AUTO_SPEECH_MODEL_ID.to_owned()),
            id => Some(id.to_owned()),
        }
    }

    /// Whether the speech model can serve now, and if not why, in the words
    /// the Playground shows (spec speech/02 §Unavailable, Q35, Q39).
    pub fn availability(&self, group: &SpeechGroup) -> Availability {
        match self.resolve(group) {
            Ok(files) => Availability { id: Some(files.id), reason: None },
            Err(unavailable) => Availability { id: Self::served_id(group), reason: Some(unavailable.0) },
        }
    }

    /// The files to serve with, or why there is no model to serve: checked in
    /// the order a person fixes them — off, an id the catalog has, a host
    /// that can run it, the files on disk.
    pub fn resolve(&self, group: &SpeechGroup) -> Result<SpeechFiles, Unavailable> {
        let Some(id) = Self::served_id(group) else {
            return Err(Unavailable("the speech model is turned off (`speech.model` is `off`)".to_owned()));
        };
        let entry = self.models.as_ref().and_then(|models| models.catalog.entries().iter().find(|entry| entry.id == id).map(|entry| (models, entry)));
        let Some((models, entry)) = entry else {
            return Err(Unavailable(format!("`speech.model` names `{id}`, which is not in the model catalog")));
        };
        self.executor.availability().map_err(Unavailable)?;
        let files: Vec<PathBuf> = entry.files.iter().map(|file| models.dir.join(&file.name)).collect();
        if !files.iter().all(|file| file.is_file()) {
            return Err(Unavailable(format!(
                "no speech model files for `{id}` under download.path; fetch them with `ignis-server model download {id}`"
            )));
        }
        Ok(SpeechFiles { id, decoder: entry.artifact_path(&models.dir), files })
    }

    /// Queue a batch job, or say when room frees. Starts the worker, and
    /// hands it the live idle time.
    pub fn submit(&self, job: Box<dyn Job>, group: &SpeechGroup) -> Result<(), Full> {
        let worker = self.worker();
        worker.set_idle_unload(group.idle_unload);
        worker.submit(job, group.max_queue)
    }

    /// Open a Realtime session, unless `max_sessions` are open (spec
    /// speech/02 §Admission, 03). The guard closes it when dropped.
    pub fn open_session(self: &Arc<Self>, group: &SpeechGroup) -> Result<SessionGuard, Full> {
        let mut open = self.gauges.sessions.load(Ordering::SeqCst);
        loop {
            if open >= group.max_sessions as usize {
                // Room frees when a person ends a session: there is no clip
                // to estimate from, so the shortest wait.
                return Err(Full { retry_after_secs: 1 });
            }
            match self.gauges.sessions.compare_exchange(open, open + 1, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(now) => open = now,
            }
        }
        self.gauges.publish();
        Ok(SessionGuard { speech: Arc::clone(self) })
    }

    /// The Realtime sessions open now.
    pub fn sessions_open(&self) -> usize {
        self.gauges.sessions.load(Ordering::SeqCst)
    }

    /// The batch requests waiting behind the one running.
    pub fn queue_length(&self) -> usize {
        self.gauges.queue.load(Ordering::SeqCst)
    }

    /// Queue an interactive task (Realtime, spec speech/03).
    pub fn submit_interactive(&self, task: worker::Task) {
        self.worker().submit_interactive(task);
    }
}

/// An open Realtime session's place among `speech.max_sessions`.
pub struct SessionGuard {
    speech: Arc<Speech>,
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        self.speech.gauges.sessions.fetch_sub(1, Ordering::SeqCst);
        self.speech.gauges.publish();
    }
}

/// Why the speech model cannot serve, as the route and `GET /v1/models` say it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unavailable(pub String);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download::catalog::{CatalogEntry, CatalogFile, CatalogLayer};

    /// A speech catalog entry of our own (the real ones are ticket #351's):
    /// the decoder GGUF and the encoder beside it.
    pub(crate) fn fixture_models(dir: &std::path::Path) -> SpeechModels {
        let file = |name: &str| CatalogFile { name: name.to_owned(), bytes: 1, sha256: "0".repeat(64) };
        let entry = |id: &str| CatalogEntry {
            id: id.to_owned(),
            repo: "test/asr".to_owned(),
            revision: "0".repeat(40),
            artifact: format!("{id}-decoder.gguf"),
            card_class: None,
            aliases: Vec::new(),
            files: vec![file(&format!("{id}-encoder.gguf")), file(&format!("{id}-decoder.gguf"))],
            layer: CatalogLayer::BuiltIn,
        };
        SpeechModels { catalog: Arc::new(Catalog::from_entries(vec![entry("qwen3-asr-1.7b"), entry("qwen3-asr-0.6b")])), dir: dir.to_owned() }
    }

    pub(crate) fn put_files(dir: &std::path::Path, id: &str) {
        for name in [format!("{id}-encoder.gguf"), format!("{id}-decoder.gguf")] {
            std::fs::write(dir.join(name), b"gguf").unwrap();
        }
    }

    fn group(model: &str) -> SpeechGroup {
        SpeechGroup { model: model.to_owned(), ..SpeechGroup::default() }
    }

    fn slot(executor: Arc<dyn SpeechExecutor>, dir: &std::path::Path) -> Speech {
        Speech::new(executor, Some(fixture_models(dir)))
    }

    /// AC 1: `auto` is the 1.7B Q4_K_M entry, an id is its entry, `off` none.
    #[test]
    fn speech_model_resolves_auto_an_id_and_off() {
        assert_eq!(Speech::served_id(&group("auto")).as_deref(), Some("qwen3-asr-1.7b"));
        assert_eq!(Speech::served_id(&group("qwen3-asr-0.6b")).as_deref(), Some("qwen3-asr-0.6b"));
        assert_eq!(Speech::served_id(&group("off")), None);
    }

    /// AC 1, 2: the model is available when its files are on disk, with the
    /// files the executor loads, and nothing is fetched by asking.
    #[test]
    fn the_model_is_available_when_its_files_are_on_disk() {
        let dir = tempfile_dir("available");
        let speech = slot(Arc::new(executor::FakeExecutor::saying(&["hi"])), &dir);
        let missing = speech.availability(&group("auto"));
        assert!(!missing.available());
        put_files(&dir, "qwen3-asr-1.7b");
        let files = speech.resolve(&group("auto")).unwrap();
        assert_eq!(files.id, "qwen3-asr-1.7b");
        assert_eq!(files.decoder, dir.join("qwen3-asr-1.7b-decoder.gguf"));
        assert_eq!(files.files.len(), 2);
        assert_eq!(speech.availability(&group("auto")), Availability { id: Some("qwen3-asr-1.7b".to_owned()), reason: None });
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2, "asking downloaded nothing");
    }

    /// AC 2: each unavailable state reports its reason, in order of what a
    /// person fixes: off, an id the catalog lacks, a host that cannot run it,
    /// files missing, naming the command that fetches them.
    #[test]
    fn each_unavailable_state_reports_its_reason() {
        let dir = tempfile_dir("reasons");
        let fake = Arc::new(executor::FakeExecutor::saying(&["hi"]));
        let speech = slot(fake.clone(), &dir);
        let reason = |model: &str| speech.availability(&group(model)).reason.expect("unavailable");
        assert!(reason("off").contains("turned off") && reason("off").contains("speech.model"));
        assert!(reason("qwen3-asr-9b").contains("`qwen3-asr-9b`") && reason("qwen3-asr-9b").contains("not in the model catalog"));
        let files = reason("auto");
        assert!(files.contains("no speech model files") && files.contains("model download qwen3-asr-1.7b"), "{files}");
        put_files(&dir, "qwen3-asr-1.7b");
        fake.set_unavailable(Some("the CPU lacks AVX2, which the speech model needs"));
        assert_eq!(reason("auto"), "the CPU lacks AVX2, which the speech model needs");
        // The host's reason comes before the files': downloading cannot fix it.
        std::fs::remove_file(dir.join("qwen3-asr-1.7b-decoder.gguf")).unwrap();
        assert!(reason("auto").contains("AVX2"));
        assert!(Speech::default().availability(&group("auto")).reason.unwrap().contains("not in the model catalog"));
    }

    /// AC 11: past `max_sessions` open sessions the next is refused with a
    /// bounded `Retry-After`; a closed one gives its place back.
    #[test]
    fn past_max_sessions_a_session_is_refused_and_a_closed_one_frees_its_place() {
        let speech = Arc::new(Speech::default());
        let group = SpeechGroup { max_sessions: 2, ..SpeechGroup::default() };
        let first = speech.open_session(&group).unwrap();
        let _second = speech.open_session(&group).unwrap();
        assert_eq!(speech.sessions_open(), 2);
        let full = speech.open_session(&group).err().expect("a third is refused");
        assert!((1..=worker::RETRY_AFTER_MAX_SECS).contains(&full.retry_after_secs));
        drop(first);
        assert_eq!(speech.sessions_open(), 1);
        assert!(speech.open_session(&group).is_ok());
    }

    /// A unique empty directory under the system temp dir.
    pub(crate) fn tempfile_dir(name: &str) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!("ignis-speech-{name}-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
