//! The executor the server drives (spec speech/01 §3-4): availability with
//! its reason, the model loaded at first use and unloaded when idle, the
//! thread count, the prefix cache, and one unit of compute at a time.
//!
//! The server owns the `speech` config group and the routes (spec 02); the
//! executor takes its settings as plain values.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::cpu::{select, Features, Isa};
use crate::decoder::Cache;
use crate::mel::Mel;
use crate::model::{LoadError, LoadReport, Model};
use crate::pool::Pool;
use crate::session::Session;

/// `speech.threads`: `auto` = physical cores − 1 (Q38), or a count used as given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Threads {
    Auto,
    Count(usize),
}

impl Threads {
    /// The workers to run, and whether they were resolved from `auto`. `auto` counts the
    /// physical cores within the CPUs this process may run on: on Linux the physical count
    /// reads `/proc/cpuinfo`, which a container's cpuset or quota does not narrow, so it is
    /// capped by the CPUs the affinity mask and the cgroup quota allow.
    pub fn resolve(self) -> (usize, &'static str) {
        match self {
            Threads::Auto => (num_cpus::get_physical().min(num_cpus::get()).saturating_sub(1).max(1), "auto"),
            Threads::Count(n) => (n.max(1), "configured"),
        }
    }
}

/// The executor's settings: the `speech` fields it reads, as plain values.
#[derive(Clone, Debug)]
pub struct Settings {
    pub threads: Threads,
    /// `speech.pinned_weights`: a private, locked copy instead of the mapped file (Q31).
    pub pinned_weights: bool,
    /// `speech.idle_unload`: unloaded after this long with no transcription; zero never (Q53).
    pub idle_unload: Duration,
    /// `speech.partial_interval_ms` (Q37).
    pub partial_interval_ms: u64,
    /// `speech.rollback_tokens`: the provisional tail's length (Q37).
    pub rollback_tokens: usize,
    /// The first partials fix no token: the official streaming's `unfixed_chunk_num` (2), so
    /// the language label and the opening words are decided on more than one interval of
    /// audio. Fixing at the first partial, on 2 s, labelled an Italian dictation Portuguese.
    pub unfixed_partials: usize,
    /// `speech.context_max_tokens` (Q50).
    pub context_max_tokens: usize,
    /// `speech.max_clip_seconds`: the KV cache is sized for it (Q19, Q36).
    pub max_clip_seconds: u32,
}

impl Default for Settings {
    /// The `speech` group's defaults (spec speech/spec.md).
    fn default() -> Self {
        Settings {
            threads: Threads::Auto,
            pinned_weights: false,
            idle_unload: Duration::from_secs(300),
            partial_interval_ms: 2000,
            rollback_tokens: 10,
            unfixed_partials: 2,
            context_max_tokens: 512,
            max_clip_seconds: 1200,
        }
    }
}

/// What one session reads of the settings when it opens, so a live change of
/// the server's config holds from the next session (the server passes the
/// current values; `Settings` holds the defaults a plain `open` uses).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionTuning {
    pub context_max_tokens: usize,
    pub rollback_tokens: usize,
    pub unfixed_partials: usize,
}

/// The GGUF pair: the decoder, and the audio encoder found beside it.
#[derive(Clone, Debug)]
pub struct ModelFiles {
    pub decoder: PathBuf,
    pub encoder: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Availability {
    Available,
    /// Unavailable, with the reason (ADR 0066): "the CPU lacks AVX2".
    Unavailable(String),
}

#[derive(Debug)]
pub enum ExecError {
    Unavailable(String),
    Load(LoadError),
}

impl std::fmt::Display for ExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecError::Unavailable(r) => write!(f, "the speech model is unavailable: {r}"),
            ExecError::Load(e) => write!(f, "loading the speech model: {e}"),
        }
    }
}

impl std::error::Error for ExecError {}

/// A loaded model and what runs it.
pub struct Loaded {
    pub model: Model,
    pub pool: Pool,
    pub isa: Isa,
    pub mel: Mel,
}

pub(crate) struct PrefixEntry {
    pub head: Vec<u32>,
    pub cache: Cache,
}

/// What sessions share with the executor.
pub(crate) struct Core {
    pub settings: Settings,
    /// The prefix cache's one entry: the last prompt head and its K/V (Q50).
    pub prefix: Mutex<Option<PrefixEntry>>,
    compute: Mutex<()>,
    /// Units running now, and the most ever at once (acceptance 19's witness: never above 1).
    in_unit: AtomicUsize,
    max_in_unit: AtomicUsize,
    last_use: Mutex<Instant>,
    loaded: Mutex<Option<Arc<Loaded>>>,
    /// Bumped by every load and unload, so a stale idle watcher stands down.
    generation: Mutex<u64>,
}

impl Core {
    /// One unit of compute at a time (Q44): a window, a partial or a commit's decode.
    pub fn unit(&self) -> Unit<'_> {
        let guard = self.compute.lock().unwrap_or_else(|e| e.into_inner());
        let now = self.in_unit.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_unit.fetch_max(now, Ordering::SeqCst);
        Unit { core: self, _guard: guard }
    }

    pub fn touch(&self) {
        *self.last_use.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
    }

    fn idle_for(&self) -> Duration {
        self.last_use.lock().unwrap_or_else(|e| e.into_inner()).elapsed()
    }

    fn unload(&self) {
        let mut g = self.generation.lock().unwrap_or_else(|e| e.into_inner());
        *g += 1;
        self.loaded.lock().unwrap_or_else(|e| e.into_inner()).take();
        self.prefix.lock().unwrap_or_else(|e| e.into_inner()).take();
    }
}

/// A unit of compute in progress: holds the compute lock.
pub(crate) struct Unit<'a> {
    core: &'a Core,
    _guard: MutexGuard<'a, ()>,
}

impl Drop for Unit<'_> {
    fn drop(&mut self) {
        self.core.in_unit.fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct Executor {
    core: Arc<Core>,
    files: ModelFiles,
    isa: Result<Isa, &'static str>,
    watcher: Mutex<Option<JoinHandle<()>>>,
}

impl Executor {
    /// Nothing is mapped or loaded here (Q53): the model loads at the first signal.
    pub fn new(settings: Settings, files: ModelFiles, features: Features) -> Executor {
        let core = Arc::new(Core {
            settings,
            prefix: Mutex::new(None),
            compute: Mutex::new(()),
            in_unit: AtomicUsize::new(0),
            max_in_unit: AtomicUsize::new(0),
            last_use: Mutex::new(Instant::now()),
            loaded: Mutex::new(None),
            generation: Mutex::new(0),
        });
        Executor { core, files, isa: select(features), watcher: Mutex::new(None) }
    }

    pub fn settings(&self) -> &Settings {
        &self.core.settings
    }

    pub fn availability(&self) -> Availability {
        match self.isa {
            Ok(_) => Availability::Available,
            Err(r) => Availability::Unavailable(r.to_string()),
        }
    }

    /// The most units of compute that ever ran at once: 1 once anything ran (Q44).
    pub fn max_concurrent_units(&self) -> usize {
        self.core.max_in_unit.load(Ordering::SeqCst)
    }

    pub fn is_loaded(&self) -> bool {
        self.core.loaded.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    /// Loads the model if it is not loaded: the first signal (a session opening, a mic press,
    /// a batch request). Returns the load's report, or `None` when it was already loaded.
    pub fn load(&self) -> Result<Option<LoadReport>, ExecError> {
        self.loaded().map(|(_, r)| r)
    }

    fn loaded(&self) -> Result<(Arc<Loaded>, Option<LoadReport>), ExecError> {
        let isa = self.isa.map_err(|r| ExecError::Unavailable(r.to_string()))?;
        let mut slot = self.core.loaded.lock().unwrap_or_else(|e| e.into_inner());
        self.core.touch();
        if let Some(l) = slot.as_ref() {
            return Ok((l.clone(), None));
        }
        let (n, source) = self.core.settings.threads.resolve();
        // hotpath-lint-allow: one line per load.
        tracing::info!(name: "ignis.speech.threads", threads = n, source, "speech threads {n} ({source})");
        let model = Model::load(&self.files.decoder, &self.files.encoder, self.core.settings.pinned_weights).map_err(ExecError::Load)?;
        let report = model.report.clone();
        // hotpath-lint-allow: one line per load.
        tracing::info!(
            name: "ignis.speech.load",
            millis = report.millis as u64,
            repacked_bytes = report.repacked_bytes,
            mapped_bytes = report.mapped_bytes,
            pinned = report.pinned,
            locked = report.locked,
            "speech model loaded in {} ms",
            report.millis
        );
        let l = Arc::new(Loaded { model, pool: Pool::new(n), isa, mel: Mel::new() });
        *slot = Some(l.clone());
        drop(slot);
        let generation = {
            let mut g = self.core.generation.lock().unwrap_or_else(|e| e.into_inner());
            *g += 1;
            *g
        };
        self.watch(generation);
        Ok((l, Some(report)))
    }

    /// Drops the model: the mapped files, the repacked weights, the pool and the prefix cache.
    /// Sessions still open keep theirs until they end.
    pub fn unload(&self) {
        self.core.unload();
        if let Some(h) = self.watcher.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            h.thread().unpark();
        }
    }

    /// Opens a transcription with `context` as the system turn, loading the model if needed.
    /// `language` forces the official prefill (Q6); `None` lets the model detect it.
    pub fn open(&self, context: &str, language: Option<&str>) -> Result<Session, ExecError> {
        let s = &self.core.settings;
        self.open_tuned(
            context,
            language,
            SessionTuning { context_max_tokens: s.context_max_tokens, rollback_tokens: s.rollback_tokens, unfixed_partials: s.unfixed_partials },
        )
    }

    /// [`open`](Self::open) with the context capped at `context_max_tokens` (the live
    /// `speech.context_max_tokens` the server passes per request) rather than the settings'.
    pub fn open_capped(&self, context: &str, language: Option<&str>, context_max_tokens: usize) -> Result<Session, ExecError> {
        let s = &self.core.settings;
        self.open_tuned(
            context,
            language,
            SessionTuning { context_max_tokens, rollback_tokens: s.rollback_tokens, unfixed_partials: s.unfixed_partials },
        )
    }

    /// [`open`](Self::open) with the settings a session reads at its opening passed in: the
    /// server's live `speech.context_max_tokens`, `speech.rollback_tokens` and
    /// `speech.unfixed_partials`.
    pub fn open_tuned(&self, context: &str, language: Option<&str>, tuning: SessionTuning) -> Result<Session, ExecError> {
        let (l, _) = self.loaded()?;
        Ok(Session::open(l, self.core.clone(), context, language, tuning))
    }

    /// The idle unloader: wakes when the idle time may have run out and unloads once
    /// `idle_unload` passed with no transcription.
    fn watch(&self, generation: u64) {
        let idle = self.core.settings.idle_unload;
        if idle.is_zero() {
            return;
        }
        let core = self.core.clone();
        let h = thread::Builder::new()
            .name("ignis-speech-idle".into())
            .spawn(move || loop {
                if *core.generation.lock().unwrap_or_else(|e| e.into_inner()) != generation {
                    return;
                }
                let quiet = core.idle_for();
                if quiet >= idle {
                    // hotpath-lint-allow: one line per unload.
                    tracing::info!(name: "ignis.speech.unload", idle_secs = quiet.as_secs(), "speech model unloaded after {} s idle", quiet.as_secs());
                    core.unload();
                    return;
                }
                thread::park_timeout(idle - quiet);
            })
            .expect("spawn the speech idle watcher");
        let old = self.watcher.lock().unwrap_or_else(|e| e.into_inner()).replace(h);
        if let Some(old) = old {
            old.thread().unpark();
        }
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        self.core.unload();
        if let Some(h) = self.watcher.lock().unwrap_or_else(|e| e.into_inner()).take() {
            h.thread().unpark();
            let _ = h.join();
        }
    }
}
