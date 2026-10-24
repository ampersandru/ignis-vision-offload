//! The seam between the server and the speech model's executor (spec
//! speech/01 §4, speech/02): the operations the server needs of it and no
//! more. The real executor is `ignis-speech` (`crates/speech`, ticket #351);
//! the route, the slot and the queue are built and tested against
//! [`FakeExecutor`] behind this trait, and the adapter to the real crate is
//! the one place that knows both.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The model's files on disk, as the slot found them: every file of the
/// catalog entry (the decoder GGUF, `decoder`, and the encoder GGUF beside
/// it) under `download.path`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechFiles {
    /// The served id.
    pub id: String,
    /// The decoder GGUF, the entry's `artifact`.
    pub decoder: PathBuf,
    /// Every file of the entry, the decoder among them.
    pub files: Vec<PathBuf>,
}

/// What a load reads from the configuration (`speech.threads`,
/// `speech.pinned_weights`): the load's, so a change holds from the next one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LoadSettings {
    /// `None` is `auto`, the physical cores less one.
    pub threads: Option<u32>,
    pub pinned_weights: bool,
}

/// What a session opens with.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SessionOpen {
    /// The system turn: the request's `prompt`, then its `keywords[]` as one
    /// line. Empty when there is none. The executor owns the tokenizer, so
    /// it keeps the last `context_max_tokens` tokens of it (spec speech/02
    /// §Context, Q50).
    pub context: String,
    pub context_max_tokens: u32,
    /// The language forced, by the model's English name (`Italian`), the
    /// official assistant prefill `language <Name><asr_text>`. `None`
    /// auto-detects.
    pub language: Option<String>,
}

/// A live transcript so far (spec speech/01 §4, `partial()`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Partial {
    /// The stable text: only ever extended, never revised.
    pub stable: String,
    /// The provisional tail, which the next call may revise.
    pub tail: String,
}

/// A finished transcription (`commit()`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Transcript {
    /// The text, beginning with every stable text ever reported.
    pub text: String,
    /// The model's own label (`Italian`); `None` when it said `None`.
    pub language: Option<String>,
    /// The decoded token ids, for `verbose_json`'s segment.
    pub tokens: Vec<u32>,
}

/// Why the executor could not do what it was asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechError(pub String);

impl std::fmt::Display for SpeechError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SpeechError {}

/// One transcription, driven by the compute worker, one call at a time.
pub trait SpeechSession: Send {
    /// 16 kHz mono f32 samples, in order.
    fn push(&mut self, samples: &[f32]) -> Result<(), SpeechError>;
    /// The transcript so far.
    fn partial(&mut self) -> Result<Partial, SpeechError>;
    /// The last window, then the transcript to its end.
    fn commit(self: Box<Self>) -> Result<Transcript, SpeechError>;
}

/// The speech model's executor.
pub trait SpeechExecutor: Send + Sync {
    /// Where it runs: `cpu`, or `gpu_loan` (the `executor` label and log
    /// field).
    fn name(&self) -> &'static str;
    /// Whether this host can run it, and if not why, in a sentence the
    /// Playground shows (`the CPU lacks AVX2 ...`). Cheap: asked per call.
    fn availability(&self) -> Result<(), String>;
    /// Map and prepare the model. Called by the worker before the first
    /// session, and again after an unload.
    fn load(&self, files: &SpeechFiles, settings: LoadSettings) -> Result<(), SpeechError>;
    /// Give the model's host memory back.
    fn unload(&self);
    fn is_loaded(&self) -> bool;
    fn open(&self, open: SessionOpen) -> Result<Box<dyn SpeechSession>, SpeechError>;
}

/// The executor of a binary that has none yet: every host is unavailable,
/// saying so. What `main` wires until `ignis-speech` is adapted.
#[derive(Debug, Default)]
pub struct NoExecutor;

const NO_EXECUTOR: &str = "this build has no speech executor";

impl SpeechExecutor for NoExecutor {
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn availability(&self) -> Result<(), String> {
        Err(NO_EXECUTOR.to_owned())
    }
    fn load(&self, _: &SpeechFiles, _: LoadSettings) -> Result<(), SpeechError> {
        Err(SpeechError(NO_EXECUTOR.to_owned()))
    }
    fn unload(&self) {}
    fn is_loaded(&self) -> bool {
        false
    }
    fn open(&self, _: SessionOpen) -> Result<Box<dyn SpeechSession>, SpeechError> {
        Err(SpeechError(NO_EXECUTOR.to_owned()))
    }
}

/// A scripted executor for tests of everything around the real one. A
/// session's transcript is the `words` it was scripted with: one more of them
/// is stable per 8 s window pushed (128,000 samples), the last arriving only
/// at the commit. It records what it was opened with, the samples it was
/// pushed, what it was loaded with, and every call its sessions took, in
/// order ([`FakeExecutor::calls`]).
#[derive(Debug)]
pub struct FakeExecutor {
    pub words: Vec<String>,
    pub language: Option<String>,
    pub name: &'static str,
    pub unavailable: Mutex<Option<String>>,
    pub loaded: AtomicBool,
    pub loads: AtomicUsize,
    pub unloads: AtomicUsize,
    pub fail_load: AtomicBool,
    /// Read by each session at its commit.
    pub fail_commit: Arc<AtomicBool>,
    /// How long each call to `push` and `commit` takes.
    pub step_delay: Mutex<Duration>,
    pub opened: Mutex<Vec<SessionOpen>>,
    /// The length of every `push`, in order, over every session.
    pub pushed: Arc<Mutex<Vec<usize>>>,
    pub load_settings: Mutex<Vec<LoadSettings>>,
    pub load_files: Mutex<Vec<SpeechFiles>>,
    /// Every call, in order, over every session: `open <n> <context>`,
    /// `push <n> <samples>`, `partial <n>` and `commit <n>`, `n` counting
    /// the sessions opened from 0.
    pub calls: Arc<Mutex<Vec<String>>>,
}

impl FakeExecutor {
    pub fn saying(words: &[&str]) -> Self {
        Self {
            words: words.iter().map(|word| (*word).to_owned()).collect(),
            language: Some("English".to_owned()),
            name: "cpu",
            unavailable: Mutex::new(None),
            loaded: AtomicBool::new(false),
            loads: AtomicUsize::new(0),
            unloads: AtomicUsize::new(0),
            fail_load: AtomicBool::new(false),
            fail_commit: Arc::new(AtomicBool::new(false)),
            step_delay: Mutex::new(Duration::ZERO),
            opened: Mutex::new(Vec::new()),
            pushed: Arc::new(Mutex::new(Vec::new())),
            load_settings: Mutex::new(Vec::new()),
            load_files: Mutex::new(Vec::new()),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn set_unavailable(&self, reason: Option<&str>) {
        *self.unavailable.lock().unwrap() = reason.map(str::to_owned);
    }

    pub fn set_step_delay(&self, delay: Duration) {
        *self.step_delay.lock().unwrap() = delay;
    }

    /// Every sample pushed to any session, in total.
    pub fn samples_pushed(&self) -> usize {
        self.pushed.lock().unwrap().iter().sum()
    }

    /// The calls so far ([`FakeExecutor::calls`]).
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

/// Samples in one 8 s window at 16 kHz.
pub const WINDOW_SAMPLES: usize = 128_000;

struct FakeSession {
    words: Vec<String>,
    language: Option<String>,
    delay: Duration,
    fail_commit: Arc<AtomicBool>,
    pushed: usize,
    stable_words: usize,
    log: Arc<Mutex<Vec<usize>>>,
    /// Its number among the sessions opened, and the executor's call log.
    index: usize,
    calls: Arc<Mutex<Vec<String>>>,
}

impl SpeechSession for FakeSession {
    fn push(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        std::thread::sleep(self.delay);
        self.pushed += samples.len();
        self.log.lock().unwrap().push(samples.len());
        self.calls.lock().unwrap().push(format!("push {} {}", self.index, samples.len()));
        // One more stable word per window pushed, the last left to the commit.
        self.stable_words = (self.pushed / WINDOW_SAMPLES).min(self.words.len().saturating_sub(1));
        Ok(())
    }

    fn partial(&mut self) -> Result<Partial, SpeechError> {
        self.calls.lock().unwrap().push(format!("partial {}", self.index));
        let (stable, tail) = self.words.split_at(self.stable_words);
        let tail = tail.first().map(|word| format!(" {word}")).unwrap_or_default();
        Ok(Partial { stable: stable.join(" "), tail })
    }

    fn commit(self: Box<Self>) -> Result<Transcript, SpeechError> {
        self.calls.lock().unwrap().push(format!("commit {}", self.index));
        std::thread::sleep(self.delay);
        if self.fail_commit.load(Ordering::SeqCst) {
            return Err(SpeechError("the fake executor was told to fail".to_owned()));
        }
        Ok(Transcript {
            text: self.words.join(" "),
            language: self.language.clone(),
            tokens: (0..self.words.len() as u32).collect(),
        })
    }
}

impl SpeechExecutor for FakeExecutor {
    fn name(&self) -> &'static str {
        self.name
    }

    fn availability(&self) -> Result<(), String> {
        match self.unavailable.lock().unwrap().clone() {
            Some(reason) => Err(reason),
            None => Ok(()),
        }
    }

    fn load(&self, files: &SpeechFiles, settings: LoadSettings) -> Result<(), SpeechError> {
        if self.fail_load.load(Ordering::SeqCst) {
            return Err(SpeechError("the fake executor was told to fail its load".to_owned()));
        }
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.load_files.lock().unwrap().push(files.clone());
        self.load_settings.lock().unwrap().push(settings);
        self.loaded.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn unload(&self) {
        self.unloads.fetch_add(1, Ordering::SeqCst);
        self.loaded.store(false, Ordering::SeqCst);
    }

    fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    fn open(&self, open: SessionOpen) -> Result<Box<dyn SpeechSession>, SpeechError> {
        if !self.is_loaded() {
            return Err(SpeechError("opened before the model was loaded".to_owned()));
        }
        // A forced language is what the model reports it heard.
        let language = open.language.clone().or_else(|| self.language.clone());
        let mut opened = self.opened.lock().unwrap();
        let index = opened.len();
        self.calls.lock().unwrap().push(format!("open {index} {}", open.context));
        opened.push(open);
        Ok(Box::new(FakeSession {
            words: self.words.clone(),
            language,
            delay: *self.step_delay.lock().unwrap(),
            fail_commit: Arc::clone(&self.fail_commit),
            pushed: 0,
            stable_words: 0,
            log: Arc::clone(&self.pushed),
            index,
            calls: Arc::clone(&self.calls),
        }))
    }
}
