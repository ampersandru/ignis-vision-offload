//! The speech model's CPU executor in the server: `ignis-speech` (spec
//! speech/01, ADR 0068) behind [`SpeechExecutor`].
//!
//! A load builds the crate's executor from the catalog entry's files and the
//! load's settings; the server's worker keeps the idle clock (spec 02) and
//! asks for the unload.

use std::sync::{Mutex, MutexGuard};

use ignis_speech::{Features, ModelFiles, SessionTuning, Settings, Threads};

use super::executor::{LoadSettings, Partial, SessionOpen, SpeechError, SpeechExecutor, SpeechFiles, SpeechSession, Transcript};

/// The reason a person reads on a CPU without AVX2, FMA and F16C.
pub const NO_AVX2_REASON: &str = "the CPU lacks AVX2, which the speech model needs";

/// A loaded model: what it was loaded from and with, and the crate's executor.
struct Loaded {
    files: SpeechFiles,
    settings: LoadSettings,
    exec: ignis_speech::Executor,
}

pub struct CpuExecutor {
    features: Features,
    loaded: Mutex<Option<Loaded>>,
}

impl Default for CpuExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuExecutor {
    pub fn new() -> CpuExecutor {
        CpuExecutor::with_features(Features::detect())
    }

    /// With features taken away from what detection reported (a test of the unavailable state).
    fn with_features(features: Features) -> CpuExecutor {
        CpuExecutor { features, loaded: Mutex::new(None) }
    }

    fn slot(&self) -> MutexGuard<'_, Option<Loaded>> {
        self.loaded.lock().unwrap_or_else(|e| e.into_inner())
    }
}

fn err(e: impl std::fmt::Display) -> SpeechError {
    SpeechError(e.to_string())
}

/// The encoder is the entry's file under the fixed `mmproj-` name, beside the
/// decoder (spec speech/01 §6; the catalog holds an entry to having one).
fn encoder_of(files: &SpeechFiles) -> Option<&std::path::PathBuf> {
    files.files.iter().find(|f| {
        *f != &files.decoder
            && f.file_name().and_then(|n| n.to_str()).is_some_and(crate::download::catalog::is_speech_encoder)
    })
}

impl SpeechExecutor for CpuExecutor {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn availability(&self) -> Result<(), String> {
        match ignis_speech::cpu::select(self.features) {
            Ok(_) => Ok(()),
            Err(r) if r == ignis_speech::NO_AVX2 => Err(NO_AVX2_REASON.to_owned()),
            Err(r) => Err(r.to_owned()),
        }
    }

    fn load(&self, files: &SpeechFiles, settings: LoadSettings) -> Result<(), SpeechError> {
        self.availability().map_err(SpeechError)?;
        let mut slot = self.slot();
        if let Some(l) = slot.as_ref() {
            if l.files == *files && l.settings == settings && l.exec.is_loaded() {
                return Ok(());
            }
        }
        let encoder = encoder_of(files)
            .ok_or_else(|| SpeechError(format!("{}: no audio encoder (mmproj-*.gguf) beside {}", files.id, files.decoder.display())))?
            .clone();
        let model = ModelFiles { decoder: files.decoder.clone(), encoder };
        let threads = settings.threads.map_or(Threads::Auto, |n| Threads::Count(n as usize));
        let s = Settings { threads, pinned_weights: settings.pinned_weights, ..Settings::default() };
        *slot = None;
        let exec = ignis_speech::Executor::new(s, model, self.features);
        exec.load().map_err(err)?;
        *slot = Some(Loaded { files: files.clone(), settings, exec });
        Ok(())
    }

    fn unload(&self) {
        self.slot().take();
    }

    fn is_loaded(&self) -> bool {
        self.slot().as_ref().is_some_and(|l| l.exec.is_loaded())
    }

    fn open(&self, open: SessionOpen) -> Result<Box<dyn SpeechSession>, SpeechError> {
        let slot = self.slot();
        let l = slot.as_ref().ok_or_else(|| SpeechError("the speech model is not loaded".into()))?;
        let tuning = SessionTuning {
            context_max_tokens: open.context_max_tokens as usize,
            rollback_tokens: open.rollback_tokens as usize,
            unfixed_partials: open.unfixed_partials as usize,
        };
        let s = l.exec.open_tuned(&open.context, open.language.as_deref(), tuning).map_err(err)?;
        Ok(Box::new(CpuSession::Open(s)))
    }
}

/// A transcription: open, then committing a unit at a time, then spent.
enum CpuSession {
    Open(ignis_speech::Session),
    Committing(ignis_speech::Commit),
    Spent,
}

const COMMITTED: &str = "the transcription is committed and takes no more audio";

impl SpeechSession for CpuSession {
    fn push(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        match self {
            CpuSession::Open(s) => s.push(samples).map_err(err),
            _ => Err(SpeechError(COMMITTED.to_owned())),
        }
    }

    fn partial(&mut self, stop: &dyn Fn() -> bool) -> Result<Option<Partial>, SpeechError> {
        let CpuSession::Open(s) = self else { return Err(SpeechError(COMMITTED.to_owned())) };
        let p = s.partial_unless(stop).map_err(err)?;
        Ok(p.map(|p| Partial { stable: p.stable, tail: p.provisional }))
    }

    fn commit_step(&mut self) -> Result<Option<Transcript>, SpeechError> {
        if let CpuSession::Open(_) = self {
            let CpuSession::Open(s) = std::mem::replace(self, CpuSession::Spent) else { unreachable!() };
            *self = CpuSession::Committing(s.into_commit());
        }
        let CpuSession::Committing(commit) = self else { return Err(SpeechError(COMMITTED.to_owned())) };
        let Some(t) = commit.step().map_err(err)? else { return Ok(None) };
        *self = CpuSession::Spent;
        Ok(Some(Transcript { text: t.text, language: t.language, tokens: t.tokens }))
    }
}

/// A session's commit stepped to its end (tests).
#[cfg(test)]
fn commit(mut s: Box<dyn SpeechSession>) -> Result<Transcript, SpeechError> {
    loop {
        if let Some(t) = s.commit_step()? {
            return Ok(t);
        }
    }
}

// Declared here, outside the inline `tests` block, so the path is relative to
// this file's directory: inside an inline module it would go through
// `cpu/tests/`, which Linux cannot resolve because it does not exist.
#[cfg(test)]
#[allow(dead_code)]
#[path = "../../../speech/tests/common/synthetic.rs"]
mod synthetic;

#[cfg(test)]
mod tests {
    use super::*;

    fn files(dir: &std::path::Path, encoder: bool) -> SpeechFiles {
        let decoder = dir.join("Qwen3-ASR-1.7B-Q4_K_M.gguf");
        let mut all = vec![decoder.clone()];
        if encoder {
            all.push(dir.join("mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"));
        }
        SpeechFiles { id: "qwen3-asr-1.7b".into(), decoder, files: all }
    }

    /// Spec speech/01 AC 22: each built-in speech entry, as `Speech::resolve`
    /// hands its files over, has its decoder as the artifact and exactly its
    /// `mmproj-` file as the encoder this adapter picks.
    #[test]
    fn each_built_in_speech_entry_gives_the_adapter_its_decoder_and_encoder() {
        let dir = std::path::Path::new("models");
        let catalog = crate::download::Catalog::built_in();
        for (id, decoder, encoder) in [
            ("qwen3-asr-1.7b", "Qwen3-ASR-1.7B-Q4_K_M.gguf", "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"),
            ("qwen3-asr-1.7b-q8", "Qwen3-ASR-1.7B-Q8_0.gguf", "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"),
            ("qwen3-asr-0.6b", "Qwen3-ASR-0.6B-Q8_0.gguf", "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"),
        ] {
            let entry = catalog.entry(id).expect(id);
            let files = SpeechFiles { id: id.into(), decoder: entry.artifact_path(dir), files: entry.files.iter().map(|f| dir.join(&f.name)).collect() };
            assert_eq!(files.decoder, dir.join(decoder), "{id}");
            assert!(files.files.contains(&files.decoder), "{id}: the decoder is one of the entry's files");
            assert_eq!(encoder_of(&files), Some(&dir.join(encoder)), "{id}");
        }
    }

    #[test]
    fn a_cpu_without_avx2_is_unavailable_with_the_reason_the_guide_shows() {
        let e = CpuExecutor::with_features(Features::none());
        assert_eq!(e.name(), "cpu");
        assert_eq!(e.availability(), Err(NO_AVX2_REASON.to_owned()));
        let r = e.load(&files(&std::env::temp_dir(), true), LoadSettings::default());
        assert_eq!(r, Err(SpeechError(NO_AVX2_REASON.to_owned())));
        assert!(!e.is_loaded());
        assert!(e.open(SessionOpen::default()).is_err());
    }

    #[test]
    fn a_load_needs_the_encoder_beside_the_decoder_and_names_a_missing_file() {
        let e = CpuExecutor::new();
        if e.availability().is_err() {
            eprintln!("SKIP: this CPU lacks AVX2; the load is not reached");
            return;
        }
        let dir = std::env::temp_dir().join(format!("ignis-speech-cpu-{}", std::process::id()));
        let r = e.load(&files(&dir, false), LoadSettings::default());
        assert!(matches!(&r, Err(SpeechError(m)) if m.contains("no audio encoder")), "{r:?}");
        // A file beside the decoder that is not the encoder's fixed name is not taken for it.
        let mut odd = files(&dir, false);
        odd.files.push(dir.join("README.md"));
        let r = e.load(&odd, LoadSettings::default());
        assert!(matches!(&r, Err(SpeechError(m)) if m.contains("no audio encoder")), "{r:?}");
        let r = e.load(&files(&dir, true), LoadSettings { threads: Some(2), pinned_weights: false });
        assert!(matches!(&r, Err(SpeechError(m)) if m.contains("Qwen3-ASR-1.7B-Q4_K_M.gguf")), "{r:?}");
        assert!(!e.is_loaded());
    }

    use super::synthetic;

    /// Q79: the adapter hands the server's stop to the crate's session. On the crate's synthetic
    /// pair (its encoder under the fixed `mmproj-` name), a refresh whose stop says so answers
    /// `None` and leaves the session as it was: its commit is the one with no refresh tried.
    #[test]
    fn a_refresh_the_server_stops_answers_none_and_changes_nothing() {
        let e = CpuExecutor::new();
        if e.availability().is_err() {
            eprintln!("SKIP: this CPU lacks AVX2; the model is not loaded");
            return;
        }
        let pair = synthetic::pair("adapter-stop");
        let encoder = pair.encoder.with_file_name("mmproj-synthetic.gguf");
        std::fs::rename(&pair.encoder, &encoder).unwrap();
        let files = SpeechFiles { id: "synthetic".into(), decoder: pair.decoder.clone(), files: vec![pair.decoder.clone(), encoder] };
        e.load(&files, LoadSettings { threads: Some(2), pinned_weights: false }).unwrap();
        let open = SessionOpen { context_max_tokens: 512, rollback_tokens: 10, unfixed_partials: 2, ..SessionOpen::default() };
        let session = || {
            let mut s = e.open(open.clone()).unwrap();
            s.push(&synthetic::audio(9.3)).unwrap();
            s
        };
        let untried = commit(session()).unwrap();
        let mut s = session();
        assert_eq!(s.partial(&|| true).unwrap(), None);
        assert_eq!(commit(s).unwrap(), untried);
        assert!(session().partial(&|| false).unwrap().is_some(), "a refresh nothing stops answers");
    }

    /// Spec speech/01 AC 19 through the seam: the adapter's commit is the crate's, a unit a step
    /// (the last window with the audio's last frames, then the decode's prefill, then the
    /// decode), and a spent session takes no more audio and no more steps.
    #[test]
    fn the_adapters_commit_is_a_unit_a_step_and_then_spent() {
        let e = CpuExecutor::new();
        if e.availability().is_err() {
            eprintln!("SKIP: this CPU lacks AVX2; the model is not loaded");
            return;
        }
        let pair = synthetic::pair("adapter-steps");
        let encoder = pair.encoder.with_file_name("mmproj-synthetic.gguf");
        std::fs::rename(&pair.encoder, &encoder).unwrap();
        let files = SpeechFiles { id: "synthetic".into(), decoder: pair.decoder.clone(), files: vec![pair.decoder.clone(), encoder] };
        e.load(&files, LoadSettings { threads: Some(2), pinned_weights: false }).unwrap();
        let mut s = e.open(SessionOpen { context_max_tokens: 512, ..SessionOpen::default() }).unwrap();
        s.push(&synthetic::audio(9.3)).unwrap();
        assert_eq!(s.commit_step().unwrap(), None, "the last window");
        assert_eq!(s.commit_step().unwrap(), None, "the decode's prefill");
        // The synthetic decoder's one token, then EOS: one decode step ends it.
        let t = s.commit_step().unwrap().expect("the decode ends");
        assert_eq!(t.tokens.len(), 1);
        assert!(s.commit_step().is_err());
        assert!(s.push(&[0.0; 160]).is_err());
    }
}
