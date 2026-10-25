//! The speech model's CPU executor in the server: `ignis-speech` (spec
//! speech/01, ADR 0067) behind [`SpeechExecutor`].
//!
//! A load builds the crate's executor from the catalog entry's files and the
//! load's settings; the server's worker keeps the idle clock (spec 02), so the
//! crate's own idle unload stays off.

use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

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

    /// With the CPU features injected (tests).
    pub fn with_features(features: Features) -> CpuExecutor {
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
        let s = Settings { threads, pinned_weights: settings.pinned_weights, idle_unload: Duration::ZERO, ..Settings::default() };
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
        Ok(Box::new(CpuSession(s)))
    }
}

struct CpuSession(ignis_speech::Session);

impl SpeechSession for CpuSession {
    fn push(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        self.0.push(samples).map_err(err)
    }

    fn partial(&mut self, stop: &dyn Fn() -> bool) -> Result<Option<Partial>, SpeechError> {
        let p = self.0.partial_unless(stop).map_err(err)?;
        Ok(p.map(|p| Partial { stable: p.stable, tail: p.provisional }))
    }

    fn commit(self: Box<Self>) -> Result<Transcript, SpeechError> {
        let t = self.0.commit().map_err(err)?;
        Ok(Transcript { text: t.text, language: t.language, tokens: t.tokens })
    }
}

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
        let e = CpuExecutor::with_features(Features::default());
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
}
