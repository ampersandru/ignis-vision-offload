//! The speech model's CPU executor in the server: `ignis-speech` (spec
//! speech/01, ADR 0067) behind [`SpeechExecutor`].
//!
//! A load builds the crate's executor from the catalog entry's files and the
//! load's settings; the server's worker keeps the idle clock (spec 02), so the
//! crate's own idle unload stays off.

use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use ignis_speech::{Features, ModelFiles, Settings, Threads};

use super::executor::{LoadSettings, Partial, SessionOpen, SpeechError, SpeechExecutor, SpeechFiles, SpeechSession, Transcript};

/// The reason a person reads on a CPU without AVX2, FMA and F16C.
pub const NO_AVX2_REASON: &str = "the CPU lacks AVX2, which the speech model needs";

type Slot = Option<(SpeechFiles, LoadSettings, ignis_speech::Executor)>;

pub struct CpuExecutor {
    features: Features,
    loaded: Mutex<Slot>,
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

    fn slot(&self) -> MutexGuard<'_, Slot> {
        self.loaded.lock().unwrap_or_else(|e| e.into_inner())
    }
}

fn err(e: impl std::fmt::Display) -> SpeechError {
    SpeechError(e.to_string())
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
        if let Some((f, s, e)) = slot.as_ref() {
            if f == files && *s == settings && e.is_loaded() {
                return Ok(());
            }
        }
        // The encoder is the entry's other file, found beside the decoder (spec speech/01 §6).
        let encoder = files
            .files
            .iter()
            .find(|f| **f != files.decoder)
            .ok_or_else(|| SpeechError(format!("{}: no audio encoder beside {}", files.id, files.decoder.display())))?
            .clone();
        let model = ModelFiles { decoder: files.decoder.clone(), encoder };
        let threads = settings.threads.map_or(Threads::Auto, |n| Threads::Count(n as usize));
        let s = Settings { threads, pinned_weights: settings.pinned_weights, idle_unload: Duration::ZERO, ..Settings::default() };
        *slot = None;
        let exec = ignis_speech::Executor::new(s, model, self.features);
        exec.load().map_err(err)?;
        *slot = Some((files.clone(), settings, exec));
        Ok(())
    }

    fn unload(&self) {
        self.slot().take();
    }

    fn is_loaded(&self) -> bool {
        self.slot().as_ref().is_some_and(|(_, _, e)| e.is_loaded())
    }

    fn open(&self, open: SessionOpen) -> Result<Box<dyn SpeechSession>, SpeechError> {
        let slot = self.slot();
        let (_, _, exec) = slot.as_ref().ok_or_else(|| SpeechError("the speech model is not loaded".into()))?;
        let s = exec.open_capped(&open.context, open.language.as_deref(), open.context_max_tokens as usize).map_err(err)?;
        Ok(Box::new(CpuSession(s)))
    }
}

struct CpuSession(ignis_speech::Session);

impl SpeechSession for CpuSession {
    fn push(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        self.0.push(samples).map_err(err)
    }

    fn partial(&mut self) -> Result<Partial, SpeechError> {
        let p = self.0.partial().map_err(err)?;
        Ok(Partial { stable: p.stable, tail: p.provisional })
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
            return;
        }
        let dir = std::env::temp_dir().join(format!("ignis-speech-cpu-{}", std::process::id()));
        let r = e.load(&files(&dir, false), LoadSettings::default());
        assert!(matches!(&r, Err(SpeechError(m)) if m.contains("no audio encoder")), "{r:?}");
        let r = e.load(&files(&dir, true), LoadSettings { threads: Some(2), pinned_weights: false });
        assert!(matches!(&r, Err(SpeechError(m)) if m.contains("Qwen3-ASR-1.7B-Q4_K_M.gguf")), "{r:?}");
        assert!(!e.is_loaded());
    }
}
