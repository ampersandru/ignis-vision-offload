//! One transcription, incrementally (spec speech/01 §4, Q37): open with a
//! context, push 16 kHz samples, ask for partials, commit.
//!
//! 1. `open` prefills the prompt's head (system turn, user turn, `<|audio_start|>`): the
//!    prefix cache's one entry, so a transcription with the previous one's context
//!    prefills none of it (Q50).
//! 2. `push` computes each mel frame once, encodes each 8 s window once as soon as it is
//!    complete, and prefills its tokens: the audio prefix.
//! 3. `partial` encodes the incomplete window, appends it, the prompt's tail and the stable
//!    text, decodes greedily, keeps the last `rollback_tokens` of what it decoded as the
//!    provisional tail (the rest joins the stable text, never revised), and rolls the cache
//!    back to the audio prefix. The first `unfixed_partials` fix nothing, as the official
//!    streaming's first chunks do. `partial_unless` is a partial its caller can stop (Q79):
//!    it checks before each step of its work and where its decode yields the unit, and a
//!    partial stopped leaves the session as it was.
//! 4. `commit` encodes the last window and decodes from the stable text to EOS, so the
//!    transcript begins with every stable text reported (Q43).
//!
//! The mel floor (the maximum − 8) is taken over the frames computed so far: a stream cannot
//! know a clip's maximum before its end. A whole clip pushed at once has every frame but its
//! last one or two computed before its first window is encoded.

use std::sync::Arc;

use crate::decoder::{self, argmax, Cache, Logits};
use crate::encoder::encode_window;
use crate::executor::{Core, Loaded, SessionTuning};
use crate::mel::{self, audio_tokens, frames_of, normalise, CHUNK_FRAMES, HOP, MIN_SAMPLES, N_FFT, N_MELS, WINDOW_CHUNKS};
use crate::text::{parse, Prompt};

const WINDOW_FRAMES: usize = CHUNK_FRAMES * WINDOW_CHUNKS;
const WINDOW_TOKENS: usize = 104;
/// Decode steps per unit of compute: about a window's prefill, so a long decode interleaves
/// with other sessions' windows (Q44).
const DECODE_UNIT: usize = 16;

/// A partial result: the stable text (never revised) and the provisional tail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partial {
    pub stable: String,
    pub provisional: String,
    pub language: Option<String>,
}

/// A completed transcription.
#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    pub text: String,
    /// The model's language label (Q51), or the forced one.
    pub language: Option<String>,
    pub audio_seconds: f64,
    pub audio_tokens: usize,
    /// The decoded token ids, the language header included (spec 02's `verbose_json` segment).
    pub tokens: Vec<u32>,
}

/// What a session did, for acceptance 17-18.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Context tokens prefilled at open (0 when the prefix cache held the head).
    pub context_prefilled: usize,
    /// Windows encoded into the audio prefix: each window once.
    pub windows: usize,
    /// Audio tokens prefilled into the audio prefix: each once.
    pub audio_prefilled: usize,
    pub partials: usize,
}

#[derive(Debug)]
pub enum SessionError {
    /// The clip passed `max_clip_seconds`.
    TooLong { seconds: u32 },
    /// The KV cache filled before the decoder stopped.
    CacheFull,
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::TooLong { seconds } => write!(f, "the audio is longer than {seconds} s"),
            SessionError::CacheFull => write!(f, "the transcription filled its KV cache"),
        }
    }
}

impl std::error::Error for SessionError {}

pub struct Session {
    loaded: Arc<Loaded>,
    core: Arc<Core>,
    prompt: Prompt,
    language: Option<String>,
    /// Pushed samples from absolute index `sample_base` (a multiple of the hop).
    samples: Vec<f32>,
    sample_base: usize,
    total_samples: usize,
    /// log10 mel rows from absolute frame `frame_base`; `next_frame` is the first not computed.
    frames: Vec<f32>,
    frame_base: usize,
    next_frame: usize,
    max_log: f32,
    windows: usize,
    cache: Cache,
    /// Cache positions after the head and every encoded window.
    audio_prefix: usize,
    stable: Vec<u32>,
    since_partial: usize,
    stats: Stats,
    /// `speech.rollback_tokens` and `speech.unfixed_partials` as they were when this session opened.
    tuning: SessionTuning,
}

impl Session {
    pub(crate) fn open(loaded: Arc<Loaded>, core: Arc<Core>, context: &str, language: Option<&str>, tuning: SessionTuning) -> Session {
        let guard = core.clone();
        let _unit = guard.unit();
        let s = &core.settings;
        let text = &loaded.model.text;
        let prompt = Prompt::new(text, context, tuning.context_max_tokens, language);
        let dec = &loaded.model.decoder;
        let audio_cap = audio_tokens(frames_of(s.max_clip_seconds as usize * mel::SAMPLE_RATE));
        // Sized for the context cap, the longest clip's audio tokens and the decode (§2): room
        // for half as many text tokens as audio tokens, where speech gives about a third. It
        // starts at the head, a window and the tail, and grows as the transcription does.
        let max = prompt.head.len() + audio_cap + prompt.tail.len() + audio_cap / 2 + 64;
        let first = prompt.head.len() + 2 * WINDOW_TOKENS + prompt.tail.len() + 64;
        let mut cache = Cache::growing(dec, first, max);
        let mut stats = Stats::default();
        {
            let mut entry = core.prefix.lock().unwrap_or_else(|e| e.into_inner());
            match entry.as_ref() {
                Some(e) if e.head == prompt.head => cache.copy_prefix(&e.cache, e.head.len()),
                _ => {
                    let x = decoder::embed(dec, &prompt.head);
                    decoder::forward(dec, &loaded.pool, loaded.isa, &mut cache, x, Logits::None, None);
                    stats.context_prefilled = prompt.context_tokens;
                    let mut keep = Cache::new(dec, prompt.head.len());
                    keep.copy_prefix(&cache, prompt.head.len());
                    *entry = Some(crate::executor::PrefixEntry { head: prompt.head.clone(), cache: keep });
                }
            }
        }
        let audio_prefix = cache.len;
        Session {
            loaded,
            core,
            prompt,
            language: language.map(str::to_string),
            samples: Vec::new(),
            sample_base: 0,
            total_samples: 0,
            frames: Vec::new(),
            frame_base: 0,
            next_frame: 0,
            max_log: f32::NEG_INFINITY,
            windows: 0,
            cache,
            audio_prefix,
            stable: Vec::new(),
            since_partial: 0,
            stats,
            tuning,
        }
    }

    /// The tuning this session opened with.
    pub fn tuning(&self) -> SessionTuning {
        self.tuning
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// Audio pushed so far, in seconds.
    pub fn seconds(&self) -> f64 {
        self.total_samples as f64 / mel::SAMPLE_RATE as f64
    }

    /// True once `partial_interval_ms` of audio arrived since the last partial.
    pub fn partial_due(&self) -> bool {
        self.since_partial * 1000 >= self.core.settings.partial_interval_ms as usize * mel::SAMPLE_RATE
    }

    /// Appends mono 16 kHz samples; encodes and prefills each window that completes.
    pub fn push(&mut self, samples: &[f32]) -> Result<(), SessionError> {
        let max = self.core.settings.max_clip_seconds;
        if self.total_samples + samples.len() > max as usize * mel::SAMPLE_RATE {
            return Err(SessionError::TooLong { seconds: max });
        }
        self.core.touch();
        self.samples.extend_from_slice(samples);
        self.total_samples += samples.len();
        self.since_partial += samples.len();
        // Frame t is final once samples up to 160·t + 200 have arrived.
        // Frame 0 also reflects at the start: it needs sample 200.
        let ready = if self.total_samples > N_FFT / 2 { (self.total_samples - N_FFT / 2) / HOP + 1 } else { 0 };
        if ready > self.next_frame {
            let core = self.core.clone();
            let _unit = core.unit();
            self.compute_frames(ready - self.next_frame);
        }
        while self.next_frame >= (self.windows + 1) * WINDOW_FRAMES {
            let core = self.core.clone();
            let _unit = core.unit();
            let mel = self.window_mel(self.windows * WINDOW_FRAMES, WINDOW_FRAMES, &[]);
            self.prefill_window(&mel, WINDOW_FRAMES);
        }
        Ok(())
    }

    fn compute_frames(&mut self, n: usize) {
        let l = &self.loaded;
        let first = self.next_frame - self.sample_base / HOP;
        let start = self.frames.len();
        l.mel.log_frames(&l.pool, l.isa, &self.samples, first, n, false, &mut self.frames);
        self.max_log = self.frames[start..].iter().fold(self.max_log, |m, &v| m.max(v));
        self.next_frame += n;
    }

    /// Normalised mel rows of `n` frames from absolute frame `first`: computed frames, then `extra`.
    fn window_mel(&self, first: usize, n: usize, extra: &[f32]) -> Vec<f32> {
        let max = extra.iter().fold(self.max_log, |m, &v| m.max(v));
        let have = (self.next_frame - first).min(n);
        let from = (first - self.frame_base) * N_MELS;
        let mut out: Vec<f32> = self.frames[from..from + have * N_MELS].iter().map(|&v| normalise(v, max)).collect();
        out.extend(extra.iter().take((n - have) * N_MELS).map(|&v| normalise(v, max)));
        out
    }

    /// Encodes a window into the audio prefix and drops what no later frame needs.
    fn prefill_window(&mut self, mel: &[f32], frames: usize) {
        let l = &self.loaded;
        let x = encode_window(&l.model.encoder, &l.pool, l.isa, mel, frames, None);
        let n = x.len() / l.model.decoder.hidden;
        decoder::forward(&l.model.decoder, &l.pool, l.isa, &mut self.cache, x, Logits::None, None);
        self.audio_prefix = self.cache.len;
        self.windows += 1;
        self.stats.windows += 1;
        self.stats.audio_prefilled += n;
        let keep_frame = self.windows * WINDOW_FRAMES;
        if keep_frame > self.frame_base && keep_frame <= self.next_frame {
            self.frames.drain(..(keep_frame - self.frame_base) * N_MELS);
            self.frame_base = keep_frame;
        }
        // Keep from the first sample a frame not yet computed needs (frames past the clip
        // start never reflect there), at a multiple of the hop; short clips keep everything.
        let need = (self.next_frame * HOP).saturating_sub(2 * HOP);
        if self.total_samples > MIN_SAMPLES && need > self.sample_base {
            self.samples.drain(..need - self.sample_base);
            self.sample_base = need;
        }
    }

    /// The frames from `next_frame` to the end of the audio so far, as if the clip ended now:
    /// reflect padding at the end, zero padding to half a second.
    fn tail_frames(&self) -> (Vec<f32>, usize) {
        let mut x = self.samples.clone();
        if self.total_samples < MIN_SAMPLES {
            x.resize(MIN_SAMPLES - self.sample_base, 0.0);
        }
        let total = frames_of(self.total_samples);
        let n = total.saturating_sub(self.next_frame);
        let mut out = Vec::new();
        let l = &self.loaded;
        l.mel.log_frames(&l.pool, l.isa, &x, self.next_frame - self.sample_base / HOP, n, true, &mut out);
        (out, total)
    }

    /// Greedy decoding after the audio: the prompt's tail and the stable text, then up to EOS
    /// or the cache's end (§2), a unit of compute per `DECODE_UNIT` steps. `None` once `stop`
    /// says so, before the prefill, after it or where it yields the unit (Q79).
    fn decode_after_audio(&mut self, stop: &dyn Fn() -> bool) -> Result<Option<Vec<u32>>, SessionError> {
        let core = self.core.clone();
        let loaded = self.loaded.clone();
        let dec = &loaded.model.decoder;
        let mut ids = self.prompt.tail.clone();
        ids.extend(&self.stable);
        if stop() {
            return Ok(None);
        }
        if !self.cache.reserve(self.cache.len + ids.len() + 1) {
            return Err(SessionError::CacheFull);
        }
        let mut unit = core.unit();
        let mut logits = decoder::forward(dec, &loaded.pool, loaded.isa, &mut self.cache, decoder::embed(dec, &ids), Logits::Last, None);
        if stop() {
            return Ok(None);
        }
        let mut out = Vec::new();
        loop {
            let t = argmax(&logits);
            if loaded.model.text.eos.contains(&t) {
                return Ok(Some(out));
            }
            out.push(t);
            if !self.cache.reserve(self.cache.len + 1) {
                return Ok(Some(out));
            }
            if out.len() % DECODE_UNIT == 0 {
                drop(unit);
                if stop() {
                    return Ok(None);
                }
                unit = core.unit();
            }
            logits = decoder::forward(dec, &loaded.pool, loaded.isa, &mut self.cache, decoder::embed(dec, &[t]), Logits::Last, None);
        }
    }

    /// The transcript text of output ids: after the `<asr_text>` header, or all of it under a
    /// forced language. Ids that end inside the header give no text yet.
    fn body(&self, ids: &[u32]) -> (String, Option<String>) {
        let text = &self.loaded.model.text;
        if self.language.is_none() && !ids.contains(&text.asr_text) {
            return (String::new(), None);
        }
        let p = parse(&text.decode(ids), self.language.as_deref());
        (p.text, p.language)
    }

    /// A partial's incomplete window: encoded and prefilled after the audio prefix. `None`
    /// when `stop` says so between the encode and the prefill (Q79).
    fn prefill_incomplete(&mut self, stop: &dyn Fn() -> bool) -> Option<()> {
        let core = self.core.clone();
        let _unit = core.unit();
        let (tail, total) = self.tail_frames();
        let first = self.windows * WINDOW_FRAMES;
        if total > first {
            let mel = self.window_mel(first, total - first, &tail);
            let l = &self.loaded;
            let x = encode_window(&l.model.encoder, &l.pool, l.isa, &mel, total - first, None);
            if stop() {
                return None;
            }
            if self.cache.reserve(self.cache.len + x.len() / l.model.decoder.hidden) {
                decoder::forward(&l.model.decoder, &l.pool, l.isa, &mut self.cache, x, Logits::None, None);
            }
        }
        Some(())
    }

    /// Re-encodes the incomplete window, decodes, and moves all but the last `rollback_tokens`
    /// of what it decoded into the stable text. The cache is rolled back to the audio prefix.
    pub fn partial(&mut self) -> Result<Partial, SessionError> {
        Ok(self.partial_unless(&|| false)?.expect("a partial nothing stops ends"))
    }

    /// [`partial`](Self::partial), stopped once `stop` says so (Q79): it asks before the
    /// encode, before the incomplete window's prefill, before the decode's prefill and after
    /// it, and where the decode yields the unit of compute. A partial stopped returns `None`
    /// and leaves the session as it was before the call: the cache rolled back to the audio
    /// prefix, the stable text and the partials counted unchanged, so a commit next decodes
    /// from the stable text already reported.
    pub fn partial_unless(&mut self, stop: &dyn Fn() -> bool) -> Result<Option<Partial>, SessionError> {
        self.core.touch();
        if self.total_samples == 0 {
            self.since_partial = 0;
            self.stats.partials += 1;
            return Ok(Some(Partial { stable: String::new(), provisional: String::new(), language: self.language.clone() }));
        }
        if stop() {
            return Ok(None);
        }
        let decoded = match self.prefill_incomplete(stop) {
            Some(()) => self.decode_after_audio(stop),
            None => Ok(None),
        };
        self.cache.truncate(self.audio_prefix);
        let Some(decoded) = decoded? else { return Ok(None) };
        self.since_partial = 0;
        self.stats.partials += 1;
        let r = self.tuning.rollback_tokens;
        let mut cut = if self.stats.partials <= self.tuning.unfixed_partials { 0 } else { decoded.len().saturating_sub(r) };
        // The stable text ends on a whole word: the provisional tail starts where a word does
        // (a token that decodes with a leading space), so an identifier the audio has not
        // finished (`model_switch.rs`) stays provisional whole. A whole word also ends on a
        // whole character, which byte-level tokens may split.
        let text = &self.loaded.model.text;
        while cut > 0 && cut < decoded.len() && !text.decode(&decoded[cut..cut + 1]).starts_with(char::is_whitespace) {
            cut -= 1;
        }
        while cut > 0 && text.decode(&[&self.stable[..], &decoded[..cut]].concat()).ends_with('\u{FFFD}') {
            cut -= 1;
        }
        self.stable.extend_from_slice(&decoded[..cut]);
        let (stable, language) = self.body(&self.stable);
        let all: Vec<u32> = [&self.stable[..], &decoded[cut..]].concat();
        let (full, _) = self.body(&all);
        let provisional = full.strip_prefix(stable.as_str()).unwrap_or(&full).to_string();
        Ok(Some(Partial { stable, provisional, language: language.or_else(|| self.language.clone()) }))
    }

    /// Encodes the rest of the audio and decodes to EOS from the stable text.
    pub fn commit(mut self) -> Result<Transcript, SessionError> {
        self.core.touch();
        let (tail, total) = {
            let core = self.core.clone();
            let _unit = core.unit();
            self.tail_frames()
        };
        // The tail frames are final now: fold them into the computed ones.
        self.max_log = tail.iter().fold(self.max_log, |m, &v| m.max(v));
        self.frames.extend_from_slice(&tail);
        self.next_frame = total;
        while self.windows * WINDOW_FRAMES < total {
            let core = self.core.clone();
            let _unit = core.unit();
            let first = self.windows * WINDOW_FRAMES;
            let n = (total - first).min(WINDOW_FRAMES);
            let mel = self.window_mel(first, n, &[]);
            self.prefill_window(&mel, n);
        }
        let out = self.decode_after_audio(&|| false)?.expect("a commit is never stopped");
        let all: Vec<u32> = [&self.stable[..], &out[..]].concat();
        let text = &self.loaded.model.text;
        let p = parse(&text.decode(&all), self.language.as_deref());
        Ok(Transcript {
            text: p.text,
            language: p.language,
            audio_seconds: self.seconds(),
            audio_tokens: audio_tokens(total),
            tokens: all,
        })
    }
}
