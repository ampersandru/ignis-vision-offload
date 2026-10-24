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
//!    back to the audio prefix.
//! 4. `commit` encodes the last window and decodes from the stable text to EOS, so the
//!    transcript begins with every stable text reported (Q43).
//!
//! The mel floor (the maximum − 8) is taken over the frames computed so far: a stream cannot
//! know a clip's maximum before its end. A whole clip pushed at once has every frame but its
//! last one or two computed before its first window is encoded.

use std::sync::Arc;

use crate::decoder::{self, argmax, Cache, Logits};
use crate::encoder::encode_window;
use crate::executor::{Core, Loaded};
use crate::mel::{self, audio_tokens, frames_of, normalise, CHUNK_FRAMES, HOP, MIN_SAMPLES, N_FFT, N_MELS, WINDOW_CHUNKS};
use crate::text::{parse, Prompt};

const WINDOW_FRAMES: usize = CHUNK_FRAMES * WINDOW_CHUNKS;

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
    lm: Arc<Loaded>,
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
}

impl Session {
    pub(crate) fn open(lm: Arc<Loaded>, core: Arc<Core>, context: &str, language: Option<&str>) -> Session {
        let guard = core.clone();
        let _unit = guard.unit();
        let s = &core.settings;
        let text = &lm.model.text;
        let prompt = Prompt::new(text, context, s.context_max_tokens, language);
        let dec = &lm.model.decoder;
        let audio_cap = audio_tokens(frames_of(s.max_clip_seconds as usize * mel::SAMPLE_RATE));
        // Sized for the context cap, the longest clip's audio tokens and the decode (§2).
        let cap = prompt.head.len() + audio_cap + prompt.tail.len() + audio_cap / 2 + 64;
        let mut cache = Cache::new(dec, cap);
        let mut stats = Stats::default();
        {
            let mut entry = core.prefix.lock().unwrap_or_else(|e| e.into_inner());
            match entry.as_ref() {
                Some(e) if e.head == prompt.head => cache.copy_prefix(&e.cache, e.head.len()),
                _ => {
                    let x = decoder::embed(dec, &prompt.head);
                    decoder::forward(dec, &lm.pool, lm.isa, &mut cache, x, Logits::None, None);
                    stats.context_prefilled = prompt.context_tokens;
                    let mut keep = Cache::new(dec, prompt.head.len());
                    keep.copy_prefix(&cache, prompt.head.len());
                    *entry = Some(crate::executor::PrefixEntry { head: prompt.head.clone(), cache: keep });
                }
            }
        }
        let audio_prefix = cache.len;
        Session {
            lm,
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
        }
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
        let lm = &self.lm;
        let first = self.next_frame - self.sample_base / HOP;
        let start = self.frames.len();
        lm.mel.log_frames(&lm.pool, lm.isa, &self.samples, first, n, false, &mut self.frames);
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
        let lm = &self.lm;
        let x = encode_window(&lm.model.encoder, &lm.pool, lm.isa, mel, frames, None);
        let n = x.len() / lm.model.decoder.hidden;
        decoder::forward(&lm.model.decoder, &lm.pool, lm.isa, &mut self.cache, x, Logits::None, None);
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
        let lm = &self.lm;
        lm.mel.log_frames(&lm.pool, lm.isa, &x, self.next_frame - self.sample_base / HOP, n, true, &mut out);
        (out, total)
    }

    /// Greedy decoding after the audio: the prompt's tail and the stable text, then up to EOS.
    fn decode_after_audio(&mut self) -> Result<Vec<u32>, SessionError> {
        let lm = &self.lm;
        let dec = &lm.model.decoder;
        let mut ids = self.prompt.tail.clone();
        ids.extend(&self.stable);
        if self.cache.len + ids.len() >= self.cache.capacity() {
            return Err(SessionError::CacheFull);
        }
        let mut logits = decoder::forward(dec, &lm.pool, lm.isa, &mut self.cache, decoder::embed(dec, &ids), Logits::Last, None);
        let mut out = Vec::new();
        loop {
            let t = argmax(&logits);
            if lm.model.text.eos.contains(&t) {
                return Ok(out);
            }
            out.push(t);
            if self.cache.len >= self.cache.capacity() {
                return Err(SessionError::CacheFull);
            }
            logits = decoder::forward(dec, &lm.pool, lm.isa, &mut self.cache, decoder::embed(dec, &[t]), Logits::Last, None);
        }
    }

    /// The transcript text of output ids: after the `<asr_text>` header, or all of it under a
    /// forced language. Ids that end inside the header give no text yet.
    fn body(&self, ids: &[u32]) -> (String, Option<String>) {
        let text = &self.lm.model.text;
        if self.language.is_none() && !ids.contains(&text.asr_text) {
            return (String::new(), None);
        }
        let p = parse(&text.decode(ids), self.language.as_deref());
        (p.text, p.language)
    }

    /// Re-encodes the incomplete window, decodes, and moves all but the last `rollback_tokens`
    /// of what it decoded into the stable text. The cache is rolled back to the audio prefix.
    pub fn partial(&mut self) -> Result<Partial, SessionError> {
        self.core.touch();
        self.since_partial = 0;
        self.stats.partials += 1;
        if self.total_samples == 0 {
            return Ok(Partial { stable: String::new(), provisional: String::new(), language: self.language.clone() });
        }
        let core = self.core.clone();
        let _unit = core.unit();
        let (tail, total) = self.tail_frames();
        let first = self.windows * WINDOW_FRAMES;
        let decoded = if total > first {
            let mel = self.window_mel(first, total - first, &tail);
            let lm = &self.lm;
            let x = encode_window(&lm.model.encoder, &lm.pool, lm.isa, &mel, total - first, None);
            decoder::forward(&lm.model.decoder, &lm.pool, lm.isa, &mut self.cache, x, Logits::None, None);
            self.decode_after_audio()
        } else {
            self.decode_after_audio()
        };
        self.cache.truncate(self.audio_prefix);
        let decoded = decoded?;
        let r = self.core.settings.rollback_tokens;
        let mut cut = decoded.len().saturating_sub(r);
        // The stable text ends on a whole character: byte-level tokens may split one.
        while cut > 0 && self.lm.model.text.decode(&[&self.stable[..], &decoded[..cut]].concat()).ends_with('\u{FFFD}') {
            cut -= 1;
        }
        self.stable.extend_from_slice(&decoded[..cut]);
        let (stable, language) = self.body(&self.stable);
        let all: Vec<u32> = [&self.stable[..], &decoded[cut..]].concat();
        let (full, _) = self.body(&all);
        let provisional = full.strip_prefix(stable.as_str()).unwrap_or(&full).to_string();
        Ok(Partial { stable, provisional, language: language.or_else(|| self.language.clone()) })
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
        let out = {
            let core = self.core.clone();
            let _unit = core.unit();
            self.decode_after_audio()?
        };
        let all: Vec<u32> = [&self.stable[..], &out[..]].concat();
        let text = &self.lm.model.text;
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
