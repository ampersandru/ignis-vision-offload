//! `server_vad`'s voice activity detection (spec speech/05): Silero VAD on the CPU, and the turns
//! it finds in a stream of 16 kHz samples.
//!
//! - [`Vad`] is Silero VAD 6.2.3's 16 kHz model: one speech probability per 512-sample frame
//!   (32 ms), each frame read with the 64 samples before it, an LSTM state carried from frame to
//!   frame. Single-threaded, under a million multiply-adds a frame, so it runs where the audio
//!   arrives, never on the speech worker.
//! - [`Turns`] reads the stream frame by frame and finds the turns in it: a turn starts at the
//!   first frame whose speech probability reaches the threshold, with the prefix padding before
//!   it (silence, even the end of the turn before, never an earlier turn's speech), and ends
//!   once the silence after its last such frame lasts the silence duration, that silence
//!   included; with an idle timeout, as much audio without speech is a turn of its own. Every
//!   length is in samples of the stream, from its first.
//!
//! Provenance (ADR 0051, an adapted op): the graph is ported from Silero VAD
//! (<https://github.com/snakers4/silero-vad>, tag v6.2.3, commit 5cd79456): the model of
//! `src/silero_vad/tinygrad_model.py`, checked against the TorchScript model
//! `src/silero_vad/data/silero_vad.jit` it describes, and driven as `src/silero_vad/utils_vad.py`
//! drives it (a fresh state, the 64-sample context, 512-sample chunks). The weights are that
//! `.jit`'s, converted once by `tools/speech-vad/record.py` into `crates/speech/vad/`. MIT
//! License, Copyright (c) 2020-present Silero Team; the licence is in `NOTICE` and beside the
//! weights.

use std::collections::VecDeque;
use std::sync::OnceLock;

use serde_json::Value;

/// Samples in one frame at 16 kHz: 32 ms, the model's chunk.
pub const FRAME: usize = 512;

/// Samples per millisecond at 16 kHz.
const PER_MS: u64 = 16;

/// The digital silence the server puts before each `server_vad` turn's audio when it feeds the
/// turn to the speech model, in milliseconds: the default of `speech.vad_leading_silence_ms`
/// (spec speech/05, acceptance 7). Qwen3-ASR's transcript moves with the audio before the
/// speech, and a turn starts only its prefix padding before it.
pub const DEFAULT_LEADING_SILENCE_MS: u32 = 500;

/// `ms` milliseconds of digital silence at 16 kHz: what goes before a turn's audio.
pub fn leading_silence(ms: u32) -> Vec<f32> {
    vec![0.0; ms as usize * PER_MS as usize]
}

/// What reads a frame: its speech probability, from 0 to 1. The model, or a script in tests.
pub trait SpeechProbability {
    fn frame(&mut self, samples: &[f32]) -> f32;
}

impl<F: FnMut(&[f32]) -> f32> SpeechProbability for F {
    fn frame(&mut self, samples: &[f32]) -> f32 {
        self(samples)
    }
}

/// The samples before a frame that the model reads with it.
const CONTEXT: usize = 64;
/// The STFT: 256-sample windows every 128 samples, 129 bins.
const WINDOW: usize = 256;
const HOP: usize = 128;
const BINS: usize = WINDOW / 2 + 1;
/// The reflect padding after the frame, a quarter of a window.
const PAD: usize = WINDOW / 4;
/// The windows in one frame read with its context and its padding.
const STEPS: usize = (CONTEXT + FRAME + PAD - WINDOW) / HOP + 1;
/// The LSTM's width.
const HIDDEN: usize = 128;

/// The model's weights, built into the binary: Silero VAD 6.2.3's 16 kHz model
/// (`crates/speech/vad/README.md`).
static WEIGHTS: &[u8] = include_bytes!("../vad/silero-vad-16k.safetensors");

/// A 1-D convolution, kernel 3, padding 1, then a ReLU.
struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    stride: usize,
}

impl Conv {
    /// `x`, `cin` rows of `len`: the output, `cout` rows, and its length.
    fn apply(&self, x: &[f32], len: usize) -> (Vec<f32>, usize) {
        let (padded, lout) = (len + 2, (len - 1) / self.stride + 1);
        let mut xp = vec![0f32; self.cin * padded];
        for i in 0..self.cin {
            xp[i * padded + 1..][..len].copy_from_slice(&x[i * len..][..len]);
        }
        let mut y = vec![0f32; self.cout * lout];
        for o in 0..self.cout {
            for t in 0..lout {
                let mut acc = 0f32;
                for i in 0..self.cin {
                    let (w, x) = (&self.w[(o * self.cin + i) * 3..][..3], &xp[i * padded + t * self.stride..][..3]);
                    acc += w[0] * x[0] + w[1] * x[1] + w[2] * x[2];
                }
                y[o * lout + t] = (acc + self.b[o]).max(0.0);
            }
        }
        (y, lout)
    }
}

/// Silero VAD's 16 kHz model.
struct Silero {
    /// The STFT's basis: the real parts of the 129 bins, then the imaginary, 256 taps each.
    stft: Vec<f32>,
    convs: [Conv; 4],
    /// The LSTM cell's input and hidden weights, 512 rows of 128: the gates i, f, g, o.
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
    /// The 1x1 convolution to one logit.
    out_w: Vec<f32>,
    out_b: f32,
}

impl Silero {
    /// The built-in weights, read once.
    fn get() -> &'static Silero {
        static MODEL: OnceLock<Silero> = OnceLock::new();
        MODEL.get_or_init(|| Silero::read(WEIGHTS).expect("the VAD's built-in weights read"))
    }

    /// A safetensors file of the converted weights.
    fn read(file: &[u8]) -> Result<Silero, String> {
        let n = u64::from_le_bytes(file.get(..8).ok_or("no header")?.try_into().unwrap()) as usize;
        let header: Value = serde_json::from_slice(file.get(8..8 + n).ok_or("short header")?).map_err(|e| e.to_string())?;
        let data = &file[8 + n..];
        let dims = |v: &Value| -> Vec<usize> { v.as_array().map(|a| a.iter().filter_map(Value::as_u64).map(|d| d as usize).collect()).unwrap_or_default() };
        let tensor = |name: &str, shape: &[usize]| -> Result<Vec<f32>, String> {
            let t = &header[name];
            if t["dtype"] != "F32" || dims(&t["shape"]) != shape {
                return Err(format!("{name}: {} {:?}, not F32 {shape:?}", t["dtype"], dims(&t["shape"])));
            }
            let at = dims(&t["data_offsets"]);
            let bytes = (at.len() == 2)
                .then(|| data.get(at[0]..at[1]))
                .flatten()
                .filter(|b| b.len() == 4 * shape.iter().product::<usize>())
                .ok_or(format!("{name}: bad offsets"))?;
            Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
        };
        let conv = |n: usize, cin: usize, cout: usize, stride: usize| -> Result<Conv, String> {
            Ok(Conv { w: tensor(&format!("conv{n}.weight"), &[cout, cin, 3])?, b: tensor(&format!("conv{n}.bias"), &[cout])?, cin, cout, stride })
        };
        Ok(Silero {
            stft: tensor("stft_conv.weight", &[2 * BINS, 1, WINDOW])?,
            convs: [conv(1, BINS, 128, 1)?, conv(2, 128, 64, 2)?, conv(3, 64, 64, 2)?, conv(4, 64, HIDDEN, 1)?],
            w_ih: tensor("lstm_cell.weight_ih", &[4 * HIDDEN, HIDDEN])?,
            w_hh: tensor("lstm_cell.weight_hh", &[4 * HIDDEN, HIDDEN])?,
            b_ih: tensor("lstm_cell.bias_ih", &[4 * HIDDEN])?,
            b_hh: tensor("lstm_cell.bias_hh", &[4 * HIDDEN])?,
            out_w: tensor("final_conv.weight", &[1, HIDDEN, 1])?,
            out_b: tensor("final_conv.bias", &[1])?[0],
        })
    }
}

/// A dot product over lengths that are multiples of 8, in eight lanes.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert!(a.len() == b.len() && a.len() % 8 == 0);
    let mut acc = [0f32; 8];
    for (x, y) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    acc.iter().sum()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One stream's VAD: Silero's model and the state it carries from frame to frame.
pub struct Vad {
    model: &'static Silero,
    /// The last 64 samples of the frame before.
    context: [f32; CONTEXT],
    h: [f32; HIDDEN],
    c: [f32; HIDDEN],
}

impl Default for Vad {
    fn default() -> Self {
        Self::new()
    }
}

impl Vad {
    /// A fresh state, as the reference's `reset_states()` leaves it.
    pub fn new() -> Vad {
        Vad { model: Silero::get(), context: [0.0; CONTEXT], h: [0.0; HIDDEN], c: [0.0; HIDDEN] }
    }
}

impl SpeechProbability for Vad {
    /// One frame of [`FRAME`] samples: its speech probability.
    fn frame(&mut self, samples: &[f32]) -> f32 {
        assert_eq!(samples.len(), FRAME, "the VAD reads 512-sample frames");
        let m = self.model;
        // The frame after its context, reflect-padded at the end.
        let mut x = [0f32; CONTEXT + FRAME + PAD];
        x[..CONTEXT].copy_from_slice(&self.context);
        x[CONTEXT..CONTEXT + FRAME].copy_from_slice(samples);
        for i in 0..PAD {
            x[CONTEXT + FRAME + i] = x[CONTEXT + FRAME - 2 - i];
        }
        self.context.copy_from_slice(&samples[FRAME - CONTEXT..]);
        // The STFT's magnitude, 129 bins by 4 steps.
        let mut mag = vec![0f32; BINS * STEPS];
        for t in 0..STEPS {
            let window = &x[t * HOP..][..WINDOW];
            for f in 0..BINS {
                let re = dot(&m.stft[f * WINDOW..][..WINDOW], window);
                let im = dot(&m.stft[(BINS + f) * WINDOW..][..WINDOW], window);
                mag[f * STEPS + t] = (re * re + im * im).sqrt();
            }
        }
        let (mut y, mut len) = (mag, STEPS);
        for conv in &m.convs {
            (y, len) = conv.apply(&y, len);
        }
        debug_assert_eq!(len, 1);
        // The LSTM cell, then the logit from the ReLU of its output.
        let mut gates = [0f32; 4 * HIDDEN];
        for (j, gate) in gates.iter_mut().enumerate() {
            *gate = dot(&m.w_ih[j * HIDDEN..][..HIDDEN], &y) + m.b_ih[j] + dot(&m.w_hh[j * HIDDEN..][..HIDDEN], &self.h) + m.b_hh[j];
        }
        let mut logit = m.out_b;
        for k in 0..HIDDEN {
            let (i, f, g, o) = (sigmoid(gates[k]), sigmoid(gates[HIDDEN + k]), gates[2 * HIDDEN + k].tanh(), sigmoid(gates[3 * HIDDEN + k]));
            self.c[k] = f * self.c[k] + i * g;
            self.h[k] = o * self.c[k].tanh();
            logit += m.out_w[k] * self.h[k].max(0.0);
        }
        sigmoid(logit)
    }
}

/// How [`Turns`] finds a turn: `server_vad`'s fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurnSettings {
    /// The probability at or over which a frame is speech.
    pub threshold: f32,
    /// The audio a turn keeps from before its first speech frame.
    pub prefix_padding_ms: u32,
    /// The silence after a turn's last speech frame that ends it.
    pub silence_duration_ms: u32,
    /// The audio without speech after which that audio is a turn of its own; `None` never.
    pub idle_timeout_ms: Option<u32>,
}

/// What [`Turns::push`] finds, in stream order.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnEvent {
    /// Speech: a turn begins at sample `start`, its prefix padding included. Its audio follows
    /// as [`TurnEvent::Audio`].
    Started { start: u64 },
    /// The current turn's next samples.
    Audio(Vec<f32>),
    /// The turn's silence lasted the silence duration: the turn ends at sample `end`, that
    /// silence included.
    Stopped { end: u64 },
    /// No speech for the idle timeout: the samples `start..end`, `audio`, are a turn of their
    /// own, and it has ended.
    TimedOut { start: u64, end: u64, audio: Vec<f32> },
}

enum State {
    Idle,
    /// The end of the turn's last speech frame.
    Speaking { last_speech: u64 },
}

/// The turns in one stream of 16 kHz samples.
pub struct Turns<V: SpeechProbability> {
    vad: V,
    settings: TurnSettings,
    state: State,
    /// The samples read into frames so far: where `held` begins.
    position: u64,
    /// The samples of the frame not yet complete.
    held: Vec<f32>,
    /// The latest samples, as many as a turn's prefix padding takes.
    ring: VecDeque<f32>,
    /// How far back a prefix padding may reach: the end of the last speech frame, or where a
    /// commit or a clear left the stream. Padding takes silence, never an earlier item's speech.
    floor: u64,
    /// Where the audio without speech began: the stream's start, the last turn's end or the
    /// last idle timeout.
    idle_since: u64,
    /// With an idle timeout, every sample since `idle_since`.
    idle: Vec<f32>,
    /// The first samples of `held` that a commit gave to its item or a clear dropped: the
    /// model reads them, nothing routes them again.
    skip: usize,
}

impl<V: SpeechProbability> Turns<V> {
    /// No turn yet: the stream's samples so far are `at`.
    pub fn new(vad: V, settings: TurnSettings, at: u64) -> Self {
        Self {
            vad,
            settings,
            state: State::Idle,
            position: at,
            held: Vec::with_capacity(FRAME),
            ring: VecDeque::new(),
            floor: at,
            idle_since: at,
            idle: Vec::new(),
            skip: 0,
        }
    }

    /// The stream's next samples, and what they hold onto `out`.
    pub fn push(&mut self, mut samples: &[f32], out: &mut Vec<TurnEvent>) {
        while !samples.is_empty() {
            let take = (FRAME - self.held.len()).min(samples.len());
            self.held.extend_from_slice(&samples[..take]);
            samples = &samples[take..];
            if self.held.len() == FRAME {
                self.frame(out);
            }
        }
    }

    /// A turn is on.
    pub fn speaking(&self) -> bool {
        matches!(self.state, State::Speaking { .. })
    }

    /// New settings, from the next frame on. A changed idle timeout counts from here: the
    /// audio before it was not kept for one.
    pub fn set_settings(&mut self, settings: TurnSettings) {
        if settings.idle_timeout_ms != self.settings.idle_timeout_ms {
            self.idle.clear();
            self.idle_since = self.position + self.skip as u64;
        }
        self.settings = settings;
    }

    /// A commit during a turn: the turn ends here, and its samples not yet read into a frame
    /// come back for its item. With no turn on, nothing.
    pub fn end_turn(&mut self) -> Vec<f32> {
        if !self.speaking() {
            return Vec::new();
        }
        let rest = self.held[self.skip..].to_vec();
        self.clear();
        rest
    }

    /// A clear: what is not in a turn yet goes nowhere, and a turn on ends without its last
    /// samples.
    pub fn clear(&mut self) {
        self.skip = self.held.len();
        self.ring.clear();
        self.idle.clear();
        self.state = State::Idle;
        self.floor = self.position + self.held.len() as u64;
        self.idle_since = self.floor;
    }

    /// The frame in `held` is complete: read it and route its samples.
    fn frame(&mut self, out: &mut Vec<TurnEvent>) {
        let frame = std::mem::replace(&mut self.held, Vec::with_capacity(FRAME));
        let speech = self.vad.frame(&frame) >= self.settings.threshold;
        let (skip, end) = (std::mem::take(&mut self.skip), self.position + FRAME as u64);
        let (fresh, from) = (&frame[skip..], self.position + skip as u64);
        self.position = end;
        match self.state {
            State::Idle if speech => {
                let pad = ms_samples(self.settings.prefix_padding_ms).min(self.ring.len() as u64).min(from - self.floor) as usize;
                let mut audio: Vec<f32> = self.ring.range(self.ring.len() - pad..).copied().collect();
                audio.extend_from_slice(fresh);
                self.idle.clear();
                out.push(TurnEvent::Started { start: from - pad as u64 });
                out.push(TurnEvent::Audio(audio));
                self.state = State::Speaking { last_speech: end };
            }
            State::Idle => {
                if let Some(idle) = self.settings.idle_timeout_ms.map(ms_samples) {
                    self.idle.extend_from_slice(fresh);
                    if end - self.idle_since >= idle {
                        let audio = std::mem::take(&mut self.idle);
                        out.push(TurnEvent::TimedOut { start: self.idle_since, end, audio });
                        self.idle_since = end;
                    }
                }
            }
            State::Speaking { last_speech } => {
                out.push(TurnEvent::Audio(fresh.to_vec()));
                if speech {
                    self.state = State::Speaking { last_speech: end };
                } else if end - last_speech >= ms_samples(self.settings.silence_duration_ms) {
                    out.push(TurnEvent::Stopped { end });
                    self.state = State::Idle;
                    (self.floor, self.idle_since) = (last_speech, end);
                }
            }
        }
        self.ring.extend(fresh);
        let keep = ms_samples(self.settings.prefix_padding_ms) as usize;
        if self.ring.len() > keep {
            self.ring.drain(..self.ring.len() - keep);
        }
    }
}

/// `ms` milliseconds in samples at 16 kHz.
fn ms_samples(ms: u32) -> u64 {
    u64::from(ms) * PER_MS
}

#[cfg(test)]
mod tests {
    use super::*;

    const SETTINGS: TurnSettings = TurnSettings { threshold: 0.5, prefix_padding_ms: 64, silence_duration_ms: 96, idle_timeout_ms: None };

    /// A script: each sample's value is its frame's probability.
    fn scripted(samples: &[f32]) -> f32 {
        samples[0]
    }

    /// One frame of probability `p`, every sample `p`.
    fn frames(ps: &[f32]) -> Vec<f32> {
        ps.iter().flat_map(|&p| std::iter::repeat_n(p, FRAME)).collect()
    }

    fn turns(settings: TurnSettings) -> Turns<fn(&[f32]) -> f32> {
        Turns::new(scripted as fn(&[f32]) -> f32, settings, 0)
    }

    /// What a test compares: an event, with a turn's audio as its length.
    #[derive(Debug, PartialEq)]
    enum Seen {
        Started(u64),
        Audio(usize),
        Stopped(u64),
        TimedOut(u64, u64, usize),
    }

    /// The events `samples` give, each run of audio joined, and that audio in order.
    fn run(turns: &mut Turns<fn(&[f32]) -> f32>, samples: &[f32]) -> (Vec<Seen>, Vec<f32>) {
        let mut out = Vec::new();
        turns.push(samples, &mut out);
        let (mut seen, mut audio) = (Vec::new(), Vec::new());
        for event in out {
            match event {
                TurnEvent::Started { start } => seen.push(Seen::Started(start)),
                TurnEvent::Audio(more) => {
                    match seen.last_mut() {
                        Some(Seen::Audio(n)) => *n += more.len(),
                        _ => seen.push(Seen::Audio(more.len())),
                    }
                    audio.extend(more);
                }
                TurnEvent::Stopped { end } => seen.push(Seen::Stopped(end)),
                TurnEvent::TimedOut { start, end, audio: silent } => {
                    seen.push(Seen::TimedOut(start, end, silent.len()));
                    audio.extend(silent);
                }
            }
        }
        (seen, audio)
    }

    /// AC 2, 3: the first frame at the threshold starts a turn whose audio begins
    /// `prefix_padding_ms` before it (64 ms, two frames at 16 kHz).
    #[test]
    fn a_frame_at_the_threshold_starts_a_turn_with_its_prefix_padding() {
        let mut t = turns(SETTINGS);
        let stream = frames(&[0.1, 0.2, 0.3, 0.5, 0.9]);
        let (seen, audio) = run(&mut t, &stream);
        assert_eq!(seen, [Seen::Started(FRAME as u64), Seen::Audio(4 * FRAME)]);
        assert!(audio == stream[FRAME..], "the turn's audio is the padding and the frames from the speech on");
    }

    /// AC 2, 3: the turn ends once the silence after its last speech frame lasts
    /// `silence_duration_ms` (96 ms, three frames), that silence included; a shorter silence
    /// keeps it, and a frame just under the threshold is silence.
    #[test]
    fn the_silence_duration_ends_a_turn_and_its_silence_is_in_it() {
        let mut t = turns(SETTINGS);
        let stream = frames(&[0.9, 0.2, 0.49, 0.5, 0.1, 0.1, 0.1, 0.1]);
        let (seen, audio) = run(&mut t, &stream);
        assert_eq!(seen, [Seen::Started(0), Seen::Audio(7 * FRAME), Seen::Stopped(7 * FRAME as u64)]);
        assert!(audio == stream[..7 * FRAME]);
    }

    /// AC 3: a turn's prefix padding may take the silence that ended the turn before it, never
    /// its speech (128 ms of padding asked, 64 ms of silence ending a turn: the next turn, its
    /// speech right after, starts at the last turn's last speech frame's end); the positions
    /// count from the samples the stream had when the turns began; and how the samples are cut
    /// into pushes changes nothing.
    #[test]
    fn the_next_turn_pads_with_the_silence_before_it_never_the_speech_and_pushes_cut_anywhere() {
        let settings = TurnSettings { prefix_padding_ms: 128, silence_duration_ms: 64, ..SETTINGS };
        let stream = frames(&[0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1]);
        let (at, f) = (1_000, FRAME as u64);
        let want = [
            Seen::Started(at),
            Seen::Audio(4 * FRAME),
            Seen::Stopped(at + 4 * f),
            Seen::Started(at + 2 * f),
            Seen::Audio(5 * FRAME),
            Seen::Stopped(at + 7 * f),
        ];
        for cut in [stream.len(), 300, 1] {
            let mut t = Turns::new(scripted as fn(&[f32]) -> f32, settings, at);
            let (mut seen, mut audio) = (Vec::new(), Vec::new());
            for piece in stream.chunks(cut) {
                let (more, samples) = run(&mut t, piece);
                for event in more {
                    match (seen.last_mut(), event) {
                        (Some(Seen::Audio(n)), Seen::Audio(m)) => *n += m,
                        (_, event) => seen.push(event),
                    }
                }
                audio.extend(samples);
            }
            assert_eq!(seen, want, "pushes of {cut}");
            let mut twice = stream[..4 * FRAME].to_vec();
            twice.extend_from_slice(&stream[2 * FRAME..]);
            assert!(audio == twice, "pushes of {cut}: the silence between the turns is in both, the speech in one");
        }
    }

    /// `idle_timeout_ms` (96 ms, three frames): that much audio without speech, counted from
    /// the stream's start or the last turn's end, is a turn of its own, and the clock starts
    /// again after it; without the field there is none.
    #[test]
    fn the_idle_timeout_makes_a_turn_of_the_silence() {
        let idle = TurnSettings { idle_timeout_ms: Some(96), ..SETTINGS };
        let f = FRAME as u64;
        let (seen, audio) = run(&mut turns(idle), &frames(&[0.1, 0.1, 0.1, 0.1, 0.9]));
        assert_eq!(seen, [Seen::TimedOut(0, 3 * f, 3 * FRAME), Seen::Started(2 * f), Seen::Audio(3 * FRAME)], "the padding is silence, wherever it went");
        assert_eq!(audio.len(), 6 * FRAME);
        let (seen, _) = run(&mut turns(idle), &frames(&[0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]));
        assert_eq!(seen, [Seen::Started(0), Seen::Audio(4 * FRAME), Seen::Stopped(4 * f), Seen::TimedOut(4 * f, 7 * f, 3 * FRAME)]);
        let (seen, _) = run(&mut turns(SETTINGS), &frames(&[0.1; 40]));
        assert_eq!(seen, [], "no idle timeout");
    }

    /// An idle timeout set, changed or dropped mid-stream counts from the change: the silence
    /// before it is no turn, and a timed-out turn holds every sample it spans.
    #[test]
    fn a_new_idle_timeout_counts_from_the_change() {
        let f = FRAME as u64;
        let mut t = turns(SETTINGS);
        let (seen, _) = run(&mut t, &frames(&[0.1; 10]));
        assert_eq!(seen, []);
        t.set_settings(TurnSettings { idle_timeout_ms: Some(96), ..SETTINGS });
        let (seen, _) = run(&mut t, &frames(&[0.1; 4]));
        assert_eq!(seen, [Seen::TimedOut(10 * f, 13 * f, 3 * FRAME)]);
        t.set_settings(SETTINGS);
        let (seen, _) = run(&mut t, &frames(&[0.1; 2]));
        assert_eq!(seen, []);
        t.set_settings(TurnSettings { idle_timeout_ms: Some(96), ..SETTINGS });
        let (seen, _) = run(&mut t, &frames(&[0.1; 3]));
        assert_eq!(seen, [Seen::TimedOut(16 * f, 19 * f, 3 * FRAME)], "nothing from before the change, no gap");
    }

    /// A commit during a turn ends it at once: the samples it holds not yet read into a frame
    /// come back for the item, and are never given again. A clear drops what no turn holds.
    /// Either way the model still reads them, and a turn after starts no earlier.
    #[test]
    fn a_commit_ends_the_turn_and_a_clear_drops_what_is_pending() {
        let f = FRAME as u64;
        let mut t = turns(SETTINGS);
        let speech = frames(&[0.9; 3]);
        let (seen, _) = run(&mut t, &speech[..2 * FRAME + 100]);
        assert_eq!(seen, [Seen::Started(0), Seen::Audio(2 * FRAME)]);
        assert!(t.speaking());
        assert_eq!(t.end_turn().len(), 100);
        assert!(!t.speaking());
        assert!(t.end_turn().is_empty(), "no turn, nothing more");
        let (seen, audio) = run(&mut t, &speech[2 * FRAME + 100..]);
        assert_eq!(seen, [Seen::Started(2 * f + 100), Seen::Audio(FRAME - 100)], "the frame's first 100 samples went with the last turn");
        assert_eq!(audio.len(), FRAME - 100);

        let mut t = turns(SETTINGS);
        let (seen, _) = run(&mut t, &frames(&[0.1, 0.1])[..2 * FRAME - 12]);
        assert_eq!(seen, []);
        t.clear();
        let mut rest = vec![0.1; 12];
        rest.extend(frames(&[0.1, 0.9]));
        let (seen, audio) = run(&mut t, &rest);
        assert_eq!(seen, [Seen::Started(2 * f - 12), Seen::Audio(2 * FRAME + 12)], "the padding is only what came after the clear");
        assert_eq!(audio.len(), 2 * FRAME + 12);
    }

    /// A change of the settings holds from the next frame.
    #[test]
    fn new_settings_hold_from_the_next_frame() {
        let mut t = turns(SETTINGS);
        let (seen, _) = run(&mut t, &frames(&[0.6]));
        assert_eq!(seen, [Seen::Started(0), Seen::Audio(FRAME)]);
        t.set_settings(TurnSettings { threshold: 0.7, silence_duration_ms: 32, ..SETTINGS });
        let (seen, _) = run(&mut t, &frames(&[0.6]));
        assert_eq!(seen, [Seen::Audio(FRAME), Seen::Stopped(2 * FRAME as u64)], "0.6 is silence at 0.7, and 32 ms of it ends the turn");
    }
}
