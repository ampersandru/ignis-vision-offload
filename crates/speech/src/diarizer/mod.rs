//! The diarizer (spec speech/08, ADR 0069): Nemotron 3 Diarization on the CPU, who spoke when in a
//! stream of 16 kHz samples, never what was said.
//!
//! - [`mel`]: its front end, a 128-bin natural log-mel every 10 ms.
//! - [`model`]: its weights and one step's forward pass.
//! - [`cache`]: its streaming state, the arrival-order speaker cache and the FIFO.
//! - [`speakers`]: who speaks each frame, an item's speaker and a change of speaker (Q107, Q108).
//! - [`Diarizer`] and [`Stream`]: a stream fed sample by sample and scored a step at a time, each
//!   step one chunk of its [`Geometry`] with its look-ahead, as the checkpoint's processor chunks
//!   it: a step waits for the samples the processor would give it, and the stream's end scores
//!   what is left in one last step with no look-ahead. Each step gives its chunk's frames: every
//!   80 ms frame's 8 speaker probabilities, the speakers numbered in the order they first spoke.
//!
//! Provenance (ADR 0051, an adapted op): the graph and its streaming state are ported from the
//! checkpoint's Transformers code (<https://github.com/huggingface/transformers>, tag v5.20.2,
//! commit 18329f10): `src/transformers/models/nemotron3_diarization/`, which restates NVIDIA NeMo's
//! streaming Sortformer (<https://github.com/NVIDIA/NeMo>). Both are Apache-2.0, Copyright 2026 The
//! HuggingFace Inc. team and Copyright NVIDIA CORPORATION. The port is checked against that code
//! running the checkpoint `nvidia/Nemotron-3-Diarization` @ f667ed73
//! (`tools/speech-diarizer/record.py`). The weights are not in the tree: they keep their
//! OpenMDW-1.1 licence beside them. The licences are in `NOTICE`.

use std::path::Path;

use crate::cpu::Isa;
use crate::model::LoadError;
use crate::pool::Pool;

pub mod cache;
pub mod mel;
pub mod model;
pub mod speakers;

pub use cache::{Probs, SpeakerCache};
pub use model::{LoadReport, Weights};

/// The speakers the model tells apart: its channels.
pub const SPEAKERS: usize = 8;
/// Samples per 80 ms frame.
pub const FRAME_SAMPLES: usize = model::STACK * mel::HOP;

/// Called with a tensor's name and its rows: the oracle's taps.
pub type Tap<'a> = &'a mut dyn FnMut(&str, &[f32]);

/// A streaming geometry, in 80 ms frames: the speaker cache's length, the FIFO's, a step's chunk,
/// its look-ahead (right context), and the frames the FIFO gives the cache when it overflows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub speaker_cache: usize,
    pub fifo: usize,
    pub chunk: usize,
    pub right_context: usize,
    pub update_period: usize,
}

impl Geometry {
    /// Spec 08's live default (Q106): a 2 s chunk, 2.32 s of latency.
    pub const LIVE: Geometry = Geometry { speaker_cache: 264, fifo: 264, chunk: 25, right_context: 4, update_period: 222 };
    /// The card's offline geometry, for a stream fed faster than real time.
    pub const FILE: Geometry = Geometry { speaker_cache: 264, fifo: 40, chunk: 340, right_context: 40, update_period: 300 };
}

/// The speaker cache's constants (the checkpoint's `streaming_config`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constants {
    pub prediction_score_threshold: f32,
    pub latest_frames_score_boost: f32,
    pub strong_boost_rate: f32,
    pub weak_boost_rate: f32,
    pub min_positive_scores_rate: f32,
    pub speaker_cache_silence_frames_per_speaker: usize,
}

/// The diarizer loaded: its weights and its front end, shared by every stream.
pub struct Diarizer {
    weights: Weights,
    front: mel::FrontEnd,
}

impl Diarizer {
    pub fn load(path: &Path, pinned: bool) -> Result<Diarizer, LoadError> {
        Ok(Diarizer { weights: Weights::load(path, pinned)?, front: mel::FrontEnd::new() })
    }

    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// A new stream: nothing heard.
    pub fn open(&self, geometry: Geometry, constants: Constants) -> Stream {
        Stream {
            geometry,
            cache: SpeakerCache::new(self.weights.d, geometry, constants),
            width: self.weights.d,
            constants,
            history: None,
            samples: Vec::new(),
            x0: 0,
            heard: 0,
            done: 0,
            steps: 0,
            ended: false,
        }
    }
}

/// One step's frames.
#[derive(Clone, Debug)]
pub struct Step {
    /// Its first 80 ms frame, a frame of the stream from its first.
    pub first: usize,
    /// Each 80 ms frame's speaker probabilities (the mean of its 8 rows of 10 ms).
    pub frames: Vec<Probs>,
    /// Each 10 ms row's speaker probabilities: the model's output, `σ(logit)`.
    pub probs: Vec<Probs>,
}

/// A stream of 16 kHz samples, scored a step at a time. A clone finished scores what is pending
/// with no more audio, and leaves the stream as it was.
#[derive(Clone)]
pub struct Stream {
    geometry: Geometry,
    cache: SpeakerCache,
    width: usize,
    constants: Constants,
    /// Every scored frame's embedding, kept for the oracle's forcing.
    history: Option<Vec<f32>>,
    /// The samples the next steps still read: from sample `x0`, with the one before it.
    samples: Vec<f32>,
    x0: usize,
    heard: usize,
    /// The 80 ms frames scored.
    done: usize,
    steps: usize,
    ended: bool,
}

impl Stream {
    pub fn geometry(&self) -> Geometry {
        self.geometry
    }

    pub fn push(&mut self, samples: &[f32]) {
        assert!(!self.ended, "samples after the stream's end");
        self.samples.extend_from_slice(samples);
        self.heard += samples.len();
    }

    /// The samples heard.
    pub fn heard(&self) -> usize {
        self.heard
    }

    /// The 80 ms frames scored so far.
    pub fn frames_done(&self) -> usize {
        self.done
    }

    /// The samples the next step waits for: the processor's chunk, the first centred, the later
    /// ones read from half a transform before their first frame.
    pub fn samples_needed(&self) -> usize {
        let g = self.geometry;
        let mel = model::STACK * (g.chunk + g.right_context);
        if self.steps == 0 {
            (mel - 1) * mel::HOP + mel::WIN / 2
        } else {
            (model::STACK * self.done * mel::HOP + mel * mel::HOP + mel::WIN).saturating_sub(mel::N_FFT / 2)
        }
    }

    /// Whether a step can run.
    pub fn ready(&self) -> bool {
        !self.ended && self.heard >= self.samples_needed()
    }

    /// The next step, once its samples arrived.
    pub fn step(&mut self, d: &Diarizer, pool: &Pool, isa: Isa) -> Option<Step> {
        if !self.ready() {
            return None;
        }
        let g = self.geometry;
        let mel = model::STACK * (g.chunk + g.right_context);
        Some(self.run(d, pool, isa, mel, g.chunk, None))
    }

    /// The stream ended: what is left, scored in one last step with no look-ahead (none when
    /// nothing is left). Nothing can be pushed after it.
    pub fn finish(&mut self, d: &Diarizer, pool: &Pool, isa: Isa) -> Option<Step> {
        if std::mem::replace(&mut self.ended, true) {
            return None;
        }
        let total = mel::frames_of(self.heard);
        let left = total.checked_sub(model::STACK * self.done).filter(|&n| n > 0)?;
        Some(self.run(d, pool, isa, left, left.div_ceil(model::STACK), None))
    }

    /// [`step`](Self::step), with every tensor of its forward pass handed to `tap`.
    pub fn step_tapped(&mut self, d: &Diarizer, pool: &Pool, isa: Isa, tap: Tap) -> Option<Step> {
        if !self.ready() {
            return None;
        }
        let g = self.geometry;
        let mel = model::STACK * (g.chunk + g.right_context);
        Some(self.run(d, pool, isa, mel, g.chunk, Some(tap)))
    }

    /// Keeps every scored frame's embedding from now on, for [`force`](Self::force).
    #[doc(hidden)]
    pub fn keep_history(&mut self) {
        self.history.get_or_insert_with(Vec::new);
    }

    /// Sets the speaker cache and the FIFO to these frames (−1 a silence slot), as this stream
    /// embedded them: the oracle's forcing, so that a step is compared with the same state.
    #[doc(hidden)]
    pub fn force(&mut self, d: &Diarizer, cache: &[i32], fifo: &[i32], compressed: bool) {
        let w = self.width;
        let h = self.history.as_ref().expect("keep_history first");
        let rows = |frames: &[i32]| -> Vec<f32> {
            frames.iter().flat_map(|&f| if f < 0 { d.weights.silence.clone() } else { h[f as usize * w..(f as usize + 1) * w].to_vec() }).collect()
        };
        let (c, f) = (rows(cache), rows(fifo));
        // The stored probabilities only score the next compression, which the next forcing undoes.
        self.cache.set(c, vec![[0.0; SPEAKERS]; cache.len()], f, compressed);
    }

    /// Empties the speaker cache and the FIFO: the oracle's negative control, a stream that
    /// forgets between steps.
    #[doc(hidden)]
    pub fn forget(&mut self) {
        self.cache = SpeakerCache::new(self.width, self.geometry, self.constants);
    }

    /// One step over `mel` log-mel frames from the next unscored one, `chunk` of their rows scored
    /// and the rest look-ahead.
    fn run(&mut self, d: &Diarizer, pool: &Pool, isa: Isa, mel: usize, chunk: usize, mut tap: Option<Tap>) -> Step {
        let w = &d.weights;
        let first_mel = model::STACK * self.done;
        let mut rows = Vec::with_capacity(mel * crate::mel::N_MELS);
        d.front.log_frames(pool, isa, &self.samples, self.x0, first_mel, mel, self.ended, &mut rows);
        let embeds = w.embed(pool, isa, &rows);
        if let Some(f) = tap.as_mut() {
            f("embedder", &embeds);
        }
        let mut inputs: Vec<f32> = Vec::with_capacity((self.cache.cache_len() + self.cache.fifo_len()) * w.d + embeds.len());
        for r in self.cache.rows() {
            inputs.extend_from_slice(r);
        }
        let cached = inputs.len() / w.d;
        inputs.extend_from_slice(&embeds);
        let n = inputs.len() / w.d;
        let logits = w.forward(pool, isa, &inputs, n, n, tap);
        let sig: Vec<Probs> = logits.chunks_exact(SPEAKERS).map(|r| std::array::from_fn(|s| (1.0 / (1.0 + (-(r[s] as f64)).exp())) as f32)).collect();
        let pooled: Vec<Probs> = sig
            .chunks_exact(model::STACK)
            .map(|g| std::array::from_fn(|s| g.iter().map(|r| r[s]).sum::<f32>() / model::STACK as f32))
            .collect();
        let probs = sig[cached * model::STACK..][..(chunk * model::STACK).min(mel)].to_vec();
        let frames = pooled[cached..cached + chunk].to_vec();
        self.cache.update(&inputs, &pooled, &w.silence, chunk);
        if let Some(h) = self.history.as_mut() {
            h.extend_from_slice(&embeds[..chunk * w.d]);
        }
        let step = Step { first: self.done, frames, probs };
        self.done += chunk;
        self.steps += 1;
        // Keep what the next frames read, and the sample before it.
        let keep = mel::first_sample(model::STACK * self.done);
        if keep > self.x0 {
            let base = self.x0.saturating_sub(1);
            let from = keep - 1;
            self.samples.drain(..(from - base).min(self.samples.len()));
            self.x0 = keep;
        }
        step
    }
}
