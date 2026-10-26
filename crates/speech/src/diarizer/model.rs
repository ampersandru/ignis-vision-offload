//! The diarizer's weights and its forward pass over one step's frames (spec speech/08).
//!
//! The graph (Transformers' `Nemotron3DiarizationForAudioFrameClassification`, see the module's
//! provenance): the embedder stacks 8 log-mel frames (80 ms) into a row of 1,024 and projects it to
//! 512, with no bias. A step's rows (the speaker cache's, the FIFO's, the chunk's and its look-ahead)
//! go through the input LayerNorm and 31 pre-LN layers: bidirectional attention of 8 heads of 64
//! over all the rows, RoPE on the full head (θ 10⁴, the positions restarting at 0 every step), q, k
//! and v with no bias, the output with one; then a 2,048-wide MLP with the exact GELU. A final
//! LayerNorm, a projection to 192, and a kernel-3 convolution to 8 × 192 per row upsample each row
//! to 8 rows of 10 ms; the head (ReLU, 192 → 192, ReLU, 192 → 8) gives each its 8 speakers' logits.

use std::path::Path;
use std::time::Instant;

use crate::cpu::Isa;
use crate::encoder::linear;
use crate::gguf::Value;
use crate::matmul::{matmul, Weight};
use crate::mel::N_MELS;
use crate::model::{LoadError, Linear, Loader};
use crate::ops::{self, AttnShape, Kv, Mask, Rope};
use crate::pool::Pool;

use super::{Constants, Geometry, SPEAKERS};

const ARCH: &str = "nemotron3_diarization";
/// Log-mel frames per row (80 ms).
pub const STACK: usize = 8;

pub struct Layer {
    pub ln1: (Vec<f32>, Vec<f32>),
    pub q: Weight,
    pub k: Weight,
    pub v: Weight,
    pub o: Linear,
    pub ln2: (Vec<f32>, Vec<f32>),
    pub fc1: Linear,
    pub fc2: Linear,
}

/// What [`Weights::load`] did.
#[derive(Clone, Debug)]
pub struct LoadReport {
    pub millis: u128,
    pub mapped_bytes: usize,
    pub pinned_bytes: usize,
}

pub struct Weights {
    pub d: usize,
    pub heads: usize,
    pub eps: f32,
    pub theta: f32,
    pub embed: Weight,
    pub input_ln: (Vec<f32>, Vec<f32>),
    pub layers: Vec<Layer>,
    pub post_ln: (Vec<f32>, Vec<f32>),
    pub proj: Linear,
    /// The upsampling convolution, rows of [tap][channel]: frame t reads frames t − 1, t, t + 1.
    pub upsample: Linear,
    pub dense: Linear,
    pub out: Linear,
    /// What fills the speaker cache's silence slots.
    pub silence: Vec<f32>,
    /// The checkpoint's `streaming_config`: its constants and its default sizes.
    pub constants: Constants,
    pub streaming: Geometry,
    /// The checkpoint's offline sizes.
    pub offline: Geometry,
    pub report: LoadReport,
}

fn err<T>(m: impl Into<String>) -> Result<T, LoadError> {
    Err(LoadError(m.into()))
}

impl Weights {
    pub fn load(path: &Path, pinned: bool) -> Result<Weights, LoadError> {
        let t0 = Instant::now();
        let (mut l, _) = Loader::open(path, pinned)?;
        let arch = l.gguf().get("general.architecture").and_then(Value::as_str).map(str::to_string);
        if arch.as_deref() != Some(ARCH) {
            return err(format!("{}: architecture {arch:?}, not {ARCH}", path.display()));
        }
        let k = |s: &str| format!("{ARCH}.{s}");
        let d = l.kv_u(&k("embedding_length"))?;
        let heads = l.kv_u(&k("attention.head_count"))?;
        let n_layers = l.kv_u(&k("block_count"))?;
        let eps = l.kv_f(&k("attention.layer_norm_epsilon"))? as f32;
        let theta = l.kv_f(&k("rope.freq_base"))? as f32;
        if l.kv_u(&k("num_mel_bins"))? != N_MELS || l.kv_u(&k("subsampling_factor"))? != STACK {
            return err(format!("{}: not 128 mel bins stacked by 8", path.display()));
        }
        if l.kv_u(&k("head.num_speakers"))? != SPEAKERS {
            return err(format!("{}: not {SPEAKERS} speakers", path.display()));
        }
        let front = (l.kv_u(&k("front_end.n_fft"))?, l.kv_u(&k("front_end.win_length"))?, l.kv_u(&k("front_end.hop_length"))?);
        if front != (super::mel::N_FFT, super::mel::WIN, super::mel::HOP) || l.kv_u(&k("front_end.sampling_rate"))? != 16_000 {
            return err(format!("{}: a front end of {front:?}, not the one this port computes", path.display()));
        }
        let constants = Constants {
            prediction_score_threshold: l.kv_f(&k("streaming.prediction_score_threshold"))? as f32,
            latest_frames_score_boost: l.kv_f(&k("streaming.latest_frames_score_boost"))? as f32,
            strong_boost_rate: l.kv_f(&k("streaming.strong_boost_rate"))? as f32,
            weak_boost_rate: l.kv_f(&k("streaming.weak_boost_rate"))? as f32,
            min_positive_scores_rate: l.kv_f(&k("streaming.min_positive_scores_rate"))? as f32,
            speaker_cache_silence_frames_per_speaker: l.kv_u(&k("streaming.speaker_cache_silence_frames_per_speaker"))?,
        };
        let cache = l.kv_u(&k("streaming.speaker_cache_length"))?;
        let streaming = Geometry {
            speaker_cache: cache,
            fifo: l.kv_u(&k("streaming.fifo_length"))?,
            chunk: Geometry::LIVE.chunk,
            right_context: Geometry::LIVE.right_context,
            update_period: l.kv_u(&k("streaming.speaker_cache_update_period"))?,
        };
        let offline = Geometry {
            speaker_cache: cache,
            fifo: l.kv_u(&k("offline.fifo_length"))?,
            chunk: l.kv_u(&k("offline.chunk_length"))?,
            right_context: l.kv_u(&k("offline.chunk_right_context"))?,
            update_period: l.kv_u(&k("offline.speaker_cache_update_period"))?,
        };
        let embed = l.weight("enc.embd.weight")?;
        if embed.k != STACK * N_MELS || embed.n != d {
            return err(format!("enc.embd.weight: {} x {}", embed.n, embed.k));
        }
        let input_ln = l.ln("enc.input_ln")?;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("enc.blk.{i}.");
            layers.push(Layer {
                ln1: l.ln(&format!("{p}ln1"))?,
                q: l.weight(&format!("{p}attn_q.weight"))?,
                k: l.weight(&format!("{p}attn_k.weight"))?,
                v: l.weight(&format!("{p}attn_v.weight"))?,
                o: l.linear(&format!("{p}attn_out"))?,
                ln2: l.ln(&format!("{p}ln2"))?,
                fc1: l.linear(&format!("{p}ffn_up"))?,
                fc2: l.linear(&format!("{p}ffn_down"))?,
            });
        }
        let post_ln = l.ln("enc.post_ln")?;
        let proj = l.linear("head.proj")?;
        let hidden = proj.w.n;
        let upsample = l.linear("head.upsample")?;
        if upsample.w.k != 3 * hidden || upsample.w.n != STACK * hidden {
            return err(format!("head.upsample.weight: {} x {} for a head of {hidden}", upsample.w.n, upsample.w.k));
        }
        let dense = l.linear("head.dense")?;
        let out = l.linear("head.out")?;
        let silence = l.f32s("silence_embd")?;
        if silence.len() != d || out.w.n != SPEAKERS {
            return err("silence_embd or head.out: the wrong size");
        }
        let report = LoadReport { millis: t0.elapsed().as_millis(), mapped_bytes: l.mapped, pinned_bytes: l.pinned_bytes() };
        Ok(Weights { d, heads, eps, theta, embed, input_ln, layers, post_ln, proj, upsample, dense, out, silence, constants, streaming, offline, report })
    }

    /// The rows of `mel` (log-mel frames, rows of 128): one per 8 frames, the last group zero-padded
    /// as the checkpoint's feature stacking pads it.
    pub fn embed(&self, pool: &Pool, isa: Isa, mel: &[f32]) -> Vec<f32> {
        let frames = mel.len() / N_MELS;
        let rows = frames.div_ceil(STACK);
        let mut stacked = mel.to_vec();
        stacked.resize(rows * STACK * N_MELS, 0.0);
        let mut out = vec![0f32; rows * self.d];
        matmul(pool, isa, &self.embed, &stacked, rows, &mut out);
        out
    }

    /// The logits of `n` rows of embeddings: `8·n` rows of 8 speakers, one per 10 ms. Only the first
    /// `keys` rows are attended to (the others are padding the checkpoint masks).
    pub fn forward(&self, pool: &Pool, isa: Isa, x: &[f32], n: usize, keys: usize, mut tap: Option<super::Tap>) -> Vec<f32> {
        let d = self.d;
        assert!(x.len() == n * d && keys <= n && keys > 0);
        let hd = d / self.heads;
        let rope = Rope::text(0, n, [hd / 2, 0, 0], self.theta, hd);
        let shape = AttnShape { hq: self.heads, hkv: self.heads, hd };
        let mut h = x.to_vec();
        ops::layer_norm(isa, &mut h, &self.input_ln.0, &self.input_ln.1, self.eps);
        if let Some(f) = tap.as_mut() {
            f("input_ln", &h);
        }
        let mut att = vec![0f32; n * d];
        for (i, l) in self.layers.iter().enumerate() {
            let mut a = h.clone();
            ops::layer_norm(isa, &mut a, &l.ln1.0, &l.ln1.1, self.eps);
            let (mut q, mut k, mut v) = (vec![0f32; n * d], vec![0f32; n * d], vec![0f32; n * d]);
            matmul(pool, isa, &l.q, &a, n, &mut q);
            matmul(pool, isa, &l.k, &a, n, &mut k);
            matmul(pool, isa, &l.v, &a, n, &mut v);
            rope.apply(isa, &mut q, self.heads);
            rope.apply(isa, &mut k, self.heads);
            ops::attention(pool, isa, shape, &q, n, Kv::F32 { k: &k, v: &v }, keys, Mask::Full, &mut att);
            let o = linear(pool, isa, &l.o, &att, n);
            ops::add(isa, &mut h, &o);
            let mut a = h.clone();
            ops::layer_norm(isa, &mut a, &l.ln2.0, &l.ln2.1, self.eps);
            let mut f = linear(pool, isa, &l.fc1, &a, n);
            ops::gelu(isa, &mut f);
            let y = linear(pool, isa, &l.fc2, &f, n);
            ops::add(isa, &mut h, &y);
            if let Some(f) = tap.as_mut() {
                f(&format!("layer.{i}"), &h);
            }
        }
        ops::layer_norm(isa, &mut h, &self.post_ln.0, &self.post_ln.1, self.eps);
        if let Some(f) = tap.as_mut() {
            f("final_ln", &h);
        }
        let p = linear(pool, isa, &self.proj, &h, n);
        if let Some(f) = tap.as_mut() {
            f("proj", &p);
        }
        // Kernel 3, padding 1: each row's input is the rows before and after it, zero at the ends.
        let c = self.proj.w.n;
        let mut cols = vec![0f32; n * 3 * c];
        for t in 0..n {
            for (j, s) in [t.wrapping_sub(1), t, t + 1].into_iter().enumerate() {
                if s < n {
                    cols[(t * 3 + j) * c..][..c].copy_from_slice(&p[s * c..][..c]);
                }
            }
        }
        // n rows of 8 × 192 are 8·n rows of 192: sub-frame g of row t is channels g·192..(g+1)·192.
        let mut u = linear(pool, isa, &self.upsample, &cols, n);
        if let Some(f) = tap.as_mut() {
            f("upsampled", &u);
        }
        relu(&mut u);
        let mut z = linear(pool, isa, &self.dense, &u, n * STACK);
        relu(&mut z);
        let logits = linear(pool, isa, &self.out, &z, n * STACK);
        if let Some(f) = tap.as_mut() {
            f("logits", &logits);
        }
        logits
    }
}

fn relu(x: &mut [f32]) {
    for v in x {
        *v = v.max(0.0);
    }
}
