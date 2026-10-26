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
//!
//! Provenance: the module's (`diarizer/mod.rs`, ADR 0051).

use std::path::Path;
use std::time::Instant;

use crate::cpu::Isa;
use crate::encoder::linear;
use crate::gguf::Value;
use crate::matmul::{matmul, Weight};
use crate::mel::N_MELS;
use crate::model::{LoadError, Linear, Loader};
use crate::ops::{self, Rope};
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
        let norm = |x: &mut [f32], ln: &(Vec<f32>, Vec<f32>)| rows(pool, x, d, &|r| ops::layer_norm(isa, r, &ln.0, &ln.1, self.eps));
        let mut h = x.to_vec();
        norm(&mut h, &self.input_ln);
        if let Some(f) = tap.as_mut() {
            f("input_ln", &h);
        }
        let mut att = vec![0f32; n * d];
        for (i, l) in self.layers.iter().enumerate() {
            let mut a = h.clone();
            norm(&mut a, &l.ln1);
            let (mut q, mut k, mut v) = (vec![0f32; n * d], vec![0f32; n * d], vec![0f32; n * d]);
            matmul(pool, isa, &l.q, &a, n, &mut q);
            matmul(pool, isa, &l.k, &a, n, &mut k);
            matmul(pool, isa, &l.v, &a, n, &mut v);
            rope.apply(isa, &mut q, self.heads);
            rope.apply(isa, &mut k, self.heads);
            attention(pool, isa, self.heads, &q, &k, &v, n, keys, &mut att);
            let o = linear(pool, isa, &l.o, &att, n);
            ops::add(isa, &mut h, &o);
            let mut a = h.clone();
            norm(&mut a, &l.ln2);
            let mut f = linear(pool, isa, &l.fc1, &a, n);
            rows(pool, &mut f, l.fc1.w.n, &|r| ops::gelu(isa, r));
            let y = linear(pool, isa, &l.fc2, &f, n);
            ops::add(isa, &mut h, &y);
            if let Some(f) = tap.as_mut() {
                f(&format!("layer.{i}"), &h);
            }
        }
        norm(&mut h, &self.post_ln);
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

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
// SAFETY: each task of `rows` touches its own rows.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// Rows a task of [`rows`] takes.
const ROWS_PER_TASK: usize = 16;

/// `f` over `x`'s rows of `width`, a few rows per task on the pool: a step's rows are hundreds.
#[doc(hidden)]
pub fn rows(pool: &Pool, x: &mut [f32], width: usize, f: &(dyn Fn(&mut [f32]) + Sync)) {
    let n = x.len() / width;
    let p = SendPtr(x.as_mut_ptr());
    pool.run(n.div_ceil(ROWS_PER_TASK), &|t| {
        let (a, b) = (t * ROWS_PER_TASK, ((t + 1) * ROWS_PER_TASK).min(n));
        // SAFETY: rows a..b are this task's alone, inside x.
        f(unsafe { std::slice::from_raw_parts_mut(p.get().add(a * width), (b - a) * width) });
    });
}

/// Query rows an attention task takes.
const QUERIES_PER_TASK: usize = 16;

/// Bidirectional attention over a step's `n` rows (rows of `heads` heads): each row's scores
/// against every key, `q·kᵀ / √d`, softmaxed over the first `keys` (the rest weigh 0), then times
/// `v`. One task per head and block of query rows, on the pool at once: a step has hundreds of
/// rows, where the decoder's attention op serves a few queries. The keys are transposed per head
/// first, so that a query's scores over 8 keys are one vector.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn attention(pool: &Pool, isa: Isa, heads: usize, q: &[f32], k: &[f32], v: &[f32], n: usize, keys: usize, out: &mut [f32]) {
    let d = q.len() / n;
    let hd = d / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    // kt[h][c][j]: key j's element c of head h.
    let mut kt = vec![0f32; heads * hd * n];
    for (j, row) in k.chunks_exact(d).enumerate() {
        for (i, &x) in row.iter().enumerate() {
            kt[i * n + j] = x;
        }
    }
    let blocks = n.div_ceil(QUERIES_PER_TASK);
    let o = SendPtr(out.as_mut_ptr());
    pool.run(heads * blocks, &|t| {
        let (h, b) = (t / blocks, t % blocks);
        let (r0, r1) = (b * QUERIES_PER_TASK, ((b + 1) * QUERIES_PER_TASK).min(n));
        let kth = &kt[h * hd * n..(h + 1) * hd * n];
        let mut scores = vec![0f32; QROWS * n];
        let mut acc = vec![0f32; QROWS * hd];
        for r in (r0..r1).step_by(QROWS) {
            let m = QROWS.min(r1 - r);
            head_rows(isa, q, kth, v, d, h * hd, hd, r, m, n, keys, scale, &mut scores, &mut acc);
            for i in 0..m {
                // SAFETY: row r + i's head h is this task's alone, inside out.
                unsafe { std::slice::from_raw_parts_mut(o.get().add((r + i) * d + h * hd), hd) }.copy_from_slice(&acc[i * hd..][..hd]);
            }
        }
    });
}

/// Query rows the attention's inner loops take together: each key and value they load serves
/// them all.
const QROWS: usize = 4;
/// Keys (in the scores) and value elements (in the sum) per register tile.
const LANES: usize = 16;

/// `m` (at most [`QROWS`]) query rows of one head from row `r0`: their scores against the keys
/// (`kt`, the head's keys transposed, rows of `n`), the softmax over the first `keys`, and the
/// weighted sums of the values (`v`, rows of `d`, the head's from column `at`) into `acc`, rows of
/// `hd`.
#[allow(clippy::too_many_arguments)]
fn head_rows(isa: Isa, q: &[f32], kt: &[f32], v: &[f32], d: usize, at: usize, hd: usize, r0: usize, m: usize, n: usize, keys: usize, scale: f32, scores: &mut [f32], acc: &mut [f32]) {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn body(isa: Isa, q: &[f32], kt: &[f32], v: &[f32], d: usize, at: usize, hd: usize, r0: usize, m: usize, n: usize, keys: usize, scale: f32, scores: &mut [f32], acc: &mut [f32]) {
        let qrow = |i: usize| &q[(r0 + i.min(m - 1)) * d + at..][..hd];
        let qs: [&[f32]; QROWS] = std::array::from_fn(qrow);
        // Scores: a tile of QROWS rows by LANES keys stays in registers over the head's elements.
        let full = keys / LANES * LANES;
        for j in (0..full).step_by(LANES) {
            let mut t = [[0f32; LANES]; QROWS];
            for c in 0..hd {
                let k: &[f32; LANES] = kt[c * n + j..][..LANES].try_into().unwrap();
                for (i, row) in t.iter_mut().enumerate() {
                    let x = qs[i][c];
                    for l in 0..LANES {
                        row[l] += x * k[l];
                    }
                }
            }
            for (i, row) in t.iter().enumerate() {
                scores[i * n + j..][..LANES].copy_from_slice(row);
            }
        }
        for j in full..keys {
            for (i, qr) in qs.iter().enumerate() {
                scores[i * n + j] = (0..hd).map(|c| qr[c] * kt[c * n + j]).sum();
            }
        }
        for i in 0..m {
            let s = &mut scores[i * n..i * n + keys];
            for a in s.iter_mut() {
                *a *= scale;
            }
            ops::softmax(isa, s);
        }
        // Sums: a tile of QROWS rows by LANES elements stays in registers over the keys.
        for e in (0..hd).step_by(LANES) {
            let mut t = [[0f32; LANES]; QROWS];
            for j in 0..keys {
                let vj: &[f32; LANES] = v[j * d + at + e..][..LANES].try_into().unwrap();
                for (i, row) in t.iter_mut().enumerate() {
                    let p = scores[i.min(m - 1) * n + j];
                    for l in 0..LANES {
                        row[l] += p * vj[l];
                    }
                }
            }
            for (i, row) in t.iter().enumerate().take(m) {
                acc[i * hd + e..][..LANES].copy_from_slice(row);
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c")]
    #[allow(clippy::too_many_arguments)]
    fn avx2(isa: Isa, q: &[f32], kt: &[f32], v: &[f32], d: usize, at: usize, hd: usize, r0: usize, m: usize, n: usize, keys: usize, scale: f32, scores: &mut [f32], acc: &mut [f32]) {
        body(isa, q, kt, v, d, at, hd, r0, m, n, keys, scale, scores, acc)
    }
    assert!(hd % LANES == 0 && m >= 1 && m <= QROWS);
    #[cfg(target_arch = "x86_64")]
    if isa.is_avx2() {
        // SAFETY: an AVX2 Isa exists only after detection.
        unsafe { avx2(isa, q, kt, v, d, at, hd, r0, m, n, keys, scale, scores, acc) };
        return;
    }
    body(isa, q, kt, v, d, at, hd, r0, m, n, keys, scale, scores, acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{AttnShape, Kv, Mask};

    /// The step's attention gives the decoder's op's result, over all keys and over the first ones.
    #[test]
    fn the_step_attention_is_the_attention_op() {
        let mut rng = crate::testutil::Rng::new(7);
        let (n, heads, hd) = (37, 4, 16);
        let d = heads * hd;
        let [q, k, v] = std::array::from_fn(|_| rng.activations(n * d));
        let pool = Pool::new(3);
        for isa in [Isa::scalar()].into_iter().chain(crate::cpu::select(crate::cpu::Features::detect()).ok()) {
            for keys in [n, 20] {
                let (mut mine, mut theirs) = (vec![0f32; n * d], vec![0f32; n * d]);
                attention(&pool, isa, heads, &q, &k, &v, n, keys, &mut mine);
                ops::attention(&pool, isa, AttnShape { hq: heads, hkv: heads, hd }, &q, n, Kv::F32 { k: &k, v: &v }, keys, Mask::Full, &mut theirs);
                let worst = mine.iter().zip(&theirs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                assert!(worst < 1e-5, "{isa:?}, {keys} keys: {worst}");
            }
        }
    }
}
