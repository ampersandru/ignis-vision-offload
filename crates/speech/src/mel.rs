//! The front end: the official log-mel (spec speech/01 §3, Q41) and the
//! official chunking's audio-token count (Q52).
//!
//! 128 slaney mel bins over a 400-point Hann-windowed STFT of hop 160 at
//! 16 kHz, reflect padding at both ends, the last frame dropped, log10 of the
//! power clamped at 1e-10, floored at the maximum − 8 and mapped by
//! (x + 4) / 4 (Whisper's normalisation). The transform runs in fp64, so the
//! error against the checkpoint's fp32 front end is its own rounding.
//!
//! A frame needs 200 samples on each side: frame t is final once samples up
//! to 160·t + 200 have arrived, whatever comes after, so a stream computes
//! each frame once. Only the last frames of a clip see the reflect padding.

use crate::cpu::Isa;
use crate::pool::Pool;

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const N_MELS: usize = 128;
const N_BINS: usize = N_FFT / 2 + 1;
/// Clips shorter than this are zero-padded to it (half a second), as the official processor pads them.
pub const MIN_SAMPLES: usize = 8000;
/// Mel frames per encoder chunk (1 s), and the tokens a full chunk gives.
pub const CHUNK_FRAMES: usize = 100;
pub const CHUNK_TOKENS: usize = 13;
/// Chunks per encoder window (8 s, 104 tokens): the encoder attends only inside a window.
pub const WINDOW_CHUNKS: usize = 8;

/// One stride-2, kernel-3, padding-1 convolution's output length.
fn conv_len(n: usize) -> usize {
    if n == 0 { 0 } else { (n - 1) / 2 + 1 }
}

/// Tokens a chunk of `frames` mel frames (≤ 100) gives: three convolutions of stride 2.
pub fn chunk_tokens(frames: usize) -> usize {
    conv_len(conv_len(conv_len(frames)))
}

/// Mel frames of a clip of `samples` samples, after the half-second pad.
pub fn frames_of(samples: usize) -> usize {
    samples.max(MIN_SAMPLES) / HOP
}

/// Audio tokens of `frames` mel frames, counted as the official processor counts them: 13 per
/// full chunk, and the last partial chunk by its own length (not padded to a full chunk).
pub fn audio_tokens(frames: usize) -> usize {
    frames / CHUNK_FRAMES * CHUNK_TOKENS + chunk_tokens(frames % CHUNK_FRAMES)
}

pub struct Mel {
    window: Vec<f64>,
    /// cos and sin of the DFT, [bin][n].
    cos: Vec<f64>,
    sin: Vec<f64>,
    /// Each mel filter: its first non-zero bin and its weights.
    filters: Vec<(usize, Vec<f64>)>,
}

fn hz_to_mel(f: f64) -> f64 {
    // The slaney scale: linear under 1 kHz, logarithmic above.
    if f >= 1000.0 { 15.0 + (f / 1000.0).ln() * (27.0 / 6.4f64.ln()) } else { 3.0 * f / 200.0 }
}

fn mel_to_hz(m: f64) -> f64 {
    if m >= 15.0 { 1000.0 * ((m - 15.0) * (6.4f64.ln() / 27.0)).exp() } else { 200.0 * m / 3.0 }
}

/// The 128 triangular slaney filters of an `n_fft`-point transform at 16 kHz: between
/// N_MELS + 2 points evenly spaced in mel from 0 to 8 kHz, each scaled by 2 / its width in Hz.
/// Each is its first non-zero bin and its weights.
pub(crate) fn slaney_filters(n_fft: usize) -> Vec<(usize, Vec<f64>)> {
    let bins = n_fft / 2 + 1;
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE as f64 / 2.0));
    let pts: Vec<f64> = (0..N_MELS + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64)).collect();
    let fft_hz: Vec<f64> = (0..bins).map(|k| (SAMPLE_RATE / 2) as f64 * k as f64 / (bins - 1) as f64).collect();
    (0..N_MELS)
        .map(|m| {
            let enorm = 2.0 / (pts[m + 2] - pts[m]);
            let w: Vec<f64> = fft_hz
                .iter()
                .map(|&f| {
                    let down = (f - pts[m]) / (pts[m + 1] - pts[m]);
                    let up = (pts[m + 2] - f) / (pts[m + 2] - pts[m + 1]);
                    down.min(up).max(0.0) * enorm
                })
                .collect();
            let first = w.iter().position(|&v| v != 0.0).unwrap_or(0);
            let last = w.iter().rposition(|&v| v != 0.0).map_or(first, |l| l + 1);
            (first, w[first..last].to_vec())
        })
        .collect()
}

impl Default for Mel {
    fn default() -> Self {
        Self::new()
    }
}

impl Mel {
    pub fn new() -> Mel {
        let window: Vec<f64> = (0..N_FFT).map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos()).collect();
        let (mut cos, mut sin) = (vec![0f64; N_BINS * N_FFT], vec![0f64; N_BINS * N_FFT]);
        for k in 0..N_BINS {
            for n in 0..N_FFT {
                // Reduced mod N_FFT so the angle stays exact.
                let a = 2.0 * std::f64::consts::PI * ((k * n) % N_FFT) as f64 / N_FFT as f64;
                cos[k * N_FFT + n] = a.cos();
                sin[k * N_FFT + n] = a.sin();
            }
        }
        Mel { window, cos, sin, filters: slaney_filters(N_FFT) }
    }

    /// log10 of the mel power (clamped at 1e-10) of frames `first..first + n` of the signal
    /// `x`, appended to `out` as rows of 128. With `ended`, `x` is the whole clip and the frames
    /// past its end see the reflect padding; otherwise every frame must lie inside `x`.
    pub fn log_frames(&self, pool: &Pool, isa: Isa, x: &[f32], first: usize, n: usize, ended: bool, out: &mut Vec<f32>) {
        if n == 0 {
            return;
        }
        let len = x.len();
        assert!(ended || (first + n - 1) * HOP + N_FFT / 2 <= len, "frame {} needs samples not yet pushed", first + n - 1);
        assert!(first > 1 || len > N_FFT / 2, "frame {first} reflects at the start: it needs sample {}", N_FFT / 2);
        let base = out.len();
        out.resize(base + n * N_MELS, 0.0);
        let op = OutPtr(out[base..].as_mut_ptr());
        pool.run(n, &|i| {
            let t = first + i;
            let mut frame = [0f64; N_FFT];
            for (j, v) in frame.iter_mut().enumerate() {
                // Padded index j of frame t is sample 160·t + j − 200, reflected at both ends.
                let s = (t * HOP + j) as isize - (N_FFT / 2) as isize;
                let s = if s < 0 { -s } else if s >= len as isize { 2 * (len as isize - 1) - s } else { s };
                *v = x[s as usize] as f64 * self.window[j];
            }
            let mut power = [0f64; N_BINS];
            dft_power(isa, &frame, &self.cos, &self.sin, &mut power);
            // SAFETY: row i of the output is this task's alone.
            let row = unsafe { std::slice::from_raw_parts_mut(op.get().add(i * N_MELS), N_MELS) };
            for (m, (f0, w)) in self.filters.iter().enumerate() {
                let e: f64 = w.iter().zip(&power[*f0..]).map(|(a, b)| a * b).sum();
                row[m] = e.max(1e-10).log10() as f32;
            }
        });
    }
}

/// Whisper's normalisation of log10 mel values: floored at `max − 8`, then (x + 4) / 4.
pub fn normalise(log: f32, max: f32) -> f32 {
    (log.max(max - 8.0) + 4.0) / 4.0
}

#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: each task writes its own frame's row.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

impl OutPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

fn dft_power(isa: Isa, x: &[f64; N_FFT], cos: &[f64], sin: &[f64], power: &mut [f64; N_BINS]) {
    #[inline(always)]
    fn body(x: &[f64; N_FFT], cos: &[f64], sin: &[f64], power: &mut [f64; N_BINS]) {
        for (k, p) in power.iter_mut().enumerate() {
            let (c, s) = (&cos[k * N_FFT..(k + 1) * N_FFT], &sin[k * N_FFT..(k + 1) * N_FFT]);
            let (mut re, mut im) = ([0f64; 4], [0f64; 4]);
            for n in (0..N_FFT).step_by(4) {
                for l in 0..4 {
                    re[l] += x[n + l] * c[n + l];
                    im[l] += x[n + l] * s[n + l];
                }
            }
            let (r, i) = ((re[0] + re[2]) + (re[1] + re[3]), (im[0] + im[2]) + (im[1] + im[3]));
            *p = r * r + i * i;
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn avx2(x: &[f64; N_FFT], cos: &[f64], sin: &[f64], power: &mut [f64; N_BINS]) {
        body(x, cos, sin, power)
    }
    #[cfg(target_arch = "x86_64")]
    if isa.is_avx2() {
        // SAFETY: an AVX2 Isa exists only after detection.
        unsafe { avx2(x, cos, sin, power) };
        return;
    }
    let _ = isa;
    body(x, cos, sin, power)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_tokens_follow_the_official_processor() {
        // d01, 10.34 s: 1,034 frames, 10 full chunks and 34 frames → 135 (llama.cpp counts 143).
        assert_eq!(frames_of(165_440), 1034);
        assert_eq!(audio_tokens(1034), 135);
        assert_eq!(audio_tokens(906), 118);
        assert_eq!(audio_tokens(1000), 130);
        assert_eq!(audio_tokens(1), 1);
        assert_eq!(chunk_tokens(100), 13);
        // Under half a second the clip is padded to 50 frames.
        assert_eq!(frames_of(100), 50);
        assert_eq!(audio_tokens(frames_of(100)), 7);
    }

    #[test]
    fn the_filters_are_slaney_and_cover_0_to_8_khz() {
        let m = Mel::new();
        assert_eq!(m.filters.len(), N_MELS);
        // Slaney normalisation gives each triangle an area of about 1 in Hz·(1/Hz) units: its peak
        // is 2 / width, so peak · width / 2 = 1.
        let (f0, w) = &m.filters[0];
        assert_eq!(*f0, 1);
        assert!(w.iter().all(|&v| v > 0.0));
        assert!(m.filters.last().unwrap().0 > 190);
    }

    #[test]
    fn a_tone_lands_in_its_mel_bin_and_frames_are_independent_of_later_samples() {
        let pool = Pool::new(2);
        let isa = crate::cpu::select(crate::cpu::Features::detect()).unwrap_or(Isa::scalar());
        let mel = Mel::new();
        let x: Vec<f32> = (0..32_000).map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16_000.0).sin() * 0.5).collect();
        let mut whole = Vec::new();
        mel.log_frames(&pool, isa, &x, 0, frames_of(x.len()), true, &mut whole);
        assert_eq!(whole.len(), 200 * N_MELS);
        let row = &whole[100 * N_MELS..101 * N_MELS];
        let peak = (0..N_MELS).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
        let hz = |m: usize| mel_to_hz(hz_to_mel(8000.0) * (m + 1) as f64 / 129.0);
        assert!((hz(peak) - 1000.0).abs() < 40.0, "peak at mel bin {peak} ({} Hz)", hz(peak));
        // The same frames computed from a prefix of the signal, before the clip ended.
        let mut part = Vec::new();
        mel.log_frames(&pool, isa, &x[..20_000], 10, 100, false, &mut part);
        assert_eq!(&part[..], &whole[10 * N_MELS..110 * N_MELS]);
        // And the scalar reference gives the same frames to within fp64 rounding.
        let mut scalar = Vec::new();
        mel.log_frames(&pool, Isa::scalar(), &x, 0, 200, true, &mut scalar);
        let worst = scalar.iter().zip(&whole).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-5, "{worst}");
    }
}
