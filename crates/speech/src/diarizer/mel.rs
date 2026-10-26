//! The diarizer's front end: Nemotron 3 Diarization's log-mel (spec speech/08).
//!
//! The stream is preemphasised (y₀ = x₀, yᵢ = xᵢ − 0.97·xᵢ₋₁) and read as zeros outside its
//! samples. Frame t is a 400-sample symmetric Hann window over y[160·t − 200, 160·t + 200), centred
//! in a 512-point transform; its power spectrum goes through the 128 slaney mel filters of that
//! transform (0 to 8 kHz), and each value is ln(e + 2⁻²⁴), unnormalised. A stream of n samples has
//! n / 160 frames. The transform runs in fp64, so the error against the checkpoint's fp32 front end
//! is its own rounding.
//!
//! This is the full-stream pass that the checkpoint's processor reproduces chunk by chunk (its
//! first chunk centred, the later ones not, the last zero-padded): the same frames, each computed
//! once. Frame t is final once samples up to 160·t + 200 have arrived, whatever comes after.

use crate::cpu::Isa;
use crate::mel::{slaney_filters, N_MELS};
use crate::pool::Pool;

pub const N_FFT: usize = 512;
pub const WIN: usize = 400;
pub const HOP: usize = 160;
/// The samples a frame reads on each side of 160·t.
pub const HALF: usize = WIN / 2;
const BINS: usize = N_FFT / 2 + 1;
const PREEMPHASIS: f64 = 0.97;
/// What the log is taken of at least: the checkpoint's `LOG_ZERO_GUARD_VALUE`.
const LOG_GUARD: f64 = 1.0 / (1u64 << 24) as f64;

/// The frames of a stream of `samples` samples, once it ended.
pub fn frames_of(samples: usize) -> usize {
    samples / HOP
}

/// The frames final once `samples` samples arrived, the stream going on.
pub fn frames_ready(samples: usize) -> usize {
    if samples < HALF { 0 } else { (samples - HALF) / HOP + 1 }
}

/// The first sample frame `t` reads, a sample of the stream; frames before it may be dropped.
pub fn first_sample(t: usize) -> usize {
    (t * HOP).saturating_sub(HALF)
}

pub struct FrontEnd {
    window: Vec<f64>,
    /// cos and sin of the 512-point transform over the window's 400 samples, [bin][n]: a window
    /// centred in the transform only shifts each bin's phase, which the power drops.
    cos: Vec<f64>,
    sin: Vec<f64>,
    filters: Vec<(usize, Vec<f64>)>,
}

impl Default for FrontEnd {
    fn default() -> Self {
        Self::new()
    }
}

impl FrontEnd {
    pub fn new() -> FrontEnd {
        // torch.hann_window(400, periodic=False).
        let window: Vec<f64> = (0..WIN).map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (WIN - 1) as f64).cos()).collect();
        let (mut cos, mut sin) = (vec![0f64; BINS * WIN], vec![0f64; BINS * WIN]);
        for k in 0..BINS {
            for n in 0..WIN {
                // Reduced mod N_FFT so the angle stays exact.
                let a = 2.0 * std::f64::consts::PI * ((k * n) % N_FFT) as f64 / N_FFT as f64;
                cos[k * WIN + n] = a.cos();
                sin[k * WIN + n] = a.sin();
            }
        }
        FrontEnd { window, cos, sin, filters: slaney_filters(N_FFT) }
    }

    /// The log-mel of frames `first..first + n`, appended to `out` as rows of 128. `x` holds the
    /// stream's samples from sample `x0` on, with the one before it when `x0 > 0` (the
    /// preemphasis reads it): `x[0]` is sample `x0 - 1` then. With `ended`, the stream ends at
    /// `x`'s end and the frames past it read zeros; otherwise every frame must lie inside `x`.
    #[allow(clippy::too_many_arguments)]
    pub fn log_frames(&self, pool: &Pool, isa: Isa, x: &[f32], x0: usize, first: usize, n: usize, ended: bool, out: &mut Vec<f32>) {
        if n == 0 {
            return;
        }
        // Index of sample s in x.
        let lead = usize::from(x0 > 0);
        let end = x0 + x.len() - lead;
        assert!(ended || (first + n - 1) * HOP + HALF <= end, "frame {} needs samples not yet pushed", first + n - 1);
        assert!(first_sample(first) >= x0, "frame {first} reads sample {} before the kept {x0}", first_sample(first));
        let y = |s: isize| -> f64 {
            if s < 0 || s as usize >= end {
                return 0.0;
            }
            let i = s as usize - x0 + lead;
            let v = x[i] as f64;
            if s == 0 { v } else { v - PREEMPHASIS * x[i - 1] as f64 }
        };
        let base = out.len();
        out.resize(base + n * N_MELS, 0.0);
        let op = OutPtr(out[base..].as_mut_ptr());
        pool.run(n, &|i| {
            let t = first + i;
            let mut frame = [0f64; WIN];
            for (j, v) in frame.iter_mut().enumerate() {
                *v = y((t * HOP + j) as isize - HALF as isize) * self.window[j];
            }
            let mut power = [0f64; BINS];
            dft_power(isa, &frame, &self.cos, &self.sin, &mut power);
            // SAFETY: row i of the output is this task's alone.
            let row = unsafe { std::slice::from_raw_parts_mut(op.get().add(i * N_MELS), N_MELS) };
            for (m, (f0, w)) in self.filters.iter().enumerate() {
                let e: f64 = w.iter().zip(&power[*f0..]).map(|(a, b)| a * b).sum();
                row[m] = (e + LOG_GUARD).ln() as f32;
            }
        });
    }
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

fn dft_power(isa: Isa, x: &[f64; WIN], cos: &[f64], sin: &[f64], power: &mut [f64; BINS]) {
    #[inline(always)]
    fn body(x: &[f64; WIN], cos: &[f64], sin: &[f64], power: &mut [f64; BINS]) {
        for (k, p) in power.iter_mut().enumerate() {
            let (c, s) = (&cos[k * WIN..(k + 1) * WIN], &sin[k * WIN..(k + 1) * WIN]);
            let (mut re, mut im) = ([0f64; 4], [0f64; 4]);
            for n in (0..WIN).step_by(4) {
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
    fn avx2(x: &[f64; WIN], cos: &[f64], sin: &[f64], power: &mut [f64; BINS]) {
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

    fn isa() -> Isa {
        crate::cpu::select(crate::cpu::Features::detect()).unwrap_or(Isa::scalar())
    }

    fn tone(n: usize, hz: f32) -> Vec<f32> {
        (0..n).map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / 16_000.0).sin() * 0.5).collect()
    }

    #[test]
    fn frames_are_counted_as_the_processor_counts_them() {
        assert_eq!(frames_of(256_000), 1600);
        assert_eq!(frames_of(159), 0);
        // Frame t needs samples up to 160·t + 200.
        assert_eq!(frames_ready(199), 0);
        assert_eq!(frames_ready(200), 1);
        assert_eq!(frames_ready(359), 1);
        assert_eq!(frames_ready(360), 2);
        for n in [0, 1, 199, 200, 239, 240, 4_000, 37_160] {
            assert!(frames_ready(n) <= frames_of(n), "{n}");
        }
        // A first chunk of the live geometry, 29 encoder frames of 8: (232 − 1)·160 + 200 samples.
        assert_eq!(frames_ready(37_160), 232);
        assert_eq!(first_sample(0), 0);
        assert_eq!(first_sample(2), 120);
    }

    #[test]
    fn a_tone_lands_in_its_mel_bin_and_frames_do_not_depend_on_later_samples() {
        let pool = Pool::new(2);
        let fe = FrontEnd::new();
        let x = tone(32_000, 1000.0);
        let mut whole = Vec::new();
        fe.log_frames(&pool, isa(), &x, 0, 0, frames_of(x.len()), true, &mut whole);
        assert_eq!(whole.len(), 200 * N_MELS);
        let row = &whole[100 * N_MELS..101 * N_MELS];
        let peak = (0..N_MELS).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
        let (f0, w) = &fe.filters[peak];
        let centre = (f0 + w.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0) as f64 * 8000.0 / 256.0;
        assert!((centre - 1000.0).abs() < 60.0, "peak at mel bin {peak} ({centre} Hz)");
        // The same frames from a prefix of the stream, before it ended.
        let mut part = Vec::new();
        fe.log_frames(&pool, isa(), &x[..20_000], 0, 10, 100, false, &mut part);
        assert_eq!(&part[..], &whole[10 * N_MELS..110 * N_MELS]);
        // And from a buffer that dropped the samples no later frame reads, keeping the one before.
        let x0 = first_sample(50);
        let mut tail = Vec::new();
        fe.log_frames(&pool, isa(), &x[x0 - 1..], x0, 50, 150, true, &mut tail);
        assert_eq!(&tail[..], &whole[50 * N_MELS..]);
        // The scalar reference gives the same frames to within fp64 rounding.
        let mut scalar = Vec::new();
        fe.log_frames(&pool, Isa::scalar(), &x, 0, 0, 200, true, &mut scalar);
        let worst = scalar.iter().zip(&whole).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-5, "{worst}");
    }

    #[test]
    fn silence_is_the_guard_and_the_end_reads_zeros() {
        let pool = Pool::new(1);
        let fe = FrontEnd::new();
        let mut out = Vec::new();
        fe.log_frames(&pool, isa(), &[0.0; 1600], 0, 0, 10, true, &mut out);
        let guard = (LOG_GUARD.ln()) as f32;
        assert!(out.iter().all(|&v| v == guard), "{:?}", &out[..4]);
        // A stream that ended: its last frame reads past the end as zeros, not as the preemphasis
        // of a last sample.
        let x = tone(1600, 440.0);
        let (mut ended, mut padded) = (Vec::new(), Vec::new());
        fe.log_frames(&pool, isa(), &x, 0, 9, 1, true, &mut ended);
        let mut longer = x.clone();
        longer.extend([0.0; 400]);
        fe.log_frames(&pool, isa(), &longer, 0, 9, 1, false, &mut padded);
        // The zero after the end is preemphasised to −0.97·x[last] in `longer`, so only the
        // sample at the end differs: the frames are close, not equal.
        let worst = ended.iter().zip(&padded).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst > 0.0 && worst < 1.0, "{worst}");
    }
}
