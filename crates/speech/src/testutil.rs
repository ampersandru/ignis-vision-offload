//! Deterministic test inputs: a SplitMix64 generator and generated blocks of every format.

use crate::quant::*;

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn u8(&mut self) -> u8 {
        self.u64() as u8
    }

    /// Uniform in [0, 1).
    pub fn f64(&mut self) -> f64 {
        (self.u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Roughly normal (Irwin-Hall of 4), mean 0, sd `sd`.
    pub fn normal(&mut self, sd: f32) -> f32 {
        let s: f64 = (0..4).map(|_| self.f64()).sum::<f64>() - 2.0;
        (s * 3f64.sqrt()) as f32 * sd
    }

    /// Activations with the occasional outlier, as residual streams have.
    pub fn activations(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| if self.u64() % 97 == 0 { self.normal(8.0) } else { self.normal(1.0) }).collect()
    }
}

fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
    let n = std::mem::size_of_val(v);
    // SAFETY: plain-old-data blocks viewed as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n) }.to_vec()
}

/// `n` rows of `k` generated weights in the file's layout of `ty` (Q4K, Q6K, Q80, F16 or F32).
pub fn weights(rng: &mut Rng, ty: WType, n: usize, k: usize) -> Vec<u8> {
    let rows = n * k;
    match ty {
        WType::Q4K => bytes_of(
            &(0..rows / QK_K)
                .map(|_| BlockQ4K {
                    d: f32_to_f16(0.002 + 0.01 * rng.f64() as f32),
                    dmin: f32_to_f16(0.001 + 0.01 * rng.f64() as f32),
                    scales: std::array::from_fn(|_| rng.u8()),
                    qs: std::array::from_fn(|_| rng.u8()),
                })
                .collect::<Vec<_>>(),
        ),
        WType::Q6K => bytes_of(
            &(0..rows / QK_K)
                .map(|_| BlockQ6K {
                    ql: std::array::from_fn(|_| rng.u8()),
                    qh: std::array::from_fn(|_| rng.u8()),
                    scales: std::array::from_fn(|_| (rng.u8() as i8).max(-127)),
                    d: f32_to_f16(0.0005 + 0.002 * rng.f64() as f32),
                })
                .collect::<Vec<_>>(),
        ),
        WType::Q80 => bytes_of(
            &(0..rows / QK8_0)
                .map(|_| BlockQ80 {
                    d: f32_to_f16(0.001 + 0.01 * rng.f64() as f32),
                    qs: std::array::from_fn(|_| (rng.u8() as i8).max(-127)),
                })
                .collect::<Vec<_>>(),
        ),
        WType::F16 => bytes_of(&(0..rows).map(|_| f32_to_f16(rng.normal(0.05))).collect::<Vec<_>>()),
        WType::F32 => bytes_of(&(0..rows).map(|_| rng.normal(0.05)).collect::<Vec<_>>()),
        WType::Q4Kx8 => panic!("generate Q4K and repack it"),
    }
}

/// Copies `bytes` into a 64-byte aligned buffer.
pub fn aligned(bytes: &[u8]) -> crate::mem::AlignedBuf {
    let mut b = crate::mem::AlignedBuf::zeroed(bytes.len());
    b.as_mut_slice().copy_from_slice(bytes);
    b
}
