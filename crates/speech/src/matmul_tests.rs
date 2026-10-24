//! Acceptance 3-5: the quantizers bit-identical between the scalar and the
//! AVX2 path, every kernel and tile shape within 1e-5 of the fp64 reference,
//! and a doubled block scale caught on every format and both Q4_K layouts.

use super::*;
use crate::cpu::{select, Features};
use crate::testutil::*;

/// The gate fixed before the spike ran: |y − ref| / Σ|w·a| per output element.
const GATE: f64 = 1e-5;

fn avx2() -> Option<Isa> {
    match select(Features::detect()) {
        Ok(isa) => Some(isa),
        Err(_) => {
            eprintln!("SKIP: this CPU lacks AVX2; only the scalar kernels are checked");
            None
        }
    }
}

/// fp64 reference: per output element the dot product of the dequantized weights with the
/// dequantized activations, and Σ|w·a|.
fn reference(w: &Weight, act: &[u8], at: AType, m: usize) -> (Vec<f64>, Vec<f64>) {
    let (n, k) = (w.n, w.k);
    let mut wd = vec![0f64; n * k];
    for c in 0..n {
        dequant_row_f64(w.ty, w.bytes(), c, k, &mut wd[c * k..(c + 1) * k]);
    }
    let mut ad = vec![0f64; k];
    let (mut y, mut norm) = (vec![0f64; m * n], vec![0f64; m * n]);
    for r in 0..m {
        dequant_act_f64(at, act, r, k, &mut ad);
        for c in 0..n {
            let wr = &wd[c * k..(c + 1) * k];
            y[r * n + c] = wr.iter().zip(&ad).map(|(a, b)| a * b).sum();
            norm[r * n + c] = wr.iter().zip(&ad).map(|(a, b)| (a * b).abs()).sum();
        }
    }
    (y, norm)
}

fn worst(out: &[f32], y: &[f64], norm: &[f64]) -> f64 {
    out.iter().zip(y).zip(norm).map(|((&o, &r), &s)| if s == 0.0 { (o as f64).abs() } else { (o as f64 - r).abs() / s }).fold(0.0, f64::max)
}

/// The activation rows the kernels see, quantized by the scalar path.
fn quantized(x: &[f32], m: usize, k: usize, at: AType) -> Vec<u8> {
    if at == AType::F32 {
        return x[..m * k].iter().flat_map(|v| v.to_le_bytes()).collect();
    }
    let pool = Pool::new(1);
    let mut out = vec![0u8; m * at.row_bytes(k) + 64];
    let off = out.as_ptr().align_offset(64);
    quantize(&pool, Isa::scalar(), at, x, m, k, &mut out[off..]);
    out[off..off + m * at.row_bytes(k)].to_vec()
}

fn weight(ty: WType, n: usize, k: usize, seed: u64) -> Weight {
    let mut rng = Rng::new(seed);
    if ty == WType::Q4Kx8 {
        let file = aligned(&weights(&mut rng, WType::Q4K, n, k));
        return Weight::repack_q4k("w", n, k, file.as_slice());
    }
    Weight::new("w", ty, n, k, Blob::owned(aligned(&weights(&mut rng, ty, n, k))))
}

/// Every tile shape the driver can pick for `ty`, for m = 1 and m > 1.
fn tile_sweep(ty: WType, m: usize) -> Vec<Tiles> {
    #[cfg(target_arch = "x86_64")]
    {
        let shapes: &[(usize, usize)] = if ty == WType::Q4Kx8 { &crate::avx2::TILES_X8 } else { &crate::avx2::TILES };
        let d = Tiles::default_for(ty);
        if m == 1 {
            shapes.iter().filter(|s| s.0 == 1).map(|&s| Tiles { gemv: s, gemm: d.gemm }).collect()
        } else {
            shapes.iter().map(|&s| Tiles { gemv: d.gemv, gemm: s }).collect()
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        vec![Tiles::default_for(ty)]
    }
}

fn check_format(ty: WType, n: usize, k: usize, ms: &[usize]) {
    let pool = Pool::new(4);
    let w = weight(ty, n, k, 0x5eed ^ k as u64);
    let at = ty.act();
    let mut rng = Rng::new(7 + n as u64);
    for &m in ms {
        let x = rng.activations(m * k);
        let act = quantized(&x, m, k, at);
        let (y, norm) = reference(&w, &act, at, m);
        let mut out = vec![f32::NAN; m * n];
        matmul(&pool, Isa::scalar(), &w, &x, m, &mut out);
        let e = worst(&out, &y, &norm);
        assert!(e <= GATE, "{ty:?} scalar m={m}: {e:.3e}");
        if let Some(isa) = avx2() {
            for tiles in tile_sweep(ty, m) {
                out.fill(f32::NAN);
                matmul_tiled(&pool, isa, &w, &x, m, &mut out, tiles);
                let e = worst(&out, &y, &norm);
                assert!(e <= GATE, "{ty:?} AVX2 m={m} {tiles:?}: {e:.3e}");
            }
        }
    }
}

/// M = 1, a remainder, 104 (an encoder window) and 512.
const MS: [usize; 4] = [1, 7, 104, 512];

#[test]
fn q4k_file_layout_matches_fp64() {
    check_format(WType::Q4K, 70, 512, &MS);
}

#[test]
fn q4k_interleaved_layout_matches_fp64() {
    check_format(WType::Q4Kx8, 72, 512, &MS);
}

#[test]
fn q6k_matches_fp64() {
    check_format(WType::Q6K, 70, 512, &MS);
}

#[test]
fn q80_matches_fp64() {
    check_format(WType::Q80, 70, 512, &MS);
}

#[test]
fn f16_matches_fp64() {
    // k = 100 leaves a tail under the 8 lanes; k = 9 is the first convolution's.
    check_format(WType::F16, 70, 100, &MS);
    check_format(WType::F16, 13, 9, &[1, 7, 104]);
}

#[test]
fn f32_matches_fp64() {
    check_format(WType::F32, 70, 100, &MS);
}

/// Acceptance 5: doubling one block scale makes 4's check fail, in each format and each Q4_K layout.
#[test]
fn a_doubled_block_scale_fails_the_check() {
    let pool = Pool::new(2);
    let (n, k, m) = (16, 512, 3);
    for ty in [WType::Q4K, WType::Q4Kx8, WType::Q6K, WType::Q80, WType::F16, WType::F32] {
        let w = weight(ty, n, k, 99);
        let at = ty.act();
        let x = Rng::new(3).activations(m * k);
        let act = quantized(&x, m, k, at);
        let (y, norm) = reference(&w, &act, at, m);
        // Row 5's second block: its scale doubled (F16 and F32 have none: one weight doubled).
        let mut bytes = w.bytes().to_vec();
        let rb = ty.row_bytes(k);
        let double_f16 = |b: &mut [u8], at: usize| {
            let h = u16::from_le_bytes([b[at], b[at + 1]]);
            b[at..at + 2].copy_from_slice(&f32_to_f16(2.0 * f16_to_f32(h)).to_le_bytes());
        };
        match ty {
            WType::Q4K => double_f16(&mut bytes, 5 * rb + 144),
            WType::Q4Kx8 => double_f16(&mut bytes, 1184 + 2 * 5),
            WType::Q6K => double_f16(&mut bytes, 5 * rb + 210 + 208),
            WType::Q80 => double_f16(&mut bytes, 5 * rb + 34),
            WType::F16 => double_f16(&mut bytes, 5 * rb + 2 * 300),
            WType::F32 => {
                let at = 5 * rb + 4 * 300;
                let v = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
                bytes[at..at + 4].copy_from_slice(&(2.0 * v).to_le_bytes());
            }
        }
        let bad = Weight { name: "bad".into(), ty, n, k, data: Blob::owned(aligned(&bytes)) };
        let mut isas = vec![Isa::scalar()];
        isas.extend(avx2());
        for isa in isas {
            let mut out = vec![0f32; m * n];
            matmul(&pool, isa, &bad, &x, m, &mut out);
            for r in 0..m {
                let row = r * n + 5;
                let e = worst(&out[row..row + 1], &y[row..row + 1], &norm[row..row + 1]);
                assert!(e > GATE, "{ty:?} {isa:?}: a doubled scale gave {e:.3e}, inside the gate");
            }
            // The other rows still pass.
            let e = worst(&out[..5], &y[..5], &norm[..5]);
            assert!(e <= GATE, "{ty:?} {isa:?}: untouched rows {e:.3e}");
        }
    }
}

/// Acceptance 3: the activation quantizers give bit-identical blocks on both paths.
#[test]
fn quantizers_are_bit_identical_on_both_paths() {
    let Some(isa) = avx2() else { return };
    let pool = Pool::new(1);
    let k = 1024;
    let mut rng = Rng::new(11);
    let mut inputs: Vec<Vec<f32>> = (0..20).map(|_| rng.activations(k)).collect();
    inputs.push(vec![0.0; k]);
    // Exact ties after scaling, a negative maximum, a repeated maximum of both signs,
    // subnormals and very large values.
    inputs.push((0..k).map(|i| (i % 255) as f32 - 127.0).collect());
    inputs.push((0..k).map(|i| if i % 64 == 0 { -3.0 } else { (i % 7) as f32 * 0.25 }).collect());
    inputs.push((0..k).map(|i| if i % 2 == 0 { 2.5 } else { -2.5 }).collect());
    inputs.push((0..k).map(|i| f32::from_bits(1 + i as u32)).collect());
    inputs.push((0..k).map(|i| (i as f32 - 512.0) * 1e30).collect());
    inputs.push((0..k).map(|i| (i as f32 * 0.37).sin() * 0.5 + 0.5 / 127.0 * (i % 3) as f32).collect());
    for at in [AType::Q8K, AType::Q80] {
        for (t, x) in inputs.iter().enumerate() {
            let rb = at.row_bytes(k);
            let mut s = aligned(&vec![0u8; rb]);
            let mut v = aligned(&vec![0xAAu8; rb]);
            quantize(&pool, Isa::scalar(), at, x, 1, k, s.as_mut_slice());
            quantize(&pool, isa, at, x, 1, k, v.as_mut_slice());
            assert!(s.as_slice() == v.as_slice(), "{at:?} input {t}: the scalar and AVX2 blocks differ");
        }
    }
}

/// Acceptance 7's layout: 1,184 bytes per 8 × 256 weights, and the same products as the file's.
#[test]
fn the_interleaved_layout_is_1184_bytes_per_group_block() {
    let (n, k) = (16, 512);
    let file = aligned(&weights(&mut Rng::new(5), WType::Q4K, n, k));
    let w = Weight::repack_q4k("w", n, k, file.as_slice());
    assert_eq!(w.bytes().len(), n / 8 * k / QK_K * 1184);
    assert_eq!(w.bytes().len() as f64 / file.len() as f64, 1184.0 / 1152.0);
    let mut a = vec![0f64; k];
    let mut b = vec![0f64; k];
    for c in 0..n {
        dequant_row_f64(WType::Q4K, file.as_slice(), c, k, &mut a);
        dequant_row_f64(WType::Q4Kx8, w.bytes(), c, k, &mut b);
        assert_eq!(a, b, "row {c}");
    }
}
