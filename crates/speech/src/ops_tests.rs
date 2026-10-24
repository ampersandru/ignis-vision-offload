//! Acceptance 9: every op within a threshold of an fp64 reference at the
//! model's real geometry, scalar and AVX2, each with a negative control (a
//! plausible wrong variant) that the same check rejects.

use super::*;
use crate::cpu::{select, Features};
use crate::mem::Blob;
use crate::quant::{f32_to_f16, WType};
use crate::testutil::*;

/// Max |y − ref| over the row, normalised by the row's max |ref|.
const TOL: f64 = 1e-5;

fn isas() -> Vec<Isa> {
    let mut v = vec![Isa::scalar()];
    match select(Features::detect()) {
        Ok(isa) => v.push(isa),
        Err(_) => eprintln!("SKIP: this CPU lacks AVX2; only the scalar ops are checked"),
    }
    v
}

/// The worst row error, each row normalised by its own max |ref|.
fn err(y: &[f32], r: &[f64], row: usize) -> f64 {
    y.chunks(row)
        .zip(r.chunks(row))
        .map(|(a, b)| {
            let scale = b.iter().fold(0f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
            a.iter().zip(b).map(|(&x, &y)| (x as f64 - y).abs()).fold(0.0, f64::max) / scale
        })
        .fold(0.0, f64::max)
}

fn pass(name: &str, isa: Isa, y: &[f32], r: &[f64], row: usize) {
    let e = err(y, r, row);
    assert!(e <= TOL, "{name} {isa:?}: {e:.3e} over {TOL:.0e}");
}

fn fail(name: &str, y: &[f32], r: &[f64], row: usize) {
    let e = err(y, r, row);
    assert!(e > TOL, "{name}'s negative control passed the check ({e:.3e})");
}

fn layer_norm_ref(x: &[f32], w: &[f32], b: &[f32], eps: f64, ddof: f64) -> Vec<f64> {
    let d = w.len();
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(d) {
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
        let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / (d as f64 - ddof);
        out.extend(row.iter().zip(w).zip(b).map(|((&v, &w), &b)| (v as f64 - mean) / (var + eps).sqrt() * w as f64 + b as f64));
    }
    out
}

#[test]
fn layer_norm_matches_fp64() {
    let mut rng = Rng::new(1);
    for d in [1024, 896] {
        let x = rng.activations(104 * d);
        let w: Vec<f32> = (0..d).map(|_| 1.0 + rng.normal(0.2)).collect();
        let b: Vec<f32> = (0..d).map(|_| rng.normal(0.1)).collect();
        let r = layer_norm_ref(&x, &w, &b, 1e-5, 0.0);
        for isa in isas() {
            let mut y = x.clone();
            layer_norm(isa, &mut y, &w, &b, 1e-5);
            pass("layer_norm", isa, &y, &r, d);
        }
        // The unbiased variance is the wrong one.
        let bad: Vec<f32> = layer_norm_ref(&x, &w, &b, 1e-5, 1.0).iter().map(|&v| v as f32).collect();
        fail("layer_norm", &bad, &r, d);
    }
}

#[test]
fn rms_norm_matches_fp64() {
    let mut rng = Rng::new(2);
    // The decoder's hidden 2048 and 1024, and QK-norm over heads of 128.
    for d in [2048, 1024, 128] {
        let x = rng.activations(104 * 16 * 128);
        let w: Vec<f32> = (0..d).map(|_| 1.0 + rng.normal(0.3)).collect();
        let reference = |ddof: f64| -> Vec<f64> {
            x.chunks(d)
                .flat_map(|row| {
                    let ms = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / (d as f64 - ddof);
                    row.iter().zip(&w).map(move |(&v, &w)| v as f64 / (ms + 1e-6).sqrt() * w as f64).collect::<Vec<_>>()
                })
                .collect()
        };
        let r = reference(0.0);
        for isa in isas() {
            let mut y = x.clone();
            rms_norm(isa, &mut y, &w, 1e-6);
            pass("rms_norm", isa, &y, &r, d);
        }
        let bad: Vec<f32> = reference(1.0).iter().map(|&v| v as f32).collect();
        fail("rms_norm", &bad, &r, d);
    }
}

#[test]
fn gelu_matches_fp64_and_rejects_the_tanh_form() {
    // The encoder FFN's 104 × 4096, with the stem's wider range.
    let x: Vec<f32> = Rng::new(3).activations(104 * 4096).iter().map(|v| v * 2.0).collect();
    let r: Vec<f64> = x.iter().map(|&v| math::gelu_f64(v as f64)).collect();
    for isa in isas() {
        let mut y = x.clone();
        gelu(isa, &mut y);
        pass("gelu", isa, &y, &r, 4096);
        let mut y = x.clone();
        let b = vec![0f32; 4096];
        add_bias_gelu(isa, &mut y, &b);
        pass("add_bias_gelu", isa, &y, &r, 4096);
    }
    let tanh: Vec<f32> = x
        .iter()
        .map(|&v| {
            let v = v as f64;
            (0.5 * v * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (v + 0.044715 * v * v * v)).tanh())) as f32
        })
        .collect();
    fail("gelu", &tanh, &r, 4096);
}

#[test]
fn silu_mul_matches_fp64() {
    let mut rng = Rng::new(4);
    let (g, u) = (rng.activations(104 * 6144), rng.activations(104 * 6144));
    let r: Vec<f64> = g.iter().zip(&u).map(|(&a, &b)| math::silu_f64(a as f64) * b as f64).collect();
    for isa in isas() {
        let mut y = g.clone();
        silu_mul(isa, &mut y, &u);
        pass("silu_mul", isa, &y, &r, 6144);
    }
    // The gate and the up projection swapped.
    let bad: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| (math::silu_f64(b as f64) * a as f64) as f32).collect();
    fail("silu_mul", &bad, &r, 6144);
}

#[test]
fn softmax_matches_fp64() {
    let mut rng = Rng::new(5);
    for n in [104, 1000, 13] {
        // Scores spread like attention logits, so no single element takes all the mass.
        let x: Vec<f32> = (0..n).map(|_| rng.normal(1.5)).collect();
        let max = x.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
        let den: f64 = x.iter().map(|&v| (v as f64 - max).exp()).sum();
        let r: Vec<f64> = x.iter().map(|&v| (v as f64 - max).exp() / den).collect();
        for isa in isas() {
            let mut y = x.clone();
            softmax(isa, &mut y);
            pass("softmax", isa, &y, &r, n);
        }
        // Base 2 where the softmax takes e: a fast-exponential slip.
        let den2: f64 = x.iter().map(|&v| (v as f64 - max).exp2()).sum();
        let bad: Vec<f32> = x.iter().map(|&v| ((v as f64 - max).exp2() / den2) as f32).collect();
        fail("softmax", &bad, &r, n);
    }
}

fn rope_ref(x: &[f32], pos: &[[u32; 3]], heads: usize, hd: usize, sections: [usize; 3], gptj: bool) -> Vec<f64> {
    let half = hd / 2;
    let inv = Rope::inv_freq(1e6, hd);
    let mut out = vec![0f64; x.len()];
    for (r, p) in pos.iter().enumerate() {
        for h in 0..heads {
            let base = (r * heads + h) * hd;
            for i in 0..half {
                let a = (p[Rope::stream(i, sections)] as f32 * inv[i]) as f64;
                let (c, s) = (a.cos(), a.sin());
                let (i0, i1) = if gptj { (base + 2 * i, base + 2 * i + 1) } else { (base + i, base + i + half) };
                let (x0, x1) = (x[i0] as f64, x[i1] as f64);
                out[i0] = x0 * c - x1 * s;
                out[i1] = x1 * c + x0 * s;
            }
        }
    }
    out
}

#[test]
fn rope_matches_fp64_on_text_and_on_three_streams() {
    let (heads, hd, sec) = (16, 128, [24, 20, 20]);
    let mut rng = Rng::new(6);
    // A 104-row window late in a long transcription, then rows whose three streams differ.
    let text: Vec<[u32; 3]> = (900..1004).map(|p| [p; 3]).collect();
    let mixed: Vec<[u32; 3]> = (0..104).map(|i| [i, 3 * i % 17, 50 + i / 3]).collect();
    for pos in [&text, &mixed] {
        let x = rng.activations(pos.len() * heads * hd);
        let r = rope_ref(&x, pos, heads, hd, sec, false);
        let rope = Rope::new(pos, sec, 1e6, hd);
        for isa in isas() {
            let mut y = x.clone();
            rope.apply(isa, &mut y, heads);
            pass("rope", isa, &y, &r, hd);
        }
        // The GPT-J pairing (2i, 2i + 1) is the wrong layout for this checkpoint.
        let bad: Vec<f32> = rope_ref(&x, pos, heads, hd, sec, true).iter().map(|&v| v as f32).collect();
        fail("rope", &bad, &r, hd);
    }
    // With equal streams M-RoPE is plain RoPE: the table does not depend on the sections.
    let a = Rope::text(900, 104, sec, 1e6, hd);
    let b = Rope::text(900, 104, [64, 0, 0], 1e6, hd);
    assert!(a.cos == b.cos && a.sin == b.sin);
}

fn attention_ref(shape: AttnShape, q: &[f32], m: usize, k: &[f64], v: &[f64], t: usize, mask: Mask, scale: f64) -> Vec<f64> {
    let AttnShape { hq, hkv, hd } = shape;
    let mut out = vec![0f64; m * hq * hd];
    for i in 0..m {
        let len = match mask {
            Mask::Causal { past } => (past + i + 1).min(t),
            Mask::Full => t,
        };
        for h in 0..hq {
            let kvh = h / (hq / hkv);
            let qr = &q[(i * hq + h) * hd..][..hd];
            let s: Vec<f64> =
                (0..len).map(|j| (0..hd).map(|d| qr[d] as f64 * k[(j * hkv + kvh) * hd + d]).sum::<f64>() * scale).collect();
            let max = s.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let den: f64 = s.iter().map(|&x| (x - max).exp()).sum();
            for j in 0..len {
                let p = (s[j] - max).exp() / den;
                for d in 0..hd {
                    out[(i * hq + h) * hd + d] += p * v[(j * hkv + kvh) * hd + d];
                }
            }
        }
    }
    out
}

#[test]
fn decoder_attention_matches_fp64() {
    // GQA 16/8 heads of 128 over the f16 cache: a 104-row audio window after 150 positions,
    // then one decode row against a 1,000-position cache.
    let shape = AttnShape { hq: 16, hkv: 8, hd: 128 };
    let pool = Pool::new(4);
    let mut rng = Rng::new(7);
    for (m, past) in [(104, 150), (1, 999)] {
        let t = past + m;
        let q = rng.activations(m * 16 * 128);
        let kh: Vec<u16> = rng.activations(t * 8 * 128).iter().map(|&v| f32_to_f16(v)).collect();
        let vh: Vec<u16> = rng.activations(t * 8 * 128).iter().map(|&v| f32_to_f16(v)).collect();
        let kf: Vec<f64> = kh.iter().map(|&h| f16_to_f32(h) as f64).collect();
        let vf: Vec<f64> = vh.iter().map(|&h| f16_to_f32(h) as f64).collect();
        let scale = 1.0 / (128f64).sqrt();
        let mask = Mask::Causal { past };
        let r = attention_ref(shape, &q, m, &kf, &vf, t, mask, scale);
        for isa in isas() {
            let mut y = vec![f32::NAN; m * 16 * 128];
            attention(&pool, isa, shape, &q, m, Kv::F16 { k: &kh, v: &vh }, t, mask, &mut y);
            pass("decoder attention", isa, &y, &r, 128);
        }
        if m > 1 {
            // Each row seeing one key too many.
            let bad: Vec<f32> =
                attention_ref(shape, &q, m, &kf, &vf, t, Mask::Causal { past: past + 1 }, scale).iter().map(|&v| v as f32).collect();
            fail("decoder attention", &bad, &r, 128);
        }
    }
}

#[test]
fn encoder_attention_matches_fp64() {
    // 16 heads of 64 (14 for the 0.6B's 896), bidirectional inside one 104-token window.
    let pool = Pool::new(4);
    let mut rng = Rng::new(8);
    for hq in [16, 14] {
        let shape = AttnShape { hq, hkv: hq, hd: 64 };
        let (m, t) = (104, 104);
        let q = rng.activations(m * hq * 64);
        let k = rng.activations(t * hq * 64);
        let v = rng.activations(t * hq * 64);
        let (kf, vf): (Vec<f64>, Vec<f64>) = (k.iter().map(|&x| x as f64).collect(), v.iter().map(|&x| x as f64).collect());
        let r = attention_ref(shape, &q, m, &kf, &vf, t, Mask::Full, 0.125);
        for isa in isas() {
            let mut y = vec![f32::NAN; m * hq * 64];
            attention(&pool, isa, shape, &q, m, Kv::F32 { k: &k, v: &v }, t, Mask::Full, &mut y);
            pass("encoder attention", isa, &y, &r, 64);
        }
        // The 1/√d scale left out.
        let bad: Vec<f32> = attention_ref(shape, &q, m, &kf, &vf, t, Mask::Full, 1.0).iter().map(|&v| v as f32).collect();
        fail("encoder attention", &bad, &r, 64);
    }
}

fn conv_ref(w: &[f64], bias: &[f32], x: &[f32], c_in: usize, c_out: usize, h: usize, wd: usize, flip: bool) -> Vec<f64> {
    let (oh, ow) = (h.div_ceil(2), wd.div_ceil(2));
    let mut out = vec![0f64; c_out * oh * ow];
    for co in 0..c_out {
        for y in 0..oh {
            for xo in 0..ow {
                let mut s = bias[co] as f64;
                for c in 0..c_in {
                    for kh in 0..3 {
                        for kw in 0..3 {
                            let (iy, ix) = ((2 * y + kh) as isize - 1, (2 * xo + kw) as isize - 1);
                            if iy < 0 || ix < 0 || iy >= h as isize || ix >= wd as isize {
                                continue;
                            }
                            let (fh, fw) = if flip { (2 - kh, 2 - kw) } else { (kh, kw) };
                            s += w[co * c_in * 9 + c * 9 + fh * 3 + fw] * x[(c * h + iy as usize) * wd + ix as usize] as f64;
                        }
                    }
                }
                out[(co * oh + y) * ow + xo] = math::gelu_f64(s);
            }
        }
    }
    out
}

#[test]
fn the_convolution_stem_matches_fp64() {
    // The first convolution on a whole 1 s chunk (1 → 480, 128 × 100), and the third (480 → 480,
    // 32 × 25), F16 weights as the GGUF holds them.
    let pool = Pool::new(4);
    let mut rng = Rng::new(9);
    for (c_in, h, wd) in [(1, 128, 100), (480, 32, 25)] {
        let c_out = 480;
        let k = c_in * 9;
        let wb = aligned(&weights(&mut rng, WType::F16, c_out, k));
        let wf: Vec<f64> = crate::quant::cast_slice::<u16>(wb.as_slice(), c_out * k).iter().map(|&h| f16_to_f32(h) as f64).collect();
        let w = Weight::new("conv", WType::F16, c_out, k, Blob::owned(wb));
        let bias: Vec<f32> = (0..c_out).map(|_| rng.normal(0.1)).collect();
        let x = rng.activations(c_in * h * wd);
        let r = conv_ref(&wf, &bias, &x, c_in, c_out, h, wd, false);
        let plane = h.div_ceil(2) * wd.div_ceil(2);
        for isa in isas() {
            let mut y = Vec::new();
            conv3x3s2_gelu(&pool, isa, &w, &bias, &x, c_in, h, wd, &mut y);
            pass("conv", isa, &y, &r, plane);
        }
        // A flipped kernel: convolution where the checkpoint cross-correlates.
        let bad: Vec<f32> = conv_ref(&wf, &bias, &x, c_in, c_out, h, wd, true).iter().map(|&v| v as f32).collect();
        fail("conv", &bad, &r, plane);
    }
}

#[test]
fn to_f16_is_bit_identical_on_both_paths() {
    let mut rng = Rng::new(10);
    let mut x: Vec<f32> = rng.activations(4099);
    // Ties at f16 precision, subnormals, overflow and infinities.
    x.extend([1.0 + 1.0 / 2048.0, 1.0 + 3.0 / 2048.0, 6e-8, -3e-5, 65520.0, 1e9, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0]);
    let isa = isas();
    let mut s = vec![0u16; x.len()];
    to_f16(Isa::scalar(), &x, &mut s);
    for v in &isa {
        let mut a = vec![1u16; x.len()];
        to_f16(*v, &x, &mut a);
        assert_eq!(a, s, "{v:?}");
    }
}
