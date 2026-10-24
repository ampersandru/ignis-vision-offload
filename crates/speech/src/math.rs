//! Scalar functions the ops apply per element, written branch-free from
//! arithmetic and bit operations only, so one body serves the scalar
//! reference and, compiled under `#[target_feature]`, the AVX2 path with the
//! same bits (Rust never contracts `a * b + c` into an FMA).
//!
//! Also the fp64 functions the op tests take as the reference.

/// e^x, relative error ≲ 3e-7; the input is clamped to [−87.3, 88], so the result is normal and finite.
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    // Round to nearest without a libm call: the 1.5·2^23 trick.
    const MAGIC: f32 = 12_582_912.0;
    let x = x.clamp(-87.3, 88.0);
    let n = (x * LOG2E + MAGIC) - MAGIC;
    let r = x - n * LN2_HI - n * LN2_LO;
    // Minimax polynomial for e^r on [−ln2/2, ln2/2] (Cephes' expf).
    let mut p = 1.987_569_1e-4f32;
    p = p * r + 1.398_199_9e-3;
    p = p * r + 8.333_452e-3;
    p = p * r + 4.166_579_6e-2;
    p = p * r + 1.666_666_5e-1;
    p = p * r + 5e-1;
    let e = p * r * r + r + 1.0;
    let scale = f32::from_bits((((n as i32) + 127) as u32) << 23);
    e * scale
}

/// erfc(z) for z ≥ 0, fractional error < 1.2e-7 (the Chebyshev fit of Numerical Recipes' `erfcc`).
#[inline(always)]
fn erfc_pos(z: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.5 * z);
    let mut p = 0.170_872_77f32;
    p = p * t - 0.822_152_23;
    p = p * t + 1.488_515_9;
    p = p * t - 1.135_203_98;
    p = p * t + 0.278_868_07;
    p = p * t - 0.186_288_06;
    p = p * t + 0.096_784_18;
    p = p * t + 0.374_091_96;
    p = p * t + 1.000_023_7;
    p = p * t - 1.265_512_2;
    t * exp(-z * z + p)
}

/// GELU with the exact erf: x · Φ(x) = ½ x · erfc(−x/√2).
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    let z = x * -std::f32::consts::FRAC_1_SQRT_2;
    let c = erfc_pos(z.abs());
    let erfc = if z >= 0.0 { c } else { 2.0 - c };
    0.5 * x * erfc
}

/// SiLU: x · σ(x).
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + exp(-x))
}

// ---------------------------------------------------------------- fp64 references

/// erf in fp64 to about 1e-14: the Maclaurin series under 3, the continued fraction of erfc above.
pub fn erf_f64(x: f64) -> f64 {
    let z = x.abs();
    let r = if z < 3.0 {
        let (mut term, mut sum, mut n) = (z, z, 0.0f64);
        while term.abs() > 1e-17 * sum.abs() {
            n += 1.0;
            term *= -z * z / n;
            sum += term / (2.0 * n + 1.0);
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else if z < 7.0 {
        // erfc(z) = e^{−z²}/√π · 1/(z + (1/2)/(z + 1/(z + (3/2)/(z + …))))
        let mut f = z;
        for k in (1..80).rev() {
            f = z + (k as f64 / 2.0) / f;
        }
        1.0 - (-z * z).exp() / std::f64::consts::PI.sqrt() / f
    } else {
        1.0
    };
    r.copysign(x)
}

pub fn gelu_f64(x: f64) -> f64 {
    0.5 * x * (1.0 + erf_f64(x / std::f64::consts::SQRT_2))
}

pub fn silu_f64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_is_within_three_ulp_of_fp64() {
        let mut x = -87.0f32;
        while x < 88.0 {
            let r = (x as f64).exp();
            let e = ((exp(x) as f64 - r) / r).abs();
            assert!(e < 3.6e-7, "exp({x}): {e:.2e}");
            x += 0.0137;
        }
        assert_eq!(exp(-200.0), exp(-87.3));
        assert!(exp(-200.0) < 1e-37);
    }

    #[test]
    fn erf_f64_matches_known_values() {
        // Abramowitz & Stegun table 7.1, and erfc(5) = 1.5374597944280e-12.
        assert!((erf_f64(0.5) - 0.520_499_877_813_046_5).abs() < 1e-15);
        assert!((erf_f64(1.0) - 0.842_700_792_949_714_9).abs() < 1e-15);
        assert!((erf_f64(2.0) - 0.995_322_265_018_952_7).abs() < 1e-15);
        assert!((erf_f64(-3.5) + 0.999_999_256_901_627_7).abs() < 1e-15);
        assert!((erf_f64(5.0) - (1.0 - 1.537_459_794_428_035e-12)).abs() < 1e-15);
    }

    #[test]
    fn gelu_and_silu_track_fp64_and_tanh_gelu_does_not() {
        let mut worst = 0f64;
        let mut tanh_worst = 0f64;
        let mut x = -12.0f32;
        while x < 12.0 {
            let r = gelu_f64(x as f64);
            let scale = (x as f64).abs().max(1e-3);
            worst = worst.max((gelu(x) as f64 - r).abs() / scale);
            let t = 0.5 * x as f64 * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x as f64 + 0.044715 * (x as f64).powi(3))).tanh());
            tanh_worst = tanh_worst.max((t - r).abs() / scale);
            let s = silu_f64(x as f64);
            assert!((silu(x) as f64 - s).abs() / scale < 1e-6, "silu({x})");
            x += 0.001;
        }
        assert!(worst < 1e-6, "erf GELU {worst:.2e}");
        assert!(tanh_worst > 1e-4, "the tanh approximation is {tanh_worst:.2e} away");
    }
}
