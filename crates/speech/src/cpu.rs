//! Which kernels run: AVX2, FMA and F16C detected at run time (spec
//! speech/01 §1, Q28, Q35, Q60).
//!
//! An [`Isa`] that names the AVX2 kernels exists only when [`select`] built
//! it from features that report all three, so no AVX2 instruction runs
//! outside a function selected after `is_x86_feature_detected!`. Safe code
//! cannot claim a feature the CPU lacks: [`Features`] comes from detection,
//! and a test can only take features away from it. The scalar kernels are
//! the reference the tests compare against; no setting serves them on x86-64,
//! not even for debugging (Q64).

#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicU64, Ordering};

/// The reason the executor gives on a CPU without the three features (ADR 0067's state with a reason).
pub const NO_AVX2: &str = "the CPU lacks AVX2";

/// The CPU features the kernels need, as detected, or fewer: the fields are private, so
/// nothing outside this module can report a feature detection did not. `Default` is none.
///
/// ```compile_fail
/// let _ = ignis_speech::Features { avx2: true, fma: true, f16c: true };
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Features {
    avx2: bool,
    fma: bool,
    f16c: bool,
}

impl Features {
    /// No feature: what a CPU without AVX2 reports.
    pub fn none() -> Features {
        Features::default()
    }

    /// These features without AVX2 (a test of the unavailable state).
    pub fn without_avx2(self) -> Features {
        Features { avx2: false, ..self }
    }

    /// These features without FMA.
    pub fn without_fma(self) -> Features {
        Features { fma: false, ..self }
    }

    /// These features without F16C.
    pub fn without_f16c(self) -> Features {
        Features { f16c: false, ..self }
    }

    pub fn detect() -> Features {
        #[cfg(target_arch = "x86_64")]
        {
            Features {
                avx2: is_x86_feature_detected!("avx2"),
                fma: is_x86_feature_detected!("fma"),
                f16c: is_x86_feature_detected!("f16c"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Features::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

/// The kernel family a computation dispatches to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Isa(Kind);

/// Counted only with debug assertions (the test builds): a shared counter bumped per row
/// inside pool tasks would put every worker on one cache line in a served build.
#[cfg(debug_assertions)]
static AVX2_DISPATCHES: AtomicU64 = AtomicU64::new(0);

impl Isa {
    /// The scalar reference. The executor never serves it (Q64); the tests compare against it.
    pub fn scalar() -> Isa {
        Isa(Kind::Scalar)
    }

    pub fn is_avx2(self) -> bool {
        self.0 != Kind::Scalar
    }

    /// Answers whether to take an AVX2 kernel, counting the dispatch in a test build.
    #[inline]
    pub(crate) fn avx2(self) -> bool {
        let yes = self.is_avx2();
        #[cfg(debug_assertions)]
        if yes {
            AVX2_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        }
        yes
    }
}

/// How many times this process dispatched to an AVX2 kernel (acceptance 6's witness).
/// Counted with debug assertions only, as `cargo test` builds; 0 in a release build.
pub fn avx2_dispatches() -> u64 {
    #[cfg(debug_assertions)]
    return AVX2_DISPATCHES.load(Ordering::Relaxed);
    #[cfg(not(debug_assertions))]
    0
}

/// The kernels to serve with: AVX2 when the features report AVX2, FMA and
/// F16C, otherwise the executor is unavailable with [`NO_AVX2`].
pub fn select(f: Features) -> Result<Isa, &'static str> {
    #[cfg(target_arch = "x86_64")]
    {
        if f.avx2 && f.fma && f.f16c {
            return Ok(Isa(Kind::Avx2));
        }
        Err(NO_AVX2)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = f;
        Err("the CPU executor has no kernels for this architecture")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx2_is_selected_only_with_all_three_features() {
        let all = Features { avx2: true, fma: true, f16c: true };
        #[cfg(target_arch = "x86_64")]
        assert!(select(all).unwrap().is_avx2());
        for missing in 0..3 {
            let mut f = all;
            match missing {
                0 => f.avx2 = false,
                1 => f.fma = false,
                _ => f.f16c = false,
            }
            assert_eq!(select(f), Err(NO_AVX2));
        }
        assert_eq!(select(Features::default()), Err(NO_AVX2));
        assert!(!Isa::scalar().is_avx2());
    }

    /// Safe code can only take features away from what detection reported, never add one: an
    /// AVX2 `Isa` needs a CPU that has AVX2.
    #[test]
    fn features_outside_detection_only_lose_what_was_detected() {
        let detected = Features::detect();
        for f in [Features::none(), detected.without_avx2(), detected.without_fma(), detected.without_f16c()] {
            assert!(!f.avx2 || detected.avx2);
            assert!(!f.fma || detected.fma);
            assert!(!f.f16c || detected.f16c);
            assert_eq!(select(f), Err(NO_AVX2), "{f:?}");
        }
        assert_eq!(select(detected.without_avx2().without_fma().without_f16c()), select(Features::none()));
    }
}
