//! Which kernels run: AVX2, FMA and F16C detected at run time (spec
//! speech/01 §1, Q28, Q35, Q60).
//!
//! An [`Isa`] that names the AVX2 kernels exists only when [`select`] built
//! it from features that report all three, so no AVX2 instruction runs
//! outside a function selected after `is_x86_feature_detected!`. The scalar
//! kernels are the reference the tests compare against; no setting serves
//! them on x86-64, not even for debugging (Q64).

use std::sync::atomic::{AtomicU64, Ordering};

/// The reason the executor gives on a CPU without the three features (ADR 0066's state with a reason).
pub const NO_AVX2: &str = "the CPU lacks AVX2";

/// The CPU features the kernels need, detected or injected by a test.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Features {
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
}

impl Features {
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

static AVX2_DISPATCHES: AtomicU64 = AtomicU64::new(0);

impl Isa {
    /// The scalar reference. The executor never serves it (Q64); the tests compare against it.
    pub fn scalar() -> Isa {
        Isa(Kind::Scalar)
    }

    pub fn is_avx2(self) -> bool {
        self.0 != Kind::Scalar
    }

    /// Counts one dispatch to an AVX2 kernel and answers whether to take it.
    #[inline]
    pub(crate) fn avx2(self) -> bool {
        let yes = self.is_avx2();
        if yes {
            AVX2_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        }
        yes
    }
}

/// How many times this process dispatched to an AVX2 kernel (acceptance 6's witness).
pub fn avx2_dispatches() -> u64 {
    AVX2_DISPATCHES.load(Ordering::Relaxed)
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
}
