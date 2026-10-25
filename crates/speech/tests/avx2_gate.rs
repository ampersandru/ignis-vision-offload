//! Acceptance 6, in its own test binary so no other test runs a kernel meanwhile: with feature
//! detection injected as reporting none of AVX2, FMA and F16C, the executor is unavailable with
//! the reason "the CPU lacks AVX2", it loads nothing, and no AVX2 kernel runs.

mod common;

use ignis_speech::{avx2_dispatches, Availability, ExecError, Executor, Features, Settings, NO_AVX2};

#[test]
fn without_avx2_the_executor_is_unavailable_and_runs_no_avx2_kernel() {
    let files = common::synthetic::pair("noavx2");
    let before = avx2_dispatches();
    for f in [Features::none(), Features::detect().without_f16c(), Features::detect().without_avx2(), Features::detect().without_fma()] {
        let e = Executor::new(Settings::default(), files.clone(), f);
        assert_eq!(e.availability(), Availability::Unavailable(NO_AVX2.to_string()));
        assert_eq!(NO_AVX2, "the CPU lacks AVX2");
        assert!(matches!(e.load(), Err(ExecError::Unavailable(r)) if r == NO_AVX2));
        assert!(matches!(e.open("", None), Err(ExecError::Unavailable(_))));
        assert!(!e.is_loaded());
    }
    assert_eq!(avx2_dispatches(), before, "an AVX2 kernel ran on an executor without AVX2");
    // The witness counts (in a test build, where the counter exists): a served executor does
    // dispatch to AVX2.
    let e = Executor::new(Settings::default(), files, Features::detect());
    if e.availability() == Availability::Available {
        let mut s = e.open("", None).unwrap();
        s.push(&common::synthetic::audio(1.0)).unwrap();
        s.commit().unwrap();
        assert!(avx2_dispatches() > before || !cfg!(debug_assertions));
    }
}
