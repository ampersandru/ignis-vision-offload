//! The front end against the recorded fixture, in the default build (spec speech/01 testing
//! decisions): the log-mel of the FLEURS fixture clip within `thresholds.json`'s bound of
//! Transformers FP32, and its audio-token count the official processor's.

mod common;

use common::*;
use ignis_speech::cpu::{select, Features, Isa};
use ignis_speech::mel::{audio_tokens, frames_of, normalise, Mel, N_MELS};
use ignis_speech::pool::Pool;

#[test]
fn the_log_mel_matches_transformers_fp32() {
    let audio = read_wav(&fixtures().join("it_it-1703.wav"));
    let bound = json(&fixtures().join("thresholds.json"))["mel"]["bound"].as_f64().unwrap();
    let pool = Pool::new(4);
    let mut isas = vec![Isa::scalar()];
    isas.extend(select(Features::detect()).ok());
    for size in ["1.7b", "0.6b"] {
        let fx = layers(size);
        let (shape, reference) = &fx.tensors["mel"];
        assert_eq!(shape, &[N_MELS, 906]);
        for &isa in &isas {
            let n = frames_of(audio.len());
            assert_eq!(n, 906);
            let mut log = Vec::new();
            Mel::new().log_frames(&pool, isa, &audio, 0, n, true, &mut log);
            let max = log.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let mut worst = 0f64;
            for t in 0..n {
                for b in 0..N_MELS {
                    let ours = normalise(log[t * N_MELS + b], max) as f64;
                    worst = worst.max((ours - reference[b * n + t] as f64).abs());
                }
            }
            assert!(worst <= bound, "{size} {isa:?}: the log-mel is {worst:.3e} from Transformers', over {bound:.3e}");
            // A negative control: one frame late is far outside the bound.
            let mut shifted = 0f64;
            for t in 1..n {
                for b in 0..N_MELS {
                    shifted = shifted.max((normalise(log[(t - 1) * N_MELS + b], max) as f64 - reference[b * n + t] as f64).abs());
                }
            }
            assert!(shifted > 100.0 * bound, "a one-frame shift is only {shifted:.3e} away");
        }
    }
}

#[test]
fn the_fixture_clip_has_the_official_processors_token_count() {
    let audio = read_wav(&fixtures().join("it_it-1703.wav"));
    for size in ["1.7b", "0.6b"] {
        let n = layers(size).meta["n_audio_tokens"].as_u64().unwrap() as usize;
        assert_eq!(audio_tokens(frames_of(audio.len())), n);
    }
}
