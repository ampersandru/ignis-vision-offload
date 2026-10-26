//! The diarizer against its oracle (spec speech/08, acceptance 1 and 17): the checkpoint's
//! Transformers code, recorded by `tools/speech-diarizer/record.py` on the fixture clip
//! (`fixtures/diarizer/`, a 16 s span of one of the twenty synthetic conversations, VoxPopuli
//! audio, CC0) and on three of the conversations (the study disk).
//!
//! The default build reads only what needs no weights: the log-mel. The `--ignored` legs read the
//! converted GGUF and the conversations from the study disk (`IGNIS_SPEECH_STUDY`):
//!
//! cargo test --release -p ignis-speech --test diarizer -- --ignored --test-threads=1

mod common;

use std::path::{Path, PathBuf};

use common::*;
use ignis_speech::cpu::{select, Features, Isa};
use ignis_speech::diarizer::mel::{frames_of, FrontEnd};
use ignis_speech::diarizer::{Constants, Diarizer, Geometry, Probs, SpeakerCache, Stream, SPEAKERS};
use ignis_speech::mel::N_MELS;
use ignis_speech::pool::Pool;

fn diarizer_fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("diarizer")
}

fn isas() -> Vec<Isa> {
    let mut isas = vec![Isa::scalar()];
    isas.extend(select(Features::detect()).ok());
    isas
}

/// Acceptance 1's front end, on the fixture clip: every recorded frame (its first 400 and its
/// last 400, as the streaming chunks carried them) within the bound of Transformers FP32.
#[test]
fn the_log_mel_matches_transformers_fp32() {
    let dir = diarizer_fixtures();
    let audio = read_wav(&dir.join("conv04-28-44.wav"));
    let bound = json(&dir.join("thresholds.json"))["mel"]["bound"].as_f64().unwrap();
    let fx = blob(&dir.join("fixture"));
    let frames = &fx.i32["mel.frames"].1;
    let (shape, rows) = &fx.f32["mel.rows"];
    assert_eq!(shape, &[frames.len(), N_MELS]);
    assert_eq!(frames_of(audio.len()) as u64, fx.meta["mel_frames"].as_u64().unwrap());
    let pool = Pool::new(4);
    let fe = FrontEnd::new();
    for isa in isas() {
        let mut mel = Vec::new();
        fe.log_frames(&pool, isa, &audio, 0, 0, frames_of(audio.len()), true, &mut mel);
        let mut worst = 0f64;
        for (i, &t) in frames.iter().enumerate() {
            let (a, b) = (&mel[t as usize * N_MELS..][..N_MELS], &rows[i * N_MELS..][..N_MELS]);
            for (x, y) in a.iter().zip(b) {
                worst = worst.max((*x as f64 - *y as f64).abs());
            }
        }
        println!("{isa:?}: log-mel {worst:.3e} from Transformers FP32 (bound {bound:.3e})");
        assert!(worst <= bound, "{isa:?}: {worst:.3e} over {bound:.3e}");
    }
}

// ---------------------------------------------------------------- the speaker cache, replayed

fn geometry_of(meta: &serde_json::Value, g: &str) -> Geometry {
    let v: Vec<usize> = meta["geometries"][g].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    Geometry { speaker_cache: v[0], fifo: v[1], chunk: v[2], right_context: v[3], update_period: v[4] }
}

/// The checkpoint's `streaming_config` constants.
const CHECKPOINT: Constants = Constants {
    prediction_score_threshold: 0.25,
    latest_frames_score_boost: 0.05,
    strong_boost_rate: 0.75,
    weak_boost_rate: 1.5,
    min_positive_scores_rate: 0.5,
    speaker_cache_silence_frames_per_speaker: 1,
};

fn probs_rows(v: &[f32]) -> Vec<Probs> {
    v.chunks_exact(SPEAKERS).map(|r| r.try_into().unwrap()).collect()
}

/// Acceptance 1's streaming state, with no weights: fed each recorded step's probabilities, the
/// cache and the FIFO hold, step after step, the frames the reference's hold, in its order, with
/// the probabilities it stored, compressing when it does. The rows are frame indices (the
/// silence slot −1), so what is compared is which frames are kept. The small geometry overflows
/// and compresses the cache many times on the fixture clip; the live one fills only its FIFO.
#[test]
fn the_speaker_cache_keeps_the_frames_the_reference_keeps() {
    let dir = diarizer_fixtures();
    let fx = blob(&dir.join("fixture"));
    let ties = &json(&dir.join("thresholds.json"))["ties"];
    for g in ["small", "live"] {
        assert_eq!(ties[format!("fixture.{g}")]["boundary_ties"], 0, "a tie at the boundary: the order of equal scores decides");
        let geometry = geometry_of(&fx.meta, g);
        let mut c = SpeakerCache::new(1, geometry, CHECKPOINT);
        let steps = fx.meta["steps"][g].as_array().unwrap();
        for (i, s) in steps.iter().enumerate() {
            let at = |k: &str| s[k].as_u64().unwrap() as usize;
            let (first, inputs, cached, chunk) = (at("first_frame"), at("inputs"), at("cached"), at("chunk"));
            assert_eq!(c.cache_len() + c.fifo_len(), cached, "{g} step {i}: the rows before the chunk");
            let mut rows: Vec<f32> = c.rows().map(|r| r[0]).collect();
            rows.extend((first..first + inputs - cached).map(|f| f as f32));
            let probs = probs_rows(&fx.f32[&format!("{g}.step.{i}.probs")].1);
            assert_eq!(probs.len(), inputs);
            c.update(&rows, &probs, &[-1.0], chunk);
            let as_rows = |v: &[i32]| v.iter().map(|&f| f as f32).collect::<Vec<_>>();
            assert_eq!(c.cache_rows(), as_rows(&fx.i32[&format!("{g}.step.{i}.cache")].1), "{g} step {i}: the cache");
            assert_eq!(c.fifo_rows(), as_rows(&fx.i32[&format!("{g}.step.{i}.fifo")].1), "{g} step {i}: the FIFO");
            assert_eq!(c.compressed(), s["compressed"].as_bool().unwrap(), "{g} step {i}");
            if c.compressed() {
                assert_eq!(c.cache_probs(), probs_rows(&fx.f32[&format!("{g}.step.{i}.cache_probs")].1), "{g} step {i}: the stored probabilities");
            }
        }
        println!("{g}: {} steps replayed, compressed: {}", steps.len(), c.compressed());
        assert_eq!(c.compressed(), g == "small");
    }
}

// ---------------------------------------------------------------- the forward pass (--ignored)

/// The converted checkpoint, F32: `tools/speech-diarizer/convert.py --type F32` on the study disk.
fn f32_gguf() -> PathBuf {
    let p = study().join("models").join("diarizer").join("Nemotron-3-Diarization-F32.gguf");
    assert!(p.is_file(), "{} missing", p.display());
    p
}

fn oracle_records() -> PathBuf {
    let p = study().join("nemotron-spike").join("oracle-361");
    assert!(p.is_dir(), "{} missing: tools/speech-diarizer/record.py record", p.display());
    p
}

/// A step's speaker-cache state, as the reference recorded it: the cache's frames, the FIFO's
/// (−1 a silence slot), and whether the cache was compressed.
type State = (Vec<i32>, Vec<i32>, bool);

fn recorded_states(b: &Blob, g: &str) -> Vec<State> {
    let steps = b.meta["steps"][g].as_array().unwrap();
    (0..steps.len())
        .map(|i| (b.i32[&format!("{g}.step.{i}.cache")].1.clone(), b.i32[&format!("{g}.step.{i}.fifo")].1.clone(), steps[i]["compressed"].as_bool().unwrap()))
        .collect()
}

/// How a stream's state goes from step to step.
#[derive(Clone, Copy)]
enum Run<'a> {
    /// Its own.
    Free,
    /// Emptied before every step: the negative control.
    Forget,
    /// The reference's, set after every step: acceptance 1's "with the same streaming state".
    Forced(&'a [State]),
}

/// A stream of `audio` fed in pieces of `piece` samples, every step taken as soon as it can run.
fn stream_probs(d: &Diarizer, pool: &Pool, isa: Isa, audio: &[f32], geometry: Geometry, piece: usize, run: Run) -> Vec<Probs> {
    let mut s: Stream = d.open(geometry, CHECKPOINT);
    s.keep_history();
    let mut out = Vec::new();
    let mut steps = 0;
    let after = |s: &mut Stream, steps: &mut usize| {
        if let Run::Forced(states) = run {
            let (c, f, z) = &states[*steps];
            s.force(d, c, f, *z);
        }
        *steps += 1;
    };
    for p in audio.chunks(piece) {
        s.push(p);
        while s.ready() {
            if let Run::Forget = run {
                s.forget();
            }
            out.extend(s.step(d, pool, isa).unwrap().probs);
            after(&mut s, &mut steps);
        }
    }
    if let Run::Forget = run {
        s.forget();
    }
    if let Some(last) = s.finish(d, pool, isa) {
        out.extend(last.probs);
        after(&mut s, &mut steps);
    }
    if let Run::Forced(states) = run {
        assert_eq!(steps, states.len(), "the reference's steps");
    }
    out
}

fn worst_probs(a: &[Probs], b: &[f32]) -> f64 {
    assert_eq!(a.len() * SPEAKERS, b.len(), "frames");
    a.iter().flatten().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).fold(0.0, f64::max)
}

fn threads() -> Pool {
    Pool::new(std::thread::available_parallelism().map_or(4, |n| n.get()).min(8))
}

fn best_isa() -> Isa {
    select(Features::detect()).unwrap_or(Isa::scalar())
}

/// The forward pass of the live geometry's first step on the fixture clip, tensor by tensor
/// (the oracle's taps): each row's relative L2 error against the reference.
#[test]
#[ignore = "reads the converted weights and the oracle's taps from the study disk"]
fn the_first_step_matches_the_reference_tensor_by_tensor() {
    let d = Diarizer::load(&f32_gguf(), false).unwrap();
    let taps = blob(&oracle_records().join("fixture-taps"));
    let rows: Vec<usize> = taps.meta["tap_rows"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let audio = read_wav(&diarizer_fixtures().join("conv04-28-44.wav"));
    let pool = threads();
    let mut s = d.open(Geometry::LIVE, CHECKPOINT);
    s.push(&audio[..s.samples_needed()]);
    let mut got = std::collections::HashMap::new();
    let mut tap = |name: &str, t: &[f32]| drop(got.insert(name.to_owned(), t.to_vec()));
    s.step_tapped(&d, &pool, best_isa(), &mut tap).unwrap();
    let mut worst = 0f64;
    let mut names: Vec<&String> = taps.f32.keys().collect();
    names.sort();
    for name in names {
        let (shape, reference) = &taps.f32[name];
        let key = name.strip_prefix("taps.").unwrap();
        let width = *shape.last().unwrap();
        let mine = got.get(key).unwrap_or_else(|| panic!("no tap {key}"));
        let picked: Vec<f32> = if mine.len() != reference.len() { rows.iter().flat_map(|&r| mine[r * width..(r + 1) * width].to_vec()).collect() } else { mine.clone() };
        assert_eq!(picked.len(), reference.len(), "{key}: {shape:?}");
        let e = row_rel_l2(&picked, reference, width);
        println!("{key:>10}: {e:.3e}");
        worst = worst.max(e);
    }
    assert!(worst < 1e-4, "a tensor {worst:.3e} away from the reference");
}

/// Acceptance 1 on the fixture clip: the live and the small geometry streamed chunk by chunk, every
/// 10 ms row's 8 probabilities within the bound of the reference, with the reference's state set
/// after every step, and with the port's own (the fixture has no tie for the order of equal scores
/// to decide); a stream that forgets its state between steps is far outside it.
#[test]
#[ignore = "reads the converted weights from the study disk"]
fn the_port_matches_the_reference_on_the_fixture_clip() {
    let d = Diarizer::load(&f32_gguf(), false).unwrap();
    let dir = diarizer_fixtures();
    let fx = blob(&dir.join("fixture"));
    let bounds = &json(&dir.join("thresholds.json"))["bound"];
    let audio = read_wav(&dir.join("conv04-28-44.wav"));
    let pool = threads();
    for g in ["live", "small"] {
        let bound = bounds[g].as_f64().unwrap();
        let reference = &fx.f32[&format!("{g}.probs")].1;
        let geometry = geometry_of(&fx.meta, g);
        let states = recorded_states(&fx, g);
        // Fed in 20 ms pieces, and all at once: the steps are the same.
        for piece in [320, audio.len()] {
            for (run, name) in [(Run::Forced(&states), "forced"), (Run::Free, "free")] {
                let e = worst_probs(&stream_probs(&d, &pool, best_isa(), &audio, geometry, piece, run), reference);
                println!("{g}, {name}, pieces of {piece}: {e:.3e} (bound {bound:.3e})");
                assert!(e <= bound, "{g} {name}: {e:.3e} from the reference, over {bound:.3e}");
            }
        }
        let forgetful = worst_probs(&stream_probs(&d, &pool, best_isa(), &audio, geometry, 320, Run::Forget), reference);
        println!("{g}, forgetting: {forgetful:.3e}");
        assert!(forgetful > 100.0 * bound, "{g}: a stream that forgets its state is only {forgetful:.3e} away");
    }
}

/// The conversations the oracle recorded (the study disk), with their name.
fn conversation_records() -> Vec<(String, Blob)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(oracle_records()).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        if name.starts_with("conv") && path.extension().is_some_and(|e| e == "json") {
            out.push((name, blob(&path.with_extension(""))));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(out.len() >= 3, "{} conversations recorded", out.len());
    out
}

/// Acceptance 1 on the conversations the oracle recorded, at the live and the file geometry:
/// minutes of speech, the cache compressed again and again. Every step is fed the reference's
/// state; the port's own run is reported beside it.
#[test]
#[ignore = "reads the converted weights and the conversations from the study disk"]
fn the_port_matches_the_reference_on_the_conversations() {
    let d = Diarizer::load(&f32_gguf(), false).unwrap();
    let bounds = json(&diarizer_fixtures().join("thresholds.json"))["bound"].clone();
    let pool = threads();
    let convs = study().join("nemotron-spike").join("data").join("conv");
    for (name, rec) in conversation_records() {
        let audio = read_wav(&convs.join(format!("{name}.wav")));
        for g in ["live", "file"] {
            let bound = bounds[g].as_f64().unwrap();
            let reference = &rec.f32[&format!("{g}.probs")].1;
            let geometry = geometry_of(&rec.meta, g);
            let states = recorded_states(&rec, g);
            let e = worst_probs(&stream_probs(&d, &pool, best_isa(), &audio, geometry, 1600, Run::Forced(&states)), reference);
            let free = worst_probs(&stream_probs(&d, &pool, best_isa(), &audio, geometry, 1600, Run::Free), reference);
            println!("{name} {g}: {e:.3e} with the reference's state, {free:.3e} with its own (bound {bound:.3e})");
            assert!(e <= bound, "{name} {g}: {e:.3e} over {bound:.3e}");
        }
    }
}

/// Acceptance 1's streaming state on the conversations: every step, set to the reference's state
/// before it and fed its recorded probabilities, keeps the frames the reference kept. Only a step
/// where torch's top-k met a tie at its boundary may keep others: their order is the one thing the
/// port defines (equal scores by the lower index) and torch leaves undefined.
#[test]
#[ignore = "reads the conversations' records from the study disk"]
fn the_speaker_cache_keeps_the_frames_the_reference_keeps_on_the_conversations() {
    let ties = json(&diarizer_fixtures().join("thresholds.json"))["ties"].clone();
    for (name, rec) in conversation_records() {
        for g in ["live", "file"] {
            let geometry = geometry_of(&rec.meta, g);
            let states = recorded_states(&rec, g);
            let steps = rec.meta["steps"][g].as_array().unwrap();
            let mut c = SpeakerCache::new(1, geometry, CHECKPOINT);
            let as_rows = |v: &[i32]| v.iter().map(|&f| f as f32).collect::<Vec<_>>();
            let mut differ = Vec::new();
            for (i, s) in steps.iter().enumerate() {
                if i > 0 {
                    let (cache, fifo, z) = &states[i - 1];
                    c.set(as_rows(cache), probs_rows(&rec.f32[&format!("{g}.step.{}.cache_probs", i - 1)].1), as_rows(fifo), *z);
                }
                let at = |k: &str| s[k].as_u64().unwrap() as usize;
                let (first, inputs, cached) = (at("first_frame"), at("inputs"), at("cached"));
                let mut rows: Vec<f32> = c.rows().map(|r| r[0]).collect();
                assert_eq!(rows.len(), cached, "{name} {g} step {i}");
                rows.extend((first..first + inputs - cached).map(|f| f as f32));
                let probs = probs_rows(&rec.f32[&format!("{g}.step.{i}.probs")].1);
                c.update(&rows, &probs, &[-1.0], at("chunk"));
                let (cache, fifo, z) = &states[i];
                if c.cache_rows() != as_rows(cache).as_slice() || c.fifo_rows() != as_rows(fifo).as_slice() || c.compressed() != *z {
                    differ.push(i);
                }
            }
            let tied = ties[format!("{name}.{g}")]["boundary_ties"].as_u64().unwrap() as usize;
            println!("{name} {g}: {} steps, {} differ ({differ:?}), {tied} boundary ties", steps.len(), differ.len());
            assert!(differ.len() <= tied, "{name} {g}: steps {differ:?} keep other frames, with {tied} ties");
        }
    }
}
