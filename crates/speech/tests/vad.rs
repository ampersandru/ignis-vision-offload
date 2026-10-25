//! `server_vad`'s VAD against Silero's published model (spec speech/05, acceptance 1): per-frame
//! speech probabilities within the recorded bound of `silero_vad.jit`'s, on the fixture clip here
//! and, in the `--ignored` leg, on the 40 FLEURS clips and the owner's dictations (the study disk,
//! `IGNIS_SPEECH_STUDY`). The records and the bound are `tools/speech-vad/record.py`'s. Acceptance
//! 7's `--ignored` leg, `server_vad`'s items against a manual commit on the dictations, runs the
//! default variant: alone on the machine.
//!
//! ```text
//! cargo test --release -p ignis-speech --test vad -- --ignored --test-threads=1
//! ```

mod common;

use std::path::{Path, PathBuf};

use common::*;
use ignis_speech::vad::{leading_silence, SpeechProbability, TurnEvent, TurnSettings, Turns, Vad, DEFAULT_LEADING_SILENCE_MS, FRAME};
use ignis_speech::{Executor, Features, Settings, Threads};

fn vad_fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("vad")
}

fn bound() -> f64 {
    json(&vad_fixtures().join("thresholds.json"))["bound"].as_f64().unwrap()
}

/// A record's clips: each id and its probabilities, from `<path>.json` and `<path>.bin`.
fn records(path: &Path) -> Vec<(String, Vec<f32>)> {
    let index = json(&path.with_extension("json"));
    let raw = std::fs::read(path.with_extension("bin")).unwrap();
    index["clips"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let (offset, frames) = (c["offset"].as_u64().unwrap() as usize, c["frames"].as_u64().unwrap() as usize);
            let p = raw[offset..offset + 4 * frames].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            (c["id"].as_str().unwrap().to_owned(), p)
        })
        .collect()
}

/// The clip's probabilities as the reference reads it: a fresh state, one per 512 samples, the
/// last frame zero-padded. `fresh_each` starts every frame from a fresh state instead.
fn probabilities(audio: &[f32], fresh_each: bool) -> Vec<f32> {
    let mut x = audio.to_vec();
    x.resize(audio.len().div_ceil(FRAME) * FRAME, 0.0);
    let mut vad = Vad::new();
    x.chunks_exact(FRAME)
        .map(|frame| {
            if fresh_each {
                vad = Vad::new();
            }
            vad.frame(frame)
        })
        .collect()
}

fn worst(ours: &[f32], reference: &[f32]) -> f64 {
    assert_eq!(ours.len(), reference.len());
    ours.iter().zip(reference).map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs()).fold(0.0, f64::max)
}

/// Acceptance 1, the fixture clip: every frame within the bound of the `.jit`'s probability. A
/// negative control: the same port with its state reset at every frame (no memory between
/// frames) is far outside it.
#[test]
fn the_port_matches_silero_on_the_fixture_clip() {
    let bound = bound();
    let audio = read_wav(&fixtures().join("it_it-1703.wav"));
    let [(id, reference)] = <[_; 1]>::try_from(records(&vad_fixtures().join("fixture"))).unwrap();
    assert_eq!(id, "it_it-1703");
    let err = worst(&probabilities(&audio, false), &reference);
    println!("it_it-1703: {err:.3e} (bound {bound:.3e})");
    assert!(err <= bound, "{err:.3e} from the .jit, over {bound:.3e}");
    let forgetful = worst(&probabilities(&audio, true), &reference);
    assert!(forgetful > 100.0 * bound, "a port that forgets between frames is only {forgetful:.3e} away");
}

/// Acceptance 1, every clip: the 40 FLEURS clips and the 11 dictations, every frame within the
/// bound.
#[test]
#[ignore]
fn the_port_matches_silero_on_fleurs_and_the_dictations() {
    let bound = bound();
    let s = study();
    let wavs: std::collections::HashMap<String, PathBuf> = {
        let text = std::fs::read_to_string(s.join("data").join("clips").join("manifest.csv")).unwrap();
        text.lines().skip(1).map(|l| l.split(',').collect::<Vec<_>>()).map(|p| (p[0].to_owned(), s.join(p[2].replace('\\', "/")))).collect()
    };
    let mut over = Vec::new();
    let mut checked = 0;
    for (set, path) in [("fleurs", vad_fixtures().join("fleurs")), ("dictations", s.join("results").join("vad").join("dictations"))] {
        let mut set_worst = 0f64;
        for (id, reference) in records(&path) {
            let wav = if set == "fleurs" { wavs[&id].clone() } else { s.join("dictations").join("16k").join(format!("{id}.wav")) };
            let err = worst(&probabilities(&read_wav(&wav), false), &reference);
            set_worst = set_worst.max(err);
            checked += 1;
            if err > bound {
                over.push(format!("{id}: {err:.3e}"));
            }
        }
        println!("{set}: worst {set_worst:.3e} (bound {bound:.3e})");
    }
    assert_eq!(checked, 51, "40 FLEURS clips and 11 dictations");
    assert!(over.is_empty(), "over the bound: {over:?}");
}

/// One item through the incremental path at real-time pace, as a Realtime session drives it:
/// 100 ms pushes, a partial whenever one is due, the commit.
struct Item {
    session: ignis_speech::Session,
}

impl Item {
    fn open(exec: &Executor) -> Item {
        Item { session: exec.open("", None).unwrap() }
    }

    fn push(&mut self, audio: &[f32]) {
        for chunk in audio.chunks(1600) {
            self.session.push(chunk).unwrap();
            if self.session.partial_due() {
                self.session.partial().unwrap();
            }
        }
    }

    fn commit(self) -> String {
        self.session.commit().unwrap().text
    }
}

/// The dictation as `server_vad` cuts it at its defaults (0.5, 300 ms, 500 ms), fed 100 ms at a
/// time and followed by a second of silence, as an open microphone goes on after the last
/// word, each item fed to the model after `lead`, as the server feeds it
/// (`speech.vad_leading_silence_ms`): each item's transcript, in order.
fn by_server_vad(exec: &Executor, audio: &[f32], lead: &[f32]) -> Vec<String> {
    let settings = TurnSettings { threshold: 0.5, prefix_padding_ms: 300, silence_duration_ms: 500, idle_timeout_ms: None };
    let mut turns = Turns::new(Vad::new(), settings, 0);
    let (mut item, mut texts, mut events) = (None, Vec::new(), Vec::new());
    let mut stream = audio.to_vec();
    stream.extend(std::iter::repeat_n(0.0, 16_000));
    for chunk in stream.chunks(1600) {
        turns.push(chunk, &mut events);
        for event in events.drain(..) {
            match event {
                TurnEvent::Started { .. } => {
                    let mut opened = Item::open(exec);
                    opened.push(lead);
                    item = Some(opened);
                }
                TurnEvent::Audio(samples) => item.as_mut().expect("audio inside a turn").push(&samples),
                TurnEvent::Stopped { .. } => texts.push(item.take().expect("a turn to stop").commit()),
                TurnEvent::TimedOut { .. } => unreachable!("no idle timeout"),
            }
        }
    }
    if let Some(mut last) = item {
        last.push(&turns.end_turn());
        texts.push(last.commit());
    }
    texts
}

/// The leading silences to measure, in milliseconds: `IGNIS_SPEECH_LEAD_MS` (`0,500,1000,1500`), and
/// always the default, `speech.vad_leading_silence_ms`'s, which the criterion is asserted on.
fn lead_arms() -> Vec<u32> {
    let asked = std::env::var("IGNIS_SPEECH_LEAD_MS").unwrap_or_default();
    let mut arms: Vec<u32> =
        asked.split(',').map(str::trim).filter(|arm| !arm.is_empty()).map(|arm| arm.parse().expect("IGNIS_SPEECH_LEAD_MS: milliseconds, comma-separated")).collect();
    if !arms.contains(&DEFAULT_LEADING_SILENCE_MS) {
        arms.push(DEFAULT_LEADING_SILENCE_MS);
    }
    arms
}

/// Acceptance 7: on each of the owner's dictations, the items `server_vad` commits, joined, have
/// a pooled WER at most 1 pt worse than a manual commit of the whole dictation through the same
/// incremental path (spec 03's), on the default variant (`qwen3-asr-1.7b`), each item fed after
/// the server's leading silence.
///
/// On the i9, 2026-10-25 (spec 05, As built): manual commit 4.03%; `server_vad` 7.58% with no
/// leading silence (the criterion failed), 4.68% with 500 ms (the default), 5.00% with 1000 ms,
/// 4.68% with 1500 ms.
#[test]
#[ignore]
fn server_vad_items_joined_are_within_a_point_of_a_manual_commit() {
    let v = &VARIANTS[0];
    assert_eq!(v.id, "qwen3-asr-1.7b");
    let threads = Threads::Auto.resolve().0;
    let exec = Executor::new(Settings { threads: Threads::Count(threads), ..Settings::default() }, v.files(), Features::detect());
    let d = study().join("dictations");
    let arms = lead_arms();
    let (mut manual, mut vad) = (Vec::new(), vec![Vec::new(); arms.len()]);
    for i in 1..=11 {
        let id = format!("d{i:02}");
        let reference = norm_dictation(&std::fs::read_to_string(d.join(format!("{id}.txt"))).unwrap());
        let audio = read_wav(&d.join("16k").join(format!("{id}.wav")));
        let mut whole = Item::open(&exec);
        whole.push(&audio);
        let m = norm_dictation(&whole.commit());
        let mut line = format!("{id}: of {} words, manual {} edits", reference.len(), edits(&reference, &m));
        for (&arm, pairs) in arms.iter().zip(&mut vad) {
            let items = by_server_vad(&exec, &audio, &leading_silence(arm));
            let s = norm_dictation(&items.join(" "));
            line += &format!("; lead {arm} ms: {} items, {} edits", items.len(), edits(&reference, &s));
            pairs.push((reference.clone(), s));
        }
        println!("{line}");
        manual.push((reference, m));
    }
    let m = pooled_wer(&manual);
    for (arm, pairs) in arms.iter().zip(&vad) {
        println!("{}: pooled WER manual commit {:.2}%, server_vad with {arm} ms of leading silence {:.2}%", v.id, 100.0 * m, 100.0 * pooled_wer(pairs));
    }
    let s = pooled_wer(&vad[arms.iter().position(|&arm| arm == DEFAULT_LEADING_SILENCE_MS).unwrap()]);
    assert!(s <= m + 0.01, "server_vad's items joined are {:.2} pt worse than a manual commit", 100.0 * (s - m));
}
