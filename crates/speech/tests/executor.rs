//! The executor and the session on a tiny synthetic GGUF pair, in the default suite (spec
//! speech/01 testing decisions): load at first use and unload (idle or asked), pinned weights,
//! the Q4_K repack as the only copy, the prefix cache, the incremental session against a
//! whole-clip run, partials and their stable text, and one unit of compute at a time.

mod common;

use std::time::{Duration, Instant};

use common::synthetic;
use ignis_speech::{Availability, Executor, Features, ModelFiles, SessionTuning, Settings, Threads};

fn settings() -> Settings {
    Settings { threads: Threads::Count(2), idle_unload: Duration::ZERO, max_clip_seconds: 60, ..Settings::default() }
}

fn exec(files: &ModelFiles, s: Settings) -> Executor {
    let e = Executor::new(s, files.clone(), Features::detect());
    if e.availability() != Availability::Available {
        panic!("this CPU lacks AVX2: the executor cannot serve here");
    }
    e
}

/// Acceptance 21 and 7: nothing is mapped before the first signal; the Q4_K weights are repacked
/// at load and their file bytes are not counted among the mapped ones; after an unload the
/// process holds nothing of the model (on Windows a mapped file cannot be deleted: deleting it
/// shows the mapping is gone).
#[test]
fn the_model_loads_at_first_use_and_unloads_whole() {
    let files = synthetic::pair("load");
    let e = exec(&files, settings());
    assert!(!e.is_loaded(), "loaded before the first signal");
    let r = e.load().unwrap().expect("the first load reports");
    assert!(e.is_loaded());
    assert_eq!(r.repacked_bytes, synthetic::q4k_bytes(), "the Q4_K weights are repacked");
    let file_bytes = std::fs::metadata(&files.decoder).unwrap().len() as usize + std::fs::metadata(&files.encoder).unwrap().len() as usize;
    assert!(r.mapped_bytes + r.repacked_bytes < file_bytes, "the repacked tensors are not also mapped");
    assert!(!r.pinned);
    assert!(e.load().unwrap().is_none(), "a second load is a no-op");
    e.unload();
    assert!(!e.is_loaded());
    synthetic::remove(&files).expect("the files are still mapped after the unload");
}

/// Acceptance 21: unloaded after `idle_unload` with no transcription, and a later session loads
/// it again.
#[test]
fn the_model_unloads_when_idle() {
    let files = synthetic::pair("idle");
    let e = exec(&files, Settings { idle_unload: Duration::from_millis(300), ..settings() });
    let mut s = e.open("", None).unwrap();
    s.push(&synthetic::audio(1.0)).unwrap();
    s.commit().unwrap();
    let t0 = Instant::now();
    while e.is_loaded() {
        assert!(t0.elapsed() < Duration::from_secs(10), "still loaded after 10 s idle");
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(t0.elapsed() >= Duration::from_millis(200), "unloaded before idle_unload ran out");
    // Every session ended: the files are unmapped.
    std::fs::rename(&files.decoder, files.decoder.with_extension("moved")).expect("the decoder is still mapped");
    std::fs::rename(files.decoder.with_extension("moved"), &files.decoder).unwrap();
    let mut s = e.open("", None).unwrap();
    assert!(e.is_loaded());
    s.push(&synthetic::audio(0.5)).unwrap();
    s.commit().unwrap();
}

/// Acceptance 21: pinned weights are a private copy; the files are not held while it serves.
#[test]
fn pinned_weights_hold_no_mapping_of_the_files() {
    let files = synthetic::pair("pinned");
    let mapped = exec(&files, settings());
    let mut s = mapped.open("", None).unwrap();
    s.push(&synthetic::audio(2.0)).unwrap();
    let expected = s.commit().unwrap();
    drop(mapped);
    let e = exec(&files, Settings { pinned_weights: true, ..settings() });
    let r = e.load().unwrap().unwrap();
    assert!(r.pinned);
    // The copy holds every tensor but the repacked Q4_K ones, whose only copy is the repack.
    assert_eq!(r.repacked_bytes, synthetic::q4k_bytes());
    let file_bytes = std::fs::metadata(&files.decoder).unwrap().len() as usize + std::fs::metadata(&files.encoder).unwrap().len() as usize;
    assert!(r.pinned_bytes > 0 && r.pinned_bytes + r.repacked_bytes <= file_bytes, "{} + {} over {file_bytes}", r.pinned_bytes, r.repacked_bytes);
    let moved = files.decoder.with_extension("moved");
    std::fs::rename(&files.decoder, &moved).expect("pinned weights still map the decoder");
    let mut s = e.open("", None).unwrap();
    s.push(&synthetic::audio(2.0)).unwrap();
    assert_eq!(s.commit().unwrap(), expected, "the pinned copy transcribes as the mapping does");
    std::fs::rename(&moved, &files.decoder).unwrap();
}

/// Acceptance 17's cache: the same context twice prefills it once; a new one replaces the entry.
#[test]
fn the_prefix_cache_holds_one_context() {
    let files = synthetic::pair("prefix");
    let e = exec(&files, settings());
    let prefilled = |c: &str| e.open(c, None).unwrap().stats().context_prefilled;
    let a = prefilled("first context, a few words");
    assert!(a > 0);
    assert_eq!(prefilled("first context, a few words"), 0);
    assert!(prefilled("another one") > 0);
    assert_eq!(prefilled("first context, a few words"), a, "the new context did not replace the entry");
    // Over the cap the context keeps its last tokens.
    let capped = Executor::new(Settings { context_max_tokens: 4, ..settings() }, files.clone(), Features::detect());
    assert_eq!(capped.open("a context longer than four tokens", None).unwrap().stats().context_prefilled, 4);
}

/// The incremental session against a whole-clip run: pushed 100 ms at a time or all at once, a
/// clip gives the same transcript, each window encoded once into the audio prefix. 40 s outgrow
/// the KV cache's first allocation, so it grows on the way.
#[test]
fn pushing_in_pieces_gives_the_whole_clip_transcript() {
    let files = synthetic::pair("pieces");
    let e = exec(&files, settings());
    let audio = synthetic::audio(40.0);
    let mut whole = e.open("", None).unwrap();
    whole.push(&audio).unwrap();
    let ws = whole.stats().clone();
    let w = whole.commit().unwrap();
    let mut s = e.open("", None).unwrap();
    for c in audio.chunks(1600) {
        s.push(c).unwrap();
    }
    let ps = s.stats().clone();
    let p = s.commit().unwrap();
    assert_eq!(p, w);
    // 40 s is 4,000 frames: four whole windows while pushing (the fifth needs the last 200
    // samples' frame), the fifth at commit.
    assert_eq!((ws.windows, ps.windows), (4, 4));
    assert_eq!((ws.audio_prefilled, ps.audio_prefilled), (416, 416));
    assert_eq!(w.audio_tokens, 520);
    assert_eq!(w.tokens.len(), 1, "the synthetic decoder emits one token");
}

/// Acceptance 18's mechanics: a partial every `partial_interval_ms` of audio, the cache rolled
/// back after each (the transcript is the whole-clip one), and with no rollback tokens the
/// decoded token joins the stable text, which the transcript begins with.
#[test]
fn partials_roll_back_and_the_transcript_keeps_the_stable_text() {
    let files = synthetic::pair("partials");
    let audio = synthetic::audio(9.3);
    let e = exec(&files, settings());
    let mut whole = e.open("", None).unwrap();
    whole.push(&audio).unwrap();
    let w = whole.commit().unwrap();
    for rollback in [5, 0] {
        let e = exec(&files, Settings { rollback_tokens: rollback, unfixed_partials: 0, ..settings() });
        let mut s = e.open("", None).unwrap();
        let mut partials = Vec::new();
        for c in audio.chunks(1600) {
            s.push(c).unwrap();
            if s.partial_due() {
                partials.push(s.partial().unwrap());
            }
        }
        assert_eq!(partials.len(), 4, "a partial every 2 s of 9.3 s");
        assert_eq!(s.stats().partials, 4);
        let t = s.commit().unwrap();
        for p in &partials {
            assert!(t.text.starts_with(&p.stable), "the transcript does not begin with {:?}", p.stable);
            if rollback > 0 {
                // One decoded token, all of it provisional: nothing is stable.
                assert_eq!(p.stable, "");
            }
        }
        if rollback > 0 {
            assert_eq!(t, w, "partials changed the transcript: the cache was not rolled back");
        }
    }
}

/// The server passes `speech.context_max_tokens`, `speech.rollback_tokens` and
/// `speech.unfixed_partials` per session: a session holds the tuning it opened with, not the
/// executor's settings, so a live change of the config holds from the next session. (The
/// synthetic decoder's one token decodes to no text, so the tuning is read back rather than seen
/// in the stable text; `partials_roll_back_and_the_transcript_keeps_the_stable_text` shows the
/// rollback itself.)
#[test]
fn a_session_holds_the_tuning_it_opened_with() {
    let files = synthetic::pair("tuning");
    let e = exec(&files, Settings { rollback_tokens: 10, unfixed_partials: 2, context_max_tokens: 512, ..settings() });
    let tuned = SessionTuning { context_max_tokens: 64, rollback_tokens: 3, unfixed_partials: 0 };
    assert_eq!(e.open_tuned("", None, tuned).unwrap().tuning(), tuned);
    assert_eq!(e.open("", None).unwrap().tuning(), SessionTuning { context_max_tokens: 512, rollback_tokens: 10, unfixed_partials: 2 });
    assert_eq!(e.open_capped("", None, 7).unwrap().tuning(), SessionTuning { context_max_tokens: 7, rollback_tokens: 10, unfixed_partials: 2 });
}

/// Acceptance 19: one unit of compute at a time, whatever the number of sessions driving it.
#[test]
fn sessions_on_two_threads_never_run_two_units_at_once() {
    let files = synthetic::pair("units");
    let e = std::sync::Arc::new(exec(&files, settings()));
    let audio = synthetic::audio(12.0);
    let mut one = e.open("", None).unwrap();
    one.push(&audio).unwrap();
    let alone = one.commit().unwrap();
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let (e, audio) = (e.clone(), audio.clone());
            std::thread::spawn(move || {
                let mut s = e.open("", None).unwrap();
                for c in audio.chunks(3200) {
                    s.push(c).unwrap();
                    if s.partial_due() {
                        s.partial().unwrap();
                    }
                }
                s.commit().unwrap()
            })
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap().audio_tokens, alone.audio_tokens);
    }
    assert_eq!(e.max_concurrent_units(), 1);
}

/// Acceptance 20: `auto` is the physical cores less one, an explicit count is used as given.
#[test]
fn threads_resolve_auto_and_counts() {
    let usable = num_cpus::get_physical().min(num_cpus::get());
    assert_eq!(Threads::Auto.resolve(), ((usable - 1).max(1), "auto"));
    assert_eq!(Threads::Count(3).resolve(), (3, "configured"));
    assert_eq!(Threads::Count(0).resolve(), (1, "configured"));
}
