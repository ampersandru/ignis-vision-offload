//! Transcribes 16 kHz mono 16-bit WAV files with the CPU executor, through the batch path
//! (one push, one commit) or at real-time pace with partials, and prints each transcript with
//! its times: the tool behind spec speech/01's "As built" numbers.
//!
//! usage: transcribe DECODER.gguf ENCODER.gguf [--threads N] [--realtime] [--rollback N] [--unfixed N] [--reps N] FILE.wav...

use std::time::Instant;

use ignis_speech::{Executor, Features, ModelFiles, Settings, Threads};

fn read_wav(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert!(&b[0..4] == b"RIFF" && &b[8..12] == b"WAVE", "{path}: not a WAV file");
    let mut i = 12;
    let (mut fmt_ok, mut data) = (false, None);
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let len = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
        let body = &b[i + 8..(i + 8 + len).min(b.len())];
        if id == b"fmt " {
            let (fmt, ch, rate, bits) = (u16::from_le_bytes([body[0], body[1]]), u16::from_le_bytes([body[2], body[3]]),
                u32::from_le_bytes(body[4..8].try_into().unwrap()), u16::from_le_bytes([body[14], body[15]]));
            assert!(fmt == 1 && ch == 1 && rate == 16_000 && bits == 16, "{path}: need 16 kHz mono 16-bit PCM");
            fmt_ok = true;
        } else if id == b"data" {
            data = Some(body);
        }
        i += 8 + len + (len & 1);
    }
    assert!(fmt_ok, "{path}: no fmt chunk");
    data.expect("no data chunk").chunks_exact(2).map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut flag = |name: &str| args.iter().position(|a| a == name).map(|i| { args.remove(i); });
    let realtime = flag("--realtime").is_some();
    let take = |args: &mut Vec<String>, name: &str| args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) });
    let threads = take(&mut args, "--threads").map(|v| Threads::Count(v.parse().expect("--threads N"))).unwrap_or(Threads::Auto);
    let reps: usize = take(&mut args, "--reps").map(|v| v.parse().expect("--reps N")).unwrap_or(1);
    let d = Settings::default();
    let rollback_tokens = take(&mut args, "--rollback").map(|v| v.parse().expect("--rollback N")).unwrap_or(d.rollback_tokens);
    let unfixed_partials = take(&mut args, "--unfixed").map(|v| v.parse().expect("--unfixed N")).unwrap_or(d.unfixed_partials);
    assert!(args.len() >= 3, "usage: transcribe DECODER.gguf ENCODER.gguf [--threads N] [--realtime] [--rollback N] [--unfixed N] [--reps N] FILE.wav...");
    let files = ModelFiles { decoder: args[0].clone().into(), encoder: args[1].clone().into() };
    let settings = Settings { threads, idle_unload: std::time::Duration::ZERO, rollback_tokens, unfixed_partials, ..d };
    let exec = Executor::new(settings, files, Features::detect());
    let t0 = Instant::now();
    let report = exec.load()?.expect("a fresh load");
    println!("load {} ms (wall {} ms), repacked {} MB, mapped {} MB, threads {:?}",
        report.millis, t0.elapsed().as_millis(), report.repacked_bytes >> 20, report.mapped_bytes >> 20, exec.settings().threads.resolve());
    for _ in 0..reps {
        for wav in &args[2..] {
            let audio = read_wav(wav);
            let t = Instant::now();
            let mut s = exec.open("", None)?;
            let mut partials = 0;
            if realtime {
                for chunk in audio.chunks(1600) {
                    s.push(chunk)?;
                    if s.partial_due() {
                        s.partial()?;
                        partials += 1;
                    }
                }
            } else {
                s.push(&audio)?;
            }
            let tc = Instant::now();
            let r = s.commit()?;
            let secs = audio.len() as f64 / 16_000.0;
            let total = t.elapsed().as_secs_f64();
            println!("{wav}\t{secs:.2} s\t{total:.3} s\t{:.1}x\tcommit {:.3} s\t{} audio tokens\t{} out\t{partials} partials\t{:?}\t{}",
                secs / total, tc.elapsed().as_secs_f64(), r.audio_tokens, r.tokens.len(), r.language, r.text);
        }
    }
    Ok(())
}
