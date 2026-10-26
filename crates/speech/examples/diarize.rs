//! Runs the diarizer on WAV files (spec speech/08's measurements): each file streamed in 100 ms
//! pieces at a geometry, every step timed, the probabilities and the speakers written out.
//!
//! usage: diarize MODEL.gguf --geometry live|file --out DIR [--threads N] [--ghost 0,3,6] WAV...
//!
//! Per file, in DIR: `<name>.<geometry>.probs` (every 10 ms row's 8 probabilities, f32
//! little-endian) and `<name>.<geometry>.speakers.json` (per ghost-rule length in seconds, the 80 ms
//! frames' speakers after spec 08's rules, as runs `[first, end, speaker]`, and the step times).
//! One summary line per file on stdout.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ignis_speech::cpu::{select, Features};
use ignis_speech::diarizer::speakers::Speakers;
use ignis_speech::diarizer::{Diarizer, Geometry, Probs};
use ignis_speech::pool::Pool;
use ignis_speech::Threads;

fn read_wav(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(&b[0..4] == b"RIFF" && &b[8..12] == b"WAVE", "{}: not a WAV file", path.display());
    let mut i = 12;
    while i + 8 <= b.len() {
        let len = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
        let body = &b[i + 8..(i + 8 + len).min(b.len())];
        if &b[i..i + 4] == b"fmt " {
            let ok = u16::from_le_bytes([body[0], body[1]]) == 1 && u16::from_le_bytes([body[2], body[3]]) == 1 && u32::from_le_bytes(body[4..8].try_into().unwrap()) == 16_000 && u16::from_le_bytes([body[14], body[15]]) == 16;
            assert!(ok, "{}: need 16 kHz mono 16-bit PCM", path.display());
        } else if &b[i..i + 4] == b"data" {
            return body.chunks_exact(2).map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0).collect();
        }
        i += 8 + len + (len & 1);
    }
    panic!("{}: no data chunk", path.display())
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let take = |args: &mut Vec<String>, name: &str| args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) });
    let threads = take(&mut args, "--threads").map(|v| v.parse().unwrap()).unwrap_or(Threads::Auto.resolve().0);
    let geometry_name = take(&mut args, "--geometry").unwrap_or_else(|| "live".into());
    let out = PathBuf::from(take(&mut args, "--out").expect("--out DIR"));
    let ghosts: Vec<f64> = take(&mut args, "--ghost").unwrap_or_else(|| "0".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let geometry = match geometry_name.as_str() {
        "live" => Geometry::LIVE,
        "file" => Geometry::FILE,
        other => panic!("--geometry live or file, not {other}"),
    };
    assert!(!args.is_empty(), "usage: diarize MODEL.gguf --geometry live|file --out DIR [--threads N] [--ghost 0,3,6] WAV...");
    let model = PathBuf::from(args.remove(0));
    std::fs::create_dir_all(&out).unwrap();
    let isa = select(Features::detect()).expect("AVX2");
    let pool = Pool::new(threads);
    let t = Instant::now();
    let d = Diarizer::load(&model, false).unwrap();
    eprintln!("loaded {} in {:.2} s, {} threads", model.display(), t.elapsed().as_secs_f64(), threads);
    let constants = d.weights().constants;
    for wav in &args {
        let path = PathBuf::from(wav);
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let audio = read_wav(&path);
        let mut s = d.open(geometry, constants);
        let (mut probs, mut frames): (Vec<Probs>, Vec<Probs>) = (Vec::new(), Vec::new());
        let mut times = Vec::new();
        let wall = Instant::now();
        for piece in audio.chunks(1600) {
            s.push(piece);
            while s.ready() {
                let t = Instant::now();
                let step = s.step(&d, &pool, isa).unwrap();
                times.push(t.elapsed().as_secs_f64());
                probs.extend(step.probs);
                frames.extend(step.frames);
            }
        }
        let t = Instant::now();
        if let Some(step) = s.finish(&d, &pool, isa) {
            times.push(t.elapsed().as_secs_f64());
            probs.extend(step.probs);
            frames.extend(step.frames);
        }
        let wall = wall.elapsed().as_secs_f64();
        let mut f = std::fs::File::create(out.join(format!("{name}.{geometry_name}.probs"))).unwrap();
        for p in &probs {
            for v in p {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        let mut speakers = serde_json::Map::new();
        for &g in &ghosts {
            let mut sp = Speakers::new(g);
            sp.push(&frames);
            let mut runs: Vec<[usize; 3]> = Vec::new();
            for i in 0..frames.len() {
                if let Some(c) = sp.speaker(i) {
                    match runs.last_mut() {
                        Some(r) if r[1] == i && r[2] == c as usize => r[1] = i + 1,
                        _ => runs.push([i, i + 1, c as usize]),
                    }
                }
            }
            speakers.insert(format!("{g}"), serde_json::json!(runs));
        }
        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        let seconds = audio.len() as f64 / 16_000.0;
        let summary = serde_json::json!({
            "file": name, "geometry": geometry_name, "seconds": seconds, "threads": threads,
            "steps": times.len(), "wall_s": wall, "compute_s": times.iter().sum::<f64>(),
            "step_ms_median": 1e3 * sorted[sorted.len() / 2], "step_ms_max": 1e3 * sorted[sorted.len() - 1],
            "x_realtime": seconds / times.iter().sum::<f64>(),
        });
        let doc = serde_json::json!({ "summary": summary, "speakers": speakers, "step_s": times });
        std::fs::write(out.join(format!("{name}.{geometry_name}.speakers.json")), serde_json::to_string(&doc).unwrap()).unwrap();
        println!("{summary}");
    }
}
