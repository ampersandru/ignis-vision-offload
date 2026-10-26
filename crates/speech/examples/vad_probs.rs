//! Writes `server_vad`'s per-frame speech probabilities of WAV files (spec speech/08's length-guard
//! measurement): one f32 per 512-sample frame (32 ms), little-endian, the last frame zero-padded, into
//! `<OUT>/<name>.vad` beside a line of the frame count on stdout.
//!
//! usage: vad_probs OUT_DIR FILE.wav...

use std::io::Write;
use std::path::{Path, PathBuf};

use ignis_speech::vad::{SpeechProbability, Vad, FRAME};

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
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(args.len() >= 2, "usage: vad_probs OUT_DIR FILE.wav...");
    let out = PathBuf::from(&args[0]);
    std::fs::create_dir_all(&out).unwrap();
    for wav in &args[1..] {
        let path = PathBuf::from(wav);
        let mut audio = read_wav(&path);
        audio.resize(audio.len().div_ceil(FRAME) * FRAME, 0.0);
        let mut vad = Vad::new();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let mut f = std::fs::File::create(out.join(format!("{name}.vad"))).unwrap();
        for frame in audio.chunks_exact(FRAME) {
            f.write_all(&vad.frame(frame).to_le_bytes()).unwrap();
        }
        println!("{name} {}", audio.len() / FRAME);
    }
}
