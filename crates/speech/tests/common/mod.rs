//! Helpers the speech tests share: the fixtures, the study disk, WAV reading and WER.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

pub mod synthetic;

pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("oracle")
}

/// The study disk (`IGNIS_SPEECH_STUDY`, default `H:/speech-study`): the GGUF files, the FLEURS
/// clips and the dictations. The legs that need it are `--ignored`, and they fail without it:
/// a compute check that skips proves nothing (docs/agents/testing.md).
pub fn study() -> PathBuf {
    let p = PathBuf::from(std::env::var("IGNIS_SPEECH_STUDY").unwrap_or_else(|_| "H:/speech-study".into()));
    assert!(p.join("models").is_dir(), "the study disk is not at {} (set IGNIS_SPEECH_STUDY)", p.display());
    p
}

/// A catalog variant: its id, size, and the GGUF pair converted from the official weights.
pub struct Variant {
    pub id: &'static str,
    pub size: &'static str,
    pub decoder: &'static str,
    pub encoder: &'static str,
}

pub const VARIANTS: [Variant; 3] = [
    Variant { id: "qwen3-asr-1.7b", size: "1.7b", decoder: "Qwen3-ASR-1.7B-Q4_K_M.gguf", encoder: "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf" },
    Variant { id: "qwen3-asr-1.7b-q8", size: "1.7b", decoder: "Qwen3-ASR-1.7B-Q8_0.gguf", encoder: "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf" },
    Variant { id: "qwen3-asr-0.6b", size: "0.6b", decoder: "Qwen3-ASR-0.6B-Q8_0.gguf", encoder: "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf" },
];

impl Variant {
    pub fn files(&self) -> ignis_speech::ModelFiles {
        let d = study().join("models").join("ignis-conv");
        let f = ignis_speech::ModelFiles { decoder: d.join(self.decoder), encoder: d.join(self.encoder) };
        assert!(f.decoder.is_file() && f.encoder.is_file(), "{}: {} or {} missing", self.id, f.decoder.display(), f.encoder.display());
        f
    }
}

pub fn read_wav(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(&b[0..4] == b"RIFF" && &b[8..12] == b"WAVE", "{}: not a WAV file", path.display());
    let mut i = 12;
    let mut data = None;
    while i + 8 <= b.len() {
        let len = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
        let body = &b[i + 8..(i + 8 + len).min(b.len())];
        if &b[i..i + 4] == b"fmt " {
            let ok = u16::from_le_bytes([body[0], body[1]]) == 1
                && u16::from_le_bytes([body[2], body[3]]) == 1
                && u32::from_le_bytes(body[4..8].try_into().unwrap()) == 16_000
                && u16::from_le_bytes([body[14], body[15]]) == 16;
            assert!(ok, "{}: need 16 kHz mono 16-bit PCM", path.display());
        } else if &b[i..i + 4] == b"data" {
            data = Some(body);
        }
        i += 8 + len + (len & 1);
    }
    data.expect("no data chunk").chunks_exact(2).map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0).collect()
}

pub fn json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap()
}

/// `<size>/layers.{json,bin}`: the tapped tensors, as rows.
pub struct Layers {
    pub meta: Value,
    pub tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

pub fn layers(size: &str) -> Layers {
    let base = fixtures().join(size).join("layers");
    let meta = json(&base.with_extension("json"));
    let raw = std::fs::read(base.with_extension("bin")).unwrap();
    let mut tensors = HashMap::new();
    for (k, e) in meta["tensors"].as_object().unwrap() {
        let shape: Vec<usize> = e["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let off = e["offset"].as_u64().unwrap() as usize;
        let n: usize = shape.iter().product();
        let v = raw[off..off + 4 * n].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        tensors.insert(k.clone(), (shape, v));
    }
    Layers { meta, tensors }
}

pub fn ids(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

/// Each row's relative L2 error against the reference, then the RMS over rows (the oracle's metric).
pub fn row_rel_l2(y: &[f32], r: &[f32], width: usize) -> f64 {
    let rows = r.len() / width;
    let mut s = 0f64;
    for i in 0..rows {
        let (a, b) = (&y[i * width..(i + 1) * width], &r[i * width..(i + 1) * width]);
        let num: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum();
        let den: f64 = b.iter().map(|&y| (y as f64).powi(2)).sum();
        s += num / den.max(1e-60);
    }
    (s / rows as f64).sqrt()
}

// ---------------------------------------------------------------- WER

/// The finding's FLEURS normaliser (`analyze.py`): NFKC, lower case, punctuation and symbols
/// to spaces, apostrophes kept only inside words.
pub fn norm_fleurs(s: &str) -> Vec<String> {
    use unicode_normalization_lite::nfkc;
    let s = nfkc(s).to_lowercase().replace('\u{2019}', "'").replace('`', "'");
    let spaced: String = s.chars().map(|c| if c == '\'' || c.is_alphanumeric() || is_mark(c) { c } else { ' ' }).collect();
    spaced.split_whitespace().map(|w| w.trim_matches('\'').to_string()).filter(|w| !w.is_empty()).collect()
}

/// The finding's dictation normaliser (`dict_score.py`): NFC, lower case, typographic
/// apostrophes to ', `_./-:` and backtick to spaces, every other character that is not a
/// letter, digit, mark or apostrophe to a space, apostrophes kept only inside words.
pub fn norm_dictation(s: &str) -> Vec<String> {
    use unicode_normalization_lite::nfc;
    let s = nfc(s).to_lowercase().replace(['\u{2019}', '\u{2018}', '\u{2bc}'], "'");
    let spaced: String = s.chars().map(|c| if "_./-:`".contains(c) { ' ' } else if c == '\'' || c.is_alphanumeric() || is_mark(c) { c } else { ' ' }).collect();
    spaced.split_whitespace().map(|w| w.trim_matches('\'').to_string()).filter(|w| !w.is_empty()).collect()
}

fn is_mark(c: char) -> bool {
    // Combining marks (Unicode category M) in the ranges the two languages use.
    matches!(c as u32, 0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F)
}

/// Word edit distance.
pub fn edits(r: &[String], h: &[String]) -> usize {
    let mut d: Vec<usize> = (0..=h.len()).collect();
    for i in 1..=r.len() {
        let mut prev = d[0];
        d[0] = i;
        for j in 1..=h.len() {
            let cur = (d[j] + 1).min(d[j - 1] + 1).min(prev + usize::from(r[i - 1] != h[j - 1]));
            prev = d[j];
            d[j] = cur;
        }
    }
    d[h.len()]
}

/// Pooled WER: total edits over total reference words.
pub fn pooled_wer(pairs: &[(Vec<String>, Vec<String>)]) -> f64 {
    let (e, n) = pairs.iter().fold((0, 0), |(e, n), (r, h)| (e + edits(r, h), n + r.len()));
    e as f64 / n as f64
}

/// The few Unicode normal forms the WER normalisers need, without a dependency: the test
/// texts are Italian and English, whose precomposed letters NFC keeps and NFKC also keeps.
mod unicode_normalization_lite {
    pub fn nfc(s: &str) -> String {
        compose(s)
    }
    pub fn nfkc(s: &str) -> String {
        // Compatibility forms the texts can carry: ligatures and full-width punctuation map to
        // their plain letters; nothing else in these languages differs from NFC.
        compose(&s.replace('\u{FB01}', "fi").replace('\u{FB02}', "fl").replace('\u{00A0}', " "))
    }
    fn compose(s: &str) -> String {
        // Base letter + combining grave/acute/diaeresis/circumflex/tilde → the precomposed letter.
        let mut out = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if let Some(&m) = chars.peek() {
                if let Some(p) = precompose(c, m) {
                    out.push(p);
                    chars.next();
                    continue;
                }
            }
            out.push(c);
        }
        out
    }
    fn precompose(c: char, m: char) -> Option<char> {
        let table: &[(char, char, char)] = &[
            ('a', '\u{300}', 'à'), ('e', '\u{300}', 'è'), ('i', '\u{300}', 'ì'), ('o', '\u{300}', 'ò'), ('u', '\u{300}', 'ù'),
            ('a', '\u{301}', 'á'), ('e', '\u{301}', 'é'), ('i', '\u{301}', 'í'), ('o', '\u{301}', 'ó'), ('u', '\u{301}', 'ú'),
            ('A', '\u{300}', 'À'), ('E', '\u{300}', 'È'), ('I', '\u{300}', 'Ì'), ('O', '\u{300}', 'Ò'), ('U', '\u{300}', 'Ù'),
            ('E', '\u{301}', 'É'), ('a', '\u{308}', 'ä'), ('o', '\u{308}', 'ö'), ('u', '\u{308}', 'ü'), ('n', '\u{303}', 'ñ'),
        ];
        table.iter().find(|t| t.0 == c && t.1 == m).map(|t| t.2)
    }
}

/// The rows of a CSV file with a header, as maps: RFC 4180 quoting (the study's results files).
pub fn csv_rows(path: &Path) -> Vec<HashMap<String, String>> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut records: Vec<Vec<String>> = Vec::new();
    let (mut field, mut record, mut quoted) = (String::new(), Vec::new(), false);
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, quoted) {
            ('"', true) if chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            ('"', _) => quoted = !quoted,
            (',', false) => record.push(std::mem::take(&mut field)),
            ('\n', false) => {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
            }
            ('\r', false) => {}
            _ => field.push(c),
        }
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    let header = records.remove(0);
    records.into_iter().map(|r| header.iter().cloned().zip(r).collect()).collect()
}
