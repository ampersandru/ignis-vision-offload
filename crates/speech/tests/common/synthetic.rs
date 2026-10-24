//! A tiny Qwen3-ASR-shaped GGUF pair, written in the test's temp directory: the real metadata
//! keys, tensor names and formats (Q4_K, Q6_K, Q8_0, F16, F32) at toy sizes, so the default
//! suite drives the executor, the session and the loader without the real model.
//!
//! Its decoder emits one token then EOS: every token embedding is +e₀ except `Ċ` (the newline
//! that ends the prompt), which is −e₀, and the head's EOS row is 10·e₀. After the prompt the
//! EOS logit is negative and the token the audio steers through the layers wins; after any
//! token EOS wins.

use std::path::{Path, PathBuf};

use ignis_speech::ModelFiles;

pub const HIDDEN: usize = 256;
const ENC_D: usize = 64;
const CONV: usize = 8;
const SPECIALS: [&str; 7] = ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<asr_text>"];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        ((self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
    }
}

fn f16(v: f32) -> [u8; 2] {
    ignis_speech::quant::f32_to_f16(v).to_le_bytes()
}

fn s(v: &str) -> Vec<u8> {
    let mut b = (v.len() as u64).to_le_bytes().to_vec();
    b.extend_from_slice(v.as_bytes());
    b
}

enum Kv {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    Strs(Vec<String>),
    I32s(Vec<i32>),
}

fn write(path: &Path, kv: &[(&str, Kv)], tensors: &[(String, Vec<u64>, u32, Vec<u8>)]) {
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    b.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    for (k, v) in kv {
        b.extend(s(k));
        match v {
            Kv::U32(x) => b.extend(4u32.to_le_bytes().into_iter().chain(x.to_le_bytes())),
            Kv::F32(x) => b.extend(6u32.to_le_bytes().into_iter().chain(x.to_le_bytes())),
            Kv::Bool(x) => b.extend(7u32.to_le_bytes().into_iter().chain([*x as u8])),
            Kv::Str(x) => b.extend(8u32.to_le_bytes().into_iter().chain(s(x))),
            Kv::Strs(xs) => {
                b.extend(9u32.to_le_bytes().into_iter().chain(8u32.to_le_bytes()).chain((xs.len() as u64).to_le_bytes()));
                for x in xs {
                    b.extend(s(x));
                }
            }
            Kv::I32s(xs) => {
                b.extend(9u32.to_le_bytes().into_iter().chain(5u32.to_le_bytes()).chain((xs.len() as u64).to_le_bytes()));
                for x in xs {
                    b.extend(x.to_le_bytes());
                }
            }
        }
    }
    let mut off = 0u64;
    let mut offs = Vec::new();
    for (name, dims, ty, data) in tensors {
        b.extend(s(name));
        b.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend(d.to_le_bytes());
        }
        b.extend(ty.to_le_bytes());
        b.extend(off.to_le_bytes());
        offs.push(off);
        off = (off + data.len() as u64).div_ceil(32) * 32;
    }
    b.resize(b.len().div_ceil(32) * 32, 0);
    let base = b.len();
    for ((_, _, _, data), o) in tensors.iter().zip(offs) {
        b.resize(base + o as usize, 0);
        b.extend_from_slice(data);
    }
    std::fs::write(path, b).unwrap();
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `n` rows of `k` weights in ggml type `ty` (0 F32, 1 F16, 8 Q8_0, 12 Q4_K, 14 Q6_K), scale `sd`.
fn weight(r: &mut Rng, ty: u32, k: usize, n: usize, sd: f32) -> Vec<u8> {
    let mut out = Vec::new();
    match ty {
        0 => (0..k * n).for_each(|_| out.extend((r.f() * sd).to_le_bytes())),
        1 => (0..k * n).for_each(|_| out.extend(f16(r.f() * sd))),
        8 => (0..k * n / 32).for_each(|_| {
            out.extend(f16(sd / 64.0));
            out.extend((0..32).map(|_| ((r.next() % 255) as i32 - 127) as i8 as u8));
        }),
        12 => (0..k * n / 256).for_each(|_| {
            out.extend(f16(sd / 128.0));
            out.extend(f16(sd / 256.0));
            out.extend((0..140).map(|_| r.next() as u8));
        }),
        14 => (0..k * n / 256).for_each(|_| {
            out.extend((0..192).map(|_| r.next() as u8));
            out.extend((0..16).map(|_| (r.next() % 64) as u8));
            out.extend(f16(sd / 512.0));
        }),
        _ => unreachable!(),
    }
    out
}

/// GPT-2's byte-to-character table: the tokenizer's byte-level alphabet.
fn byte_chars() -> Vec<String> {
    let mut bs: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut pairs: Vec<(u32, u32)> = bs.into_iter().zip(cs).collect();
    pairs.sort();
    pairs.into_iter().map(|(_, c)| char::from_u32(c).unwrap().to_string()).collect()
}

/// Writes the pair into a fresh directory named after `tag` and returns its files.
pub fn pair(tag: &str) -> ModelFiles {
    let dir = std::env::temp_dir().join(format!("ignis-speech-synth-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut r = Rng(0x2545_F491_4F6C_DD1D);
    let mut tokens: Vec<String> = SPECIALS.iter().map(|t| t.to_string()).collect();
    let mut types = vec![3i32; SPECIALS.len()];
    types[6] = 4;
    for c in byte_chars() {
        tokens.push(c);
        types.push(1);
    }
    let v = tokens.len();
    let nl = tokens.iter().position(|t| t == "\u{10a}").unwrap();
    let mut embd = vec![0f32; v * HIDDEN];
    for t in 0..v {
        embd[t * HIDDEN] = if t == nl { -1.0 } else { 1.0 };
    }
    let mut head = vec![0f32; v * HIDDEN];
    head[0] = 10.0;
    for t in 1..v {
        for d in 1..HIDDEN {
            head[t * HIDDEN + d] = r.f();
        }
    }
    let ones = |n: usize| f32s(&vec![1.0; n]);
    let h = HIDDEN as u64;
    let mut dec_t: Vec<(String, Vec<u64>, u32, Vec<u8>)> = vec![
        ("token_embd.weight".into(), vec![h, v as u64], 0, f32s(&embd)),
        ("output_norm.weight".into(), vec![h], 0, ones(HIDDEN)),
        ("output.weight".into(), vec![h, v as u64], 0, f32s(&head)),
    ];
    let p = "blk.0.";
    dec_t.extend([
        (format!("{p}attn_norm.weight"), vec![h], 0, ones(HIDDEN)),
        (format!("{p}attn_q.weight"), vec![h, 64], 0, weight(&mut r, 0, HIDDEN, 64, 0.2)),
        (format!("{p}attn_k.weight"), vec![h, 32], 0, weight(&mut r, 0, HIDDEN, 32, 0.2)),
        (format!("{p}attn_v.weight"), vec![h, 32], 0, weight(&mut r, 0, HIDDEN, 32, 0.5)),
        // The layer's updates stay small beside the embedding, so the head sees ±e₀ plus a
        // small audio-dependent part.
        (format!("{p}attn_output.weight"), vec![64, h], 0, weight(&mut r, 0, 64, HIDDEN, 0.01)),
        (format!("{p}attn_q_norm.weight"), vec![32], 0, ones(32)),
        (format!("{p}attn_k_norm.weight"), vec![32], 0, ones(32)),
        (format!("{p}ffn_norm.weight"), vec![h], 0, ones(HIDDEN)),
        (format!("{p}ffn_gate.weight"), vec![h, h], 12, weight(&mut r, 12, HIDDEN, HIDDEN, 0.008)),
        (format!("{p}ffn_up.weight"), vec![h, h], 12, weight(&mut r, 12, HIDDEN, HIDDEN, 0.008)),
        (format!("{p}ffn_down.weight"), vec![h, h], 14, weight(&mut r, 14, HIDDEN, HIDDEN, 0.0005)),
    ]);
    let dec_path = dir.join("decoder.gguf");
    write(
        &dec_path,
        &[
            ("general.architecture", Kv::Str("qwen3vl".into())),
            ("qwen3vl.block_count", Kv::U32(1)),
            ("qwen3vl.embedding_length", Kv::U32(HIDDEN as u32)),
            ("qwen3vl.feed_forward_length", Kv::U32(HIDDEN as u32)),
            ("qwen3vl.attention.head_count", Kv::U32(2)),
            ("qwen3vl.attention.head_count_kv", Kv::U32(1)),
            ("qwen3vl.attention.key_length", Kv::U32(32)),
            ("qwen3vl.attention.layer_norm_rms_epsilon", Kv::F32(1e-6)),
            ("qwen3vl.rope.freq_base", Kv::F32(1e6)),
            ("qwen3vl.rope.dimension_sections", Kv::I32s(vec![6, 5, 5, 0])),
            ("qwen3vl.n_deepstack_layers", Kv::U32(0)),
            ("tokenizer.ggml.model", Kv::Str("gpt2".into())),
            ("tokenizer.ggml.pre", Kv::Str("qwen2".into())),
            ("tokenizer.ggml.tokens", Kv::Strs(tokens)),
            ("tokenizer.ggml.token_type", Kv::I32s(types)),
            ("tokenizer.ggml.merges", Kv::Strs(Vec::new())),
        ],
        &dec_t,
    );

    let (d, c) = (ENC_D as u64, CONV as u64);
    let mut enc_t: Vec<(String, Vec<u64>, u32, Vec<u8>)> = vec![
        ("a.conv2d.1.weight".into(), vec![3, 3, 1, c], 1, weight(&mut r, 1, 9, CONV, 0.3)),
        ("a.conv2d.1.bias".into(), vec![1, 1, c], 0, weight(&mut r, 0, 1, CONV, 0.1)),
        ("a.conv2d.2.weight".into(), vec![3, 3, c, c], 1, weight(&mut r, 1, 9 * CONV, CONV, 0.3)),
        ("a.conv2d.2.bias".into(), vec![1, 1, c], 0, weight(&mut r, 0, 1, CONV, 0.1)),
        ("a.conv2d.3.weight".into(), vec![3, 3, c, c], 1, weight(&mut r, 1, 9 * CONV, CONV, 0.3)),
        ("a.conv2d.3.bias".into(), vec![1, 1, c], 0, weight(&mut r, 0, 1, CONV, 0.1)),
        ("a.conv_out.weight".into(), vec![16 * c, d], 1, weight(&mut r, 1, 16 * CONV, ENC_D, 0.3)),
        ("a.position_embd.weight".into(), vec![d, 13], 0, weight(&mut r, 0, ENC_D, 13, 0.5)),
        ("a.post_ln.weight".into(), vec![d], 0, ones(ENC_D)),
        ("a.post_ln.bias".into(), vec![d], 0, f32s(&vec![0.0; ENC_D])),
        ("mm.a.mlp.1.weight".into(), vec![d, d], 8, weight(&mut r, 8, ENC_D, ENC_D, 1.0)),
        ("mm.a.mlp.1.bias".into(), vec![d], 0, f32s(&vec![0.0; ENC_D])),
        ("mm.a.mlp.2.weight".into(), vec![d, h], 0, weight(&mut r, 0, ENC_D, HIDDEN, 0.5)),
        ("mm.a.mlp.2.bias".into(), vec![h], 0, f32s(&vec![0.0; HIDDEN])),
    ];
    let p = "a.blk.0.";
    for (g, ty) in [("attn_q", 8), ("attn_k", 8), ("attn_v", 8), ("attn_out", 8), ("ffn_up", 8), ("ffn_down", 8)] {
        enc_t.push((format!("{p}{g}.weight"), vec![d, d], ty, weight(&mut r, ty, ENC_D, ENC_D, 1.0)));
        enc_t.push((format!("{p}{g}.bias"), vec![d], 0, weight(&mut r, 0, 1, ENC_D, 0.1)));
    }
    for g in ["ln1", "ln2"] {
        enc_t.push((format!("{p}{g}.weight"), vec![d], 0, ones(ENC_D)));
        enc_t.push((format!("{p}{g}.bias"), vec![d], 0, f32s(&vec![0.0; ENC_D])));
    }
    let enc_path = dir.join("mmproj.gguf");
    write(
        &enc_path,
        &[
            ("general.architecture", Kv::Str("clip".into())),
            ("clip.has_audio_encoder", Kv::Bool(true)),
            ("clip.audio.projector_type", Kv::Str("qwen3a".into())),
            ("clip.audio.embedding_length", Kv::U32(ENC_D as u32)),
            ("clip.audio.feed_forward_length", Kv::U32(ENC_D as u32)),
            ("clip.audio.block_count", Kv::U32(1)),
            ("clip.audio.attention.head_count", Kv::U32(1)),
            ("clip.audio.num_mel_bins", Kv::U32(128)),
            ("clip.audio.attention.layer_norm_epsilon", Kv::F32(1e-5)),
            ("clip.audio.projection_dim", Kv::U32(HIDDEN as u32)),
        ],
        &enc_t,
    );
    ModelFiles { decoder: dec_path, encoder: enc_path }
}

/// The bytes of the pair's Q4_K tensors in the file (the two FFN projections).
pub fn q4k_bytes() -> usize {
    2 * HIDDEN * HIDDEN / 256 * 144
}

/// A chirp with noise: speech-like enough to drive the encoder, deterministic.
pub fn audio(seconds: f32) -> Vec<f32> {
    let n = (seconds * 16_000.0) as usize;
    let mut r = Rng(7);
    (0..n)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            0.3 * (2.0 * std::f32::consts::PI * (200.0 + 300.0 * t) * t).sin() + 0.02 * r.f()
        })
        .collect()
}

pub fn remove(files: &ModelFiles) -> std::io::Result<()> {
    std::fs::remove_file(&files.decoder)?;
    std::fs::remove_file(&files.encoder)
}

pub fn dir_of(files: &ModelFiles) -> PathBuf {
    files.decoder.parent().unwrap().to_path_buf()
}
