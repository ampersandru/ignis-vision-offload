//! The speech model loaded from its GGUF pair (spec speech/01 §1-2, §6): the
//! Qwen3 decoder (`qwen3vl` with no deepstack layers) and the AuT audio
//! encoder (a `clip` file with a `qwen3a` projector). The family is read from
//! the metadata, never from a file name (ADR 0047).
//!
//! Q4_K matmul weights are repacked into the interleaved layout at load, the
//! only copy kept; every other tensor is read from the mapped file, or from a
//! private copy when the weights are pinned (Q31, Q53).

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use memmap2::MmapMut;

use crate::gguf::{Gguf, TensorInfo, TensorType, Value};
use crate::matmul::Weight;
use crate::mem::Blob;
use crate::quant::{dequant_row_f64, WType};
use crate::text::Text;

/// A matmul weight and its bias.
pub struct Linear {
    pub w: Weight,
    pub b: Option<Vec<f32>>,
}

pub struct EncLayer {
    pub ln1: (Vec<f32>, Vec<f32>),
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub ln2: (Vec<f32>, Vec<f32>),
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct Encoder {
    pub d: usize,
    pub heads: usize,
    pub eps: f32,
    /// The three 3×3 stride-2 convolutions (1 → 480 → 480 → 480) and their biases.
    pub conv: [Linear; 3],
    pub conv_channels: usize,
    /// 480 channels × 16 mel bins → d.
    pub conv_out: Weight,
    /// The sinusoidal positions of a chunk's 13 tokens, rows of d.
    pub pos: Vec<f32>,
    pub layers: Vec<EncLayer>,
    pub post_ln: (Vec<f32>, Vec<f32>),
    pub proj1: Linear,
    pub proj2: Linear,
    /// The projector's output: the decoder's hidden size.
    pub out_dim: usize,
}

pub struct DecLayer {
    pub attn_norm: Vec<f32>,
    pub q: Weight,
    pub k: Weight,
    pub v: Weight,
    pub o: Weight,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
}

/// Token embedding rows, read (dequantized) from the file's bytes.
pub struct Embedding {
    pub ty: WType,
    pub dim: usize,
    pub rows: usize,
    data: Blob,
}

impl Embedding {
    pub fn row(&self, id: u32, out: &mut [f32]) {
        let mut tmp = vec![0f64; self.dim];
        dequant_row_f64(self.ty, self.data.as_slice(), id as usize, self.dim, &mut tmp);
        for (o, t) in out.iter_mut().zip(tmp) {
            *o = t as f32;
        }
    }
}

pub struct Decoder {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub theta: f32,
    pub sections: [usize; 3],
    pub embed: Embedding,
    pub layers: Vec<DecLayer>,
    pub norm: Vec<f32>,
    /// The LM head: `output.weight`, or the embedding when the file ties them.
    pub head: Weight,
    pub vocab: usize,
}

impl Decoder {
    /// Bytes of K and V per token: layers × 2 × KV heads × head dim × 2 (f16).
    pub fn kv_bytes_per_token(&self) -> usize {
        self.layers.len() * 2 * self.kv_heads * self.head_dim * 2
    }
}

/// What a load did, for the log and the "As built" numbers.
#[derive(Clone, Debug)]
pub struct LoadReport {
    pub millis: u128,
    /// Bytes of Q4_K weights repacked into private memory.
    pub repacked_bytes: usize,
    /// Bytes read from the mapped files (or copied, when pinned).
    pub mapped_bytes: usize,
    pub pinned: bool,
    /// Whether the pinned copy's pages are locked in RAM (best effort).
    pub locked: bool,
}

pub struct Model {
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub text: Text,
    pub report: LoadReport,
}

#[derive(Debug)]
pub struct LoadError(pub String);

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LoadError {}

fn err<T>(m: impl Into<String>) -> Result<T, LoadError> {
    Err(LoadError(m.into()))
}

/// Where a file's tensor bytes come from: the mapping, or a private copy of its data section.
struct Source {
    gguf: Arc<Gguf>,
    copy: Option<Arc<memmap2::Mmap>>,
}

impl Source {
    /// With `pinned`, the data section is copied into anonymous memory and its pages locked
    /// (best effort); the second value says whether the lock held.
    fn new(gguf: Gguf, pinned: bool) -> Result<(Source, bool), LoadError> {
        let gguf = Arc::new(gguf);
        if !pinned {
            return Ok((Source { gguf, copy: None }, false));
        }
        let section = gguf.data_section();
        let fail = |e: std::io::Error| LoadError(format!("pinning {}: {e}", gguf.path().display()));
        let mut anon = MmapMut::map_anon(section.len().max(1)).map_err(fail)?;
        anon[..section.len()].copy_from_slice(section);
        let ro = anon.make_read_only().map_err(fail)?;
        // mlock where the OS offers it to a mapping; elsewhere the private copy is what pins.
        #[cfg(unix)]
        let locked = ro.lock().is_ok();
        #[cfg(not(unix))]
        let locked = false;
        Ok((Source { gguf: gguf.clone(), copy: Some(Arc::new(ro)) }, locked))
    }

    fn bytes(&self, t: &TensorInfo) -> &[u8] {
        match &self.copy {
            None => self.gguf.data(t),
            Some(c) => &c[t.offset as usize..(t.offset + t.size) as usize],
        }
    }

    fn owner(&self) -> Arc<dyn Send + Sync> {
        match &self.copy {
            None => self.gguf.clone(),
            Some(c) => c.clone(),
        }
    }
}

struct Loader {
    src: Source,
    repacked: usize,
    mapped: usize,
}

impl Loader {
    fn tensor(&self, name: &str) -> Result<&TensorInfo, LoadError> {
        self.src.gguf.tensor(name).ok_or_else(|| LoadError(format!("{}: no tensor {name}", self.src.gguf.path().display())))
    }

    fn f32s(&mut self, name: &str) -> Result<Vec<f32>, LoadError> {
        let t = self.tensor(name)?.clone();
        if t.ty != TensorType::F32 {
            return err(format!("{name}: {} where F32 is expected", t.ty.name()));
        }
        Ok(self.src.bytes(&t).chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    }

    /// A matmul weight of `n` rows: ggml order puts the row length first, and every further
    /// dimension but the last is folded into it (the convolutions' 3 × 3 × c_in).
    fn weight(&mut self, name: &str) -> Result<Weight, LoadError> {
        let t = self.tensor(name)?.clone();
        let n = *t.dims.last().unwrap() as usize;
        let k = (t.elements() / n as u64) as usize;
        let bytes = self.src.bytes(&t);
        let ty = match t.ty {
            TensorType::Q4K if n % 8 == 0 => {
                self.repacked += bytes.len();
                return Ok(Weight::repack_q4k(name, n, k, bytes));
            }
            TensorType::Q4K => WType::Q4K,
            TensorType::Q6K => WType::Q6K,
            TensorType::Q8_0 => WType::Q80,
            TensorType::F16 => WType::F16,
            TensorType::F32 => WType::F32,
            other => return err(format!("{name}: {} weights are not served by the CPU executor", other.name())),
        };
        self.mapped += bytes.len();
        Ok(Weight::new(name, ty, n, k, Blob::borrowed(bytes, self.src.owner())))
    }

    fn linear(&mut self, base: &str) -> Result<Linear, LoadError> {
        let w = self.weight(&format!("{base}.weight"))?;
        let bias = format!("{base}.bias");
        let b = if self.src.gguf.tensor(&bias).is_some() { Some(self.f32s(&bias)?) } else { None };
        Ok(Linear { w, b })
    }

    fn ln(&mut self, base: &str) -> Result<(Vec<f32>, Vec<f32>), LoadError> {
        Ok((self.f32s(&format!("{base}.weight"))?, self.f32s(&format!("{base}.bias"))?))
    }

    fn kv_u(&self, key: &str) -> Result<usize, LoadError> {
        self.src.gguf.get(key).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| LoadError(format!("{}: no {key}", self.src.gguf.path().display())))
    }

    fn kv_f(&self, key: &str) -> Result<f64, LoadError> {
        self.src.gguf.get(key).and_then(Value::as_f64).ok_or_else(|| LoadError(format!("{}: no {key}", self.src.gguf.path().display())))
    }
}

/// Checks the pair is Qwen3-ASR by its metadata: a `qwen3vl` decoder with no deepstack layers and
/// a `clip` encoder with an audio tower and the `qwen3a` projector.
pub fn check_family(dec: &Gguf, enc: &Gguf) -> Result<(), LoadError> {
    let s = |g: &Gguf, k: &str| g.get(k).and_then(Value::as_str).map(str::to_string);
    if s(dec, "general.architecture").as_deref() != Some("qwen3vl") {
        return err(format!("{}: not a qwen3vl decoder", dec.path().display()));
    }
    if dec.get("qwen3vl.n_deepstack_layers").and_then(Value::as_u64).unwrap_or(0) != 0 {
        return err(format!("{}: a decoder with deepstack layers is a vision model, not Qwen3-ASR", dec.path().display()));
    }
    if s(enc, "general.architecture").as_deref() != Some("clip") || enc.get("clip.has_audio_encoder") != Some(&Value::Bool(true)) {
        return err(format!("{}: not an audio encoder", enc.path().display()));
    }
    if s(enc, "clip.audio.projector_type").as_deref() != Some("qwen3a") {
        return err(format!("{}: projector {:?}, not qwen3a", enc.path().display(), s(enc, "clip.audio.projector_type")));
    }
    Ok(())
}

impl Model {
    pub fn load(decoder: &Path, encoder: &Path, pinned: bool) -> Result<Model, LoadError> {
        let t0 = Instant::now();
        let dec = Gguf::open(decoder).map_err(|e| LoadError(e.to_string()))?;
        let enc = Gguf::open(encoder).map_err(|e| LoadError(e.to_string()))?;
        check_family(&dec, &enc)?;
        let text = Text::from_gguf(&dec).map_err(LoadError)?;
        let (src, dl) = Source::new(dec, pinned)?;
        let mut d = Loader { src, repacked: 0, mapped: 0 };
        let (src, el) = Source::new(enc, pinned)?;
        let mut e = Loader { src, repacked: 0, mapped: 0 };
        let decoder = load_decoder(&mut d)?;
        let encoder = load_encoder(&mut e)?;
        if encoder.out_dim != decoder.hidden {
            return err(format!("the encoder projects to {}, the decoder's hidden size is {}", encoder.out_dim, decoder.hidden));
        }
        let report = LoadReport {
            millis: t0.elapsed().as_millis(),
            repacked_bytes: d.repacked + e.repacked,
            mapped_bytes: d.mapped + e.mapped,
            pinned,
            locked: pinned && dl && el,
        };
        Ok(Model { encoder, decoder, text, report })
    }
}

fn load_decoder(l: &mut Loader) -> Result<Decoder, LoadError> {
    let a = "qwen3vl";
    let layers_n = l.kv_u(&format!("{a}.block_count"))?;
    let hidden = l.kv_u(&format!("{a}.embedding_length"))?;
    let heads = l.kv_u(&format!("{a}.attention.head_count"))?;
    let kv_heads = l.kv_u(&format!("{a}.attention.head_count_kv"))?;
    let head_dim = l.kv_u(&format!("{a}.attention.key_length")).unwrap_or(hidden / heads);
    let eps = l.kv_f(&format!("{a}.attention.layer_norm_rms_epsilon"))? as f32;
    let theta = l.kv_f(&format!("{a}.rope.freq_base"))? as f32;
    let sec: Vec<usize> = match l.src.gguf.get(&format!("{a}.rope.dimension_sections")).and_then(Value::as_array) {
        Some(v) => v.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect(),
        None => vec![head_dim / 2, 0, 0],
    };
    let sections = [sec.first().copied().unwrap_or(0), sec.get(1).copied().unwrap_or(0), sec.get(2).copied().unwrap_or(0)];
    let et = l.tensor("token_embd.weight")?.clone();
    let ety = match et.ty {
        TensorType::Q4K => WType::Q4K,
        TensorType::Q6K => WType::Q6K,
        TensorType::Q8_0 => WType::Q80,
        TensorType::F16 => WType::F16,
        TensorType::F32 => WType::F32,
        other => return err(format!("token_embd.weight: {} is not served by the CPU executor", other.name())),
    };
    let vocab = et.dims[1] as usize;
    let embed = Embedding { ty: ety, dim: et.dims[0] as usize, rows: vocab, data: Blob::borrowed(l.src.bytes(&et), l.src.owner()) };
    let mut layers = Vec::with_capacity(layers_n);
    for i in 0..layers_n {
        let p = format!("blk.{i}.");
        layers.push(DecLayer {
            attn_norm: l.f32s(&format!("{p}attn_norm.weight"))?,
            q: l.weight(&format!("{p}attn_q.weight"))?,
            k: l.weight(&format!("{p}attn_k.weight"))?,
            v: l.weight(&format!("{p}attn_v.weight"))?,
            o: l.weight(&format!("{p}attn_output.weight"))?,
            q_norm: l.f32s(&format!("{p}attn_q_norm.weight"))?,
            k_norm: l.f32s(&format!("{p}attn_k_norm.weight"))?,
            ffn_norm: l.f32s(&format!("{p}ffn_norm.weight"))?,
            gate: l.weight(&format!("{p}ffn_gate.weight"))?,
            up: l.weight(&format!("{p}ffn_up.weight"))?,
            down: l.weight(&format!("{p}ffn_down.weight"))?,
        });
    }
    let norm = l.f32s("output_norm.weight")?;
    // The checkpoint ties the head to the embedding; a file may hold it as its own tensor.
    let head = if l.src.gguf.tensor("output.weight").is_some() {
        l.weight("output.weight")?
    } else {
        let w = Weight::new("token_embd.weight", ety, vocab, hidden, Blob::borrowed(l.src.bytes(&et), l.src.owner()));
        l.mapped += w.bytes().len();
        w
    };
    Ok(Decoder { hidden, heads, kv_heads, head_dim, eps, theta, sections, embed, layers, norm, head, vocab })
}

fn load_encoder(l: &mut Loader) -> Result<Encoder, LoadError> {
    let d = l.kv_u("clip.audio.embedding_length")?;
    let heads = l.kv_u("clip.audio.attention.head_count")?;
    let n_layers = l.kv_u("clip.audio.block_count")?;
    let eps = l.kv_f("clip.audio.attention.layer_norm_epsilon").unwrap_or(1e-5) as f32;
    let mels = l.kv_u("clip.audio.num_mel_bins")?;
    if mels != crate::mel::N_MELS {
        return err(format!("{mels} mel bins (the front end makes {})", crate::mel::N_MELS));
    }
    let conv = [l.linear("a.conv2d.1")?, l.linear("a.conv2d.2")?, l.linear("a.conv2d.3")?];
    let conv_channels = conv[0].w.n;
    let conv_out = l.weight("a.conv_out.weight")?;
    if conv_out.k != conv_channels * mels.div_ceil(8) {
        return err(format!("a.conv_out.weight: k {} for {conv_channels} channels", conv_out.k));
    }
    let pos_all = l.f32s("a.position_embd.weight")?;
    if pos_all.len() < crate::mel::CHUNK_TOKENS * d {
        return err("a.position_embd.weight: fewer than 13 positions");
    }
    let pos = pos_all[..crate::mel::CHUNK_TOKENS * d].to_vec();
    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let p = format!("a.blk.{i}.");
        layers.push(EncLayer {
            ln1: l.ln(&format!("{p}ln1"))?,
            q: l.linear(&format!("{p}attn_q"))?,
            k: l.linear(&format!("{p}attn_k"))?,
            v: l.linear(&format!("{p}attn_v"))?,
            o: l.linear(&format!("{p}attn_out"))?,
            ln2: l.ln(&format!("{p}ln2"))?,
            fc1: l.linear(&format!("{p}ffn_up"))?,
            fc2: l.linear(&format!("{p}ffn_down"))?,
        });
    }
    let post_ln = l.ln("a.post_ln")?;
    let proj1 = l.linear("mm.a.mlp.1")?;
    let proj2 = l.linear("mm.a.mlp.2")?;
    let out_dim = proj2.w.n;
    Ok(Encoder { d, heads, eps, conv, conv_channels, conv_out, pos, layers, post_ln, proj1, proj2, out_dim })
}
