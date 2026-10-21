//! GitHub #335 (spec 27b-16gb/03 solution 6, ADR 0061): the load option
//! `prefill_attention` reaches the prompt attention's band and nothing else.
//! On the card, one prompt prefills in 128-token chunks under hq-e8-2b, once
//! with the 8-bit band (the default) and once with the FP16 fallback:
//!
//! - every chunk that starts at or before 512 attends its whole history
//!   through the residual window (the 32 sinks, the 512-key ring, the chunk
//!   itself), the BF16 pass both loads share, so its span logits are the same
//!   bit for bit;
//! - from the chunk at 640 on, the keys between the sinks and the ring go
//!   through the band, so the logits move -- the option reached the kernel --
//!   and stay close: the two greedy readings agree almost everywhere.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{bind_text_scope_27b, materialize, CudaDevice, FrontendSet, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b_with_prefill_attention;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::prefill_program_span_logits;
use ignis_core::{KvFormat, PrefillAttention, RopeScaling};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const CHUNK: u32 = 128;
const MAX_CONTEXT: u32 = 2048;
/// The first chunk whose history reaches past the sinks and the ring: its
/// start minus the 512-key ring clears the 32 sinks.
const FIRST_BAND_CHUNK_START: usize = 640;
/// The least share of band positions whose argmax the two precisions agree on.
const ARGMAX_AGREEMENT_FLOOR: f64 = 0.9;

fn new_pool() -> SeqPool {
    SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: KvFormat::HqE8_2b,
            kv_page_group_count: 64,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 1,
            retained_slot_count: 0,
            retained_host_slot_count: 0,
        },
    )
    .unwrap_or_else(|e| panic!("seq pool: {e}"))
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn argmax(row: &[u16]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if bf16(v) > bf16(row[best]) {
            best = i;
        }
    }
    best
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_fp16_prefill_attention_moves_only_the_band() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let text = "A paged KV cache hands out 64-token pages from a free list; a sequence that grows past its \
                last page takes another, and one that ends gives every page back. Prefix sharing lets two \
                requests point at the same pages until one of them writes. "
        .repeat(24);
    let prompt: Vec<i32> = frontend
        .tokenizer()
        .encode(&text)
        .unwrap_or_else(|e| panic!("tokenize prompt: {e}"))
        .into_iter()
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();
    let prompt = &prompt[..prompt.len().min(MAX_CONTEXT as usize - 1)];
    assert!(prompt.len() >= FIRST_BAND_CHUNK_START + 2 * CHUNK as usize, "{} tokens", prompt.len());

    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!();
        }
    };
    let mut artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let mut logits = Vec::new();
    for precision in [PrefillAttention::Fp8, PrefillAttention::Fp16] {
        let model = load_qwen38_27b_with_prefill_attention(
            &reader,
            &artifact,
            &handles,
            CHUNK,
            MAX_CONTEXT,
            KvFormat::HqE8_2b,
            None,
            None,
            RopeScaling::NONE,
            precision,
        )
        .unwrap_or_else(|e| panic!("{precision}: model load: {e}"));
        let pool = new_pool();
        let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("{precision}: alloc: {e}"));
        let mut span = vec![0u16; prompt.len() * vocab];
        prefill_program_span_logits(&model, &pool, &mut sequence, prompt, 0, &mut span)
            .unwrap_or_else(|e| panic!("{precision}: prefill: {e}"));
        logits.push(span);
    }
    artifact.release_arena(&mut device).expect("release");

    let (fp8, fp16) = (&logits[0], &logits[1]);
    let window = FIRST_BAND_CHUNK_START * vocab;
    let window_differing = fp8[..window].iter().zip(&fp16[..window]).filter(|(a, b)| a != b).count();
    assert_eq!(window_differing, 0, "the window-only chunks differ at {window_differing} logits");

    let band_rows = prompt.len() - FIRST_BAND_CHUNK_START;
    let band_differing = fp8[window..].iter().zip(&fp16[window..]).filter(|(a, b)| a != b).count();
    assert!(band_differing > 0, "the FP16 band computed the 8-bit one's logits: the option did not reach it");
    let agreeing = (FIRST_BAND_CHUNK_START..prompt.len())
        .filter(|&p| argmax(&fp8[p * vocab..(p + 1) * vocab]) == argmax(&fp16[p * vocab..(p + 1) * vocab]))
        .count();
    let agreement = agreeing as f64 / band_rows as f64;
    assert!(
        agreement >= ARGMAX_AGREEMENT_FLOOR,
        "the two bands agree on the argmax at {agreeing} of {band_rows} positions"
    );
    println!(
        "prefill attention: {} prompt tokens in {} chunks; the window-only {FIRST_BAND_CHUNK_START} positions \
         identical, {band_differing} of {} band logits moved, argmax agreed at {agreeing} of {band_rows}",
        prompt.len(),
        prompt.len().div_ceil(CHUNK as usize),
        band_rows * vocab
    );
}
