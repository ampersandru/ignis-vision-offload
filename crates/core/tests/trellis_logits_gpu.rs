//! GitHub #339 (spec 27b-16gb/05a): the 16 GB artifact's whole program --
//! every layer, the head, the chunked prefill -- against the served artifact's
//! on the same span: top-1 agreement over every position of a stretch of
//! code. A wiring error anywhere in the trellis body (a projection's rows in
//! the wrong place, a residual not added, the wrong head) leaves the logits
//! nothing to agree with; two quantizations of one checkpoint agree on most
//! positions.
//!
//! An error detector, not a quality number (engine agreement is not quality,
//! the KLD gates are spec 05b's): the floor sits well under what the two
//! artifacts measured and far above what a broken body reaches. The mean
//! KL(served || trellis) is printed beside it.
//!
//! Both loads are this test's, one after the other on the card: BF16 KV (the
//! oracle format, ADR 0022), one 1,024-token chunk, the span's logits read
//! through the measurement readout (`prefill_program_span_logits`).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

#[path = "support/logit_drift.rs"]
mod logit_drift;

use std::path::Path;

use ignis_artifact::{CudaDevice, FrontendSet, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::prefill_program_span_logits;

use logit_drift::{argmax, bf16};

const TRELLIS: &str = "F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/qwen3_8_27b_trellis_sc30-v2.ninfer";
const SERVED: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const SPAN_TOKENS: usize = 768;
const PREFILL_CHUNK: u32 = 1024;
const MAX_CONTEXT: u32 = 1024;
/// The share of positions whose argmax the trellis program must share with
/// the served one. Measured 2026-10-23 on the 5090 over the frozen span: 683
/// of 768 (0.889), mean KL(served || trellis) 0.071 nats (printed below, and
/// again on a rerun); a body wired wrong agrees on almost none.
const TOP1_FLOOR: f64 = 0.7;

fn log_softmax(row: &[u16]) -> Vec<f64> {
    let max = row.iter().map(|&b| bf16(b)).fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = row.iter().map(|&b| (bf16(b) as f64 - max).exp()).sum();
    let lse = max + sum.ln();
    row.iter().map(|&b| bf16(b) as f64 - lse).collect()
}

/// The span's `[SPAN_TOKENS][vocab]` logits on `path`'s program, or `None`
/// where the profile allows a skip. The device and the load are dropped
/// before it returns, so the next load has the card.
fn span_logits(path: &str, span: &[i32]) -> Option<Vec<u16>> {
    let reader = Reader::open(Path::new(path)).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind {path}: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}"));
            return None;
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("materialize {path}: {e}"));
            return None;
        }
    };
    let model = load_qwen38_27b(&reader, &artifact, &handles, PREFILL_CHUNK, MAX_CONTEXT, ignis_core::KvFormat::Bf16)
        .unwrap_or_else(|e| panic!("load {path}: {e}"));
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: ignis_core::KvFormat::Bf16,
            kv_page_group_count: MAX_CONTEXT.div_ceil(64),
            max_context_tokens: MAX_CONTEXT,
            slot_count: 1,
            retained_slot_count: 0,
            retained_host_slot_count: 0,
        },
    )
    .unwrap_or_else(|e| panic!("pool: {e}"));
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let mut rows = vec![0u16; span.len() * vocab];
    let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    if let Err(e) = prefill_program_span_logits(&model, &pool, &mut sequence, span, 0, &mut rows) {
        gpu_profile::skip_or_fail(&format!("span logits on {path}: {e}"));
        return None;
    }
    Some(rows)
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_trellis_program_agrees_with_the_served_one_on_code() {
    for path in [TRELLIS, SERVED] {
        if !Path::new(path).exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {path}")) {
            return;
        }
    }
    let served = Reader::open(Path::new(SERVED)).unwrap_or_else(|e| panic!("open: {e}"));
    let frontend = FrontendSet::from_reader(&served).unwrap_or_else(|e| panic!("frontend: {e}"));
    // Real code, frozen: the trellis op's Rust binding as of f3985f17
    // (`crates/core/src/trellis.rs`), so an edit of the live file moves no
    // position.
    let span: Vec<i32> = frontend
        .tokenizer()
        .encode(include_str!("fixtures/trellis_logits/code_span.txt"))
        .unwrap_or_else(|e| panic!("tokenize: {e}"))
        .into_iter()
        .take(SPAN_TOKENS)
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();
    assert_eq!(span.len(), SPAN_TOKENS, "the source is long enough");
    drop(served);

    let Some(theirs) = span_logits(SERVED, &span) else { return };
    let Some(ours) = span_logits(TRELLIS, &span) else { return };
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let (mut agree, mut kl) = (0usize, 0f64);
    for position in 0..SPAN_TOKENS {
        let row = position * vocab..(position + 1) * vocab;
        agree += usize::from(argmax(&ours[row.clone()]).0 == argmax(&theirs[row.clone()]).0);
        let (p, q) = (log_softmax(&theirs[row.clone()]), log_softmax(&ours[row]));
        kl += p.iter().zip(&q).map(|(&lp, &lq)| lp.exp() * (lp - lq)).sum::<f64>();
    }
    let share = agree as f64 / SPAN_TOKENS as f64;
    println!(
        "top-1 agreement {agree}/{SPAN_TOKENS} = {share:.4} (floor {TOP1_FLOOR}); mean KL(served || trellis) {:.4} nats",
        kl / SPAN_TOKENS as f64
    );
    assert!(share >= TOP1_FLOOR, "the trellis program agrees with the served one on {share} of the positions");
}
