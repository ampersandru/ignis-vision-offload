//! GitHub #341 (spec prefill-5090/01 lever 1): the 27B's nine BF16 exception
//! tensors held in FP8, converted at load. On the card:
//!
//! - each converted slot holds, byte for byte, what `ignis_artifact::fp8`
//!   writes from the artifact's BF16 -- the GPU's quantizer and the host's are
//!   one contract;
//! - the leaf counts the conversions' bytes, not the stored ones, and the
//!   plan's arena shrinks by the difference;
//! - a three-chunk prefill (the W8A8 route) reads the prompt's own next
//!   token as often as the BF16 load does, within 0.05 (the inter-engine
//!   top-1 agreement is reported, not gated: see the test's note), and greedy
//!   decoding through the captured graphs (the W8A16 GEMV route) produces
//!   what eager decoding does. The quality gate is the KLD against the BF16
//!   checkpoint (spec prefill-5090/01 AC 3), not this.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{
    bind_model_scope_27b_with, fp8, materialize, CudaDevice, DeviceEncoding, FrontendSet, MaterializationPlan,
    ModelScope, NumericFormat, ObjectHandle, Reader, StorageLayout,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::{capture_decode_graphs, decode_program_batch, prefill_program_span_logits, program_stats};
use ignis_core::KvFormat;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// The `*_input_scale_divisor` scalars: bound host-side, never a descriptor
/// (`model_load_gpu.rs`).
const DIVISOR_COUNT: u64 = 247;
const CHUNK: u32 = 128;
const MAX_CONTEXT: u32 = 1024;
const GENERATED: usize = 16;

const SHORT: [usize; 4] = [1, 8, 9, 40];

struct Output {
    /// The span logits of the prompt's first SHORT[i] tokens, prefilled alone.
    short: Vec<Vec<u16>>,
    span_logits: Vec<u16>,
    eager: Vec<i32>,
    graph: Vec<i32>,
}

fn new_pool() -> SeqPool {
    SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: KvFormat::HqE8_2b,
            kv_page_group_count: 32,
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

#[allow(clippy::too_many_arguments)]
fn run(
    device: &mut CudaDevice,
    reader: &Reader,
    plan: &MaterializationPlan,
    handles: &[ObjectHandle],
    prompt: &[i32],
    converted: bool,
    chunk: u32,
    prefill_attention: ignis_core::PrefillAttention,
) -> Option<Output> {
    let label = if converted { "fp8" } else { "bf16" };
    let mut artifact = match materialize(reader, plan, device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("{label}: materialize: {e}")) {
                return None;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let stats = *artifact.stats();
    if converted {
        // Every converted slot is the host contract's bytes.
        let arena = artifact.arena().expect("arena");
        let mut checked = 0;
        for placed in &plan.device_objects {
            let DeviceEncoding::Fp8Rows(blocks) = placed.encoding else { continue };
            let object = &reader.objects()[placed.handle.index];
            let cols = match object {
                ignis_artifact::Object::Tensor(t) => t.shape[1] as u32,
                ignis_artifact::Object::Resource(_) => unreachable!(),
            };
            let want = fp8::quantize_blocks(reader.payload_at(object).expect("payload").data, blocks, cols);
            let view = artifact.device_view(placed.handle).expect("view");
            assert_eq!((view.format, view.layout), (NumericFormat::Fp8E4M3FnRowBf16S, StorageLayout::RowScaleV1));
            let mut got = vec![0u8; want.len()];
            use ignis_artifact::Device as _;
            device.copy_d2h(arena, view.offset_in_arena, &mut got).expect("d2h");
            device.synchronize().expect("sync");
            assert!(got == want, "{}: the device's FP8 payload is not the host contract's", object.name());
            checked += 1;
        }
        assert_eq!(checked, 9, "the nine BF16 exception tensors");
        let planned: u64 =
            plan.device_objects.iter().filter(|p| p.encoding != DeviceEncoding::Stored).map(|p| p.bytes).sum();
        assert_eq!(stats.fp8_device_bytes, planned, "the slots the plan gave them");
        assert_eq!(stats.fp8_source_bytes, 6 * 14_336 * 5_120 * 2 + 3 * 5_120 * 6_144 * 2, "their stored bytes");
    } else {
        assert_eq!((stats.fp8_source_bytes, stats.fp8_device_bytes), (0, 0));
    }
    let output = {
        let model = ignis_core::model_load::load_qwen38_27b_with_prefill_attention(
            reader,
            &artifact,
            handles,
            chunk,
            MAX_CONTEXT,
            KvFormat::HqE8_2b,
            None,
            None,
            ignis_core::RopeScaling::NONE,
            prefill_attention,
        )
        .unwrap_or_else(|e| panic!("{label}: model load: {e}"));
        assert_eq!(
            model.stats().vram_bytes,
            stats.h2d_bytes - DIVISOR_COUNT * 4 - (stats.fp8_source_bytes - stats.fp8_device_bytes),
            "{label}: the leaf counts the conversions' bytes"
        );
        let vocab = ModelConfig::qwen38_27b().vocab as usize;
        let short: Vec<Vec<u16>> = SHORT
            .iter()
            .map(|&n| {
                let pool = new_pool();
                let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("{label}: alloc: {e}"));
                let mut logits = vec![0u16; n * vocab];
                prefill_program_span_logits(&model, &pool, &mut sequence, &prompt[..n], 0, &mut logits)
                    .unwrap_or_else(|e| panic!("{label}: short prefill: {e}"));
                logits
            })
            .collect();
        let mut span_logits = vec![0u16; prompt.len() * vocab];
        let eager = {
            let pool = new_pool();
            let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("{label}: alloc: {e}"));
            prefill_program_span_logits(&model, &pool, &mut sequence, prompt, 0, &mut span_logits)
                .unwrap_or_else(|e| panic!("{label}: prefill: {e}"));
            let mut tokens = Vec::with_capacity(GENERATED);
            for _ in 0..GENERATED {
                tokens.extend(decode_program_batch(&model, &pool, &mut [&mut sequence]).expect("eager decode"));
            }
            tokens
        };
        let graph = {
            let pool = new_pool();
            let capture = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("{label}: capture: {e}"));
            assert!(capture.is_ready(1), "{label}: no width-1 decode graph");
            let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("{label}: alloc: {e}"));
            let mut scratch = vec![0u16; prompt.len() * vocab];
            prefill_program_span_logits(&model, &pool, &mut sequence, prompt, 0, &mut scratch)
                .unwrap_or_else(|e| panic!("{label}: prefill: {e}"));
            assert_eq!(scratch, span_logits, "{label}: a second prefill agrees bit for bit");
            let mut tokens = Vec::with_capacity(GENERATED);
            for _ in 0..GENERATED {
                tokens.extend(decode_program_batch(&model, &pool, &mut [&mut sequence]).expect("graph decode"));
                assert_eq!(program_stats(&model, &pool).expect("stats").graph_launches, 1, "{label}: replayed");
            }
            tokens
        };
        Output { short, span_logits, eager, graph }
    };
    artifact.release_arena(device).expect("release");
    Some(output)
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_fp8_exception_tensors_are_the_host_contracts_bytes_and_read_the_text_as_bf16_does() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let text = "Explain, step by step, how a paged KV cache with 64-token pages is allocated and freed in \
                an LLM inference server, what fragmentation it avoids, and how prefix sharing between \
                requests changes the bookkeeping. "
        .repeat(8);
    let prompt: Vec<i32> = frontend
        .tokenizer()
        .encode(&text)
        .unwrap_or_else(|e| panic!("tokenize prompt: {e}"))
        .into_iter()
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();
    assert!(prompt.len() > 2 * CHUNK as usize, "three chunks: {} tokens", prompt.len());

    let scope = ModelScope::default();
    let (bf16_plan, handles) = bind_model_scope_27b_with(&reader, scope).unwrap_or_else(|e| panic!("bind: {e}"));
    let (fp8_plan, fp8_handles) = bind_model_scope_27b_with(&reader, ModelScope { fp8_exceptions: true, ..scope })
        .unwrap_or_else(|e| panic!("bind: {e}"));
    assert_eq!(handles, fp8_handles, "the same objects");
    let saved = bf16_plan.device_capacity_bytes - fp8_plan.device_capacity_bytes;
    // Six [14336][5120] and three [5120][6144] matrices, each about halved.
    let stored: u64 = 6 * 14_336 * 5_120 * 2 + 3 * 5_120 * 6_144 * 2;
    assert!(saved > stored / 2 - 1_000_000 && saved < stored / 2, "the arena shrinks by {saved} B");

    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!();
        }
    };
    use ignis_core::PrefillAttention;
    let Some(reference) = run(&mut device, &reader, &bf16_plan, &handles, &prompt, false, CHUNK, PrefillAttention::Fp8) else {
        return;
    };
    let Some(converted) = run(&mut device, &reader, &fp8_plan, &fp8_handles, &prompt, true, CHUNK, PrefillAttention::Fp8) else {
        return;
    };

    // Reported, not gated: the quality gate is the KLD against the BF16
    // checkpoint (spec prefill-5090/01 AC 3). Two engines that both read a
    // repeated text with peaked distributions are nats apart wherever their
    // argmaxes differ, and the 4-bit activation codes of every NVFP4 layer
    // past four tokens turn a small difference upstream into a flip
    // downstream, so the agreement between them measures that amplifier more
    // than either engine. What each one gets of the text's own next token
    // does not have that problem.
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let argmax = |row: &[u16]| {
        row.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(i, m), (j, &b)| if bf16(b) > m { (j, bf16(b)) } else { (i, m) }).0
    };
    let next_token = |logits: &[u16]| {
        let hits = (0..prompt.len() - 1).filter(|&t| argmax(&logits[t * vocab..(t + 1) * vocab]) as i32 == prompt[t + 1]).count();
        hits as f64 / (prompt.len() - 1) as f64
    };
    let agree = (0..prompt.len())
        .filter(|&t| argmax(&reference.span_logits[t * vocab..(t + 1) * vocab]) == argmax(&converted.span_logits[t * vocab..(t + 1) * vocab]))
        .count() as f64
        / prompt.len() as f64;
    let (bf16_hits, fp8_hits) = (next_token(&reference.span_logits), next_token(&converted.span_logits));
    let short_hits: Vec<String> = SHORT
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let a = argmax(&reference.short[i][(n - 1) * vocab..]);
            let b = argmax(&converted.short[i][(n - 1) * vocab..]);
            format!("{n}:{}", if a == b { "same" } else { "differs" })
        })
        .collect();
    eprintln!(
        "over {} positions: the text's next token, bf16 {bf16_hits:.3} fp8 {fp8_hits:.3}; top-1 agreement {agree:.3}; \
         short prompts' last token {short_hits:?}; greedy bf16 {:?} fp8 {:?}",
        prompt.len(),
        reference.eager,
        converted.eager
    );
    assert!(fp8_hits >= bf16_hits - 0.05, "the text's next token: fp8 {fp8_hits:.3} against bf16 {bf16_hits:.3}");
    assert_eq!(converted.eager, converted.graph, "fp8: the decode graphs decode what eager decoding does");
    assert_eq!(reference.eager, reference.graph, "bf16: the same");
}
