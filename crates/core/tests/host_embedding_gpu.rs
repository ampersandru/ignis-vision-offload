//! ADR 0062 (GitHub #335, spec 27b-16gb/03 AC 6): the 27B's token embedding
//! lives in mapped pinned host memory. On the card, a load binds the table
//! from the mapped allocation -- the leaf finds exactly the table's bytes in
//! host memory and every other bound tensor in VRAM -- and computes what the
//! all-on-device load, the placement before ADR 0062, computes: every span
//! logit of a multi-chunk prefill bit for bit, then the same greedy tokens
//! decoded eagerly and through the captured decode graphs.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{
    bind_model_scope_27b_on_device, bind_text_scope_27b, materialize, CudaDevice, FrontendSet,
    MaterializationPlan, ModelScope, ObjectHandle, Reader, Residence,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::{capture_decode_graphs, decode_program_batch, prefill_program_span_logits, program_stats};
use ignis_core::KvFormat;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// `text/token_embedding`, W8G32 `[248320, 5120]`: 5,440 bytes a row.
const TABLE_BYTES: u64 = 248_320 * 5_440;
/// The `*_input_scale_divisor` scalars: bound host-side, never a descriptor
/// (`model_load_gpu.rs`).
const DIVISOR_COUNT: u64 = 247;
/// A 128-token chunk, so the prompt below prefills in three.
const CHUNK: u32 = 128;
const MAX_CONTEXT: u32 = 1024;
const GENERATED: usize = 16;

/// What one placement computes: the prompt's span logits and the greedy
/// tokens decoded eagerly and through the decode graphs.
struct Output {
    span_logits: Vec<u16>,
    eager: Vec<i32>,
    graph: Vec<i32>,
}

fn new_pool() -> SeqPool {
    SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: KvFormat::Bf16,
            kv_page_group_count: 32,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 1,
            retained_slot_count: 0,
            retained_host_slot_count: 0,
        },
    )
    .unwrap_or_else(|e| panic!("seq pool: {e}"))
}

/// Materialize `plan`, load it, check where the leaf found the table, and
/// run the prompt; the device memory is released before returning.
fn run(
    device: &mut CudaDevice,
    reader: &Reader,
    plan: &MaterializationPlan,
    handles: &[ObjectHandle],
    prompt: &[i32],
    mapped: bool,
) -> Output {
    let label = if mapped { "host-mapped" } else { "on-device" };
    let mut artifact = materialize(reader, plan, device, None).unwrap_or_else(|e| panic!("{label}: materialize: {e}"));
    let table = artifact.device_view(handles[0]).expect("the token embedding's view");
    assert_eq!(table.name, "text/token_embedding");
    assert_eq!(table.residence, if mapped { Residence::HostMapped } else { Residence::Device });
    let output = {
        let model = load_qwen38_27b(reader, &artifact, handles, CHUNK, MAX_CONTEXT, KvFormat::Bf16)
            .unwrap_or_else(|e| panic!("{label}: model load: {e}"));
        let stats = model.stats();
        // The leaf asks the runtime where each tensor lives: the table in
        // host memory, the rest -- what the materializer uploaded -- in VRAM.
        assert_eq!(stats.host_mapped_bytes, if mapped { TABLE_BYTES } else { 0 }, "{label}");
        assert_eq!(stats.vram_bytes, artifact.stats().h2d_bytes - DIVISOR_COUNT * 4, "{label}");
        assert_eq!(artifact.stats().host_mapped_bytes, if mapped { TABLE_BYTES } else { 0 }, "{label}");

        let vocab = ModelConfig::qwen38_27b().vocab as usize;
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
            assert_eq!(program_stats(&model, &pool).expect("stats").graph_launches, 0, "{label}: eager");
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
            let mut tokens = Vec::with_capacity(GENERATED);
            for _ in 0..GENERATED {
                tokens.extend(decode_program_batch(&model, &pool, &mut [&mut sequence]).expect("graph decode"));
                assert_eq!(program_stats(&model, &pool).expect("stats").graph_launches, 1, "{label}: replayed");
            }
            tokens
        };
        Output { span_logits, eager, graph }
    };
    artifact.release_arena(device).expect("release");
    output
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_host_mapped_token_embedding_computes_what_the_device_one_does() {
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

    let (mapped_plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind: {e}"));
    let (device_plan, device_handles) =
        bind_model_scope_27b_on_device(&reader, ModelScope::default()).unwrap_or_else(|e| panic!("bind: {e}"));
    assert_eq!(handles, device_handles);
    assert!(device_plan.device_capacity_bytes - mapped_plan.device_capacity_bytes >= TABLE_BYTES);

    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!();
        }
    };
    let on_device = run(&mut device, &reader, &device_plan, &device_handles, &prompt, false);
    let host_mapped = run(&mut device, &reader, &mapped_plan, &handles, &prompt, true);
    let differing = on_device.span_logits.iter().zip(&host_mapped.span_logits).filter(|(a, b)| a != b).count();
    assert_eq!(differing, 0, "span logits differ at {differing} of {} positions x vocab", prompt.len());
    assert_eq!(on_device.eager, host_mapped.eager, "eager greedy tokens");
    assert_eq!(on_device.graph, host_mapped.graph, "graph-replayed greedy tokens");
    assert_eq!(host_mapped.eager, host_mapped.graph, "graph replay against eager");
    println!(
        "host-mapped token embedding: {} prompt tokens in {} chunks, {} span logits identical, \
         {GENERATED} greedy tokens identical eager and replayed: {:?}",
        prompt.len(),
        prompt.len().div_ceil(CHUNK as usize),
        host_mapped.span_logits.len(),
        host_mapped.eager
    );
}
