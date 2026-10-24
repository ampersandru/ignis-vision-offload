//! The 27B load the DFlash2 replay examples run on (GitHub #340): planned the
//! way the server plans its own (`ignis_server::runtime::qwen27b_load`) --
//! the leaf's lines at the load's options, the CUDA context and the measured
//! residual, and a KV pool of the rest of the VRAM budget -- then loaded the
//! way `CudaLeaf::load_model` loads, with every plan line logged and the
//! leaf's buffers read back against it.
//!
//! The shape follows the artifact's body. The trellis one runs
//! `rtx50-16gb`'s plan in the 16 GB class's budget, as a real 16 GB card
//! would (owner, 2026-10-24); the NVFP4 one runs the same engine shape --
//! context, chunk, rope, window, head -- on the 5090's derived budget at its
//! eight lanes, so a replay on both differs only in the body and the card.
//! Neither adapts: a plan that does not fit is refused, naming its shortfall.
//!
//! No retained slots: the replay never publishes or captures a prefix. The
//! server's host ones (`--reuse-retained-host`, 16) are pinned host memory,
//! no line of the VRAM plan, and its device ones default to none.

// Each including example takes what it needs.
#![allow(dead_code)]

use ignis_artifact::{bind_model_scope_27b_with, body_27b, materialize, Body27b, CudaDevice, MaterializedArtifact, Reader};
use ignis_core::model_load::{self, Model};
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::capture_decode_graphs;
use ignis_core::{
    plan_vram, ExceptionWeights, KvFormat, KvGeometry, ModelConfig, PrefillAttention, ProposalHead, Residency,
    RopeScaling, Speculation, SpeculativeBackend, VramLines, VramMode, VramPlan, VramRequest,
};
use ignis_runtime::{CudaLeafConfig, ReservedBytes};

/// The 16 GB class's VRAM budget (spec 27b-16gb/05), every byte the process
/// holds included.
pub const SIXTEEN_GIB: u64 = 17_179_869_184;
/// `vram.qwen38.headroom_bytes` of the built-in profiles, `1200M`.
pub const QWEN38_HEADROOM: u64 = 1200 << 20;

/// What a load is asked for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadShape {
    pub vram: VramMode,
    pub max_context: u32,
    /// The decode lanes: the pool's slots and the widest round.
    pub lanes: u32,
    pub prefill_chunk: u32,
    /// The DFlash2 window (full head); 0 for a plain load.
    pub window: u32,
    pub rope: RopeScaling,
}

impl LoadShape {
    /// The shape `body`'s replay runs at: `rtx50-16gb`'s plan in 16 GiB for
    /// the trellis body, and the same engine at eight lanes on the 5090's
    /// derived budget for the NVFP4 one.
    pub fn for_body(body: Body27b) -> Self {
        let sixteen_gb = Self {
            vram: VramMode::Explicit { budget_bytes: SIXTEEN_GIB, allow_oversubscription: false },
            max_context: 262_144,
            lanes: 2,
            prefill_chunk: 1024,
            window: 7,
            rope: RopeScaling::NONE,
        };
        match body {
            Body27b::Trellis => sixteen_gb,
            Body27b::Nvfp4 => Self { vram: VramMode::Derived { headroom_bytes: QWEN38_HEADROOM }, lanes: 8, ..sixteen_gb },
        }
    }

    /// [`LoadShape::for_body`], at the prefill chunk the server's own
    /// `IGNIS_MODEL_PREFILL_CHUNK` names, when it names one: on the NVFP4
    /// body only (its `rtx5090` profile serves 2048 since GitHub #341), since
    /// the trellis body loads as a 16 GB card does, at 1024, and is refused
    /// any other.
    pub fn from_env(body: Body27b) -> Result<Self, String> {
        let shape = Self::for_body(body);
        let Ok(raw) = std::env::var("IGNIS_MODEL_PREFILL_CHUNK") else {
            return Ok(shape);
        };
        let chunk: u32 = raw.trim().parse().map_err(|e| format!("IGNIS_MODEL_PREFILL_CHUNK `{raw}`: {e}"))?;
        if body == Body27b::Trellis && chunk != shape.prefill_chunk {
            return Err(format!(
                "IGNIS_MODEL_PREFILL_CHUNK {chunk}: the trellis body loads as a 16 GB card does, at {}",
                shape.prefill_chunk
            ));
        }
        Ok(Self { prefill_chunk: chunk, ..shape })
    }

    fn speculation(&self) -> Option<Speculation> {
        (self.window > 0).then(|| {
            Speculation::new(SpeculativeBackend::Dflash2, self.window)
                .unwrap_or_else(|e| panic!("window {}: {e}", self.window))
                .with_proposal_head(ProposalHead::Full)
        })
    }

    /// The leaf config the server's `leaf_config_for_shape` builds for this
    /// shape, with a KV pool of `kv_pool_bytes`.
    fn leaf_config(&self, kv_pool_bytes: u64) -> CudaLeafConfig {
        CudaLeafConfig {
            max_context_tokens: self.max_context,
            kv_format: KvFormat::HqE8_2b,
            kv_pool_bytes,
            slot_count: self.lanes,
            retained_device_slots: 0,
            retained_host_slots: 0,
            prefill_chunk_tokens: self.prefill_chunk,
            speculation: self.speculation(),
            vision: None,
            rope_scaling: self.rope,
            prefill_attention: PrefillAttention::default(),
        }
    }
}

/// A loaded model, its pool and what they stand on.
pub struct Loaded {
    pub model: Model,
    pub pool: SeqPool,
    pub plan: VramPlan,
    artifact: MaterializedArtifact,
    device: CudaDevice,
}

impl Loaded {
    /// Give the model, the pool and the weights back, in that order (the
    /// arena has no `Drop` of its own), so the next load starts from the
    /// memory this one took.
    pub fn release(self) {
        let Loaded { model, pool, mut artifact, mut device, .. } = self;
        drop(model);
        drop(pool);
        let _ = artifact.release_arena(&mut device);
    }
}

/// Plan `reader`'s load at `shape`, log the plan, and load it -- refusing a
/// plan its budget cannot hold, or a load whose buffers are not its plan's.
pub fn load(reader: &Reader, shape: LoadShape) -> Result<Loaded, String> {
    let body = body_27b(reader);
    // What is free before this process holds anything, read before the
    // context exists, as the server reads it: the context is a plan line. A
    // later load in the process adds back what it still holds (GitHub #324).
    let (nvml_free, _) = CudaDevice::nvml_memory(0).map_err(|e| format!("NVML: {e}"))?;
    let free_at_start =
        nvml_free + ignis_runtime::held_before_load(CudaDevice::context_created(), ignis_runtime::qwen27b_loaded());
    let mut device = CudaDevice::create(0).map_err(|e| format!("CUDA device: {e}"))?;
    let speculation = shape.speculation();
    let scope = model_load::model_scope(speculation, None, ExceptionWeights::default());
    let (bound, handles) = bind_model_scope_27b_with(reader, scope).map_err(|e| format!("bind: {e}"))?;

    let planning = shape.leaf_config(0);
    let reservations = planning.plan_reservations(reader, &bound, &handles)?;
    planning.kv_pool_arena_bytes(1)?;
    let kv_arena_bytes = |pages: u32| planning.kv_pool_arena_bytes(pages).unwrap_or(u64::MAX);
    let reserved = reservations.reserved;
    let lines = VramLines {
        weights: bound.device_capacity_bytes,
        cuda_context: ignis_runtime::CUDA_CONTEXT_BYTES,
        workspace: reserved.workspace,
        trellis_workspace: reserved.trellis_workspace,
        media_embedding: reserved.media_embedding,
        sampling: reserved.sampling,
        decode_graph: reserved.decode_graph,
        verify_round: reserved.verify_round,
        drafter_round: reserved.drafter_round,
        lane_state: reserved.lane_state,
        retained_slots: reserved.retained_slots,
        hq_residual_window: reserved.hq_residual_window,
        residual: ignis_runtime::LOAD_RESIDUAL_BYTES,
    };
    let geometry = KvGeometry::qwen38_27b();
    let format = planning.kv_format;
    let plan = plan_vram(&VramRequest {
        mode: shape.vram,
        free_at_start_bytes: free_at_start,
        lines,
        kv_page_bytes: format.page_bytes(geometry),
        max_context_tokens: shape.max_context,
        retained_slots: 0,
        kv_pool: None,
        residency: Residency::Resident,
        embedding_pool_named: false,
        kv_arena_bytes: &kv_arena_bytes,
        can_page: cfg!(windows),
    })
    .map_err(|e| format!("VRAM plan: {e}"))?;
    println!("load shape: {body:?} body, {shape:?}, kv {}, lines (bytes):", format.as_str());
    for (name, bytes) in plan.lines.entries() {
        println!("  {name:<20} {bytes:>14}");
    }
    println!(
        "  {:<20} {:>14} ({} pages, {} tokens)\n  total {} of a {} budget ({}; {} free at start)",
        "kv_pool",
        plan.kv_pool_bytes,
        plan.kv_page_count,
        plan.kv_token_capacity(),
        plan.total_bytes,
        plan.budget_bytes,
        plan.mode.as_str(),
        plan.free_at_start_bytes
    );
    for warning in &plan.warnings {
        println!("  warning: {warning}");
    }
    if plan.total_bytes > plan.budget_bytes {
        return Err(format!("the plan holds {} bytes, past its {}-byte budget", plan.total_bytes, plan.budget_bytes));
    }

    let artifact = materialize(reader, &bound, &mut device, None).map_err(|e| format!("materialize: {e}"))?;
    let config = shape.leaf_config(plan.kv_budget_bytes(format, geometry));
    let pages = config.kv_pool_plan().map_err(|e| e.to_string())?.page_count;
    let model = model_load::load_qwen38_27b_with_decode_lanes(
        reader,
        &artifact,
        &handles,
        config.prefill_chunk_tokens,
        config.max_context_tokens,
        config.kv_format,
        config.speculation,
        config.vision,
        config.rope_scaling,
        config.prefill_attention,
        config.slot_count,
    )
    .map_err(|e| format!("model load: {e}"))?;
    ignis_runtime::note_qwen27b_loaded();
    let pool = SeqPool::create_with_speculation(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: config.kv_format,
            kv_page_group_count: pages,
            max_context_tokens: config.max_context_tokens,
            slot_count: config.slot_count,
            retained_slot_count: config.retained_device_slots,
            retained_host_slot_count: config.retained_host_slots,
        },
        config.speculation.map(|s| s.backend()),
    )
    .map_err(|e| format!("seq pool: {e}"))?;
    let capture = capture_decode_graphs(&model, &pool).map_err(|e| format!("capture: {e}"))?;
    println!("decode graphs: {} of {} widths ready in {} us", capture.ready_count(), shape.lanes, capture.capture_micros);

    // What the leaf built, line by line, against what the plan laid out: the
    // server refuses a start where the two differ, and so does this.
    let model_lines = model.stats().reserved;
    let pool_stats = pool.stats();
    let built = ReservedBytes {
        workspace: model_lines.workspace_bytes,
        trellis_workspace: model_lines.trellis_workspace_bytes,
        media_embedding: model_lines.media_embedding_bytes,
        sampling: model_lines.sampling_bytes,
        decode_graph: model_lines.decode_graph_bytes,
        verify_round: model_lines.verify_round_bytes,
        drafter_round: model_lines.drafter_round_bytes,
        lane_state: pool_stats.lane_state_bytes,
        retained_slots: pool_stats.retained_state_bytes,
        hq_residual_window: pool_stats.hq_residual_bytes,
        kv_pool: pool_stats.kv_arena_bytes,
    };
    let planned = ReservedBytes { kv_pool: plan.kv_pool_bytes, ..reserved };
    if built != planned {
        return Err(format!("the load holds {built:?} beside its weights, not the {planned:?} its plan laid out"));
    }
    if let Ok((free_after, _)) = CudaDevice::nvml_memory(0) {
        println!(
            "loaded: {} bytes planned, NVML free {nvml_free} -> {free_after} ({} taken)",
            plan.total_bytes,
            nvml_free.saturating_sub(free_after)
        );
    }
    Ok(Loaded { model, pool, plan, artifact, device })
}
