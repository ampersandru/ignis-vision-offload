//! Expert residency's device side (`kernel/include/ignis_residency.h`,
//! `crates/core/src/residency/device.rs`) against the CPU policy model it
//! implements (spec flash-next/03, acceptance 4, GitHub #301).
//!
//! - The leaf's own plan line for the tables stays within the CPU plan's
//!   upper bound, its descriptor refuses a prefill width past the decode
//!   width, and its layout is the Rust mirror's (host arithmetic, no GPU:
//!   runs with `--features cuda`).
//! - On random traces -- decode of one to three lanes, prefill chunks, a
//!   lookahead with holes, a budget or none, a warm start, refused steps,
//!   forwards restarted at a layer --
//!   every step's outcome on the device equals `ResidencyModel::step`'s, and
//!   the counters agree at the end. Half the steps hand the lookahead over as
//!   the next router's logits, so the device's own ranking (BF16-rounded,
//!   ties to the lower id) is held to the CPU's. The same again with a pinned
//!   hot set beside the host pool (GitHub #329 item 12): the device's pinned
//!   keys against `ResidencyModel::pin`.
//! - The committed recorded trace (`kernel/tests/fixtures/residency/trace_v1.txt`)
//!   replayed with its favourite experts pinned, step for step against
//!   `ResidencyModel::pin` (acceptance 4 on a recorded trace, with pins).
//!
//! The kernel leaf's residency CTest replays the committed fixture unpinned
//! with the copy and slot-table checks; this test covers what one fixed
//! trace cannot.
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing GPU is a skip, under it a hard failure (`ignis_core::gpu_profile`).

#![cfg(feature = "cuda")]

use std::os::raw::c_void;

use ignis_artifact::{CudaDevice, Device, DeviceBuffer};
use ignis_core::gpu_profile;
use ignis_core::residency::device::{self, DeviceReport, DeviceResidency, NO_BUDGET, ResidencyDesc};
use ignis_core::residency::{
    ExpertCatalog, KBits, KClass, LayerStep, Phase, PolicyConfig, PrefetchBudget, ProjectionId,
    Projection, ResidencyModel, StepError, StepOutcome, residency_table_bytes,
};

const TOP_K: usize = 10;

#[test]
fn the_leaf_s_tables_line_stays_within_the_cpu_plan_s_upper_bound() {
    // Flash-Next's geometry, every projection given a slot (the bound's own
    // worst case), an 8192-token chunk and W = 16.
    let record_bytes = [827_392, 1_032_192, 1_236_992, 1_646_592, 417_792, 520_192, 622_592, 827_392];
    let desc = ResidencyDesc {
        layers: 48,
        experts: 512,
        capacity: [6144; 8],
        record_bytes,
        max_tokens: 8192,
        lookahead_width: 16,
        prefill_lookahead_width: 10,
        prefetch_budget_one_row_bytes: NO_BUDGET,
        prefetch_budget_per_row_bytes: 0,
        staging_half_bytes: 800_000_000,
        host_pool_bytes: 38_000_000_000,
        copy_blocks: 16,
        report: 0,
        pinned: [0; 8],
    };
    let plan = device::plan_bytes(&desc).expect("plan");
    let bound = residency_table_bytes(48, 512, 8192, 16);
    assert!(plan.tables <= bound, "leaf {} B, CPU bound {bound} B", plan.tables);
    assert!(plan.tables * 10 >= bound * 8, "a loose bound: leaf {} B, CPU {bound} B", plan.tables);
    let round = |b: u64| b.div_ceil(256) * 256;
    let pools: u64 = record_bytes.iter().map(|&b| round(6144 * b)).sum();
    assert_eq!(plan.pools, pools);
    assert_eq!(plan.staging, round(1_600_000_000));
    assert_eq!(plan.total, plan.pools + plan.staging + plan.tables);
}

#[test]
fn a_prefill_width_past_the_decode_width_is_refused_by_name() {
    // The ranked lookahead's scratch is sized for the decode width.
    let desc = ResidencyDesc {
        layers: 2,
        experts: 16,
        capacity: [4; 8],
        record_bytes: [4096; 8],
        max_tokens: 4,
        lookahead_width: 3,
        prefill_lookahead_width: 4,
        prefetch_budget_one_row_bytes: NO_BUDGET,
        prefetch_budget_per_row_bytes: 0,
        staging_half_bytes: 4096 * 32,
        host_pool_bytes: 4096 * 64,
        copy_blocks: 4,
        report: 0,
        pinned: [0; 8],
    };
    let refused = device::plan_bytes(&desc).expect_err("a prefill width past the decode width");
    assert!(refused.contains("prefill_lookahead_width exceeds lookahead_width"), "{refused}");
    device::plan_bytes(&ResidencyDesc { prefill_lookahead_width: 3, ..desc }).expect("equal widths");
}

/// `struct ignis_residency_desc` has no `size` field, so both sides pin its
/// layout: kernel/include/ignis_residency.h static_asserts the same.
#[test]
fn the_residency_descriptor_is_the_leaf_s_layout() {
    use std::mem::{offset_of, size_of};
    assert_eq!(offset_of!(ResidencyDesc, prefill_lookahead_width), 112);
    assert_eq!(offset_of!(ResidencyDesc, prefetch_budget_one_row_bytes), 120);
    assert_eq!(offset_of!(ResidencyDesc, prefetch_budget_per_row_bytes), 128);
    assert_eq!(offset_of!(ResidencyDesc, copy_blocks), 152);
    assert_eq!(offset_of!(ResidencyDesc, pinned), 160);
    assert_eq!(size_of::<ResidencyDesc>(), 192);
}

/// GitHub #329 item 12: a class's pinned slots are pool bytes past its LRU's
/// in the leaf's plan line.
#[test]
fn the_leaf_s_pools_line_holds_the_pinned_slots() {
    let desc = ResidencyDesc {
        layers: 2,
        experts: 16,
        capacity: [4; 8],
        record_bytes: [4096; 8],
        max_tokens: 4,
        lookahead_width: 3,
        prefill_lookahead_width: 3,
        prefetch_budget_one_row_bytes: NO_BUDGET,
        prefetch_budget_per_row_bytes: 0,
        staging_half_bytes: 4096 * 32,
        host_pool_bytes: 4096 * 64,
        copy_blocks: 4,
        report: 0,
        pinned: [0; 8],
    };
    let unpinned = device::plan_bytes(&desc).expect("plan").pools;
    let pinned = device::plan_bytes(&ResidencyDesc { pinned: [3, 0, 0, 0, 1, 0, 0, 0], ..desc }).expect("plan").pools;
    assert_eq!(unpinned, 8 * 4 * 4096);
    assert_eq!(pinned, unpinned + 4 * 4096);
}

#[test]
fn a_policy_budget_becomes_the_descriptor_s_two_words() {
    assert_eq!(device::budget_words(None), (NO_BUDGET, 0));
    let budget = PrefetchBudget { one_row_bytes: 1_172_500, per_row_bytes: 726_250 };
    assert_eq!(device::budget_words(Some(budget)), (1_172_500, 726_250));
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// f32 rounded to BF16 (nearest, ties to even) and back, as the router ranks.
fn bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

/// The router's ranking of one row of logits: BF16-rounded, ties to the
/// lower id, best first.
fn rank(logits: &[f32], width: usize) -> Vec<u16> {
    let mut order: Vec<u16> = (0..logits.len() as u16).collect();
    order.sort_by(|&a, &b| {
        bf16(logits[usize::from(b)])
            .total_cmp(&bf16(logits[usize::from(a)]))
            .then(a.cmp(&b))
    });
    order.truncate(width);
    order
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn ptr(buf: &DeviceBuffer) -> *mut c_void {
    buf.base_ptr() as *mut c_void
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn the_device_steps_equal_the_policy_model_on_random_traces() {
    if let Some(steps) = replay_random_traces(false) {
        println!("{steps} steps equal on the device and in the policy model");
    }
}

/// GitHub #329 item 12: the same with a pinned hot set beside the host pool
/// -- the routing's favourite experts whole, and a few projections alone.
#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn the_device_steps_equal_the_policy_model_with_pinned_keys() {
    if let Some(steps) = replay_random_traces(true) {
        println!("{steps} steps with pinned keys equal on the device and in the policy model");
    }
}

/// One step's report on the device against the policy model's outcome.
/// Returns the pinned keys among its hits.
fn assert_step_equal(report: &DeviceReport, outcome: Result<StepOutcome, StepError>, pinned: &[ProjectionId], at: &str) -> usize {
    match outcome {
        Ok(o) => {
            assert_eq!(report.status, 0, "{at}");
            assert_eq!(report.hits, o.hits, "{at}: hits");
            assert_eq!(report.prefetch_hits, o.prefetch_hits, "{at}: prefetch hits");
            assert_eq!(report.misses, o.misses, "{at}: misses");
            assert_eq!(report.evictions, o.evictions, "{at}: evictions");
            assert_eq!(report.prefetches, o.prefetches, "{at}: prefetches");
            assert_eq!(report.prefetch_dropped, o.prefetch_dropped, "{at}: dropped");
            assert_eq!(report.bytes_moved, o.bytes_moved, "{at}: bytes");
            o.hits.iter().filter(|id| pinned.contains(id)).count()
        }
        Err(StepError::NoEvictableSlot { class, .. }) => {
            assert_eq!(report.status, 1 + class.index() as u32, "{at}: refused");
            0
        }
        Err(e) => panic!("{at}: {e}"),
    }
}

/// The device's counters against the model's at the end of a replay. The
/// device times its demand copies; the model has no clock. A phase that
/// copied a miss in waited for it, one that copied none did not.
fn assert_counters_equal(gpu: &DeviceResidency, model: &ResidencyModel, at: &str) {
    let mut counters = gpu.counters().expect("counters");
    for phase in [Phase::Decode, Phase::Prefill] {
        let copied = KClass::ALL.iter().any(|c| model.counters().misses[c.index()][phase.index()] > 0);
        assert_eq!(counters.stall_nanos[phase.index()] > 0, copied, "{at}: {phase:?} stall");
    }
    counters.stall_nanos = [0, 0];
    assert_eq!(&counters, model.counters(), "{at}: counters");
}

/// Spec flash-next/03 acceptance 4 on a recorded trace, with a pinned hot set
/// (GitHub #329 item 12, review 2026-10-18): the committed fixture the leaf's
/// residency CTest replays unpinned, through the device with the trace's
/// favourite experts pinned out of the pools' slots (as a load pins them),
/// step for step against `ResidencyModel::pin`. The fixture's expectations
/// are the unpinned model's, so only its inputs are read.
#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn the_recorded_trace_replays_pinned_as_in_the_policy_model() {
    if let Some((steps, pinned_hits)) = replay_recorded_trace_pinned() {
        println!("{steps} recorded steps with pinned keys equal on the device and in the policy model ({pinned_hits} pinned hits)");
    }
}

/// One `step` record of the fixture: its selection and its lookahead rows
/// (`-1` a hole), as the CTest uploads them.
struct RecordedStep {
    layer: u16,
    prefill: bool,
    tokens: usize,
    ids: Vec<i32>,
    rows: usize,
    stride: usize,
    lookahead: Vec<i32>,
}

/// The inputs of `kernel/tests/fixtures/residency/trace_v1.txt`
/// (`crates/core/tests/expert_residency_fixture.rs` documents the format).
struct RecordedTrace {
    catalog: ExpertCatalog,
    record_bytes: [u64; 8],
    capacity: [u32; 8],
    width: usize,
    prefill_width: usize,
    budget: Option<PrefetchBudget>,
    warm: Vec<ProjectionId>,
    steps: Vec<RecordedStep>,
}

fn read_recorded_trace() -> RecordedTrace {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernel/tests/fixtures/residency/trace_v1.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (mut layers, mut experts) = (0u16, 0u16);
    let (mut record_bytes, mut capacity) = ([0u64; 8], [0u32; 8]);
    let (mut width, mut prefill_width, mut budget) = (0, 0, None);
    let (mut k2, mut warm_keys, mut steps) = (Vec::new(), Vec::new(), Vec::new());
    for line in text.lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        let n = |i: usize| -> u64 { w[i].parse().unwrap_or_else(|e| panic!("{line}: {e}")) };
        match w.first().copied() {
            Some("ignis-residency-fixture") => assert_eq!(n(1), 3, "fixture version"),
            Some("layers") => {
                (layers, experts) = (n(1) as u16, n(3) as u16);
                assert_eq!(n(5), TOP_K as u64, "top_k");
            }
            Some("record_bytes") => record_bytes = std::array::from_fn(|i| n(1 + i)),
            Some("capacity") => capacity = std::array::from_fn(|i| n(1 + i) as u32),
            Some("width") => {
                (width, prefill_width) = (n(1) as usize, n(3) as usize);
                budget = (n(5) != u64::MAX).then(|| PrefetchBudget { one_row_bytes: n(5), per_row_bytes: n(7) });
            }
            Some("k2") => k2 = (1..w.len()).map(|i| n(i) as u32).collect(),
            Some("warm") => warm_keys = (2..w.len()).map(|i| n(i) as u32).collect(),
            Some("step") => {
                let tokens = n(3) as usize;
                let ids: Vec<i32> = (4..4 + tokens * TOP_K).map(|i| n(i) as i32).collect();
                let at = 4 + tokens * TOP_K;
                let (rows, stride) = (n(at) as usize, n(at + 1) as usize);
                let lookahead = w[at + 2..at + 2 + rows * stride].iter().map(|v| v.parse().expect("lookahead")).collect();
                steps.push(RecordedStep { layer: n(1) as u16, prefill: n(2) == 1, tokens, ids, rows, stride, lookahead });
            }
            _ => {}
        }
    }
    let k_of = |half_bits: u32| *KBits::ALL.iter().find(|k| k.half_bits() == half_bits).expect("a K the catalog knows");
    let map = k2.chunks_exact(2).map(|pair| (k_of(pair[0]), k_of(pair[1]))).collect();
    let catalog = ExpertCatalog::new(layers, experts, map, record_bytes).expect("the fixture's catalog");
    let id_of = |key: u32| {
        let expert = key / 2;
        ProjectionId::new((expert / u32::from(experts)) as u16, (expert % u32::from(experts)) as u16, Projection::ALL[(key % 2) as usize])
    };
    let warm = warm_keys.iter().map(|&key| id_of(key)).collect();
    RecordedTrace { catalog, record_bytes, capacity, width, prefill_width, budget, warm, steps }
}

/// The recorded trace through the device and the policy model with a pinned
/// hot set: the steps run and the pinned keys they hit, or `None` on a skip.
fn replay_recorded_trace_pinned() -> Option<(usize, usize)> {
    let mut dev = match CudaDevice::create(0) {
        Ok(d) => d,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("no CUDA device: {e}"));
            return None;
        }
    };
    let trace = read_recorded_trace();
    let catalog = &trace.catalog;
    let experts = catalog.experts();
    // The trace routes the low experts most (a quarter of the ids two times
    // in three): pin experts 0-2 of every layer whole, as long as each class
    // keeps at least half its slots for the LRU.
    let mut pinned: Vec<ProjectionId> = Vec::new();
    let mut taken = [0u32; 8];
    for expert in 0..3u16 {
        for layer in 0..catalog.layers() {
            let both = Projection::ALL.map(|p| ProjectionId::new(layer, expert, p));
            let mut after = taken;
            for id in both {
                after[catalog.class_of(id).index()] += 1;
            }
            if (0..8).all(|c| after[c] <= trace.capacity[c] / 2) {
                taken = after;
                pinned.extend(both);
            }
        }
    }
    assert!(pinned.len() >= 8, "a pinned set to test: {pinned:?}");
    // A load pins out of the pools' slots: the LRU keeps the rest.
    let lru: [u32; 8] = std::array::from_fn(|c| trace.capacity[c] - taken[c]);
    let mut model = ResidencyModel::new(
        catalog.clone(),
        PolicyConfig {
            capacity: lru,
            prefetch_width: trace.width,
            prefill_prefetch_width: trace.prefill_width,
            prefetch_budget: trace.budget,
        },
    );
    assert_eq!(model.pin(&pinned), pinned.len());
    assert_eq!(model.pinned_counts(), taken);
    let (k2, offsets, pool_bytes) = device::packed_layout_without(catalog, |id| pinned.contains(&id));
    let heaviest = (0..catalog.layers()).map(|l| catalog.layer_bytes(l)).max().unwrap();
    let max_rows = trace.steps.iter().map(|s| s.tokens.max(s.rows)).max().unwrap();
    let (one_row, per_row) = device::budget_words(trace.budget);
    let desc = ResidencyDesc {
        layers: u32::from(catalog.layers()),
        experts: u32::from(experts),
        capacity: lru,
        record_bytes: trace.record_bytes,
        max_tokens: max_rows as u32,
        lookahead_width: trace.width as u32,
        prefill_lookahead_width: trace.prefill_width as u32,
        prefetch_budget_one_row_bytes: one_row,
        prefetch_budget_per_row_bytes: per_row,
        staging_half_bytes: heaviest,
        host_pool_bytes: pool_bytes,
        copy_blocks: 4,
        report: 1,
        pinned: model.pinned_counts(),
    };
    let mut gpu = match DeviceResidency::new(&desc, &k2, &offsets) {
        Ok(g) => g,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("ignis_residency_create: {e}"));
            return None;
        }
    };
    for (i, b) in gpu.host_pool_mut().expect("host pool").iter_mut().enumerate() {
        *b = (i * 7 + 3) as u8;
    }
    // The pinned records, from a buffer of their own: the pool has no copy.
    let keys: Vec<u32> = pinned.iter().map(|&id| device::key(experts, id)).collect();
    let (mut src, mut at) = (Vec::new(), Vec::new());
    for &id in &pinned {
        at.push(src.len() as u64);
        src.extend((0..catalog.bytes(id)).map(|i| (i * 5 + 1) as u8));
    }
    gpu.admit(&keys, &src, &at).expect("admit the pinned keys");
    gpu.admit_finish().expect("finish the pinned admission");
    assert_eq!(gpu.warm_start(&trace.warm).expect("warm start"), model.warm_start(&trace.warm), "warm start");
    assert_eq!(gpu.slots_in_use().expect("occupancy"), model.occupancy(), "warm occupancy");

    let widest_ranking = trace.steps.iter().map(|s| s.rows * s.stride).max().unwrap();
    let ids_buf = dev.allocate((max_rows * TOP_K * 4) as u64).expect("ids");
    let look_buf = dev.allocate((widest_ranking.max(1) * 4) as u64).expect("lookahead");
    let mut pinned_hits = 0;
    for (i, s) in trace.steps.iter().enumerate() {
        dev.copy_h2d(&ids_buf, 0, as_bytes(&s.ids)).expect("ids upload");
        if s.rows > 0 {
            dev.copy_h2d(&look_buf, 0, as_bytes(&s.lookahead)).expect("lookahead upload");
        }
        dev.synchronize().expect("uploads");
        let phase = if s.prefill { Phase::Prefill } else { Phase::Decode };
        let result = unsafe {
            if s.rows == 0 {
                gpu.step(u32::from(s.layer), phase, ptr(&ids_buf) as *const i32, s.tokens as u32, std::ptr::null(), std::ptr::null_mut())
            } else {
                gpu.step_ranked(
                    u32::from(s.layer),
                    phase,
                    ptr(&ids_buf) as *const i32,
                    s.tokens as u32,
                    ptr(&look_buf) as *const i32,
                    s.rows as u32,
                    s.stride as u32,
                    std::ptr::null_mut(),
                )
            }
        };
        if let Err(e) = result {
            gpu_profile::skip_or_fail(&format!("ignis_residency_step: {e}"));
            return None;
        }
        let report = gpu.last_report(u32::from(s.layer)).expect("report");
        let selected: Vec<u16> = s.ids.iter().map(|&e| e as u16).collect();
        // The model takes a lane's ranking without its holes, as the device's walk skips them.
        let lanes: Vec<Vec<u16>> = s
            .lookahead
            .chunks(s.stride.max(1))
            .map(|row| row.iter().filter(|&&e| e >= 0).map(|&e| e as u16).collect())
            .collect();
        let lane_refs: Vec<&[u16]> = lanes.iter().map(Vec::as_slice).collect();
        let step = if s.prefill { LayerStep::prefill(s.layer, &selected) } else { LayerStep::decode(s.layer, &selected) };
        let at = format!("recorded step {i} (layer {})", s.layer);
        pinned_hits += assert_step_equal(&report, model.step(&step.lookahead(&lane_refs)), &pinned, &at);
        assert_eq!(gpu.slots_in_use().expect("occupancy"), model.occupancy(), "{at}: occupancy");
    }
    assert_counters_equal(&gpu, &model, "recorded trace");
    assert!(pinned_hits > 0, "the trace selected no pinned key: the replay proves nothing about pins");
    Some((trace.steps.len(), pinned_hits))
}

/// Random traces through the device and the policy model, step for step,
/// with a pinned hot set when `pin`. The steps run, or `None` on a skip.
fn replay_random_traces(pin: bool) -> Option<usize> {
    let mut dev = match CudaDevice::create(0) {
        Ok(d) => d,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("no CUDA device: {e}"));
            return None;
        }
    };
    const LAYERS: u16 = 3;
    const EXPERTS: u16 = 32;
    const WIDTH: usize = 3;
    // A chunk's own, narrower width: the device must take it from the same rankings.
    const PREFILL_WIDTH: usize = 2;
    const MAX_ROWS: usize = 8;
    let record_bytes = [4096, 8192, 8192, 12288, 4096, 4096, 8192, 8192];
    let ids_buf = dev.allocate((MAX_ROWS * TOP_K * 4) as u64).expect("ids");
    let look_buf = dev.allocate((MAX_ROWS * usize::from(EXPERTS) * 4) as u64).expect("lookahead");
    let (mut steps_run, mut pinned_hits) = (0, 0);

    for seed in 0..6u64 {
        let mut rng = Rng(1000 + seed);
        let map = (0..usize::from(LAYERS) * usize::from(EXPERTS))
            .map(|_| (KBits::ALL[rng.below(4) as usize], KBits::ALL[rng.below(4) as usize]))
            .collect();
        let catalog = ExpertCatalog::new(LAYERS, EXPERTS, map, record_bytes).expect("catalog");
        let capacity: [u32; 8] = std::array::from_fn(|_| 6 + rng.below(20) as u32);
        // Even seeds budget the decode steps, by their rows on two of the three (GitHub #306).
        let budget = (seed % 2 == 0).then(|| PrefetchBudget {
            one_row_bytes: 10_000 + rng.below(30_000),
            per_row_bytes: if seed == 0 { 0 } else { 2_000 + rng.below(10_000) },
        });
        let mut model = ResidencyModel::new(
            catalog.clone(),
            PolicyConfig { capacity, prefetch_width: WIDTH, prefill_prefetch_width: PREFILL_WIDTH, prefetch_budget: budget },
        );
        // The pinned hot set: the routing's favourite experts (it picks ids
        // under 12 two times in three) whole, and a few projections alone.
        let mut pinned = Vec::new();
        if pin {
            for l in 0..LAYERS {
                for e in 0..2u16 {
                    pinned.extend(Projection::ALL.map(|p| ProjectionId::new(l, e, p)));
                }
            }
            for _ in 0..4 {
                let layer = rng.below(u64::from(LAYERS)) as u16;
                let expert = 2 + rng.below(u64::from(EXPERTS) - 2) as u16;
                let id = ProjectionId::new(layer, expert, Projection::ALL[rng.below(2) as usize]);
                if !pinned.contains(&id) {
                    pinned.push(id);
                }
            }
        }
        assert_eq!(model.pin(&pinned), pinned.len(), "seed {seed}");
        let (k2, offsets, pool_bytes) = device::packed_layout_without(&catalog, |id| pinned.contains(&id));
        let heaviest = (0..LAYERS).map(|l| catalog.layer_bytes(l)).max().unwrap();
        let (one_row, per_row) = device::budget_words(budget);
        let desc = ResidencyDesc {
            layers: u32::from(LAYERS),
            experts: u32::from(EXPERTS),
            capacity,
            record_bytes,
            max_tokens: MAX_ROWS as u32,
            lookahead_width: WIDTH as u32,
            prefill_lookahead_width: PREFILL_WIDTH as u32,
            prefetch_budget_one_row_bytes: one_row,
            prefetch_budget_per_row_bytes: per_row,
            staging_half_bytes: heaviest,
            host_pool_bytes: pool_bytes,
            copy_blocks: 4,
            report: 1,
            pinned: model.pinned_counts(),
        };
        let mut gpu = match DeviceResidency::new(&desc, &k2, &offsets) {
            Ok(g) => g,
            Err(e) => {
            gpu_profile::skip_or_fail(&format!("ignis_residency_create: {e}"));
            return None;
        }
        };
        for (i, b) in gpu.host_pool_mut().expect("host pool").iter_mut().enumerate() {
            *b = (i * 7 + 3) as u8;
        }
        if !pinned.is_empty() {
            // The pinned records, from a buffer of their own: the pool has no copy.
            let keys: Vec<u32> = pinned.iter().map(|&id| device::key(EXPERTS, id)).collect();
            let (mut src, mut at) = (Vec::new(), Vec::new());
            for &id in &pinned {
                at.push(src.len() as u64);
                src.extend((0..catalog.bytes(id)).map(|i| (i * 5 + 1) as u8));
            }
            gpu.admit(&keys, &src, &at).expect("admit the pinned keys");
            gpu.admit_finish().expect("finish the pinned admission");
        }
        let warm: Vec<ProjectionId> = (0..6u16)
            .flat_map(|e| (0..LAYERS).map(move |l| (l, e)))
            .flat_map(|(l, e)| Projection::ALL.map(|p| ProjectionId::new(l, e, p)))
            .collect();
        assert_eq!(gpu.warm_start(&warm).expect("warm start"), model.warm_start(&warm), "seed {seed}");
        assert_eq!(gpu.slots_in_use().expect("occupancy"), model.occupancy(), "seed {seed}: warm occupancy");

        for round in 0..30 {
            let prefill = round % 4 == 3;
            let rows = if prefill { 4 + rng.below(5) as usize } else { 1 + rng.below(3) as usize };
            // Now and then a forward restarted at a layer: the same layer twice in a row.
            let mut order = Vec::new();
            for layer in 0..LAYERS {
                order.push(layer);
                if rng.below(8) == 0 {
                    order.push(layer);
                }
            }
            for layer in order {
                let mut ids = Vec::with_capacity(rows * TOP_K);
                for _ in 0..rows {
                    let mut row: Vec<u16> = Vec::new();
                    while row.len() < TOP_K {
                        let e = if rng.below(3) == 0 { rng.below(u64::from(EXPERTS)) } else { rng.below(12) } as u16;
                        if !row.contains(&e) {
                            row.push(e);
                        }
                    }
                    ids.extend(row.iter().map(|&e| i32::from(e)));
                }
                dev.copy_h2d(&ids_buf, 0, as_bytes(&ids)).expect("ids upload");
                let selected: Vec<u16> = ids.iter().map(|&e| e as u16).collect();
                let last = layer + 1 == LAYERS;
                let by_logits = rng.below(2) == 0;
                // The CPU model's lanes, and what the device is given.
                let lanes: Vec<Vec<u16>> = if last {
                    Vec::new()
                } else if by_logits {
                    // A few distinct values, so BF16 ties are common.
                    let logits: Vec<f32> = (0..rows * usize::from(EXPERTS))
                        .map(|_| (rng.below(9) as f32 - 4.0) * 0.375 + (rng.below(3) as f32) * 1e-4)
                        .collect();
                    dev.copy_h2d(&look_buf, 0, as_bytes(&logits)).expect("logits upload");
                    logits.chunks(usize::from(EXPERTS)).map(|row| rank(row, WIDTH)).collect()
                } else {
                    let stride = WIDTH + 1;
                    let mut raw = vec![-1i32; rows * stride];
                    let mut lanes = Vec::new();
                    for r in 0..rows {
                        let mut lane = Vec::new();
                        for j in 0..stride {
                            if rng.below(6) == 0 {
                                continue; // a hole
                            }
                            let e = rng.below(u64::from(EXPERTS)) as u16;
                            raw[r * stride + j] = i32::from(e);
                            lane.push(e);
                        }
                        lanes.push(lane);
                    }
                    dev.copy_h2d(&look_buf, 0, as_bytes(&raw)).expect("lookahead upload");
                    lanes
                };
                dev.synchronize().expect("uploads");
                let phase = if prefill { Phase::Prefill } else { Phase::Decode };
                let result = unsafe {
                    if last {
                        gpu.step(u32::from(layer), phase, ptr(&ids_buf) as *const i32, rows as u32, std::ptr::null(), std::ptr::null_mut())
                    } else if by_logits {
                        gpu.step(
                            u32::from(layer),
                            phase,
                            ptr(&ids_buf) as *const i32,
                            rows as u32,
                            ptr(&look_buf) as *const f32,
                            std::ptr::null_mut(),
                        )
                    } else {
                        gpu.step_ranked(
                            u32::from(layer),
                            phase,
                            ptr(&ids_buf) as *const i32,
                            rows as u32,
                            ptr(&look_buf) as *const i32,
                            rows as u32,
                            (WIDTH + 1) as u32,
                            std::ptr::null_mut(),
                        )
                    }
                };
                if let Err(e) = result {
                    gpu_profile::skip_or_fail(&format!("ignis_residency_step: {e}"));
                    return None;
                }
                let report = gpu.last_report(u32::from(layer)).expect("report");
                let lane_refs: Vec<&[u16]> = lanes.iter().map(Vec::as_slice).collect();
                let step = if prefill { LayerStep::prefill(layer, &selected) } else { LayerStep::decode(layer, &selected) };
                let at = format!("seed {seed} round {round} layer {layer}");
                pinned_hits += assert_step_equal(&report, model.step(&step.lookahead(&lane_refs)), &pinned, &at);
                assert_eq!(gpu.slots_in_use().expect("occupancy"), model.occupancy(), "{at}: occupancy");
                steps_run += 1;
            }
        }
        assert_counters_equal(&gpu, &model, &format!("seed {seed}"));
    }
    // The pinned replay must select its pinned keys, or it says nothing about them.
    assert_eq!(pinned_hits > 0, pin, "{pinned_hits} pinned hits");
    Some(steps_run)
}
