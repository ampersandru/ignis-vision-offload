//! A device-only expert residency's Rust side (`host_pool_bytes` 0,
//! ADR 0054, GitHub #329): no host pool and no staging ring, the three refusals `admit` makes
//! before the FFI, `admit_finish` refusing while a projection is missing, and
//! a step that hits once every projection is admitted.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing GPU is a skip, under it a hard failure (`ignis_core::gpu_profile`).

#![cfg(feature = "cuda")]

use std::os::raw::c_void;

use ignis_artifact::{CudaDevice, Device};
use ignis_core::gpu_profile;
use ignis_core::moe;
use ignis_core::residency::device::{plan_bytes, DeviceResidency, NO_BUDGET, ResidencyDesc};
use ignis_core::residency::{KClass, Phase};

const LAYERS: u32 = 2;
const EXPERTS: u32 = 16;
const KEYS: u32 = LAYERS * EXPERTS * 2;
const TOP_K: usize = 10;
const RECORD_BYTES: [u64; KClass::COUNT] = [4096, 8192, 8192, 12288, 4096, 4096, 8192, 8192];

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// The K map (`k2`: 4, 5, 6 or 8) and the class of every key.
fn k_map() -> (Vec<u8>, Vec<usize>) {
    let k2: Vec<u8> = (0..KEYS).map(|key| [4u8, 5, 6, 8][(key / 2 % 4) as usize]).collect();
    let classes = (0..KEYS)
        .map(|key| {
            let k = match k2[key as usize] {
                4 => 0,
                5 => 1,
                6 => 2,
                _ => 3,
            };
            (key & 1) as usize * 4 + k
        })
        .collect();
    (k2, classes)
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn a_device_only_residency_admits_every_projection_then_hits() {
    let mut dev = match CudaDevice::create(0) {
        Ok(d) => d,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("no CUDA device: {e}"));
            return;
        }
    };
    let (k2, classes) = k_map();
    let mut capacity = [0u32; KClass::COUNT];
    for &c in &classes {
        capacity[c] += 1;
    }
    // No staging ring, as the load's desc (GitHub #329 item 2): no step
    // misses, so nothing stages.
    let desc = ResidencyDesc {
        layers: LAYERS,
        experts: EXPERTS,
        capacity,
        record_bytes: RECORD_BYTES,
        max_tokens: 4,
        lookahead_width: 2,
        prefill_lookahead_width: 2,
        prefetch_budget_one_row_bytes: NO_BUDGET,
        prefetch_budget_per_row_bytes: 0,
        staging_half_bytes: 0,
        host_pool_bytes: 0,
        copy_blocks: 4,
        report: 1,
    };
    // A device-only residency takes no pool offsets.
    let mut gpu = match DeviceResidency::new(&desc, &k2, &[]) {
        Ok(g) => g,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("ignis_residency_create: {e}"));
            return;
        }
    };
    assert!(gpu.host_pool_mut().is_none(), "a device-only residency has no host pool");
    assert_eq!(plan_bytes(&desc).expect("plan").staging, 0, "nor a ring in its plan");
    assert_eq!(gpu.layout().expect("layout").ring_bytes, 0, "nor one allocated");

    // Every record at its own offset in one source buffer.
    let mut offsets = Vec::with_capacity(KEYS as usize);
    let mut total = 0u64;
    for &c in &classes {
        offsets.push(total);
        total += RECORD_BYTES[c];
    }
    let src: Vec<u8> = (0..total).map(|i| (i * 7 + 3) as u8).collect();
    let keys: Vec<u32> = (0..KEYS).collect();

    // The three refusals made before the FFI.
    let e = gpu.admit(&keys[..4], &src, &offsets[..3]).expect_err("keys and offsets of different lengths");
    assert!(e.contains("4 keys and 3 offsets"), "{e}");
    let e = gpu.admit(&[KEYS], &src, &[0]).expect_err("a key outside the catalog");
    assert!(e.contains("outside the 64-key catalog"), "{e}");
    let last = KEYS as usize - 1;
    let e = gpu.admit(&[KEYS - 1], &src, &[total - RECORD_BYTES[classes[last]] + 1]).expect_err("a record past its buffer");
    assert!(e.contains("runs past its"), "{e}");

    // Half the keys admitted: the admission cannot be closed.
    let half = KEYS as usize / 2;
    gpu.admit(&keys[..half], &src, &offsets[..half]).expect("first half");
    let e = gpu.admit_finish().expect_err("projections still missing");
    assert!(e.contains("never admitted"), "{e}");

    // A refused finish changes nothing: the rest is admitted and it closes.
    gpu.admit(&keys[half..], &src, &offsets[half..]).expect("second half");
    gpu.admit_finish().expect("every projection admitted");
    assert_eq!(
        gpu.slots_in_use().expect("occupancy"),
        capacity,
        "every slot of every class holds a projection"
    );

    // A step hits: the first ten experts of layer 0, decode, one row.
    let ids: Vec<i32> = (0..TOP_K as i32).collect();
    let ids_buf = dev.allocate((TOP_K * 4) as u64).expect("ids");
    dev.copy_h2d(&ids_buf, 0, as_bytes(&ids)).expect("ids upload");
    dev.synchronize().expect("upload");
    moe::stream_sync(std::ptr::null_mut()).expect("legacy stream");
    let result = unsafe {
        gpu.step(0, Phase::Decode, ids_buf.base_ptr() as *const i32, 1, std::ptr::null(), std::ptr::null_mut::<c_void>())
    };
    if let Err(e) = result {
        gpu_profile::skip_or_fail(&format!("ignis_residency_step: {e}"));
        return;
    }
    let report = gpu.last_report(0).expect("report");
    assert_eq!(report.status, 0);
    assert_eq!(report.hits.len(), 2 * TOP_K, "gate/up and down of each selected expert");
    assert!(report.misses.is_empty() && report.evictions.is_empty(), "{report:?}");
    assert_eq!(report.bytes_moved, 0);
    let counters = gpu.counters().expect("counters");
    assert_eq!(counters.misses.iter().flatten().sum::<u64>(), 0);
    assert_eq!(counters.hits.iter().flatten().sum::<u64>(), 2 * TOP_K as u64);
}
