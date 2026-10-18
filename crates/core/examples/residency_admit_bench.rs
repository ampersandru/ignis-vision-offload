//! Admission throughput of a device-only expert residency (ADR 0054, GitHub
//! #329 item 16): can the residency phase of a device-only load ever be the
//! bottleneck? Builds a synthetic device-only residency of Flash-Next's
//! geometry (512 experts, its eight record sizes, a deterministic K mix) over
//! `--layers` layers and times, 3 reps each, the median in GB/s:
//!
//! - (a) `DeviceResidency::admit` a layer at a time from a pageable
//!   `Vec<u8>` layer buffer: the loader's real path, one synchronous pageable
//!   `cudaMemcpy` per record;
//! - (b) the same bytes as one pageable `cudaMemcpy` per layer;
//! - (c) the same bytes as one copy per layer from a `cudaHostAlloc` buffer.
//!
//! `cargo run -p ignis-core --release --features cuda --example residency_admit_bench -- [--layers 12]`
//! Holds the GPU for the run: take the GPU lock first (docs/agents/testing.md).

use std::ffi::c_void;
use std::time::Instant;

use ignis_core::residency::device::{DeviceResidency, NO_BUDGET, ResidencyDesc};
use ignis_core::residency::KClass;

unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    fn cudaHostAlloc(ptr: *mut *mut c_void, bytes: usize, flags: u32) -> i32;
    fn cudaFreeHost(ptr: *mut c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

const H2D: i32 = 1;
const EXPERTS: u32 = 512;
const REPS: usize = 3;
const RECORD_BYTES: [u64; KClass::COUNT] = [827_392, 1_032_192, 1_236_992, 1_646_592, 417_792, 520_192, 622_592, 827_392];

fn cuda(rc: i32, what: &str) {
    assert_eq!(rc, 0, "{what}: CUDA error {rc}");
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let layers: u32 = args.windows(2).find(|w| w[0] == "--layers").map_or(12, |w| w[1].parse().expect("--layers N"));

    // A deterministic K mix: roughly the artifact's, K = 2 the most common.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let keys = layers * EXPERTS * 2;
    let k2: Vec<u8> = (0..keys)
        .map(|_| match next() % 10 {
            0..=4 => 4u8,
            5..=6 => 5,
            7..=8 => 6,
            _ => 8,
        })
        .collect();
    let class_of = |key: u32| -> usize {
        let k = match k2[key as usize] {
            4 => 0,
            5 => 1,
            6 => 2,
            _ => 3,
        };
        (key & 1) as usize * 4 + k
    };
    let mut capacity = [0u32; KClass::COUNT];
    let mut layer_bytes = vec![0u64; layers as usize];
    for key in 0..keys {
        capacity[class_of(key)] += 1;
        layer_bytes[(key / (EXPERTS * 2)) as usize] += RECORD_BYTES[class_of(key)];
    }
    let heaviest = *layer_bytes.iter().max().unwrap();
    let total: u64 = layer_bytes.iter().sum();
    println!(
        "{layers} layers, {keys} records, {:.2} GB, mean record {:.0} B, heaviest layer {:.1} MB",
        total as f64 / 1e9,
        total as f64 / f64::from(keys),
        heaviest as f64 / 1e6
    );
    println!("class counts {capacity:?}");

    // One layer buffer per distinct layer size is overkill: fill the heaviest once and
    // admit every layer from its prefix (the records of a layer are packed in key order).
    let layer_bufs: Vec<Vec<u8>> = {
        let mut one = vec![0u8; heaviest as usize];
        for (i, b) in one.iter_mut().enumerate() {
            *b = (i.wrapping_mul(7) + 3) as u8;
        }
        vec![one]
    };
    let src = &layer_bufs[0];
    // Offsets of each layer's records, key order, within the layer buffer.
    let layer_offsets: Vec<(Vec<u32>, Vec<u64>)> = (0..layers)
        .map(|l| {
            let mut at = 0u64;
            let mut ks = Vec::new();
            let mut offs = Vec::new();
            for key in l * EXPERTS * 2..(l + 1) * EXPERTS * 2 {
                ks.push(key);
                offs.push(at);
                at += RECORD_BYTES[class_of(key)];
            }
            (ks, offs)
        })
        .collect();

    let desc = ResidencyDesc {
        layers,
        experts: EXPERTS,
        capacity,
        record_bytes: RECORD_BYTES,
        max_tokens: 8,
        lookahead_width: 4,
        prefill_lookahead_width: 4,
        prefetch_budget_one_row_bytes: NO_BUDGET,
        prefetch_budget_per_row_bytes: 0,
        staging_half_bytes: heaviest,
        host_pool_bytes: 0,
        copy_blocks: 16,
        report: 0,
    };

    // (a) the real path.
    let mut a = Vec::new();
    for rep in 0..REPS {
        let mut gpu = DeviceResidency::new(&desc, &k2, &[]).expect("create");
        let t = Instant::now();
        for (ks, offs) in &layer_offsets {
            gpu.admit(ks, src, offs).expect("admit");
        }
        cuda(unsafe { cudaDeviceSynchronize() }, "sync");
        let s = t.elapsed().as_secs_f64();
        gpu.admit_finish().expect("finish");
        println!("(a) admit, per record, pageable: rep {rep}: {s:.3} s, {:.2} GB/s", total as f64 / 1e9 / s);
        a.push(total as f64 / 1e9 / s);
    }

    // (b) and (c): one copy per layer into a device buffer of the heaviest layer's size.
    let mut dst: *mut c_void = std::ptr::null_mut();
    cuda(unsafe { cudaMalloc(&mut dst, heaviest as usize) }, "cudaMalloc");
    let mut pinned: *mut c_void = std::ptr::null_mut();
    cuda(unsafe { cudaHostAlloc(&mut pinned, heaviest as usize, 0) }, "cudaHostAlloc");
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), pinned.cast::<u8>(), heaviest as usize) };
    let (mut b, mut c) = (Vec::new(), Vec::new());
    for rep in 0..REPS {
        for (name, from, out) in [("(b) one copy per layer, pageable", src.as_ptr().cast::<c_void>(), &mut b), ("(c) one copy per layer, pinned", pinned.cast_const(), &mut c)] {
            let t = Instant::now();
            for &bytes in &layer_bytes {
                cuda(unsafe { cudaMemcpy(dst, from, bytes as usize, H2D) }, "cudaMemcpy");
            }
            cuda(unsafe { cudaDeviceSynchronize() }, "sync");
            let s = t.elapsed().as_secs_f64();
            println!("{name}: rep {rep}: {s:.3} s, {:.2} GB/s", total as f64 / 1e9 / s);
            out.push(total as f64 / 1e9 / s);
        }
    }
    unsafe {
        cudaFreeHost(pinned);
        cudaFree(dst);
    }
    println!(
        "median GB/s: (a) {:.2}  (b) {:.2}  (c) {:.2}",
        median(a),
        median(b),
        median(c)
    );
}
