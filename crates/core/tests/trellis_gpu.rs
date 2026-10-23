//! The dense trellis linear through its Rust binding (spec 27b-16gb/02 Acceptance 3, GitHub #337):
//! the workspace's plan line for the 27B's calls on a prepared device, and one call end to end.
//! The kernel's own tests (`kernel/tests/test_trellis_*.cu`) hold the numerics; this holds the
//! ABI and the plan line the 16 GB load will reserve (spec 05).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1` a missing GPU or a
//! kernel error is a skip; under it a hard failure (`ignis_core::gpu_profile`).

#![cfg(feature = "cuda")]

use std::os::raw::c_void;

use ignis_artifact::{CudaDevice, Device, DeviceBuffer};
use ignis_core::gpu_profile;
use ignis_core::trellis::{self, Call, TrellisWeight, TrellisWorkspace};

fn device() -> Option<CudaDevice> {
    let dev = match CudaDevice::create(0) {
        Ok(d) => d,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("no CUDA device: {e}"));
            return None;
        }
    };
    if let Err(e) = trellis::prepare() {
        gpu_profile::skip_or_fail(&format!("ignis_trellis_prepare: {e}"));
        return None;
    }
    Some(dev)
}

fn upload(dev: &mut CudaDevice, bytes: &[u8]) -> Option<DeviceBuffer> {
    let buf = dev.allocate(bytes.len() as u64);
    match buf.and_then(|b| dev.copy_h2d(&b, 0, bytes).and_then(|_| dev.synchronize()).map(|_| b)) {
        Ok(b) => Some(b),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("device upload: {e:?}"));
            None
        }
    }
}

fn download(dev: &mut CudaDevice, buf: &DeviceBuffer) -> Option<Vec<u8>> {
    let mut out = vec![0u8; buf.len() as usize];
    match dev.copy_d2h(buf, 0, &mut out).and_then(|_| dev.synchronize()) {
        Ok(()) => Some(out),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("device download: {e:?}"));
            None
        }
    }
}

fn ptr(buf: &DeviceBuffer) -> *mut c_void {
    buf.base_ptr() as *mut c_void
}

fn bf16(v: f32) -> [u8; 2] {
    ((v.to_bits() + 0x7FFF + ((v.to_bits() >> 16) & 1)) >> 16).to_le_bytes()[..2].try_into().unwrap()
}

fn f32_of_bf16(b: [u8; 2]) -> f32 {
    f32::from_bits((u16::from_le_bytes(b) as u32) << 16)
}

/// The repack's calls (spec 01): GDN qkv, z, out; GQA q, k/v, o; the MLP pair and down; the head.
const CALLS: [Call; 8] = [
    Call { in_features: 5120, out_features: 10240, weights: 1 },
    Call { in_features: 5120, out_features: 6144, weights: 1 },
    Call { in_features: 6144, out_features: 5120, weights: 1 },
    Call { in_features: 5120, out_features: 12288, weights: 1 },
    Call { in_features: 5120, out_features: 1024, weights: 1 },
    Call { in_features: 5120, out_features: 17408, weights: 2 },
    Call { in_features: 17408, out_features: 5120, weights: 1 },
    Call { in_features: 5120, out_features: 248320, weights: 1 },
];

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn the_plan_line_covers_every_call_the_27b_makes() {
    let Some(_dev) = device() else { return };
    // The 16 GB profile's widths: two lanes of DFlash2 rounds (16 columns), 2048-token chunks.
    let line = match trellis::plan_bytes(&CALLS, 16, 2048) {
        Ok(b) => b,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("plan_bytes: {e}"));
            return;
        }
    };
    for call in CALLS {
        for tokens in [1, 2, 8, 16, 2048] {
            let need = trellis::workspace_bytes(call, tokens).unwrap();
            assert!(need <= line, "{call:?} at {tokens} tokens needs {need} > the plan line {line}");
        }
    }
    // The largest call at the chunk: the down projection's rotated activations, 2048 x 17408 fp16.
    assert!(line >= 2048 * 17408 * 2, "{line}");
    println!("trellis plan line at 16 decode / 2048 prefill tokens: {line} bytes");
    assert!(trellis::workspace_bytes(Call { in_features: 100, out_features: 128, weights: 1 }, 1).is_err());
}

#[test]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
fn one_call_runs_through_the_binding() {
    let Some(mut dev) = device() else { return };
    // in 256, out 128, K = 3: all-zero words decode every weight to the codebook's value of state
    // 0, (1024 + 0) * fp16(0x1EEE) + fp16(0xC931) = -3.453125. x = e_0 rotates to 1/sqrt(128) on
    // block 0, so every inner sum is -3.453125 * sqrt(128), and the output rotation of that
    // constant is 128 * -3.453125 at column 0 and zero elsewhere (suh = svh = 1).
    let (inp, out, k2) = (256u32, 128u32, 6u32);
    let words = vec![0u8; (inp * out / 16 * k2) as usize];
    let ones = |n: u32| (0..n).flat_map(|_| 0x3C00u16.to_le_bytes()).collect::<Vec<u8>>();
    let mut x = vec![0u8; (inp * 2) as usize];
    x[..2].copy_from_slice(&bf16(1.0));
    let (Some(d_words), Some(d_suh), Some(d_svh), Some(d_x), Some(d_y)) = (
        upload(&mut dev, &words),
        upload(&mut dev, &ones(inp)),
        upload(&mut dev, &ones(out)),
        upload(&mut dev, &x),
        upload(&mut dev, &vec![0u8; (out * 2) as usize]),
    ) else {
        return;
    };
    let w = TrellisWeight {
        trellis: ptr(&d_words),
        suh: ptr(&d_suh),
        svh: ptr(&d_svh),
        k2,
        in_features: inp,
        out_features: out,
        tile_stride: 0,
    };
    let bytes = trellis::workspace_bytes(Call { in_features: inp, out_features: out, weights: 1 }, 1).unwrap();
    let Some(d_ws) = upload(&mut dev, &vec![0u8; bytes as usize]) else { return };
    let ws = TrellisWorkspace { base: ptr(&d_ws), bytes };
    let ran = unsafe {
        trellis::workspace_init(&ws, std::ptr::null_mut())
            .and_then(|_| trellis::linear(&w, ptr(&d_x), 1, ptr(&d_y), out, trellis::STORE, &ws, std::ptr::null_mut()))
    };
    // The call ran on the legacy stream; the device copies on its own non-blocking one.
    if let Err(e) = ran.and_then(|_| ignis_core::moe::stream_sync(std::ptr::null_mut())) {
        gpu_profile::skip_or_fail(&format!("ignis_trellis_linear: {e}"));
        return;
    }
    let Some(y) = download(&mut dev, &d_y) else { return };
    let y: Vec<f32> = y.chunks(2).map(|c| f32_of_bf16([c[0], c[1]])).collect();
    // Within one BF16 step of 442 (the fp16 activations and accumulation round on the way).
    assert!((y[0] - 128.0 * -3.453125).abs() <= 2.0, "column 0: {}", y[0]);
    assert!(y[1..].iter().all(|v| v.abs() < 1e-3), "{:?}", &y[1..8]);
}
