//! A pinned hot set changes where experts live, never what the model
//! computes (GitHub #329 item 12): a Flash-Next engine loaded with
//! `pinned_expert_bytes` generates, greedy, the tokens of one loaded without,
//! at one lane and at three; its residency holds the pinned slots past each
//! class's LRU, which gave them up, and its host pool is smaller by their
//! bytes.
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`). Explicit GPU profile (ADR 0006): outside
//! `IGNIS_GPU_PROFILE=1` a missing artifact or GPU is a skip, under it a
//! failure. Two loads one after the other, each needing ~38 GB of free RAM
//! for its pinned expert pool.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::flash_next::{EngineOptions, FlashNextEngine};
use ignis_core::gpu_profile;
use ignis_core::residency::device::ResidencyDesc;

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const GENERATED: usize = 64;

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

/// Arbitrary ids, as `flash_next_forward_gpu.rs` prompts the engine.
fn prompt(len: usize, salt: u32) -> Vec<u32> {
    (0..len as u32).map(|i| 1000 + (i * 7919 + salt * 104_729) % 60_000).collect()
}

/// One load at `pinned_expert_bytes`: what it generated at one lane and at
/// three, and its residency's descriptor; `None` on a skip.
fn load_and_generate(pinned_expert_bytes: u64, prompts: &[Vec<u32>]) -> Option<(Vec<Vec<u32>>, Vec<Vec<u32>>, ResidencyDesc)> {
    let options = EngineOptions { max_context_tokens: 8192, pinned_expert_bytes, ..EngineOptions::default() };
    let mut engine = match FlashNextEngine::load(&model_dir(), options) {
        Ok(engine) => engine,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next engine pinning {pinned_expert_bytes} bytes: {e}"));
            return None;
        }
    };
    let one = engine.generate(&prompts[..1], GENERATED).unwrap_or_else(|e| panic!("generate on one lane: {e}"));
    let three = engine.generate(prompts, GENERATED).unwrap_or_else(|e| panic!("generate on three lanes: {e}"));
    Some((one, three, *engine.residency().desc()))
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn a_pinned_load_generates_the_unpinned_load_s_tokens() {
    if !model_dir().join(ARTIFACT_FILE_NAME).exists() {
        gpu_profile::skip_or_fail(&format!("no Flash-Next artifact in {}", model_dir().display()));
        return;
    }
    let prompts: Vec<Vec<u32>> = (0..3).map(|lane| prompt(300 + 17 * lane, 2 + lane as u32)).collect();
    let Some((one, three, plain)) = load_and_generate(0, &prompts) else { return };
    let Some((pinned_one, pinned_three, pinned)) = load_and_generate(1 << 30, &prompts) else { return };

    assert_eq!(plain.pinned, [0; 8], "0 pins nothing");
    let pinned_bytes: u64 = (0..8).map(|c| u64::from(pinned.pinned[c]) * pinned.record_bytes[c]).sum();
    eprintln!(
        "pinned {} projections, {pinned_bytes} bytes; host pool {} -> {} bytes",
        pinned.pinned.iter().sum::<u32>(),
        plain.host_pool_bytes,
        pinned.host_pool_bytes
    );
    assert!(pinned_bytes >= 1 << 30, "whole experts until the budget is reached: {pinned_bytes}");
    for c in 0..8 {
        assert_eq!(pinned.capacity[c] + pinned.pinned[c], plain.capacity[c], "class {c}: the pins came out of the LRU's slots");
    }
    assert_eq!(plain.host_pool_bytes - pinned.host_pool_bytes, pinned_bytes, "the host pool holds no copy of a pinned record");
    assert_eq!(pinned_one, one, "one lane: the pinned load's greedy tokens");
    assert_eq!(pinned_three, three, "three lanes: the pinned load's greedy tokens");
}
