//! Flash-Next's `none` rotary table, byte for byte, across the change that
//! taught its kernels YaRN (ADR 0059, spec rope-scaling/02 AC 9): the
//! unscaled table multiplies cos and sin by exactly 1.0f, an IEEE identity,
//! so a `none` load must give the very logits it gave before the change.
//!
//! - **`none` is the logits recorded before the change.** A real-text prompt
//!   of 2,560 tokens (the converter's `long8192` window 0) in 1,024-token
//!   chunks, so the last chunk runs the indexer's sparse selection: every
//!   row's BF16 logits hash to what the capture recorded.
//! - **A `yarn:4` load moves every row but the first** (AC 10): the same
//!   prompt, every logit finite and every row's bytes but the first's other
//!   than the capture's, so the scaled table reaches the program at all.
//!   The first row attends to itself alone, and a softmax over one key is 1
//!   whatever the table makes of its score, so it must stay the capture's.
//!   The argmax agreement with `none` is printed, not judged.
//!
//! The capture is per row: the SHA-256 of the row's BF16 logits and its
//! argmax. A bit-exact comparison only means something on the card and the
//! build that recorded it -- another card's split decisions reorder the sums
//! -- so this test is behind the non-default `logits-capture` feature and
//! the GPU profile never runs it:
//!
//! ```text
//! cargo test -p ignis-core --features cuda,logits-capture --test flash_next_rope_logits_gpu -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `IGNIS_RECORD_FLASH_NEXT_LOGITS=1` records the capture instead (on the
//! build before the change); without it the test compares. The capture is
//! `.scratch/flash-next-none-logits.txt` under the checkout, or the path in
//! `IGNIS_FLASH_NEXT_LOGITS_CAPTURE`.
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`). Needs ~38 GB of free RAM for the pinned expert
//! pool.

#![cfg(all(feature = "cuda", feature = "logits-capture"))]

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use ignis_core::flash_next::{EngineOptions, FlashNextEngine};
use ignis_core::{gpu_profile, KvFormat, RopeScaling};
use sha2::{Digest, Sha256};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
/// Past the indexer's dense threshold (2,051 visible tokens), so the last
/// chunk's rows select blocks.
const PROMPT_TOKENS: usize = 2560;

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

fn capture_path() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_LOGITS_CAPTURE").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.scratch/flash-next-none-logits.txt"),
        PathBuf::from,
    )
}

/// The first [`PROMPT_TOKENS`] tokens of the converter's `long8192` window 0.
fn prompt(dir: &Path) -> Option<Vec<u32>> {
    let path = dir.join("references").join("long8192").join("tokens.u32");
    let Ok(bytes) = fs::read(&path) else {
        gpu_profile::skip_or_fail(&format!("{} does not exist", path.display()));
        return None;
    };
    let tokens: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().expect("four bytes"))).collect();
    Some(tokens[..PROMPT_TOKENS].to_vec())
}

/// One row of the capture: its logits' SHA-256 (hex) and argmax.
#[derive(Debug, Clone, PartialEq)]
struct Row {
    sha256: String,
    argmax: u32,
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Every row of `tokens` teacher-forced through a load of `options`.
fn rows(dir: &Path, options: EngineOptions, tokens: &[u32]) -> Option<Vec<Row>> {
    let mut engine = match FlashNextEngine::load(dir, options) {
        Ok(engine) => engine,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next engine: {e}"));
            return None;
        }
    };
    let vocab = engine.vocab();
    let mut out = Vec::with_capacity(tokens.len());
    engine
        .span_logits(tokens, &mut |first, logits| {
            assert_eq!(first, out.len(), "rows arrive in order");
            for row in logits.chunks_exact(vocab) {
                assert!(row.iter().all(|&v| bf16(v).is_finite()), "row {} holds a non-finite logit", out.len());
                let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
                let sha256 = Sha256::digest(&bytes).iter().fold(String::new(), |mut s, b| {
                    let _ = write!(s, "{b:02x}");
                    s
                });
                let argmax = (0..vocab).max_by(|&a, &b| bf16(row[a]).total_cmp(&bf16(row[b])).then(b.cmp(&a))).expect("a row");
                out.push(Row { sha256, argmax: argmax as u32 });
            }
            Ok(())
        })
        .unwrap_or_else(|e| panic!("span logits: {e}"));
    assert_eq!(out.len(), tokens.len());
    Some(out)
}

fn write_capture(path: &Path, rows: &[Row]) {
    let text: String = rows.iter().enumerate().map(|(i, r)| format!("{i} {} {}\n", r.sha256, r.argmax)).collect();
    fs::create_dir_all(path.parent().expect("a parent")).expect("the capture's directory");
    fs::write(path, text).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn read_capture(path: &Path) -> Vec<Row> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("read the capture {} (record it on the build before the change): {e}", path.display())
    });
    text.lines()
        .map(|line| {
            let fields: Vec<&str> = line.split(' ').collect();
            Row { sha256: fields[1].to_owned(), argmax: fields[2].parse().expect("an argmax") }
        })
        .collect()
}

/// The load both runs of the comparison make: BF16 KV (the oracle format),
/// 1,024-token chunks, no graphs (a span runs none).
fn options() -> EngineOptions {
    EngineOptions {
        prefill_chunk_tokens: 1024,
        max_context_tokens: 8192,
        kv_format: KvFormat::Bf16,
        capture_graphs: false,
        ..EngineOptions::default()
    }
}

#[test]
#[ignore = "GPU, logits-capture feature: bit-exact on the recording card and build only"]
fn a_none_load_gives_the_logits_recorded_before_the_change() {
    let dir = model_dir();
    let Some(tokens) = prompt(&dir) else { return };
    let Some(none) = rows(&dir, options(), &tokens) else { return };
    let path = capture_path();
    if std::env::var_os("IGNIS_RECORD_FLASH_NEXT_LOGITS").is_some() {
        write_capture(&path, &none);
        println!("recorded {} rows to {}", none.len(), path.display());
        return;
    }
    let recorded = read_capture(&path);
    assert_eq!(recorded.len(), none.len(), "the capture's rows");
    let differing: Vec<usize> = (0..none.len()).filter(|&i| none[i] != recorded[i]).collect();
    println!("none against the capture: {} of {} rows differ", differing.len(), none.len());
    assert!(differing.is_empty(), "a none load moved rows {:?}", &differing[..differing.len().min(16)]);
}

#[test]
#[ignore = "GPU, logits-capture feature: against the capture of the recording card and build"]
fn a_yarn_load_moves_every_row_but_the_first_from_the_none_capture() {
    let dir = model_dir();
    let Some(tokens) = prompt(&dir) else { return };
    let recorded = read_capture(&capture_path());
    let options = EngineOptions { rope_scaling: RopeScaling::yarn(4.0).expect("yarn:4"), ..options() };
    let Some(yarn) = rows(&dir, options, &tokens) else { return };
    assert_eq!(recorded.len(), yarn.len(), "the capture's rows");
    let unmoved: Vec<usize> = (0..yarn.len()).filter(|&i| yarn[i].sha256 == recorded[i].sha256).collect();
    let agree = (0..yarn.len()).filter(|&i| yarn[i].argmax == recorded[i].argmax).count();
    println!(
        "yarn:4 against the none capture: {} of {} rows unmoved, argmax agrees on {agree}",
        unmoved.len(),
        yarn.len()
    );
    assert_eq!(unmoved, [0], "the rows a yarn:4 load leaves as none's: the first alone, which attends to itself");
}
