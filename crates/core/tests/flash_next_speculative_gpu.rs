//! The verify round on Flash-Next's state (spec flash-next/07 phase C,
//! GitHub #307), proven with fake drafters on the real artifact before the
//! MTP head drafts.
//!
//! The load is `SpeculativeBackend::VerifyOnly` at 3 draft tokens and the
//! default's 8 rows: one lane verifies 3 drafts a round, three lanes 1.
//! Four drafters run the same prompts for the same number of tokens:
//! - **none** (every lane at extent 0): one committed token a round, through
//!   the verify pass's k + 1 columns;
//! - **reject** (a constant id no greedy step picks): every draft rejected;
//! - **random**: rejected at the first divergence, by chance accepted;
//! - **oracle** (none's own text): every draft accepted.
//!
//! A column's computation does not depend on what the other columns of its
//! round hold -- each op is row-independent at a fixed round shape -- so a
//! rejected column is a column never drafted: **reject and random commit
//! none's text bit for bit, and leave every lane's state as none leaves it**.
//! The state is probed where a later step reads it: one more token prefilled
//! at each lane's frontier, its logits compared bit for bit (KV and pooled
//! indexer blocks, the indexer tail, the GDN states and conv taps, the n-gram
//! conv, the hq residual window). Any component the commit restored wrongly
//! moves that row. The oracle commits several columns a round: in BF16 KV
//! the window's earlier columns are read exactly as the pages would hold
//! them, so it too commits none's text and state bit for bit; under
//! hq-e8-2b a column reads the window's own rows fresh where a later round
//! reads them back from the residual ring, so there it is held to the
//! near-tie rule instead.
//!
//! The MTP head's drafts obey the same rule: its alignment runs over all
//! k + 1 columns, the rejected ones included, and leaves rows in the head's
//! own section (its hq ring slots, its indexer tail) that its chain steps
//! past the frontier must not read, so none and reject leave the head
//! proposing the same drafts bit for bit, in both KV formats, on one lane
//! and on two.
//!
//! Spec-on against spec-off (one-token rounds, the plain decode graphs) is
//! held to the near-tie rule: the verify pass runs k + 1 columns per lane
//! through other kernel shapes (attention splits, the indexer's prefill
//! scoring), so its logits differ by summation order.
//!
//! Past the default's 8 verify rows (GitHub #329 item 5, ADR 0056) the
//! linears and mixes take their tensor-core routes and the text is no longer
//! held to spec-off's: a budget's rounds are held instead to spec-off's
//! logits row by row on spec-off's own text (teacher-forced, a tolerance in
//! BF16 ulps), to their own rows' picks (the commit, exact), to none's state
//! (exact), to themselves run to run, and to spec-off's mean log-probability
//! and a floor of the head's acceptance (`support/logit_drift.rs`).
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`). Explicit GPU profile (ADR 0006): outside
//! `IGNIS_GPU_PROFILE=1` a missing artifact or GPU is a skip, under it a
//! failure. Needs ~38 GB of free RAM for the pinned expert pool.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::flash_next::{EngineOptions, FlashNextEngine, LaneRound, RoundLogits};
use ignis_core::gpu_profile;
use ignis_core::kv_format::KvFormat;
use ignis_core::speculation::{FlashNextSpeculation, SpeculativeBackend};
use ignis_core::step::SamplingParams;

#[path = "support/logit_drift.rs"]
mod logit_drift;

use logit_drift::{
    acceptance_within_band, commit_follows_argmax, log_prob, row_drift, LaneCommit, RowDrift, ACCEPTANCE_BAND,
    HEAD_ACCEPTANCE_FLOOR, MEAN_LOG_PROB_NATS, PICK_GAP_ULPS, TOLERANCE_ULPS,
};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const DRAFT_TOKENS: u32 = 3;
const TOKENS: usize = 40;
/// A draft no greedy step picks: the vocabulary's last id, a reserved token.
const REJECTED: u32 = 248_319;

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

fn engine(kv_format: KvFormat) -> Option<FlashNextEngine> {
    engine_with(kv_format, SpeculativeBackend::VerifyOnly)
}

fn engine_with(kv_format: KvFormat, backend: SpeculativeBackend) -> Option<FlashNextEngine> {
    engine_with_context(kv_format, backend, 8192)
}

fn engine_with_context(kv_format: KvFormat, backend: SpeculativeBackend, max_context_tokens: u32) -> Option<FlashNextEngine> {
    let dir = model_dir();
    if !dir.join(ARTIFACT_FILE_NAME).exists() {
        gpu_profile::skip_or_fail(&format!("no Flash-Next artifact in {}", dir.display()));
        return None;
    }
    if backend == SpeculativeBackend::Mtp && !ignis_core::flash_next_mtp::companion_path(&dir).exists() {
        gpu_profile::skip_or_fail(&format!("no MTP companion in {}", dir.display()));
        return None;
    }
    let speculation = FlashNextSpeculation::new(backend, DRAFT_TOKENS, 0).expect("valid");
    let options =
        EngineOptions { max_context_tokens, kv_format, speculation: Some(speculation), ..EngineOptions::default() };
    match FlashNextEngine::load(&dir, options) {
        Ok(engine) => Some(engine),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next engine: {e}"));
            None
        }
    }
}

/// Every width's bit: at `DRAFT_TOKENS` under the default 8-row budget each
/// width of the default lanes drafts at least one token, so each captures
/// its verify pass and commit graphs.
fn every_width_verifies() -> u32 {
    (1u32 << ignis_core::flash_next::DEFAULT_DECODE_LANES) - 1
}

/// The first `len` tokens of reference set `set` (real text), or None with a
/// skip.
fn reference(set: &str, len: usize) -> Option<Vec<u32>> {
    let path = model_dir().join("references").join(set).join("tokens.u32");
    let Ok(bytes) = std::fs::read(&path) else {
        gpu_profile::skip_or_fail(&format!("no reference tokens at {}", path.display()));
        return None;
    };
    let tokens: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Some(tokens.get(..len).expect("the reference set holds the prompt").to_vec())
}

/// `len` tokens of window `window` of the test2048 reference set (windows
/// 2,048 apart), or None with a skip.
fn test2048_window(window: usize, len: usize) -> Option<Vec<u32>> {
    let set = reference("test2048", (window + 1) * 2048)?;
    Some(set[window * 2048..window * 2048 + len].to_vec())
}

/// The converter's G1 prompts (real text), or None with a skip.
fn g1_prompts() -> Option<Vec<Vec<u32>>> {
    let refs = model_dir().join("references").join("g1_flash_next.json");
    let Ok(text) = std::fs::read_to_string(&refs) else {
        gpu_profile::skip_or_fail(&format!("no G1 references at {}", refs.display()));
        return None;
    };
    let g1: serde_json::Value = serde_json::from_str(&text).expect("the G1 references parse");
    Some(
        g1["prompts"]
            .as_array()
            .expect("prompts")
            .iter()
            .map(|p| {
                p["prompt_token_ids"].as_array().expect("ids").iter().map(|x| x.as_u64().expect("an id") as u32).collect()
            })
            .collect(),
    )
}

/// A prompt past the dense threshold (2051 visible tokens), so every column
/// attends through the indexer's sparse selection: the G1 prompts and their
/// reference continuations, repeated.
fn long_prompt(prompts: &[Vec<u32>]) -> Vec<u32> {
    let mut out = Vec::new();
    while out.len() < 2600 {
        for p in prompts {
            out.extend_from_slice(p);
        }
    }
    out.truncate(2600);
    out
}

/// What one drafter's run left: each lane's tokens, its rounds, the head's
/// drafts each round made, and the probe row after them.
struct Outcome {
    tokens: Vec<Vec<u32>>,
    rounds: Vec<Vec<LaneRound>>,
    head_drafts: Vec<Vec<Vec<u32>>>,
    probes: Vec<Vec<f32>>,
}

fn run(engine: &FlashNextEngine, prompts: &[Vec<u32>], drafter: Option<&mut dyn FnMut(usize, &[u32], u32) -> Vec<u32>>) -> Outcome {
    run_with(engine, prompts, drafter, None)
}

/// [`run`], every round's logits handed to `on_round`.
fn run_with(
    engine: &FlashNextEngine,
    prompts: &[Vec<u32>],
    drafter: Option<&mut dyn FnMut(usize, &[u32], u32) -> Vec<u32>>,
    on_round: Option<&mut dyn FnMut(RoundLogits<'_>)>,
) -> Outcome {
    let mut run = engine
        .generate_speculative_with_logits(prompts, TOKENS, drafter, on_round)
        .unwrap_or_else(|e| panic!("generate_speculative on {} lanes: {e}", prompts.len()));
    let mut probes = Vec::new();
    for (seq, context) in run.sequences.iter_mut().zip(run.contexts.iter_mut()) {
        probes.push(engine.probe_logits(seq, context).unwrap_or_else(|e| panic!("probe: {e}")));
    }
    Outcome { tokens: run.tokens, rounds: run.rounds, head_drafts: run.head_drafts, probes }
}

fn same_bits(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn top_two(row: &[f32]) -> (u32, f32, f32) {
    let (mut best, mut first, mut second) = (0u32, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (id, &v) in row.iter().enumerate() {
        if v > first {
            second = first;
            first = v;
            best = id as u32;
        } else if v > second {
            second = v;
        }
    }
    (best, first, second)
}

/// One BF16 ulp at `x`'s magnitude (`flash_next_forward_gpu.rs`'s rule).
fn bf16_ulp(x: f32) -> f32 {
    let exponent = ((x.abs().to_bits() >> 23) & 0xff) as i32 - 127;
    2f32.powi(exponent - 7)
}

/// `on` against `off`, lane by lane: identical, or identical up to a first
/// divergence where the prefill route's logits after the agreed text put the
/// two picks within one BF16 ulp, or side with `on` (the reference's own
/// decode drifted). Returns a line per divergence; panics on a divergence
/// the prefill route lays at `on`'s door.
fn assert_near_tie(engine: &mut FlashNextEngine, prompts: &[Vec<u32>], off: &[Vec<u32>], on: &[Vec<u32>], what: &str) -> Vec<String> {
    let mut report = Vec::new();
    for (lane, ((prompt, a), b)) in prompts.iter().zip(off).zip(on).enumerate() {
        let Some(at) = a.iter().zip(b).position(|(x, y)| x != y) else {
            assert_eq!(a.len(), b.len(), "{what} lane {lane}: lengths");
            continue;
        };
        let mut text = prompt.clone();
        text.extend_from_slice(&a[..at]);
        let logits = engine.last_logits(&text).unwrap_or_else(|e| panic!("near-tie probe: {e}"));
        let (top, first, _) = top_two(&logits);
        let gap = logits[a[at] as usize] - logits[b[at] as usize];
        let line = format!(
            "{what} lane {lane}: diverges at {at}: off {} on {} (prefill pick {top}), prefill gap off - on = {gap} (ulp {})",
            a[at],
            b[at],
            bf16_ulp(first)
        );
        println!("{line}");
        assert!(gap <= bf16_ulp(first), "{line}: the prefill route sides with spec-off past a near-tie");
        report.push(line);
    }
    report
}

/// Every drafter on one set of prompts, in one KV format.
fn exercise(engine: &mut FlashNextEngine, kv_format: KvFormat, prompts: &[Vec<u32>], what: &str) {
    let lanes = prompts.len() as u32;
    let window = engine.options().speculation.expect("speculation").window(lanes);
    assert!(window > 0, "{what}: a width that verifies");
    let none = run(engine, prompts, Some(&mut |_, _, _| Vec::new()));
    for round in &none.rounds {
        assert!(round.iter().all(|l| l.extent == 0 && l.committed == 1), "{what}: none committed a draft");
    }
    let reject = run(engine, prompts, Some(&mut |_, _, w| vec![REJECTED; w as usize]));
    let mut seed = 0x9e37_79b9_u32;
    let random = run(
        engine,
        prompts,
        Some(&mut |_, _, w| {
            (0..w)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    1000 + seed % 60_000
                })
                .collect()
        }),
    );
    let reference = none.tokens.clone();
    let oracle = run(
        engine,
        prompts,
        Some(&mut |lane, emitted, w| {
            let next = emitted.len() + 1;
            reference[lane][next.min(TOKENS)..(next + w as usize).min(TOKENS)].to_vec()
        }),
    );

    for (name, outcome) in [("reject", &reject), ("random", &random)] {
        assert_eq!(outcome.tokens, none.tokens, "{what}: the {name} drafter committed other text than none");
        for (lane, (p, q)) in outcome.probes.iter().zip(&none.probes).enumerate() {
            assert_eq!(same_bits(p, q), 0, "{what}: lane {lane}'s state after the {name} drafter's rounds is not none's");
        }
    }
    let drafted: u32 = reject.rounds.iter().flatten().map(|l| l.extent).sum();
    assert!(drafted > 0, "{what}: the reject drafter's drafts were never verified");
    assert!(reject.rounds.iter().flatten().all(|l| l.committed == 1), "{what}: a rejected draft was committed");

    // The oracle's drafts are none's text: every one is accepted while the
    // oracle's own text is none's.
    let accepted: u32 = oracle.rounds.iter().flatten().map(|l| l.committed - 1).sum();
    let proposed: u32 = oracle.rounds.iter().flatten().map(|l| l.extent).sum();
    println!(
        "{what} ({kv_format:?}): oracle {} rounds for {} tokens per lane, {accepted}/{proposed} drafts accepted",
        oracle.rounds.len(),
        TOKENS
    );
    if kv_format == KvFormat::Bf16 {
        assert_eq!(accepted, proposed, "{what}: an oracle draft was rejected");
        assert_eq!(oracle.tokens, none.tokens, "{what}: the oracle committed other text than none");
        for (lane, (p, q)) in oracle.probes.iter().zip(&none.probes).enumerate() {
            assert_eq!(same_bits(p, q), 0, "{what}: lane {lane}'s state after the oracle's rounds is not none's");
        }
    } else {
        assert_near_tie(engine, prompts, &none.tokens, &oracle.tokens, &format!("{what} oracle vs none"));
    }

    // Spec-on (the verify round) against spec-off (one-token rounds).
    let off = engine.generate(prompts, TOKENS).unwrap_or_else(|e| panic!("{what}: spec-off generate: {e}"));
    assert_near_tie(engine, prompts, &off, &none.tokens, &format!("{what} spec-on vs spec-off"));
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn a_rejected_draft_is_a_draft_never_made_on_every_state_component() {
    let Some(g1) = g1_prompts() else { return };
    let long = long_prompt(&g1);
    for kv_format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
        let Some(mut engine) = engine(kv_format) else { return };
        assert_ne!(engine.graphs_ready(), 0, "{kv_format:?}: no round graph captured: {:?}", engine.graph_error());
        assert_eq!(
            engine.verify_graphs_ready(),
            Ok(every_width_verifies()),
            "{kv_format:?}: a verify graph did not capture: {:?}",
            engine.graph_error()
        );
        exercise(&mut engine, kv_format, &g1[..1], &format!("{kv_format:?} one lane, dense"));
        exercise(&mut engine, kv_format, &g1[..3], &format!("{kv_format:?} three lanes, dense"));
        exercise(&mut engine, kv_format, std::slice::from_ref(&long), &format!("{kv_format:?} one lane, sparse"));
        drop(engine);
    }
}

/// Each draft position's acceptance over `rounds`: of the rounds that
/// verified a draft at position j with every draft before it accepted, the
/// share that accepted it too (a run cut by its budget counts as far as it
/// committed).
fn acceptance(rounds: &[Vec<LaneRound>]) -> Vec<(u32, u32)> {
    let mut at = vec![(0u32, 0u32); DRAFT_TOKENS as usize];
    for lane in rounds.iter().flatten() {
        let accepted = lane.committed - 1;
        for j in 0..lane.extent.min(accepted + 1) {
            at[j as usize].1 += 1;
            if j < accepted {
                at[j as usize].0 += 1;
            }
        }
    }
    at
}

/// Spec flash-next/07 phase D: the MTP head drafts every round from its
/// own entries (prefill's, then each round's alignment and chain), and the
/// verify round keeps the text -- bit for bit what the same load commits
/// without drafts in BF16 KV (a draft-free run of the same load: the trunk
/// never sees the head), up to the near-tie rule under hq-e8-2b and against
/// spec-off. Prints each draft position's acceptance.
#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact and its MTP companion"]
fn the_mtp_head_drafts_and_the_text_is_kept() {
    let Some(g1) = g1_prompts() else { return };
    let long = long_prompt(&g1);
    for kv_format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
        let Some(mut engine) = engine_with(kv_format, SpeculativeBackend::Mtp) else { return };
        assert_eq!(
            engine.verify_graphs_ready(),
            Ok(every_width_verifies()),
            "{kv_format:?}: an MTP verify graph did not capture: {:?}",
            engine.graph_error()
        );
        for (prompts, what) in [
            (g1[..1].to_vec(), format!("{kv_format:?} MTP one lane, dense")),
            (g1[..3].to_vec(), format!("{kv_format:?} MTP three lanes, dense")),
            (vec![long.clone()], format!("{kv_format:?} MTP one lane, sparse")),
        ] {
            let none = run(&engine, &prompts, Some(&mut |_, _, _| Vec::new()));
            let head = run(&engine, &prompts, None);
            let alpha = acceptance(&head.rounds);
            let tokens: u32 = head.rounds.iter().flatten().map(|l| l.committed).sum();
            println!(
                "{what}: {} rounds for {tokens} tokens, acceptance per position (accepted/reached) {alpha:?}",
                head.rounds.len()
            );
            assert!(alpha[0].0 > 0, "{what}: the head's first drafts were never accepted: {alpha:?}");
            if kv_format == KvFormat::Bf16 {
                assert_eq!(head.tokens, none.tokens, "{what}: the head's drafts changed the committed text");
                for (lane, (p, q)) in head.probes.iter().zip(&none.probes).enumerate() {
                    assert_eq!(same_bits(p, q), 0, "{what}: lane {lane}'s state after the head's rounds is not none's");
                }
            } else {
                assert_near_tie(&mut engine, &prompts, &none.tokens, &head.tokens, &format!("{what} head vs none"));
            }
            let off = engine.generate(&prompts, TOKENS).unwrap_or_else(|e| panic!("{what}: spec-off generate: {e}"));
            assert_near_tie(&mut engine, &prompts, &off, &head.tokens, &format!("{what} spec-on vs spec-off"));
        }
        drop(engine);
    }
}

/// GitHub #307 (2c87e4c): the last prefill chunk's draw block has its own
/// arena scope, so the MTP head's entries after it reuse the arena within the
/// prefill scratch the plan sized (the arena is exactly that size and throws
/// past it). The load is cut to a long hq prompt of whole chunks, so the last
/// chunk is exactly the plan's chunk size and the head's attention peaks at
/// nearly the plan's attention term, the case the draw block's 16 MiB would
/// stack on. Measured on the real artifact at this size (arena peak after the
/// last chunk, plan 323_309_568): 305_332_224 without the scope, so the
/// scope's margin here is the 18 MB the plan keeps over the head's own peak,
/// and this guard does not go red without 2c87e4c; it holds the full-chunk
/// MTP prefill inside the plan (a bad_alloc fails the generate).
#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact and its MTP companion"]
fn the_mtp_heads_entries_after_a_full_chunks_draw_fit_the_planned_arena() {
    let Some(g1) = g1_prompts() else { return };
    let chunk = EngineOptions::default().prefill_chunk_tokens as usize;
    let prompt_tokens = 40 * chunk;
    // A page (64) multiple past the prompt: room for the generated tokens.
    let Some(engine) =
        engine_with_context(KvFormat::HqE8_2b, SpeculativeBackend::Mtp, (prompt_tokens + 2 * 64) as u32)
    else {
        return;
    };
    assert_eq!(engine.options().prefill_chunk_tokens as usize, chunk);
    let mut prompt = Vec::new();
    while prompt.len() < prompt_tokens {
        for p in &g1 {
            prompt.extend_from_slice(p);
        }
    }
    prompt.truncate(prompt_tokens);
    let head = run(&engine, &[prompt], None);
    assert_eq!(head.tokens[0].len(), TOKENS, "the full-chunk prompt generated through the head");
}

/// GitHub #307: the MTP head drafts from the committed text only. A verify
/// round's head writes its alignment at every column; under hq-e8-2b a
/// chain step at q read the rejected columns' ring rows as positions
/// q - 512 + i (a ring slot carries no position), and in both formats it
/// pooled a block from the alignment's indexer tail. Two drafters that commit
/// the same text -- none (extent 0) and reject (every draft rejected) -- must
/// leave the head proposing the same drafts, bit for bit, every round. The
/// prompts are past the ring (1,536 tokens, dense: every visible row read)
/// and past the dense threshold (2,051 visible tokens, the indexer's sparse selection; 3,072 is past it),
/// one lane each, then both as two lanes of one round (window 3 at the
/// 8-row budget: three lanes would run at window 1, where the head's chain
/// has no step and nothing is restored before it).
#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact and its MTP companion"]
fn the_heads_drafts_never_read_a_rejected_column() {
    const HEAD_TOKENS: usize = 64;
    let (Some(dense), Some(sparse)) = (reference("test2048", 1536), reference("long8192", 3072)) else { return };
    let cases = [
        (vec![dense.clone()], "dense, 1536 tokens"),
        (vec![sparse.clone()], "sparse, 3072 tokens"),
        (vec![dense, sparse], "two lanes, dense and sparse"),
    ];
    for kv_format in [KvFormat::HqE8_2b, KvFormat::Bf16] {
        let Some(engine) = engine_with(kv_format, SpeculativeBackend::Mtp) else { return };
        for (prompts, what) in &cases {
            let what = format!("{kv_format:?} {what}");
            let mut none = |_: usize, _: &[u32], _: u32| Vec::new();
            let mut reject = |_: usize, _: &[u32], window: u32| vec![REJECTED; window as usize];
            let mut a = engine
                .generate_speculative(prompts, HEAD_TOKENS, Some(&mut none))
                .unwrap_or_else(|e| panic!("{what}: none: {e}"));
            // Its lanes' slots back before the next run takes as many: the pool holds the load's three.
            a.sequences.clear();
            let b = engine
                .generate_speculative(prompts, HEAD_TOKENS, Some(&mut reject))
                .unwrap_or_else(|e| panic!("{what}: reject: {e}"));
            assert_eq!(a.tokens, b.tokens, "{what}: none and reject committed different text");
            assert!(a.rounds.iter().chain(&b.rounds).flatten().all(|l| l.committed == 1), "{what}: a round committed a draft");
            assert!(a.head_drafts.iter().flatten().all(|d| d.len() == DRAFT_TOKENS as usize), "{what}: a round made no drafts");
            let differ: Vec<usize> = (0..a.head_drafts.len()).filter(|&r| a.head_drafts[r] != b.head_drafts[r]).collect();
            assert!(
                differ.is_empty(),
                "{what}: the head drafted differently after {} of {} rounds (first at round {}: {:?} vs {:?})",
                differ.len(),
                a.head_drafts.len(),
                differ[0],
                a.head_drafts[differ[0]],
                b.head_drafts[differ[0]]
            );
        }
    }
}

/// A load of eight decode lanes with the MTP head at a verify budget of
/// `rows`, or None with a skip.
fn budget_engine(kv_format: KvFormat, rows: u32) -> Option<FlashNextEngine> {
    let dir = model_dir();
    if !dir.join(ARTIFACT_FILE_NAME).exists() || !ignis_core::flash_next_mtp::companion_path(&dir).exists() {
        gpu_profile::skip_or_fail(&format!("no Flash-Next artifact or MTP companion in {}", dir.display()));
        return None;
    }
    let speculation = FlashNextSpeculation::new(SpeculativeBackend::Mtp, DRAFT_TOKENS, rows).expect("valid");
    let options = EngineOptions {
        max_context_tokens: 8192,
        kv_format,
        decode_lanes: 8,
        speculation: Some(speculation),
        ..EngineOptions::default()
    };
    match FlashNextEngine::load(&dir, options) {
        Ok(engine) => Some(engine),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next engine at {rows} rows: {e}"));
            None
        }
    }
}

/// The G1 prompts and test2048 windows 3, 9, 17 and 33 (40 tokens each):
/// one a lane, or None with a skip.
fn eight_prompts() -> Option<Vec<Vec<u32>>> {
    let mut prompts = g1_prompts()?;
    for window in [3, 9, 17, 33] {
        prompts.push(test2048_window(window, 40)?);
    }
    assert_eq!(prompts.len(), 8, "eight prompts, one a lane");
    Some(prompts)
}

/// One round's picks, kept small: its lanes, columns a lane, each row's
/// pick (`[lanes][columns]`) and a hash of its bits.
#[derive(Debug, PartialEq)]
struct RoundPicks {
    lanes: Vec<usize>,
    columns: usize,
    picks: Vec<u32>,
    bits: u64,
}

fn picks_of(round: &RoundLogits<'_>, vocab: usize) -> RoundPicks {
    let picks = round.rows.chunks_exact(vocab).map(|row| logit_drift::argmax(row).0).collect();
    // FNV-1a over the rows' bits: two runs' rows compared without keeping them.
    let bits = round.rows.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
    RoundPicks { lanes: round.lanes.to_vec(), columns: round.columns, picks, bits }
}

/// Exact at any budget (ADR 0056): every lane of every round committed its
/// own rows' picks -- `offered(round, lane, position)` the drafts the lane
/// was offered at its anchor's position, cut to its extent here. A line per
/// lane that did not.
fn commits_follow_picks(
    run: &Outcome,
    picked: &[RoundPicks],
    offered: &dyn Fn(usize, usize, usize) -> Vec<u32>,
    what: &str,
) -> Vec<String> {
    assert_eq!(run.rounds.len(), picked.len(), "{what}: one round's picks a round");
    let mut at = vec![0usize; run.tokens.len()];
    let mut refused = Vec::new();
    for (r, (round, round_picks)) in run.rounds.iter().zip(picked).enumerate() {
        assert_eq!(round.len(), round_picks.lanes.len(), "{what}: round {r}'s lanes");
        for (a, (lane_round, &lane)) in round.iter().zip(&round_picks.lanes).enumerate() {
            let (p, committed) = (at[lane], lane_round.committed as usize);
            let drafts = offered(r, lane, p);
            assert!(lane_round.extent as usize <= drafts.len(), "{what}: round {r} lane {lane} verified more than it was offered");
            let commit = LaneCommit {
                anchor: run.tokens[lane][p],
                drafts: &drafts[..lane_round.extent as usize],
                committed: &run.tokens[lane][p..p + committed],
                next: run.tokens[lane].get(p + committed).copied(),
                picks: &round_picks.picks[a * round_picks.columns..(a + 1) * round_picks.columns],
            };
            if let Err(e) = commit_follows_argmax(&commit) {
                refused.push(format!("{what} round {r} lane {lane}: {e}"));
            }
            at[lane] = p + committed;
        }
    }
    refused
}

/// One verify row on the teacher's text: its round, lane, column and the
/// text index its column held, its drift from the plain route's row there,
/// and both routes' log-probability of the text's next token (plain, wide;
/// None at the text's last token).
struct Forced {
    round: usize,
    lane: usize,
    column: usize,
    at: usize,
    drift: RowDrift,
    worst: String,
    log_probs: Option<(f64, f64)>,
}

/// One cell of a budget: `set` on the load's lanes, every check of ADR
/// 0056's policy. Panics on an exact check; returns a line per row or lane
/// outside a tolerance, or per commit that is not its rows' picks.
fn budget_cell(engine: &mut FlashNextEngine, kv_format: KvFormat, set: &[Vec<u32>], what: &str) -> Vec<String> {
    let vocab = engine.vocab();
    let lanes = set.len();
    let mut refused = Vec::new();

    // The state, exact: a rejected draft is a draft never made, and so in
    // BF16 KV is an accepted one (the oracle: none's own text as drafts).
    let none = run(engine, set, Some(&mut |_, _, _| Vec::new()));
    let reject = run(engine, set, Some(&mut |_, _, w| vec![REJECTED; w as usize]));
    assert_eq!(reject.tokens, none.tokens, "{what}: the reject drafter committed other text than none");
    for (lane, (p, q)) in reject.probes.iter().zip(&none.probes).enumerate() {
        assert_eq!(same_bits(p, q), 0, "{what}: lane {lane}'s state after the reject drafter's rounds is not none's");
    }
    assert!(reject.rounds.iter().flatten().any(|l| l.extent > 0), "{what}: the reject drafter's drafts were never verified");
    if kv_format == KvFormat::Bf16 {
        let reference = none.tokens.clone();
        let oracle = run(
            engine,
            set,
            Some(&mut |lane, emitted, w| {
                let next = emitted.len() + 1;
                reference[lane][next.min(TOKENS)..(next + w as usize).min(TOKENS)].to_vec()
            }),
        );
        let accepted: u32 = oracle.rounds.iter().flatten().map(|l| l.committed - 1).sum();
        let proposed: u32 = oracle.rounds.iter().flatten().map(|l| l.extent).sum();
        assert_eq!(accepted, proposed, "{what}: an oracle draft was rejected");
        assert_eq!(oracle.tokens, none.tokens, "{what}: the oracle committed other text than none");
        for (lane, (p, q)) in oracle.probes.iter().zip(&none.probes).enumerate() {
            assert_eq!(same_bits(p, q), 0, "{what}: lane {lane}'s state after the oracle's rounds is not none's");
        }
    }

    // The teacher: speculation off, every round's rows kept. Its draw is its
    // rows' pick.
    let mut off_rows: Vec<Vec<u16>> = Vec::with_capacity(TOKENS);
    let off = engine
        .generate_with_logits(set, TOKENS, &mut |round| {
            assert_eq!((round.lanes.len(), round.columns), (lanes, 1), "{what}: a spec-off round's shape");
            off_rows.push(round.rows.to_vec());
        })
        .unwrap_or_else(|e| panic!("{what}: spec-off generate: {e}"));
    for (t, rows) in off_rows.iter().enumerate().take(TOKENS - 1) {
        for (lane, row) in rows.chunks_exact(vocab).enumerate() {
            assert_eq!(off[lane][t + 1], logit_drift::argmax(row).0, "{what}: spec-off lane {lane} drew past its row at {t}");
        }
    }

    // Teacher-forced: the verify rounds offered spec-off's text as drafts, so
    // every column holds the teacher's token while the lane's committed text
    // is the teacher's; each row is held to spec-off's at the same text index.
    let positions = std::cell::RefCell::new(vec![0usize; lanes]);
    let mut forced: Vec<Forced> = Vec::new();
    let mut taught_picks = Vec::new();
    let teacher = |lane: usize, p: usize| -> Vec<u32> { off[lane][(p + 1).min(TOKENS)..(p + 1 + DRAFT_TOKENS as usize).min(TOKENS)].to_vec() };
    let taught = run_with(
        engine,
        set,
        Some(&mut |lane, emitted, w| {
            positions.borrow_mut()[lane] = emitted.len();
            let mut drafts = teacher(lane, emitted.len());
            drafts.truncate(w as usize);
            drafts
        }),
        Some(&mut |round| {
            let positions = positions.borrow();
            for (a, &lane) in round.lanes.iter().enumerate() {
                for column in 0..round.columns {
                    let at = positions[lane] + column;
                    if at >= TOKENS {
                        continue;
                    }
                    let on = &round.rows[(a * round.columns + column) * vocab..][..vocab];
                    let plain = &off_rows[at][lane * vocab..][..vocab];
                    let log_probs = (at + 1 < TOKENS).then(|| {
                        let next = off[lane][at + 1];
                        (log_prob(plain, next), log_prob(on, next))
                    });
                    let drift = row_drift(plain, on);
                    let worst = format!(
                        "id {} moved {} -> {} (pick {} at {})",
                        drift.worst,
                        logit_drift::bf16(plain[drift.worst as usize]),
                        logit_drift::bf16(on[drift.worst as usize]),
                        drift.pick_off,
                        logit_drift::bf16(plain[drift.pick_off as usize])
                    );
                    forced.push(Forced { round: taught_picks.len(), lane, column, at, drift, worst, log_probs });
                }
            }
            taught_picks.push(picks_of(&round, vocab));
        }),
    );
    refused.extend(commits_follow_picks(&taught, &taught_picks, &|_, lane, p| teacher(lane, p), &format!("{what} teacher-forced")));

    // A row is on the teacher's text when its column was drafted and its
    // lane's text up to the round's anchor is the teacher's: a lane whose
    // pick moved leaves the comparison after that round (no cascade).
    let mut anchors = std::collections::HashMap::new();
    let mut valid = std::collections::HashSet::new();
    let mut at = vec![0usize; lanes];
    for (r, round) in taught.rounds.iter().enumerate() {
        for (lane_round, &lane) in round.iter().zip(&taught_picks[r].lanes) {
            let p = at[lane];
            anchors.insert((r, lane), p);
            if taught.tokens[lane][..=p] == off[lane][..=p] {
                for column in 0..=lane_round.extent as usize {
                    valid.insert((r, lane, column));
                }
            }
            at[lane] = p + lane_round.committed as usize;
        }
    }
    for f in &forced {
        assert_eq!(Some(&(f.at - f.column)), anchors.get(&(f.round, f.lane)), "{what}: round {} lane {}'s anchor", f.round, f.lane);
    }
    let left = (0..lanes).filter(|&lane| taught.tokens[lane] != off[lane]).count();
    let compared: Vec<&Forced> = forced.iter().filter(|f| valid.contains(&(f.round, f.lane, f.column))).collect();
    assert!(compared.len() >= lanes * (TOKENS - 1) / 2, "{what}: only {} teacher-forced rows compared", compared.len());
    for f in &compared {
        if let Err(e) = logit_drift::within(&f.drift, TOLERANCE_ULPS, PICK_GAP_ULPS) {
            refused.push(format!(
                "{what} teacher-forced round {} lane {} column {} (text index {}): {e}; {}",
                f.round, f.lane, f.column, f.at, f.worst
            ));
        }
    }
    let mut ulps: Vec<f32> = compared.iter().map(|f| f.drift.ulps).collect();
    ulps.sort_by(f32::total_cmp);
    let widest = compared.iter().max_by(|a, b| a.drift.ulps.total_cmp(&b.drift.ulps)).expect("a compared row");
    let moved: Vec<String> = compared
        .iter()
        .filter(|f| f.drift.pick_off != f.drift.pick_on)
        .map(|f| format!("lane {}@{} gap {} ulps", f.lane, f.at, f.drift.gap_ulps))
        .collect();
    let pairs: Vec<(f64, f64)> = compared.iter().filter_map(|f| f.log_probs).collect();
    let mean = |pick: fn(&(f64, f64)) -> f64| pairs.iter().map(pick).sum::<f64>() / pairs.len() as f64;
    let (plain_lp, wide_lp) = (mean(|p| p.0), mean(|p| p.1));
    if (wide_lp - plain_lp).abs() > MEAN_LOG_PROB_NATS {
        refused.push(format!(
            "{what} teacher-forced: mean log-probability {wide_lp} against spec-off's {plain_lp}, past {MEAN_LOG_PROB_NATS}"
        ));
    }

    // The head: it drafts and is accepted, its commits are its rows' picks,
    // and the same budget gives the same text, rounds, drafts, rows and
    // state run to run.
    let mut head_picks = Vec::new();
    let head = run_with(engine, set, None, Some(&mut |round| head_picks.push(picks_of(&round, vocab))));
    let mut again_picks = Vec::new();
    let again = run_with(engine, set, None, Some(&mut |round| again_picks.push(picks_of(&round, vocab))));
    assert_eq!((&again.tokens, &again.rounds, &again.head_drafts), (&head.tokens, &head.rounds, &head.head_drafts), "{what}: the head's run is not deterministic");
    assert!(again_picks == head_picks, "{what}: the head's rows differ run to run");
    for (lane, (p, q)) in again.probes.iter().zip(&head.probes).enumerate() {
        assert_eq!(same_bits(p, q), 0, "{what}: lane {lane}'s state after the head's run differs run to run");
    }
    let head_offered = |r: usize, lane: usize, _: usize| -> Vec<u32> {
        let Some(before) = r.checked_sub(1) else { return Vec::new() };
        let at = head_picks[before].lanes.iter().position(|&l| l == lane).expect("a lane runs every round until it ends");
        head.head_drafts[before][at].clone()
    };
    refused.extend(commits_follow_picks(&head, &head_picks, &head_offered, &format!("{what} head")));
    let alpha = acceptance(&head.rounds);
    let first = f64::from(alpha[0].0) / f64::from(alpha[0].1.max(1));
    if first < HEAD_ACCEPTANCE_FLOOR {
        refused.push(format!("{what}: the head's first drafts accepted {first:.3} ({alpha:?}), under {HEAD_ACCEPTANCE_FLOOR}"));
    }
    // The band: the head on the same prompts two lanes at a time, whose
    // rounds take 8 rows at any budget of 8 or more -- two lanes of 4
    // columns, the default's widest shape (a lone last lane, 4 rows).
    let speculation = engine.options().speculation.expect("speculation");
    assert!(2 * (speculation.window(2) as usize + 1) <= 8, "{what}: two lanes verify within 8 rows");
    let mut narrow = (0u32, 0u32);
    for pair in set.chunks(2) {
        let (accepted, offered) = acceptance(&run(engine, pair, None).rounds)[0];
        narrow = (narrow.0 + accepted, narrow.1 + offered);
    }
    if let Err(e) = acceptance_within_band(alpha[0], narrow, ACCEPTANCE_BAND) {
        refused.push(format!("{what}: {e}"));
    }

    println!(
        "{what}: teacher-forced {} rows ({left} lane(s) left at a moved pick), drift ulps max {} (lane {}@{}: {} \
         logits, {}) p99 {} median {}, picks moved [{}], mean log-prob spec-off {plain_lp:.6} wide {wide_lp:.6} (diff {:.3e}); head {} rounds, \
         acceptance {alpha:?} ({first:.3}), two lanes at a time {narrow:?} ({:.3})",
        compared.len(),
        ulps[ulps.len() - 1],
        widest.lane,
        widest.at,
        widest.drift.max_abs,
        widest.worst,
        ulps[ulps.len() * 99 / 100],
        ulps[ulps.len() / 2],
        moved.join(", "),
        wide_lp - plain_lp,
        head.rounds.len(),
        f64::from(narrow.0) / f64::from(narrow.1.max(1))
    );
    refused
}

/// GitHub #329 item 5: a round's logits copy cannot ride a permitted set
/// (its mask writes into the rows), so the leaf refuses the pair by name
/// (`program.cu`) -- and the same round without the copy runs.
fn a_permitted_set_refuses_the_logits_copy(engine: &FlashNextEngine, prompt: &[u32]) {
    let mut standing = engine
        .generate_speculative(&[prompt.to_vec()], 2, None)
        .unwrap_or_else(|e| panic!("generate_speculative: {e}"));
    let permitted = [0i32, 1];
    let lanes = [(SamplingParams::greedy(), &permitted[..])];
    let mut logits = vec![0u16; engine.vocab()];
    let err = engine
        .decode_standing(&mut standing.sequences, &mut standing.contexts, &lanes, Some(&mut logits))
        .expect_err("a permitted set with a logits copy is refused");
    assert!(err.contains("out_logits cannot ride a round with a permitted set"), "{err}");
    engine
        .decode_standing(&mut standing.sequences, &mut standing.contexts, &lanes, None)
        .unwrap_or_else(|e| panic!("the same round without the copy: {e}"));
    // A round emits the token the last draw left pending; its own draw, from
    // the set, now stands pending.
    let pending = standing.sequences[0].pending_token().expect("a pending token");
    assert!(permitted.contains(&pending), "the round drew {pending}, outside its permitted set");
}

/// Every cell of one verify budget, in both KV formats: `lane_sets` the
/// lanes of each cell, each a width whose window drafts.
fn budget(rows: u32, lane_sets: &[usize]) {
    let Some(prompts) = eight_prompts() else { return };
    let mut refused = Vec::new();
    for kv_format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
        // A load that skips outside the profile ends the cells, not the
        // refusals already collected in the other format.
        let Some(mut engine) = budget_engine(kv_format, rows) else { break };
        let speculation = engine.options().speculation.expect("speculation");
        let verifying = (1..=8u32).filter(|&w| speculation.window(w) > 0).fold(0, |mask, w| mask | 1 << (w - 1));
        assert_eq!(
            engine.verify_graphs_ready(),
            Ok(verifying),
            "{kv_format:?}: a verify graph did not capture at {rows} rows: {:?}",
            engine.graph_error()
        );
        a_permitted_set_refuses_the_logits_copy(&engine, &prompts[0]);
        for &lanes in lane_sets {
            let window = speculation.window(lanes as u32) as usize;
            assert!(window > 0 && lanes * (window + 1) <= rows as usize, "{lanes} lanes draft within {rows} rows");
            let what = format!("{kv_format:?} {lanes} lanes x {} columns at a budget of {rows} rows", window + 1);
            refused.extend(budget_cell(&mut engine, kv_format, &prompts[..lanes], &what));
        }
        drop(engine);
    }
    assert!(refused.is_empty(), "{} refusal(s) under ADR 0056's policy:\n{}", refused.len(), refused.join("\n"));
}

/// GitHub #329 item 5, ADR 0056: the default's 8 rows (two lanes of 4
/// columns, four of 2) against the same rules as a wider budget -- where the
/// tolerance's floor is measured; its text is held bit for bit and to the
/// near-tie rule by the tests above.
#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact and its MTP companion"]
fn the_default_budgets_rounds_keep_the_wide_budgets_rules() {
    budget(8, &[2, 4]);
}

/// GitHub #329 item 5, ADR 0056: past 8 rows a verify round's linears, mixes
/// and shared expert take their tensor-core routes and its MoE runs in
/// launches of 8, so its logits are spec-off's up to summation order: held
/// row by row to spec-off's within `TOLERANCE_ULPS` on spec-off's own text,
/// its commits to its own rows' picks, its state to none's, its runs to
/// themselves, its quality to spec-off's and its head's acceptance to a
/// floor. Four lanes verify 3 drafts (16 rows), five 2 (15), eight 1 (16).
#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact and its MTP companion"]
fn a_sixteen_row_budget_is_numerically_equivalent_to_speculation_off() {
    budget(16, &[4, 5, 8]);
}

/// As at 16 rows: six lanes verify 3 drafts (24 rows), eight 2 (24).
#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact and its MTP companion"]
fn a_twenty_four_row_budget_is_numerically_equivalent_to_speculation_off() {
    budget(24, &[6, 8]);
}

/// As at 16 rows: eight lanes verify 3 drafts (32 rows).
#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact and its MTP companion"]
fn a_thirty_two_row_budget_is_numerically_equivalent_to_speculation_off() {
    budget(32, &[8]);
}
