//! Spec 27b-16gb/05a (GitHub #339): the 16 GB artifact served end to end by
//! ignis, through the server's own load path (`load::prepare_model`,
//! `load::load_model`), on the 5090 inside the 16 GiB budget a 16 GB card
//! has: `--profile rtx50-16gb --vram-budget-bytes 17179869184`.
//!
//! The load binds the trellis body by its head (no flag names it), plans the
//! trellis workspace as its own VRAM line, runs the profile's two decode
//! lanes with DFlash2, and serves greedy coding completions that are coherent
//! (they hold what the prompt asks for) and deterministic (the same prompt
//! twice gives the same text). Each request's speculative rounds are read off
//! its `ignis.request.done` line, and the tokens per round printed beside the
//! plan's lines -- the figures Acceptance A1 and A2 report.
//!
//! Its own test binary: the request log is written on the engine's telemetry
//! task, so the capture is the process's global subscriber.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or an unavailable GPU is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::http::Request;
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use ignis_core::compute::ModelFamily;
use ignis_core::gpu_profile;
use ignis_server::Server;

const TRELLIS: &str = "F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/qwen3_8_27b_trellis_sc30-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
/// A 16 GB card's whole memory: spec 05a Acceptance A1's budget.
const BUDGET: &str = "17179869184";

fn the_27b(flags: &[&str]) -> ignis_server::config::Config {
    let args: Vec<String> = flags.iter().map(|flag| flag.to_string()).collect();
    let ignis_server::config::ConfigOutcome::Config(config) =
        ignis_server::config::resolve(&args, |_| None).expect("resolve")
    else {
        panic!("expected a runnable config");
    };
    config.for_family(ModelFamily::Qwen38_27b).expect("the 27B")
}

fn named<'a>(events: &'a [serde_json::Value], name: &str) -> Vec<&'a serde_json::Value> {
    events.iter().filter(|event| event["event_name"] == name).collect()
}

async fn complete(app: &axum::Router, prompt: &str, max_tokens: u32) -> String {
    let body = serde_json::json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": false,
        "enable_thinking": false
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string().into_bytes()))
        .expect("build the request");
    let response = app.clone().oneshot(request).await.expect("route the request");
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.expect("read the body");
    let text = String::from_utf8(bytes.to_vec()).expect("the body is UTF-8");
    let value: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
    assert_eq!(status, 200, "{value}");
    value["choices"][0]["message"]["content"].as_str().unwrap_or_else(|| panic!("{value}")).to_owned()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn the_16gb_artifact_serves_greedy_code_on_two_dflash2_lanes_in_a_16_gib_budget() {
    if !Path::new(TRELLIS).exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {TRELLIS}")) {
        return;
    }
    let sink = Arc::new(ignis_logging::MemorySink::new());
    tracing_subscriber::registry()
        .with(ignis_logging::JsonLayer::new(sink.clone()))
        .try_init()
        .expect("this test binary installs the only subscriber");
    let events = || -> Vec<serde_json::Value> {
        sink.lines().iter().map(|line| serde_json::from_str(line).expect("json")).collect()
    };

    let flags = [
        "--profile",
        "rtx50-16gb",
        "--vram-budget-bytes",
        BUDGET,
        "--model-idle-unload",
        "0",
        "--model-loop-guard",
        "off",
    ];
    let options = ignis_server::config::Config { model: MODEL.to_owned(), model_named: true, ..the_27b(&flags) };
    let loaded = match ignis_server::load::prepare_model(&options, Path::new(TRELLIS)).and_then(ignis_server::load::load_model)
    {
        Ok(loaded) => loaded,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the 16 GB artifact under rtx50-16gb: {e}"));
            return;
        }
    };
    let model = loaded.model;
    let logged = events();
    let plan = named(&logged, "ignis.runtime.vram_plan");
    assert_eq!(plan.len(), 1, "{logged:?}");
    println!("vram_plan: {}", plan[0]);
    let lines = &plan[0]["attributes"];
    assert_eq!(lines["budget_bytes"], 17_179_869_184u64);
    assert_eq!(lines["mode"], "explicit");
    let trellis_workspace = lines["trellis_workspace_bytes"].as_u64().expect("the trellis line");
    assert!(trellis_workspace > 0, "the trellis body plans its workspace");
    let weights = lines["weights_bytes"].as_u64().unwrap();
    // Spec 01's As built: the text scope is 9,825,087,488 bytes of device
    // arena; the drafter adds its 66 objects (the full head: no shortlist).
    assert!((9_825_087_488..11_264_998_400).contains(&weights), "{weights}");
    assert!(lines["total_bytes"].as_u64().unwrap() <= 17_179_869_184, "the plan fits the budget");
    println!(
        "plan: weights {weights}, trellis workspace {trellis_workspace}, workspace {}, kv pool {} tokens, adapted '{}'",
        lines["workspace_bytes"], lines["kv_pool_tokens"], lines["adapted"]
    );
    let load = named(&logged, "ignis.model.loaded");
    println!("model.loaded: {}", load[0]);
    assert_eq!(load[0]["attributes"]["decode_lanes"], 2);
    let speculation = load[0]["attributes"]["speculation"].as_str().unwrap_or_default();
    assert!(speculation.starts_with("dflash2 draft_tokens=7 draft_head=full"), "{speculation}");

    let app = Server::from_active(model.clone()).with_request_timeout(Duration::from_secs(600)).app();
    let prompts = [
        (
            "Write a Python function `is_prime(n)` that returns True when n is a prime number. Reply with the code only.",
            "def is_prime",
        ),
        ("Write a Rust function `fn gcd(a: u64, b: u64) -> u64` using Euclid's algorithm. Reply with the code only.", "fn gcd"),
    ];
    let mut rounds = 0u64;
    let mut accepted = 0u64;
    for (prompt, needle) in prompts {
        let first = complete(&app, prompt, 160).await;
        let again = complete(&app, prompt, 160).await;
        println!("completion: {first:?}");
        assert!(first.contains(needle), "the completion holds `{needle}`: {first:?}");
        assert_eq!(first, again, "greedy decoding is deterministic");
    }
    // The request log is written on the telemetry task after each response.
    let mut done = Vec::new();
    for _ in 0..100 {
        done = events().into_iter().filter(|e| e["event_name"] == "ignis.request.done").collect();
        if done.len() >= 2 * prompts.len() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(done.len(), 2 * prompts.len(), "every request logs its done line");
    for line in &done {
        let a = &line["attributes"];
        let (r, d, c) = (a["spec.rounds"].as_u64(), a["spec.drafted"].as_u64(), a["spec.accepted"].as_u64());
        println!(
            "request {}: tokens {}, {:.1} tok/s, spec.rounds {r:?}, drafted {d:?}, accepted {c:?}, pos {}",
            a["request_id"], a["tokens"], a["tok_s"].as_f64().unwrap_or(0.0), a["spec.pos"]
        );
        rounds += r.expect("a DFlash2 load's request runs verify rounds");
        accepted += c.unwrap_or(0);
    }
    // A round commits its accepted drafts and the target's own token.
    let per_round = (accepted + rounds) as f64 / rounds as f64;
    println!("tokens per round: {per_round:.2} over {rounds} rounds");
    assert!(per_round > 1.0, "the drafter's proposals are accepted at all: {per_round}");

    drop(app);
    tokio::task::spawn_blocking(move || model.shut_down()).await.expect("the model shuts down");
}
