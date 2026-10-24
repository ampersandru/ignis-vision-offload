//! The `small-card` preset end to end (#340, spec 27b-16gb/05 Part B AC 7):
//! the whole record assembled over the real HTTP transport against the
//! in-process mock engine, extended with the server's `/v1/tokenize`,
//! `/v1/config`, `/v1/config/origins` and `/metrics`. CPU-only (ADR 0006).
//!
//! What a unit test cannot give: that the pieces are read from the routes the
//! server really serves, that the agent turns go out as recorded chat bodies
//! (tools and all), and that a missing piece degrades to `null` + a note.

mod common;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use common::{MockEngine, MAX_MODEL_LEN};
use ignis_bench::small_card::{self, Options};
use serde_json::{json, Value};

/// `/v1/tokenize` as the mock counts a prompt: one token per word, the ids
/// hashed from the words (the server's `return_token_ids` shape).
async fn tokenize(Json(body): Json<Value>) -> Json<Value> {
    let text = body["messages"][0]["content"].as_str().unwrap_or("").to_string();
    let ids: Vec<u32> =
        text.split_whitespace().map(|w| w.bytes().fold(7u32, |a, b| a.wrapping_mul(131).wrapping_add(b as u32))).collect();
    Json(json!({ "count": ids.len(), "max_model_len": MAX_MODEL_LEN, "token_ids": ids }))
}

async fn config() -> Json<Value> {
    Json(json!({
        "server": { "bind": "0.0.0.0:8080", "api_key": "sk-must-not-leak" },
        "model": { "artifact": "C:/Users/someone/x.ninfer", "max_context": 262144, "prefill_chunk": 1024 },
        "vram": { "budget_bytes": 17179869184u64 },
        "spec": { "decode_lanes": 2, "draft_head": "auto" },
    }))
}

async fn origins() -> Json<Value> {
    Json(json!({ "object": "list", "data": [
        { "key": "model.max_context", "scope": null, "source": "profile", "scoped": false, "spelling": "rtx50-16gb" },
        { "key": "vram.budget_bytes", "scope": null, "source": "flag", "scoped": false, "spelling": "--vram-budget-bytes" },
    ]}))
}

/// Each scrape moves the speculative counters on by 10 rounds.
async fn metrics(State(calls): State<Arc<AtomicU64>>) -> String {
    let n = calls.fetch_add(1, Ordering::Relaxed) + 1;
    format!(
        "ignis_build_info{{version=\"0.7.3-test\"}} 1\n\
         ignis_vram_reserved_bytes{{line=\"weights\"}} 12000000000\n\
         ignis_vram_reserved_bytes{{line=\"trellis_workspace\"}} 805306368\n\
         ignis_vram_budget_bytes 17179869184\n\
         ignis_kv_pool_pages 512\n\
         ignis_kv_page_bytes 4194304\n\
         ignis_speculative_rounds_total {}\n\
         ignis_speculative_drafted_tokens_total {}\n\
         ignis_speculative_accepted_tokens_total {}\n",
        10 * n,
        80 * n,
        60 * n
    )
}

fn server_routes() -> Router {
    Router::new()
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/config", get(config))
        .route("/v1/config/origins", get(origins))
        .route("/metrics", get(metrics).with_state(Arc::new(AtomicU64::new(0))))
}

const SMI: &str = "NVIDIA GeForce RTX 5070 Ti, 16303 MiB, 4, 16\n";

fn opts(url: &str, metrics: Option<String>) -> Options {
    Options { endpoint: url.to_string(), metrics, cells: vec![200, 30_000], samples: 2, decode_tokens: 16 }
}

#[test]
fn the_record_is_assembled_from_what_the_server_exposes() {
    let engine = MockEngine::start_with(server_routes());
    // userinfo in the endpoint rides to the server and never into the record
    let url = engine.url().replace("http://", "http://user:secret@");
    let report = small_card::run(&opts(&url, Some(engine.url().to_string())), &|| Ok(SMI.to_string()));
    let json: Value = serde_json::to_value(&report).unwrap();

    // A sample of the record for the report: SMALL_CARD_SAMPLE_OUT=<path>.
    if let Ok(path) = std::env::var("SMALL_CARD_SAMPLE_OUT") {
        std::fs::write(path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    }

    assert_eq!(json["schema"], "ignis-small-card/1");
    assert_eq!(json["bench_version"], env!("CARGO_PKG_VERSION"));
    assert!(json["started_at"].as_str().unwrap().ends_with('Z') && json["finished_at"].as_str().unwrap().ends_with('Z'));
    assert!(json["endpoint"].as_str().unwrap().starts_with("http://127.0.0.1:"));

    assert_eq!(json["card"]["name"], "NVIDIA GeForce RTX 5070 Ti");
    assert_eq!(json["card"]["vram_total_mib"], 16303);
    assert_eq!((&json["card"]["pcie_gen_current"], &json["card"]["pcie_width_current"]), (&json!(4), &json!(16)));

    assert_eq!(json["server"]["version"], "0.7.3-test");
    assert_eq!(json["server"]["model_id"], common::MODEL);
    assert_eq!(json["server"]["profile"], "rtx50-16gb");
    assert_eq!(json["server"]["max_model_len"], MAX_MODEL_LEN);
    assert_eq!(json["plan"]["budget_bytes"], 17_179_869_184u64);
    assert_eq!(json["plan"]["reserved_bytes_by_line"]["trellis_workspace"], 805_306_368u64);
    assert_eq!(json["plan"]["kv_pool_pages"], 512);
    assert_eq!(json["config"]["spec.decode_lanes"]["value"], 2);
    assert_eq!(json["config"]["vram.budget_bytes"]["spelling"], "--vram-budget-bytes");
    // the context the config asks for is above what the mock runs: lowered
    assert_eq!(json["context"]["configured_max_context"], 262144);
    assert_eq!(json["context"]["lowered"], true);

    // decode: the four coding prompts and the three agent turns, all timed
    for (set, n) in [("coding", 4), ("agent", 3)] {
        let samples = json["decode"][set]["samples"].as_array().unwrap();
        assert_eq!(samples.len(), n, "{set}");
        assert!(samples.iter().all(|s| s["tokens_per_s"].as_f64().is_some_and(|r| r > 0.0)), "{set}: {samples:?}");
        assert!(json["decode"][set]["median_tokens_per_s"].as_f64().unwrap() > 0.0, "{set}");
        assert_eq!(json["decode"][set]["errors"], json!([]), "{set}");
    }
    // the speculative counters moved by one scrape's worth between the two
    // reads around the decode phase
    assert_eq!(json["decode"]["speculative"]["rounds"], 10);
    assert_eq!(json["decode"]["speculative"]["accepted_tokens"], 60);
    assert_eq!(json["decode"]["speculative"]["acceptance"], 0.75);

    // ttft: the small cell is measured cold, the large one does not fit
    let cells = json["ttft"].as_array().unwrap();
    assert_eq!(cells[0]["prompt_tokens"], 200);
    assert!(cells[0]["median_ttft_ms"].as_f64().unwrap() > 0.0);
    assert_eq!(cells[0]["all_cold"], true);
    assert_eq!(cells[0]["samples_ms"].as_array().unwrap().len(), 2);
    assert_eq!(cells[1]["prompt_tokens"], 30_000);
    assert_eq!(cells[1]["median_ttft_ms"], Value::Null);
    assert!(cells[1]["error"].as_str().unwrap().contains("does not fit"), "{:?}", cells[1]);

    // the agent turns went out as recorded bodies: tools and history intact
    let bodies = engine.state.bodies();
    let agent: Vec<&Value> = bodies.iter().filter(|b| b.get("tools").is_some()).collect();
    assert_eq!(agent.len(), 3, "three agent turns sent with their tools");
    assert!(agent.iter().all(|b| b["messages"].as_array().unwrap().iter().any(|m| m["role"] == "tool")));
    assert!(agent.iter().all(|b| b["max_tokens"] == 16 && b["stream"] == true));

    // nothing private reaches the record
    let text = serde_json::to_string(&json).unwrap();
    for leak in ["secret", "sk-must-not-leak", "someone", "api_key", "authorization", "Authorization", "0.0.0.0"] {
        assert!(!text.contains(leak), "{leak} leaked into the record");
    }
}

#[test]
fn a_missing_piece_is_null_with_a_note_and_the_rest_is_still_measured() {
    // The plain mock: models and completions only. No config, no tokenizer,
    // no metrics listener; and no nvidia-smi.
    let engine = MockEngine::start();
    let report = small_card::run(&opts(engine.url(), None), &|| Err("could not run nvidia-smi: not found".to_string()));
    let json: Value = serde_json::to_value(&report).unwrap();

    assert_eq!(json["card"], Value::Null);
    assert_eq!(json["config"], Value::Null);
    assert_eq!(json["server"]["version"], Value::Null);
    assert_eq!(json["server"]["profile"], Value::Null);
    assert_eq!(json["plan"]["budget_bytes"], Value::Null);
    assert_eq!(json["plan"]["reserved_bytes_by_line"], Value::Null);
    assert_eq!(json["decode"]["speculative"], Value::Null);
    assert_eq!(json["context"]["lowered"], Value::Null);
    // the same keys are there, null, not absent
    assert!(json.as_object().unwrap().contains_key("card") && json["plan"].as_object().unwrap().contains_key("kv_pool_pages"));

    let notes: Vec<&str> = json["notes"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    for want in ["card:", "config:", "metrics: not read", "tokenize"] {
        assert!(notes.iter().any(|n| n.contains(want)), "no note for `{want}` in {notes:?}");
    }
    // the cells say why they have no number
    let cells = json["ttft"].as_array().unwrap();
    assert!(cells[0]["error"].as_str().unwrap().contains("tokenizer"), "{:?}", cells[0]);
    assert_eq!(cells[0]["median_ttft_ms"], Value::Null);
    // decode needs only the completions route
    assert_eq!(json["decode"]["coding"]["samples"].as_array().unwrap().len(), 4);
    assert!(json["decode"]["agent"]["median_tokens_per_s"].as_f64().unwrap() > 0.0);
}

#[test]
fn an_unreachable_server_yields_a_record_that_says_so_and_does_not_panic() {
    // Port 1 on localhost refuses at once.
    let report = small_card::run(&opts("http://127.0.0.1:1", None), &|| Ok(SMI.to_string()));
    assert_eq!(report.server.model_id, None);
    assert!(report.decode.coding.is_none() && report.ttft.is_empty());
    assert!(report.notes.iter().any(|n| n.contains("nothing was measured")), "{:?}", report.notes);
    assert_eq!(report.card.as_ref().and_then(|c| c.vram_total_mib), Some(16303));
}
