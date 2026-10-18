//! The loop guard on `/metrics` (GitHub #330, spec server/14): what an
//! operator reads after the requests the guard watched, over chat
//! completions and `POST /v1/responses`, through the in-process router over
//! `MockCompute`. The socket's rewind is counted in `responses_socket.rs`.

#[path = "support/responses.rs"]
mod responses;

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use ignis_core::{MockCompute, SchedulerConfig};
use ignis_server::loop_guard::{LoopGuardMode, LoopGuardParams, TAG_PREFIX};
use ignis_server::metrics::Metrics;
use serde_json::{json, Value as JsonValue};
use tower::ServiceExt;

use responses::{server, sse_events, Script, MODEL};

/// POST `body` to `path`; the status and the body text.
async fn post(app: &axum::Router, path: &str, body: JsonValue) -> (u16, String) {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

/// Prose, then a loop the guard rewinds.
fn looping_reasoning() -> String {
    responses::prose(3000) + &responses::looping(4000)
}

fn rewind() -> LoopGuardParams {
    LoopGuardParams { mode: LoopGuardMode::Rewind, ..LoopGuardParams::default() }
}

/// A server with metrics on and the guard under `guard`, whose request 0
/// reasons `reasoning` and whose continuation (request 1) reasons silently
/// for 20 tokens and stops; and its metrics.
fn guarded(reasoning: &str, guard: LoopGuardParams) -> (axum::Router, Arc<Metrics>) {
    let mock = Arc::new(MockCompute::new());
    mock.eos_after(1, 20);
    let script = Script::encoding(responses::scripted_text(&mock, 0, reasoning));
    let server = server(&script, SchedulerConfig::default(), mock).with_loop_guard(guard).with_metrics();
    let metrics = Arc::clone(server.metrics.as_ref().unwrap());
    (server.app(), metrics)
}

/// The text of `series`' sample (its name and labels) in `metrics`'
/// exposition, when there is one.
fn sample(metrics: &Metrics, series: &str) -> Option<String> {
    metrics.render().lines().find_map(|line| line.strip_prefix(series)?.strip_prefix(' ').map(str::to_owned))
}

fn count(metrics: &Metrics, series: &str) -> u64 {
    sample(metrics, series).unwrap_or_else(|| panic!("no {series} in:\n{}", metrics.render())).parse().unwrap()
}

fn loops(metrics: &Metrics, outcome: &str) -> u64 {
    count(metrics, &format!("ignis_loop_guard_loops_total{{outcome=\"{outcome}\"}}"))
}

/// `POST /v1/responses` for `max_output_tokens`, streamed or not: the
/// terminal response object.
async fn respond(app: &axum::Router, max_output_tokens: u32, stream: bool) -> JsonValue {
    let body = json!({ "model": MODEL, "input": "the whole history of Rome", "max_output_tokens": max_output_tokens, "stream": stream });
    let (status, text) = post(app, "/v1/responses", body).await;
    assert_eq!(status, 200, "{text}");
    match stream {
        false => serde_json::from_str(&text).unwrap(),
        true => sse_events(&text).last().unwrap().1["response"].clone(),
    }
}

/// `POST /v1/chat/completions`, streamed or not: the reasoning the client
/// got and the usage.
async fn chat(app: &axum::Router, stream: bool) -> (String, JsonValue) {
    let body = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "the whole history of Rome" }],
        "max_tokens": 4000,
        "stream": stream,
        "stream_options": { "include_usage": true },
    });
    let (status, text) = post(app, "/v1/chat/completions", body).await;
    assert_eq!(status, 200, "{text}");
    if !stream {
        let answer: JsonValue = serde_json::from_str(&text).unwrap();
        let reasoning = answer["choices"][0]["message"]["reasoning_content"].as_str().unwrap_or_default().to_owned();
        return (reasoning, answer["usage"].clone());
    }
    let chunks: Vec<JsonValue> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    let reasoning = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["reasoning_content"].as_str())
        .collect();
    let usage = chunks.iter().rev().find(|chunk| !chunk["usage"].is_null()).expect("the usage chunk")["usage"].clone();
    (reasoning, usage)
}

/// A server that runs the guard `off` exports nothing of it, a looping
/// request served; one that runs it has nothing to say before it watched a
/// request.
#[tokio::test]
async fn nothing_of_the_guard_is_exported_before_it_watches_a_request_or_ever_when_off() {
    let (app, metrics) = guarded(&looping_reasoning(), LoopGuardParams { mode: LoopGuardMode::Off, ..rewind() });
    let response = respond(&app, 4000, false).await;
    assert!(!response["output"][0]["content"][0]["text"].as_str().unwrap().contains(TAG_PREFIX));
    assert!(!metrics.render().contains("ignis_loop_guard_"), "{}", metrics.render());

    let (_, metrics) = guarded(&looping_reasoning(), rewind());
    assert!(!metrics.render().contains("ignis_loop_guard_"), "{}", metrics.render());
}

/// A reasoning that never loops: watched, not looped, its lowest score
/// well above the threshold (prose scores 0.45-0.49).
#[tokio::test]
async fn a_reasoning_that_never_loops_is_watched_and_scored_above_the_threshold() {
    let (app, metrics) = guarded(&responses::prose(3000), rewind());
    let response = respond(&app, 400, false).await;
    assert!(response["usage"]["output_tokens_details"].get("rewound_tokens").is_none(), "{response}");
    assert_eq!(count(&metrics, "ignis_loop_guard_watched_requests_total"), 1);
    assert_eq!(count(&metrics, "ignis_loop_guard_looped_requests_total"), 0);
    assert_eq!(count(&metrics, "ignis_loop_guard_rewound_tokens_total"), 0);
    assert_eq!(count(&metrics, "ignis_loop_guard_min_ratio_count"), 1);
    assert_eq!(count(&metrics, "ignis_loop_guard_min_ratio_bucket{le=\"0.4\"}"), 0);
    assert_eq!(count(&metrics, "ignis_loop_guard_min_ratio_bucket{le=\"0.5\"}"), 1);
}

/// A rewind on `/v1/responses`, streamed and not, read as soon as the
/// response is: one watched request that looped once, its loop rewound,
/// the tokens it took back the usage's `rewound_tokens`, and its lowest
/// score the loop's. The rewind's cancel and continuation are an engine
/// request each: the request counters count them as such.
#[tokio::test]
async fn a_rewind_on_responses_is_counted_with_the_tokens_its_usage_names() {
    for stream in [false, true] {
        let (app, metrics) = guarded(&looping_reasoning(), rewind());
        let response = respond(&app, 4000, stream).await;
        let rewound = response["usage"]["output_tokens_details"]["rewound_tokens"].as_u64().expect("rewound");
        assert_eq!(count(&metrics, "ignis_loop_guard_watched_requests_total"), 1);
        assert_eq!(count(&metrics, "ignis_loop_guard_looped_requests_total"), 1);
        assert_eq!((loops(&metrics, "rewound"), loops(&metrics, "closed"), loops(&metrics, "detected")), (1, 0, 0));
        assert_eq!(count(&metrics, "ignis_loop_guard_rewound_tokens_total"), rewound);
        assert_eq!(count(&metrics, "ignis_loop_guard_min_ratio_count"), 1);
        assert_eq!(count(&metrics, "ignis_loop_guard_min_ratio_bucket{le=\"0.25\"}"), 1);
        responses::metric_reaches(&metrics, "ignis_requests_completed_total 1").await;
        assert_eq!(count(&metrics, "ignis_requests_accepted_total"), 2, "the request and its continuation");
        assert_eq!(count(&metrics, "ignis_requests_cancelled_total"), 1, "the rewind's cancel");
    }
}

/// The same rewind on chat completions, streamed and not.
#[tokio::test]
async fn a_rewind_on_chat_completions_is_counted_the_same() {
    for stream in [false, true] {
        let (app, metrics) = guarded(&looping_reasoning(), rewind());
        let (reasoning, usage) = chat(&app, stream).await;
        assert!(reasoning.contains(TAG_PREFIX), "rewound: {reasoning}");
        let rewound = usage["completion_tokens_details"]["rewound_tokens"].as_u64().expect("rewound");
        assert_eq!(count(&metrics, "ignis_loop_guard_watched_requests_total"), 1);
        assert_eq!(count(&metrics, "ignis_loop_guard_looped_requests_total"), 1);
        assert_eq!(loops(&metrics, "rewound"), 1);
        assert_eq!(count(&metrics, "ignis_loop_guard_rewound_tokens_total"), rewound);
    }
}

/// With no nudge left the guard closes the reasoning: `closed`, and not the
/// thinking budget's forced close, which the continuation, its close
/// prefilled and no budget, never reaches.
#[tokio::test]
async fn a_guard_close_is_counted_as_closed_and_not_as_a_forced_close() {
    let (app, metrics) = guarded(&looping_reasoning(), LoopGuardParams { retries: 0, ..rewind() });
    let response = respond(&app, 4000, false).await;
    let rewound = response["usage"]["output_tokens_details"]["rewound_tokens"].as_u64().expect("rewound");
    assert_eq!((loops(&metrics, "rewound"), loops(&metrics, "closed")), (0, 1));
    assert_eq!(count(&metrics, "ignis_loop_guard_rewound_tokens_total"), rewound);
    responses::metric_reaches(&metrics, "ignis_requests_completed_total 1").await;
    assert!(!metrics.render().contains("ignis_thinking_forced_closes_total"), "{}", metrics.render());
}

/// In `log` mode a loop is only logged: `detected`, nothing rewound, one
/// looped request however many times the loop is logged.
#[tokio::test]
async fn a_loop_in_log_mode_is_counted_as_detected() {
    let (app, metrics) = guarded(&looping_reasoning(), LoopGuardParams { mode: LoopGuardMode::Log, ..rewind() });
    let response = respond(&app, 1000, false).await;
    assert!(!response["output"][0]["content"][0]["text"].as_str().unwrap().contains(TAG_PREFIX));
    assert!(loops(&metrics, "detected") >= 1, "{}", metrics.render());
    assert_eq!((loops(&metrics, "rewound"), count(&metrics, "ignis_loop_guard_rewound_tokens_total")), (0, 0));
    assert_eq!(count(&metrics, "ignis_loop_guard_looped_requests_total"), 1);
}

/// A reasoning that opens in a loop leaves nothing to keep: the loop is
/// held, `too_little_kept`, and never rewound.
#[tokio::test]
async fn a_reasoning_that_opens_in_a_loop_is_counted_as_held() {
    let (app, metrics) = guarded(&responses::looping(8000), rewind());
    let response = respond(&app, 1100, false).await;
    assert!(response["usage"]["output_tokens_details"].get("rewound_tokens").is_none(), "{response}");
    assert_eq!(loops(&metrics, "too_little_kept"), 1);
    assert_eq!(loops(&metrics, "rewound"), 0);
    assert_eq!(count(&metrics, "ignis_loop_guard_looped_requests_total"), 1);
}
