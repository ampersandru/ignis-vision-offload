//! `ignis-bench replay --as-recorded` and `--retry-engine-full`
//! (GitHub #329 item 14): an agentic trace replayed as the client sent it,
//! and an engine that is momentarily full waited out instead of recorded
//! as a failure. Through the real HTTP transport against the in-process
//! mock engine; the default behaviour (flat user message, no retry) is
//! pinned alongside.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::MockEngine;
use ignis_bench::client::{replay, Endpoint, HttpEndpoint, ReplayConfig, Request, RetryPolicy};
use ignis_bench::trace::{RequestClass, Trace};

/// A raw chat request body as a Qwen Code client sends it.
fn recorded_body() -> serde_json::Value {
    serde_json::json!({
        "model": "recorded-model",
        "messages": [
            { "role": "system", "content": "You are a coding agent." },
            { "role": "user", "content": "list the files" },
        ],
        "tools": [{ "type": "function", "function": { "name": "ls", "parameters": {} } }],
        "tool_choice": "auto",
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 99999,
        "stream": false,
    })
}

fn request_for(prompt: String) -> Request {
    Request {
        id: "r".into(),
        class: RequestClass::Main,
        prompt,
        max_tokens: 6,
        stream: true,
        include_usage: true,
        enable_thinking: None,
        images: Vec::new(),
    }
}

fn trace_of(prompt: &str, n: usize) -> Trace {
    let lines: Vec<String> = (0..n)
        .map(|i| {
            serde_json::json!({
                "id": format!("r{i}"), "class": "sub", "t_arrive_ms": 0,
                "prompt": prompt, "max_tokens": 4, "stream": true,
            })
            .to_string()
        })
        .collect();
    Trace::from_jsonl(&lines.join("\n")).expect("valid trace")
}

fn no_wait() -> ReplayConfig {
    ReplayConfig { max_concurrency: 1, time_scale: 0.0 }
}

#[test]
fn by_default_a_json_prompt_is_still_one_flat_user_message() {
    let engine = MockEngine::start();
    let ep = HttpEndpoint::new(engine.url());
    let prompt = recorded_body().to_string();
    ep.complete(&request_for(prompt.clone())).expect("completes");
    let sent = &engine.state.bodies()[0];
    assert_eq!(sent["messages"].as_array().unwrap().len(), 1);
    assert_eq!(sent["messages"][0]["content"], prompt.as_str());
    assert!(sent.get("tools").is_none());
}

#[test]
fn as_recorded_sends_the_recorded_messages_and_tools_and_keeps_what_the_harness_owns() {
    let engine = MockEngine::start();
    let ep = HttpEndpoint::new(engine.url()).as_recorded(true);
    ep.complete(&request_for(recorded_body().to_string())).expect("completes");
    let sent = &engine.state.bodies()[0];
    let want = recorded_body();
    // As recorded.
    assert_eq!(sent["messages"], want["messages"]);
    assert_eq!(sent["tools"], want["tools"]);
    assert_eq!(sent["tool_choice"], "auto");
    assert_eq!(sent["top_p"], 0.8, "other request fields ride along");
    // The harness's own.
    assert_eq!(sent["max_tokens"], 6, "the trace line's budget, not the recorded one");
    assert_eq!(sent["stream"], true);
    assert_eq!(sent["stream_options"]["include_usage"], true);
    assert_eq!(sent["temperature"], 0, "greedy");
    assert_eq!(sent["seed"], 0);
    assert_eq!(sent["class"], "interactive");
    assert!(sent.get("model").is_none(), "the recorded model name is not the harness's to send");
}

#[test]
fn as_recorded_sends_a_line_that_is_not_a_chat_body_as_today() {
    let engine = MockEngine::start();
    let ep = HttpEndpoint::new(engine.url()).as_recorded(true);
    ep.complete(&request_for("just a plain prompt".into())).expect("completes");
    ep.complete(&request_for(r#"{"no":"messages here"}"#.into())).expect("completes");
    let bodies = engine.state.bodies();
    assert_eq!(bodies[0]["messages"][0]["content"], "just a plain prompt");
    assert_eq!(bodies[1]["messages"][0]["content"], r#"{"no":"messages here"}"#);
}

#[test]
fn by_default_a_full_engine_is_recorded_as_a_failure_without_a_retry() {
    let engine = MockEngine::start();
    engine.state.set_fail_503_times(1, Some(0));
    let ep: Arc<dyn Endpoint> = Arc::new(HttpEndpoint::new(engine.url()));
    let results = replay(ep, &trace_of("p", 1), &no_wait());
    assert!(!results[0].ok);
    assert_eq!(results[0].retries, 0);
    assert_eq!(engine.state.bodies().len(), 1, "sent once");
}

#[test]
fn a_refused_request_is_resent_after_retry_after_and_the_wait_is_recorded() {
    let engine = MockEngine::start();
    engine.state.set_fail_503_times(2, Some(1));
    let policy = RetryPolicy { max_retries: 3, max_wait: Duration::from_millis(60) };
    let ep: Arc<dyn Endpoint> = Arc::new(HttpEndpoint::new(engine.url()).with_retry(policy));
    let results = replay(ep, &trace_of("p", 1), &no_wait());
    let m = &results[0];
    assert!(m.ok, "{m:?}");
    assert_eq!(m.retries, 2);
    // Retry-After: 1 s, bounded to 60 ms each wait.
    assert!(m.retry_wait_ms >= 110.0 && m.retry_wait_ms < 900.0, "waited {} ms", m.retry_wait_ms);
    assert_eq!(engine.state.bodies().len(), 3, "two refusals and the admission");
    assert!(m.total_ms < m.retry_wait_ms + 5_000.0 && m.ttft_ms <= m.total_ms);
}

#[test]
fn the_retries_are_bounded_and_a_request_that_never_fits_still_fails_with_its_waiting_recorded() {
    let engine = MockEngine::start();
    engine.state.set_fail_503_times(100, None);
    let policy = RetryPolicy { max_retries: 2, max_wait: Duration::from_millis(20) };
    let ep: Arc<dyn Endpoint> = Arc::new(HttpEndpoint::new(engine.url()).with_retry(policy));
    let results = replay(ep, &trace_of("p", 1), &no_wait());
    let m = &results[0];
    assert!(!m.ok);
    assert_eq!(m.retries, 2);
    assert!(m.retry_wait_ms >= 35.0, "waited {} ms", m.retry_wait_ms);
    assert_eq!(engine.state.bodies().len(), 3, "the first send plus two retries");
}

#[test]
fn only_engine_full_is_retried() {
    // A 400 (an empty messages array) is the client's fault: sent once.
    let engine = MockEngine::start();
    let policy = RetryPolicy { max_retries: 3, max_wait: Duration::from_millis(20) };
    let ep = HttpEndpoint::new(engine.url()).as_recorded(true).with_retry(policy);
    let err = ep
        .complete(&request_for(serde_json::json!({ "messages": [] }).to_string()))
        .expect_err("a 400");
    assert!(err.contains("400"), "{err}");
    assert_eq!(engine.state.bodies().len(), 1);
}
