//! The `ignis.response.admitted` line: a response's id is random, not its
//! request's (spec responses-api/01, Output items), so this line is what
//! pairs the two in the log — for a response over HTTP, one on the socket
//! admitted at once, and one admitted from the queue.
//!
//! A binary of its own with one test, which installs the process's
//! subscriber: a subscriber scoped to one test's thread never sees what
//! another thread logs, and a callsite another test reached first stays
//! uninteresting to it.

#[path = "support/responses.rs"]
mod responses;

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use ignis_core::{MockCompute, SchedulerConfig};
use ignis_logging::{JsonLayer, MemorySink};
use serde_json::{json, Value};
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

use responses::{events_until, live, send, server, socket, terminal, Script, MODEL};

/// Every `ignis.response.admitted` line `log` holds, as `(response_id, request_id)`.
fn admitted(log: &MemorySink) -> Vec<(String, u64)> {
    log.lines()
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("json"))
        .filter(|event| event["event_name"] == "ignis.response.admitted")
        .map(|event| {
            let attributes = &event["attributes"];
            (attributes["response_id"].as_str().unwrap().to_owned(), attributes["request_id"].as_u64().unwrap())
        })
        .collect()
}

/// Whether an event is on `stream` and of type `kind`.
fn on(stream: &'static str, kind: &'static str) -> impl Fn(&Value) -> bool {
    move |event| event["stream_id"] == stream && event["type"] == kind
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_admitted_response_is_logged_with_the_request_that_serves_it() {
    let log = Arc::new(MemorySink::new());
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(JsonLayer::new(log.clone())))
        .expect("the only subscriber this binary installs");
    // One lane, and room for a request that holds it until it is cancelled.
    let config = SchedulerConfig { max_in_flight: 1, max_sequence_tokens: 2 << 20, kv_capacity_pages: (2 << 20) / 16 * 2, ..SchedulerConfig::default() };
    let server = server(&Script::silent(), config, Arc::new(MockCompute::new()));
    let short = |input: &str| json!({ "model": MODEL, "input": input, "max_output_tokens": 1, "enable_thinking": false });

    let request = Request::post("/v1/responses").header("content-type", "application/json").body(Body::from(short("over http").to_string()));
    let answered = server.clone().app().oneshot(request.unwrap()).await.unwrap();
    let body: Value = serde_json::from_slice(&to_bytes(answered.into_body(), usize::MAX).await.unwrap()).unwrap();
    let over_http = body["id"].as_str().expect("a response").to_owned();

    let live = live(server).await;
    let mut socket = socket(&live).await;
    let hog = json!({ "type": "response.create", "stream_id": "hog", "model": MODEL, "input": "hog", "max_output_tokens": 1_000_000, "enable_thinking": false });
    send(&mut socket, hog).await;
    let in_progress = events_until(&mut socket, on("hog", "response.in_progress")).await.pop().unwrap();
    let at_once = in_progress["response"]["id"].as_str().unwrap().to_owned();
    let mut queued = short("wait");
    queued["type"] = json!("response.create");
    queued["stream_id"] = json!("main");
    send(&mut socket, queued).await;
    let created = events_until(&mut socket, on("main", "response.created")).await.pop().unwrap();
    assert_eq!(created["response"]["status"], "queued");
    let from_the_queue = created["response"]["id"].as_str().unwrap().to_owned();
    send(&mut socket, json!({ "type": "response.cancel", "response_id": at_once })).await;
    events_until(&mut socket, |event| event["stream_id"] == "main" && terminal(event)).await;

    assert_eq!(admitted(&log), [(over_http, 0), (at_once, 1), (from_the_queue, 2)]);
}
