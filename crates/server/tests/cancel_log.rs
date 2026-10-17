//! The `ignis.request.cancelled` line's `cause` (GitHub #327): `timeout` for
//! a request that hit `--server-request-timeout` on every HTTP path that
//! answers one (chat completions, with and without a stop sequence, and the
//! Responses API, streamed or not), `consumer` for a client that hung up, `shutdown` for a
//! request still in flight when its model thread stops. A binary of its own
//! with one test, which installs the process's subscriber: the line is
//! written on the telemetry consumer's task, which a subscriber scoped to
//! the test's thread never sees.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{to_bytes, Body};
use axum::http::Request;
use ignis_core::mock::MockCompute;
use ignis_core::{ConcreteScheduler, RequestId, SchedulerConfig};
use ignis_logging::{JsonLayer, MemorySink};
use ignis_server::engine::Engine;
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::Server;
use serde_json::{json, Value};
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

const MODEL: &str = "test-model";

/// The mock's scheduler, except that it never runs what it takes: what it
/// admits stays in flight until it is cancelled, so every request here ends
/// by its deadline, its client or its model thread.
struct NeverRuns(ConcreteScheduler);

impl ignis_core::Scheduler for NeverRuns {
    fn submit(&mut self, input: ignis_core::RequestInput, class: ignis_core::RequestClass) -> Result<RequestId, ignis_core::SubmitError> {
        self.0.submit(input, class)
    }
    fn cancel(&mut self, request: RequestId) -> bool {
        self.0.cancel(request)
    }
    fn advance(&mut self) -> Vec<ignis_core::SchedEvent> {
        Vec::new()
    }
    fn is_idle(&self) -> bool {
        true
    }
    fn model_id(&self) -> &str {
        self.0.model_id()
    }
    fn max_sequence_tokens(&self) -> u32 {
        self.0.max_sequence_tokens()
    }
    fn mode(&self) -> ignis_core::EngineMode {
        self.0.mode()
    }
    fn occupancy(&self) -> ignis_core::Occupancy {
        self.0.occupancy()
    }
}

/// Every `ignis.request.cancelled` line `log` holds, as `(request_id, cause)`.
fn cancelled(log: &MemorySink) -> Vec<(u64, String)> {
    log.lines()
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("json"))
        .filter(|event| event["event_name"] == "ignis.request.cancelled")
        .map(|event| {
            let attributes = &event["attributes"];
            (attributes["request_id"].as_u64().unwrap(), attributes["cause"].as_str().unwrap().to_owned())
        })
        .collect()
}

/// Wait until `log` holds `count` cancelled lines.
async fn cancelled_lines(log: &MemorySink, count: usize) -> Vec<(u64, String)> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let lines = cancelled(log);
        if lines.len() >= count {
            return lines;
        }
        assert!(Instant::now() < deadline, "only {lines:?} after waiting for {count}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn post(path: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cancelled_request_says_why_it_was_cancelled() {
    let log = Arc::new(MemorySink::new());
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(JsonLayer::new(log.clone())))
        .expect("the only subscriber this binary installs");

    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), max_in_flight: 8, ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    let engine = Engine::new(Box::new(NeverRuns(scheduler)));
    let app = Server::new(engine.clone(), Box::new(SimpleTemplateProvider))
        .with_request_timeout(Duration::from_millis(300))
        .with_seedless_seed(0)
        .app();
    let chat = json!({ "model": MODEL, "messages": [{ "role": "user", "content": "hi" }], "max_tokens": 8 });

    // Past the deadline: chat completions, with a stop sequence (its own
    // collection path), and the Responses API.
    let mut stopped = chat.clone();
    stopped["stop"] = json!(["zzz"]);
    let timed_out = [
        ("/v1/chat/completions", chat.clone()),
        ("/v1/chat/completions", stopped),
        ("/v1/responses", json!({ "model": MODEL, "input": "hi", "max_output_tokens": 8 })),
    ];
    for (n, (path, body)) in timed_out.into_iter().enumerate() {
        let response = app.clone().oneshot(post(path, body)).await.unwrap();
        assert_eq!(response.status(), 504, "{path}");
        let lines = cancelled_lines(&log, n + 1).await;
        assert_eq!(lines[n].1, "timeout", "{path}: {lines:?}");
    }
    // A streamed response past its deadline ends `failed`, its request
    // cancelled for it.
    let response = app
        .clone()
        .oneshot(post("/v1/responses", json!({ "model": MODEL, "input": "hi", "max_output_tokens": 8, "stream": true })))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert!(String::from_utf8_lossy(&body).contains("request_timeout"), "{body:?}");
    let lines = cancelled_lines(&log, 4).await;
    assert_eq!(lines[3].1, "timeout", "{lines:?}");

    // A client that hangs up on a stream.
    let mut streamed = chat.clone();
    streamed["stream"] = json!(true);
    let response = app.clone().oneshot(post("/v1/chat/completions", streamed.clone())).await.unwrap();
    assert_eq!(response.status(), 200);
    drop(response);
    let lines = cancelled_lines(&log, 5).await;
    assert_eq!(lines[4].1, "consumer", "{lines:?}");

    // A stream still in flight when the model thread stops (a switch past
    // its drain window).
    let response = app.clone().oneshot(post("/v1/chat/completions", streamed)).await.unwrap();
    assert_eq!(response.status(), 200);
    engine.shutdown();
    let lines = cancelled_lines(&log, 6).await;
    assert_eq!(lines[5].1, "shutdown", "{lines:?}");
    let _ = to_bytes(response.into_body(), usize::MAX).await;

    // One line per request, and no request cancelled twice.
    let lines = cancelled(&log);
    let mut ids: Vec<u64> = lines.iter().map(|(id, _)| *id).collect();
    ids.dedup();
    assert_eq!(ids.len(), 6, "{lines:?}");
}
