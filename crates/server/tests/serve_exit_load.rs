//! GitHub #329 item 17, while serving: a stop does not wait for a model load.
//! A switch, a reload or the load after an idle unload has no cancel, and a
//! cold one outlasts a pod's grace period, so once the serve's shutdown
//! resolves:
//! - a request waiting for one is answered `503 server_stopping` with
//!   `Retry-After`, and the HTTP drain is not held by it;
//! - no request begins one any more;
//! - one still running once HTTP has drained is returned
//!   (`Served::LoadAbandoned`) with nothing torn down, for `main` to end the
//!   process under it.
//!
//! Loads are `support/switch.rs`'s mock loader's, held inside the load. That
//! an unload still tearing down is waited for is `idle_unload.rs`'s, a visit's
//! `image_enhancement.rs`'s, a switch begun over `POST /v1/models/switch`
//! `serve_joins_model_thread.rs`'s, and the real binary's `start_load.rs`'s.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{header, Method, Request};
use http_body_util::BodyExt;
use ignis_server::model_switch::{SwitchStarted, Switcher};
use ignis_server::{Served, Server};
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::{source, MockLoader};

/// Long enough that no test here reaches it unless something is wrong.
const PATIENT: Duration = Duration::from_secs(30);

/// How soon a stop must be answered: a held load never ends by itself.
const BOUND: Duration = Duration::from_secs(5);

/// A server on `start` whose requests may switch to any of `ids`.
fn implicit_on(loader: &Arc<MockLoader>, start: &str, ids: &[&str]) -> Server {
    let known: BTreeMap<String, PathBuf> = ids.iter().map(|id| (id.to_string(), source(id).artifact)).collect();
    Server::from_active(loader.model(start)).with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT).with_known_models(known))
}

/// Serve `server` on a fresh listener: its address, its stop, and the
/// serve's task.
async fn serving(
    server: Server,
) -> (std::net::SocketAddr, tokio::sync::oneshot::Sender<()>, tokio::task::JoinHandle<std::io::Result<Served>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(server.serve_on_until(listener, None, async {
        let _ = stopped.await;
    }));
    (addr, stop, task)
}

fn chat(model: &str) -> Value {
    json!({ "model": model, "messages": [{ "role": "user", "content": "hello" }], "max_tokens": 3 })
}

/// A chat completion naming `model` over HTTP to `addr`, its connection not
/// kept: the status, the `Retry-After`, the body.
async fn chat_over_http(addr: std::net::SocketAddr, model: &'static str) -> (u16, Option<String>, Value) {
    let client = reqwest::Client::builder().pool_max_idle_per_host(0).build().expect("client");
    let response = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .header(header::CONTENT_TYPE.as_str(), "application/json")
        .body(chat(model).to_string())
        .send()
        .await
        .expect("answered");
    let status = response.status().as_u16();
    let retry_after = response.headers().get("retry-after").map(|value| value.to_str().expect("ascii").to_owned());
    let body = response.text().await.expect("body");
    (status, retry_after, serde_json::from_str(&body).unwrap_or(Value::String(body)))
}

/// Wait (bounded) until `check` holds.
async fn until(what: &str, check: impl Fn() -> bool) {
    let deadline = Instant::now() + PATIENT;
    while !check() {
        assert!(Instant::now() < deadline, "never: {what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn assert_stopping(status: u16, retry_after: Option<&str>, body: &Value) {
    assert_eq!(status, 503, "{body}");
    assert_eq!(body["error"]["code"], "server_stopping", "{body}");
    assert_eq!(retry_after, Some("5"), "{body}");
}

/// The request that began an implicit switch waits inside its handler, so it
/// held the HTTP drain for the whole load. Now it is answered at the stop,
/// the drain ends, and the load is returned to be abandoned.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_request_waiting_on_an_implicit_switch_is_answered_and_the_load_abandoned() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, "mock-a", &["mock-a", "mock-b"]);
    let load = loader.hold_next_load();
    let (addr, stop, serve) = serving(server.clone()).await;
    let waiting = tokio::spawn(chat_over_http(addr, "mock-b"));
    // In the load: the old model is off the card, the new one not built.
    until("the switch's load is under way", || server.status().as_str() == "switching" && loader.resident() == 0).await;

    stop.send(()).expect("serving");
    let (status, retry_after, body) = tokio::time::timeout(BOUND, waiting).await.expect("answered at the stop").expect("task");
    assert_stopping(status, retry_after.as_deref(), &body);
    let served = tokio::time::timeout(BOUND, serve).await.expect("the serve returned with the load running").expect("task").expect("serve");
    assert_eq!(served, Served::LoadAbandoned { load: SwitchStarted { from: "mock-a".into(), to: "mock-b".into() }, phase: "switching" });
    assert!(loader.loads().is_empty(), "the load was not waited for");
    load.send(()).expect("the load is held");
}

/// The reload after an idle unload: the request that needs the model waits
/// for its load, inside its handler, as the implicit switch's does.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_request_waiting_on_the_load_after_an_idle_unload_is_answered_and_the_load_abandoned() {
    let loader = MockLoader::new();
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT))
        .with_idle_unload(Duration::from_secs(1));
    let (addr, stop, serve) = serving(server.clone()).await;
    // The serve's own clock unloads the model nobody uses.
    let switcher = Arc::clone(server.switcher.as_ref().expect("a switcher"));
    until("the model is unloaded", || server.status().as_str() == "unloaded" && switcher.running().is_none()).await;
    server.update_live(|live| live.idle_unload = Duration::ZERO);
    let load = loader.hold_next_load();
    let waiting = tokio::spawn(chat_over_http(addr, "mock-a"));
    until("the load is under way", || server.status().as_str() == "loading").await;

    stop.send(()).expect("serving");
    let (status, retry_after, body) = tokio::time::timeout(BOUND, waiting).await.expect("answered at the stop").expect("task");
    assert_stopping(status, retry_after.as_deref(), &body);
    let served = tokio::time::timeout(BOUND, serve).await.expect("the serve returned with the load running").expect("task").expect("serve");
    // Logged as `GET /v1/models` reports it: not a switch, a load.
    assert_eq!(served, Served::LoadAbandoned { load: SwitchStarted { from: "mock-a".into(), to: "mock-a".into() }, phase: "loading" });
    load.send(()).expect("the load is held");
}

/// A request that reaches its switch only after the stop — one whose body
/// was still arriving — begins none: nothing would be left to serve it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn once_the_serve_stops_a_request_begins_no_switch() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, "mock-a", &["mock-a", "mock-b"]);
    let app = server.app();
    let (_, stop, serve) = serving(server.clone()).await;
    stop.send(()).expect("serving");
    let served = tokio::time::timeout(BOUND, serve).await.expect("the serve returned").expect("task").expect("serve");
    assert_eq!(served, Served::Drained, "nothing was loading");

    let (status, retry_after, body) = post(&app, "/v1/chat/completions", chat("mock-b")).await;
    assert_stopping(status, retry_after.as_deref(), &body);
    assert!(loader.prepared_artifacts().is_empty(), "no switch began");
    assert_eq!(server.status().as_str(), "serving");
}

/// `POST` to `uri` on `app`, in process: the status, the `Retry-After`,
/// the body.
async fn post(app: &axum::Router, uri: &str, body: Value) -> (u16, Option<String>, Value) {
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let (status, retry_after) = (response.status().as_u16(), response.headers().get("retry-after").map(|v| v.to_str().unwrap().to_owned()));
    let body: Value = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    (status, retry_after, body)
}

/// `POST /v1/models/switch` after the stop is refused `503
/// server_stopping`, not accepted with a `202` for a switch the process
/// would end under: nothing is prepared, and the model is as it was.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn once_the_serve_stops_an_explicit_switch_is_refused_not_accepted() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, "mock-a", &["mock-a", "mock-b"]);
    let app = server.app();
    let (_, stop, serve) = serving(server.clone()).await;
    stop.send(()).expect("serving");
    let served = tokio::time::timeout(BOUND, serve).await.expect("the serve returned").expect("task").expect("serve");
    assert_eq!(served, Served::Drained, "nothing was loading");

    let switch = json!({ "artifact": source("mock-b").artifact, "model": "mock-b" });
    let (status, retry_after, body) = post(&app, "/v1/models/switch", switch).await;
    assert_stopping(status, retry_after.as_deref(), &body);
    assert!(loader.prepared_artifacts().is_empty(), "no switch began");
    assert_eq!(server.status().as_str(), "serving");
}

/// The model the idle clock unloaded is not loaded again by a request that
/// reaches it after the stop: `503 server_stopping`, and no load begins.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn once_the_serve_stops_a_request_loads_no_unloaded_model() {
    let loader = MockLoader::new();
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT))
        .with_idle_unload(Duration::from_secs(1));
    let app = server.app();
    let (_, stop, serve) = serving(server.clone()).await;
    let switcher = Arc::clone(server.switcher.as_ref().expect("a switcher"));
    until("the model is unloaded", || server.status().as_str() == "unloaded" && switcher.running().is_none()).await;
    stop.send(()).expect("serving");
    let served = tokio::time::timeout(BOUND, serve).await.expect("the serve returned").expect("task").expect("serve");
    assert_eq!(served, Served::Drained, "nothing was loading");

    let (status, retry_after, body) = post(&app, "/v1/chat/completions", chat("mock-a")).await;
    assert_stopping(status, retry_after.as_deref(), &body);
    assert!(loader.prepared_artifacts().is_empty(), "no load began");
    assert_eq!(server.status().as_str(), "unloaded");
}
