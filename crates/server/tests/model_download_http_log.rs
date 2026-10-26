//! GitHub #376 AC 11, the log half: a download started through `POST
//! /v1/models/download` logs the `ignis.model.download_*` events a start's
//! fetch logs, and one that fails logs `ignis.model.download_failed` with its
//! reason; no line carries the token.
//!
//! A binary of its own, with one test: `tracing` caches whether a callsite
//! is enabled, and a test fetching with no subscriber, on another thread of
//! the same binary, can settle that cache for the transfer's events before
//! this one's scoped subscriber is consulted — the capture then sees
//! nothing. Alone here, nothing races it.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_server::config::file::RealFiles;
use ignis_server::config::{resolve_with, ConfigOutcome};
use ignis_server::download_http::Downloads;
use ignis_server::model_switch::Switcher;
use ignis_server::Server;
use serde_json::{json, Value};
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

#[path = "support/repo.rs"]
mod repo;
use repo::{catalog_served, Repo};

async fn send(app: &axum::Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

#[tokio::test]
async fn a_download_from_the_api_logs_the_download_events_and_its_failure() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "logged", &[("acme-logged", 32), ("acme-refused", 32)]);
    let dir = models.dir.clone();
    let args: Vec<String> = ["--download-endpoint", &repo.endpoint(), "--download-token", "tok-secret"].iter().map(|arg| arg.to_string()).collect();
    let config = match resolve_with(&args, |_| None, &RealFiles).expect("the start options resolve") {
        ConfigOutcome::Config(config) => config,
        other => panic!("{other:?}"),
    };
    let app = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, Duration::from_secs(30)).with_catalog(models, true))
        .with_downloads(Downloads::of(&config, None))
        .app();
    let sink = Arc::new(ignis_logging::MemorySink::new());
    let _guard = tracing::subscriber::set_default(tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())));
    // Bounded scheduling turns until a line naming `event` and `model` is logged.
    let until_logged = |event: &'static str, model: &'static str| {
        let sink = Arc::clone(&sink);
        async move {
            for _ in 0..10_000 {
                if sink.lines().iter().any(|line| line.contains(event) && line.contains(model)) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            panic!("no {event} for {model}: {:?}", sink.lines());
        }
    };

    let (status, accepted) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-logged" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    until_logged("ignis.model.downloaded", "acme-logged.ninfer\"").await;
    let lines = sink.lines();
    let started: Vec<&String> = lines.iter().filter(|line| line.contains("ignis.model.download_started")).collect();
    assert_eq!(started.len(), 2, "one per file: {lines:?}");
    assert!(started.iter().all(|line| line.contains("acme-logged") && line.contains(&repo.endpoint())), "{started:?}");
    assert_eq!(lines.iter().filter(|line| line.contains("ignis.model.downloaded")).count(), 2, "{lines:?}");

    repo.answer(401);
    let (status, _) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-refused" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until_logged("ignis.model.download_failed", "acme-refused").await;
    let failed: Vec<String> = sink.lines().into_iter().filter(|line| line.contains("ignis.model.download_failed")).collect();
    assert_eq!(failed.len(), 1, "logged once: {failed:?}");
    assert!(failed[0].contains("401") && failed[0].contains("IGNIS_DOWNLOAD_TOKEN"), "{}", failed[0]);
    for line in sink.lines() {
        assert!(!line.contains("tok-secret"), "the token leaked: {line}");
    }
    let _ = std::fs::remove_dir_all(dir);
}
