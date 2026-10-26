//! GitHub #377 AC 4: a download cancelled through `POST
//! /v1/models/download/cancel` logs its own event,
//! `ignis.model.download_cancelled`, once, naming the entry and the bytes it
//! keeps for the resume; a cancel is no failure, so it logs no
//! `ignis.model.download_failed`.
//!
//! A binary of its own, with one test: `tracing` caches whether a callsite
//! is enabled, and a test cancelling with no subscriber, on another thread of
//! the same binary, can settle that cache for the event before this one's
//! scoped subscriber is consulted — the capture then sees nothing. Alone
//! here, nothing races it.

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
use repo::{catalog_served, files_of, Repo};

async fn send(app: &axum::Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

#[tokio::test]
async fn a_cancelled_download_logs_its_own_event_with_the_bytes_it_keeps() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "cancel-logged", &[("acme-cancelled", 32)]);
    let dir = models.dir.clone();
    let files = files_of("acme-cancelled");
    repo.hold(&files.artifact.0, 4096);
    let args: Vec<String> = ["--download-endpoint", &repo.endpoint()].iter().map(|arg| arg.to_string()).collect();
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
    let named = |line: &String, name: &str| serde_json::from_str::<Value>(line).is_ok_and(|event| event["event_name"] == name);
    let cancelled = || sink.lines().into_iter().filter(|line| named(line, "ignis.model.download_cancelled")).collect::<Vec<String>>();

    let (status, _) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-cancelled" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    // Bounded scheduling turns until the held artifact's first bytes are on disk.
    let part = dir.join(format!("{}.part", files.artifact.0));
    for _ in 0..10_000 {
        if std::fs::metadata(&part).is_ok_and(|meta| meta.len() == 4096) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let (status, answer) = send(&app, Method::POST, "/v1/models/download/cancel", None).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    // The download is settled, and the event logged, by the task watching it.
    for _ in 0..10_000 {
        if !cancelled().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    let lines = cancelled();
    assert_eq!(lines.len(), 1, "logged once: {:?}", sink.lines());
    let event: Value = serde_json::from_str(&lines[0]).expect("a JSON line");
    let fields = &event["attributes"];
    assert_eq!(fields["model"], json!("acme-cancelled"), "{event}");
    assert_eq!(fields["bytes_on_disk"], json!(1 + 4096), "what stays for the resume: {event}");
    assert_eq!(fields["bytes"], json!(files.total()), "{event}");
    assert!(lines[0].contains(&repo.endpoint()), "{event}");
    assert_eq!(event["severity_text"], json!("INFO"), "{event}");
    assert!(!sink.lines().iter().any(|line| named(line, "ignis.model.download_failed")), "a cancel is no failure: {:?}", sink.lines());
    let _ = std::fs::remove_dir_all(dir);
}
