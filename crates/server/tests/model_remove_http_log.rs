//! GitHub #380, the log half: a removal through `POST /v1/models/remove`
//! logs its own event, `ignis.model.removed`, with the entry, what it
//! deleted and the bytes it freed; a refused one logs nothing.
//!
//! A binary of its own, with one test: `tracing` caches whether a callsite
//! is enabled, and a test removing with no subscriber, on another thread of
//! the same binary, can settle that cache before this one's scoped
//! subscriber is consulted — the capture then sees nothing. Alone here,
//! nothing races it.

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

async fn send(app: &axum::Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let request = Request::builder().method(Method::POST).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let response = app.clone().oneshot(request.body(Body::from(body.to_string())).unwrap()).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

#[tokio::test]
async fn a_removal_from_the_api_logs_its_own_event_and_a_refused_one_nothing() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "remove-logged", &[("acme-gone", 32), ("acme-live", 32)]);
    let dir = models.dir.clone();
    for id in ["acme-gone", "acme-live"] {
        let files = files_of(id);
        std::fs::write(dir.join(&files.sidecar.0), &files.sidecar.1).unwrap();
        std::fs::write(dir.join(&files.artifact.0), &files.artifact.1).unwrap();
    }
    let args: Vec<String> = ["--download-endpoint", &repo.endpoint()].iter().map(|arg| arg.to_string()).collect();
    let config = match resolve_with(&args, |_| None, &RealFiles).expect("the start options resolve") {
        ConfigOutcome::Config(config) => config,
        other => panic!("{other:?}"),
    };
    let app = Server::from_active(loader.model("acme-live"))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, Duration::from_secs(30)).with_catalog(models, true))
        .with_downloads(Downloads::of(&config, None))
        .app();
    let sink = Arc::new(ignis_logging::MemorySink::new());
    let _guard = tracing::subscriber::set_default(tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())));

    let (status, refused) = send(&app, "/v1/models/remove", json!({ "model": "acme-live" })).await;
    assert_eq!(status, StatusCode::CONFLICT, "{refused}");
    let (status, removed) = send(&app, "/v1/models/remove", json!({ "model": "acme-gone" })).await;
    assert_eq!(status, StatusCode::OK, "{removed}");

    let lines: Vec<String> = sink.lines().into_iter().filter(|line| line.contains("ignis.model.removed")).collect();
    assert_eq!(lines.len(), 1, "one event, for the removal that took place: {:?}", sink.lines());
    let line: Value = serde_json::from_str(&lines[0]).expect("a JSON line");
    let text = line.to_string();
    let files = files_of("acme-gone");
    assert!(text.contains("\"acme-gone\"") && !text.contains("acme-live"), "{text}");
    assert!(text.contains(&files.total().to_string()), "the bytes freed: {text}");
    assert!(text.contains(&files.sidecar.0) && text.contains(&files.artifact.0), "the files deleted: {text}");
    let _ = std::fs::remove_dir_all(dir);
}
