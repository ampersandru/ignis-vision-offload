//! A download started through the API (GitHub #376, spec #374, ADR 0071),
//! over the mock-backed router with a temporary `download.path`, against a
//! local endpoint serving small fake files (`support/repo.rs`): no network,
//! no GPU, no real model. `POST /v1/models/download` starts it in the
//! background, `GET /v1/models/catalog` follows it, and what lands under the
//! models folder is the outcome. A download is held mid-transfer by holding
//! the endpoint's response, never by a timer (ADR 0006).

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_server::config::file::RealFiles;
use ignis_server::config::{resolve_with, Config, ConfigOutcome};
use ignis_server::download_http::Downloads;
use ignis_server::model_switch::{CatalogModels, Switcher};
use ignis_server::Server;
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

#[path = "support/repo.rs"]
mod repo;
use repo::{catalog_served, files_of, Repo};

const PATIENT: Duration = Duration::from_secs(30);

async fn send(app: &axum::Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned())))
}

/// The configuration a start with `argv` runs, fetching from `repo`.
fn config(repo: &Repo, argv: &[&str]) -> Config {
    let mut args = vec!["--download-endpoint".to_owned(), repo.endpoint()];
    args.extend(argv.iter().map(|arg| arg.to_string()));
    match resolve_with(&args, |_| None, &RealFiles).expect("the start options resolve") {
        ConfigOutcome::Config(config) => config,
        other => panic!("{other:?}"),
    }
}

/// A server on `mock-a`, switching to the catalog `models`, downloading as
/// `config` says.
fn server_with(loader: &Arc<MockLoader>, models: CatalogModels, config: &Config) -> Server {
    Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT).with_catalog(models, true))
        .with_downloads(Downloads::of(config, None))
}

fn row<'a>(rows: &'a Value, id: &str) -> &'a Value {
    rows["data"].as_array().expect("an array of rows").iter().find(|row| row["id"] == id).unwrap_or_else(|| panic!("no row for {id}: {rows}"))
}

/// `GET /v1/models/catalog` until `id`'s row meets `wanted` (scheduling
/// turns, bounded): the bytes the endpoint sent land on disk shortly after.
async fn until_row(app: &axum::Router, id: &str, wanted: impl Fn(&Value) -> bool) -> Value {
    let mut last = Value::Null;
    for _ in 0..10_000 {
        let (status, rows) = send(app, Method::GET, "/v1/models/catalog", None).await;
        assert_eq!(status, StatusCode::OK, "{rows}");
        last = row(&rows, id).clone();
        if wanted(&last) {
            return last;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("{id}'s row never got there: {last}");
}

fn files_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir).unwrap().map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    names
}

/// AC 1, 6, 7: the download answers 202 and runs in the background while the
/// server serves; the row says it runs, with the bytes on disk, and once it
/// ends the entry is on disk, switchable and listed as a servable model —
/// and the server did not switch to it.
#[tokio::test]
async fn a_download_answers_202_runs_in_the_background_and_leaves_a_servable_model() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "lands", &[("acme-new", 32)]);
    let dir = models.dir.clone();
    let files = files_of("acme-new");
    repo.hold(&files.artifact.0, 4096);
    let app = server_with(&loader, models, &config(&repo, &[])).app();

    let (status, accepted) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-new" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    assert_eq!((accepted["status"].clone(), accepted["model"].clone(), accepted["bytes"].clone()), (json!("downloading"), json!("acme-new"), json!(files.total())));

    let held = until_row(&app, "acme-new", |row| row["bytes_on_disk"] == json!(1 + 4096)).await;
    assert_eq!((held["download"].clone(), held["on_disk"].clone(), held["switchable"].clone()), (json!("running"), json!("partial"), json!(false)), "{held}");
    assert!(held.get("download_reason").is_none(), "{held}");
    // The loaded model serves meanwhile.
    let (status, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!((status, models["status"].clone(), models["data"][0]["id"].clone()), (StatusCode::OK, json!("serving"), json!("mock-a")));

    repo.release();
    let landed = until_row(&app, "acme-new", |row| row["on_disk"] == "complete").await;
    assert_eq!((landed["download"].clone(), landed["bytes_on_disk"].clone(), landed["switchable"].clone()), (json!("none"), json!(files.total()), json!(true)), "{landed}");
    assert_eq!(files_in(&dir), [files.artifact.0.clone(), files.sidecar.0.clone()], "every file at its name, no .part left");
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    let ids: Vec<&str> = models["data"].as_array().unwrap().iter().map(|model| model["id"].as_str().unwrap()).collect();
    assert_eq!(ids.first(), Some(&"mock-a"), "no switch: {models}");
    assert!(ids.contains(&"acme-new"), "a servable model now: {models}");
    assert!(loader.prepared_artifacts().is_empty(), "nothing was loaded");
    let _ = std::fs::remove_dir_all(dir);
}

/// AC 2: one download at a time — a second one, of another entry or the
/// same, is refused with 409 naming the one that runs; once it ends, the
/// next one begins.
#[tokio::test]
async fn a_second_download_while_one_runs_is_refused_with_409() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "one-at-a-time", &[("acme-a", 32), ("acme-b", 32)]);
    let dir = models.dir.clone();
    repo.hold(&files_of("acme-a").artifact.0, 4096);
    let app = server_with(&loader, models, &config(&repo, &[])).app();

    let (status, _) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-a" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    for id in ["acme-b", "acme-a"] {
        let (status, refused) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": id }))).await;
        assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::CONFLICT, json!("download_in_progress")), "{id}: {refused}");
        assert!(refused["error"]["message"].as_str().unwrap().contains("acme-a"), "it names the one that runs: {refused}");
    }
    let (_, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!((row(&rows, "acme-a")["download"].clone(), row(&rows, "acme-b")["download"].clone()), (json!("running"), json!("none")), "the refused one was never queued");

    repo.release();
    until_row(&app, "acme-a", |row| row["on_disk"] == "complete").await;
    let (status, accepted) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-b" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    until_row(&app, "acme-b", |row| row["on_disk"] == "complete").await;
    let _ = std::fs::remove_dir_all(dir);
}

/// AC 3, 4, 9: refused with a reason, before anything is fetched, for an id
/// that is no catalog entry, an entry above the card's class, an entry
/// already on disk, `download.enabled` off and `download.from_ui` off — the
/// last two through their config fields, `--download-enabled false` and
/// `--download-from-ui false`.
#[tokio::test]
async fn a_download_is_refused_with_a_reason_and_fetches_nothing() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let refused = |app: axum::Router, model: &'static str| async move {
        let (status, body) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": model }))).await;
        (status, body["error"]["code"].as_str().unwrap_or_default().to_owned(), body["error"]["message"].as_str().unwrap_or_default().to_owned())
    };

    let models = catalog_served(&repo, "refused", &[("acme-big", 32), ("acme-here", 16)]);
    let dir = models.dir.clone();
    let here = files_of("acme-here");
    std::fs::write(dir.join(&here.sidecar.0), &here.sidecar.1).unwrap();
    std::fs::write(dir.join(&here.artifact.0), &here.artifact.1).unwrap();
    let on_16gb = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT).with_catalog(models, true).with_card_class(Some(ignis_server::config::profile::CardClass::Gb16)))
        .with_downloads(Downloads::of(&config(&repo, &[]), None))
        .app();

    let (status, code, message) = refused(on_16gb.clone(), "nope").await;
    assert_eq!((status, code.as_str()), (StatusCode::NOT_FOUND, "model_not_found"), "{message}");
    assert!(message.contains("not a catalog entry"), "{message}");
    let (status, code, _) = refused(on_16gb.clone(), "mock-a").await;
    assert_eq!((status, code.as_str()), (StatusCode::NOT_FOUND, "model_not_found"), "the loaded model is no catalog entry here");
    let (status, code, message) = refused(on_16gb.clone(), "acme-big").await;
    assert_eq!((status, code.as_str()), (StatusCode::CONFLICT, "model_above_card_class"), "{message}");
    assert!(message.contains("32 GB") && message.contains("16 GB"), "it names both classes: {message}");
    let (status, code, message) = refused(on_16gb.clone(), "acme-here").await;
    assert_eq!((status, code.as_str()), (StatusCode::CONFLICT, "model_already_downloaded"), "{message}");
    let (status, body) = send(&on_16gb, Method::POST, "/v1/models/download", Some(json!({ "model": " " }))).await;
    assert_eq!((status, body["error"]["param"].clone()), (StatusCode::BAD_REQUEST, json!("model")));
    // Above the class: the row says so, and offers nothing to download.
    let (_, rows) = send(&on_16gb, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!((row(&rows, "acme-big")["available"].clone(), row(&rows, "acme-big")["download"].clone()), (json!(false), json!("none")));

    for (flag, field) in [("--download-enabled", "download.enabled"), ("--download-from-ui", "download.from_ui")] {
        let models = catalog_served(&repo, "refused-off", &[("acme-off", 32)]);
        let off_dir = models.dir.clone();
        let app = server_with(&loader, models, &config(&repo, &[flag, "false"])).app();
        let (status, code, message) = refused(app.clone(), "acme-off").await;
        assert_eq!((status, code.as_str()), (StatusCode::FORBIDDEN, "download_disabled"), "{flag}: {message}");
        assert!(message.contains(&format!("{field} is false")), "it names {field}: {message}");
        assert!(files_in(&off_dir).is_empty(), "{flag}: nothing landed");
        let _ = std::fs::remove_dir_all(off_dir);
    }
    assert!(repo.seen().is_empty(), "no refusal reached the endpoint: {:?}", repo.seen());
    let _ = std::fs::remove_dir_all(dir);
}

/// GitHub #378: a download the models folder has no room for — its bytes
/// still missing plus `download.free_margin` above the volume's free space —
/// is refused before any byte is written, naming both numbers, and its row
/// keeps the reason. The short disk is the margin itself, set larger than
/// any disk; a partial entry counts only what it still misses.
#[tokio::test]
async fn a_download_the_models_folder_has_no_room_for_is_refused_and_its_row_says_why() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "no-room", &[("acme-huge", 32)]);
    let dir = models.dir.clone();
    let files = files_of("acme-huge");
    // A partial entry: its sidecar (1 byte) is there, its 65,536-byte artifact is not.
    std::fs::write(dir.join(&files.sidecar.0), &files.sidecar.1).unwrap();
    let app = server_with(&loader, models, &config(&repo, &["--download-free-margin", "1000000G"])).app();

    let (status, refused) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-huge" }))).await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::INSUFFICIENT_STORAGE, json!("download_no_room")), "{refused}");
    let message = refused["error"]["message"].as_str().unwrap().to_owned();
    // 1,000,000 GiB of margin plus the 65,536 bytes still missing.
    assert!(message.contains("1073741824065536 bytes needed"), "the bytes still missing plus the margin: {message}");
    assert!(message.contains("bytes free") && message.contains("download.free_margin"), "the free space, and the field to lower: {message}");
    assert_eq!(files_in(&dir), [files.sidecar.0.clone()], "nothing written");
    assert!(repo.seen().is_empty(), "nothing fetched: {:?}", repo.seen());
    let (_, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    let refused_row = row(&rows, "acme-huge");
    assert_eq!((refused_row["download"].clone(), refused_row["download_reason"].clone()), (json!("failed"), json!(message)), "{refused_row}");
    let _ = std::fs::remove_dir_all(dir);
}

/// AC 8: a failed download keeps its reason on the row, in memory, until the
/// next attempt; a 401 or 403 from the endpoint names the token and where
/// the server reads it, and never the token itself.
#[tokio::test]
async fn a_failed_download_keeps_its_reason_on_the_row_until_the_next_attempt() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "failed", &[("acme-gated", 32)]);
    let dir = models.dir.clone();
    let app = server_with(&loader, models, &config(&repo, &[])).app();

    repo.answer(401);
    let (status, _) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-gated" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let failed = until_row(&app, "acme-gated", |row| row["download"] == "failed").await;
    let reason = failed["download_reason"].as_str().expect("a failure carries its reason").to_owned();
    assert!(reason.contains("401") && reason.contains("none was sent") && reason.contains("IGNIS_DOWNLOAD_TOKEN"), "{reason}");
    assert!(!reason.contains("HF_TOKEN"), "HF_TOKEN goes to huggingface.co only, not to this endpoint: {reason}");
    let (_, again) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!(row(&again, "acme-gated")["download_reason"], json!(reason), "kept, read after read");

    // The next attempt forgets it as it begins.
    repo.answer(0);
    let (status, _) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-gated" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let landed = until_row(&app, "acme-gated", |row| row["on_disk"] == "complete").await;
    assert_eq!(landed["download"], "none", "{landed}");
    assert!(landed.get("download_reason").is_none(), "{landed}");

    // A token the endpoint refuses: named by where it comes from, never shown.
    let models = catalog_served(&repo, "failed-token", &[("acme-private", 32)]);
    let token_dir = models.dir.clone();
    let app = server_with(&loader, models, &config(&repo, &["--download-token", "tok-secret"])).app();
    repo.answer(403);
    let (status, _) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-private" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let failed = until_row(&app, "acme-private", |row| row["download"] == "failed").await;
    let reason = failed["download_reason"].as_str().unwrap();
    assert!(reason.contains("403") && reason.contains("refused the token sent") && reason.contains("IGNIS_DOWNLOAD_TOKEN"), "{reason}");
    assert!(!failed.to_string().contains("tok-secret"), "the token never reaches the row: {failed}");
    let sent: Vec<_> = repo.seen().into_iter().filter(|seen| seen.file.starts_with("acme-private")).collect();
    assert!(!sent.is_empty() && sent.iter().all(|seen| seen.authorization.as_deref() == Some("Bearer tok-secret")), "the token went to the endpoint: {sent:?}");
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(token_dir));
}

/// AC 1: the route takes the same authentication as `PATCH /v1/config` —
/// the server's one API key — and answers the browser's preflight.
#[tokio::test]
async fn the_download_route_sits_behind_the_api_key_and_answers_the_preflight() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "keyed", &[("acme-keyed", 32)]);
    let dir = models.dir.clone();
    let app = server_with(&loader, models, &config(&repo, &[])).with_api_key(ignis_server::config::ApiKey::new("sk-test")).app();
    let keyed = |key: Option<&'static str>| {
        let mut request = Request::builder().method(Method::POST).uri("/v1/models/download").header(header::CONTENT_TYPE, "application/json");
        if let Some(key) = key {
            request = request.header(header::AUTHORIZATION, format!("Bearer {key}"));
        }
        request.body(Body::from(json!({ "model": "acme-keyed" }).to_string())).unwrap()
    };

    let (status, _) = send(&app, Method::OPTIONS, "/v1/models/download", None).await;
    assert_eq!(status, StatusCode::OK);
    for key in [None, Some("sk-wrong")] {
        let response = app.clone().oneshot(keyed(key)).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{key:?}");
        assert_eq!(response.headers().get(header::WWW_AUTHENTICATE).unwrap(), "Bearer");
    }
    assert!(files_in(&dir).is_empty() && repo.seen().is_empty(), "nothing downloaded without the key");
    let response = app.clone().oneshot(keyed(Some("sk-test"))).await.unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let _ = std::fs::remove_dir_all(dir);
}

/// AC 10: a switch still never downloads: a catalog entry not on disk is a
/// `404 model_not_found`, and nothing reaches the endpoint.
#[tokio::test]
async fn a_switch_still_never_downloads() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "no-switch-download", &[("acme-far", 32)]);
    let dir = models.dir.clone();
    let app = server_with(&loader, models, &config(&repo, &[])).app();
    let (status, refused) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": "acme-far" }))).await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::NOT_FOUND, json!("model_not_found")), "{refused}");
    assert!(repo.seen().is_empty() && files_in(&dir).is_empty(), "nothing fetched");
    let (_, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!(row(&rows, "acme-far")["download"], "none");
    let _ = std::fs::remove_dir_all(dir);
}

/// A server built with no downloads (a test's) answers `501`, as one with
/// no loader answers a switch.
#[tokio::test]
async fn a_server_without_downloads_answers_501() {
    let loader = MockLoader::new();
    let app = Server::from_active(loader.model("mock-a")).app();
    let (status, refused) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "qwen3.8-27b" }))).await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::NOT_IMPLEMENTED, json!("download_unavailable")));
}
