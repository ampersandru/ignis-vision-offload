//! The card class (ADR 0064, spec 27b-16gb/04 §6, GitHub #336) over the
//! mock-backed router: on a 16 GB card a switch to a catalog entry above
//! the card's class — explicit, or a request's implicit one — is refused
//! `409 model_above_card_class` before anything is prepared, drained or torn
//! down, and the loaded model keeps serving; a request naming the 27B's
//! default id is served by the 16 GB entry that answers to it as an alias;
//! `GET /v1/models` and `GET /v1/models/catalog` list such entries as
//! unavailable, with the reason. On a 32 GB card, or one of no known class,
//! nothing changes. The catalog is the built-in one with the 16 GB 27B as an
//! operator entry standing in for the one GitHub #338 adds; every artifact
//! is a file under `download.path`, and the models are `support/switch.rs`'s
//! mocks, loaded by id.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_server::config::file::Format;
use ignis_server::config::profile::CardClass;
use ignis_server::download::catalog::parse_operator;
use ignis_server::model_switch::{CatalogModels, Switcher};
use ignis_server::{ModelStatus, Server};
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

const PATIENT: Duration = Duration::from_secs(30);

/// The 16 GB 27B's served id, and the id it answers to.
const SIXTEEN: &str = "qwen3.8-27b-16g";
const DEFAULT_27B: &str = "qwen3.8-27b";
const FLASH_NEXT: &str = "qwen3.8-flash-next";
const IMAGE: &str = "ignis-image-2.1-turbo";

/// The built-in catalog and the 16 GB 27B, every entry's artifact a file
/// under a fresh `download.path`.
fn catalog_on_disk(tag: &str) -> CatalogModels {
    let dir = std::env::temp_dir().join(format!("ignis-card-class-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let zeros = "0".repeat(64);
    let yaml = format!(
        "models:\n  - id: {SIXTEEN}\n    repo: acme/q16\n    revision: v1\n    artifact: q16.ninfer\n    card_class: 16\n    aliases: [{DEFAULT_27B}]\n    files:\n      - {{ name: q16.ninfer.conversion.json, bytes: 1, sha256: \"{zeros}\" }}\n      - {{ name: q16.ninfer, bytes: 7, sha256: \"{zeros}\" }}\n"
    );
    let catalog = parse_operator(&yaml, Format::Yaml, "test").expect("a valid catalog");
    for entry in catalog.entries() {
        std::fs::write(entry.artifact_path(&dir), b"mock").unwrap();
    }
    CatalogModels { catalog: Arc::new(catalog), dir }
}

/// A server on `start`, on a card of class `card`, whose requests may switch
/// to any catalog entry on disk.
fn server_on(loader: &Arc<MockLoader>, start: &str, card: Option<CardClass>, tag: &str) -> Server {
    Server::from_active(loader.model(start)).with_switcher(
        Switcher::new(Arc::clone(loader) as _, PATIENT).with_catalog(catalog_on_disk(tag), true).with_card_class(card),
    )
}

async fn send(app: &axum::Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, body)
}

fn chat(model: &str) -> Value {
    json!({ "model": model, "messages": [{ "role": "user", "content": "hello there" }], "max_tokens": 3 })
}

fn response_body(model: &str) -> Value {
    json!({ "model": model, "input": "hello there", "max_output_tokens": 3, "enable_thinking": false })
}

fn decision(model: &str) -> Value {
    json!({ "state": "s", "model": model, "questions": { "q": { "type": "noul", "instructions": "Urgent?" } } })
}

/// `GET /v1/models` until its `status` is `serving` (scheduling turns,
/// bounded).
async fn until_serving(app: &axum::Router) -> Value {
    for _ in 0..10_000 {
        let (status, body) = send(app, Method::GET, "/v1/models", None).await;
        if status == StatusCode::OK && body["status"] == "serving" {
            return body;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("`GET /v1/models` never reported serving");
}

/// A refusal naming the entry's class and the card's.
fn refused_by_class(status: StatusCode, body: &Value, model: &str) {
    assert_eq!(status, StatusCode::CONFLICT, "{model}: {body}");
    assert_eq!(body["error"]["code"], "model_above_card_class", "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains(model) && message.contains("32 GB") && message.contains("16 GB"), "{message}");
}

/// The server still serves `loaded`, nothing was prepared or loaded, and a
/// request naming no model is answered by it.
async fn still_serving(app: &axum::Router, server: &Server, loader: &MockLoader, loaded: &str) {
    assert!(loader.prepared_artifacts().is_empty(), "nothing was prepared: {:?}", loader.prepared_artifacts());
    assert!(loader.loads().is_empty(), "nothing was loaded: {:?}", loader.loads());
    assert_eq!(loader.resident(), 1, "the loaded model was never torn down");
    assert_eq!(*server.status(), ModelStatus::Serving);
    let (_, models) = send(app, Method::GET, "/v1/models", None).await;
    assert_eq!((models["status"].as_str(), models["data"][0]["id"].as_str()), (Some("serving"), Some(loaded)), "{models}");
    assert!(models.get("last_switch").is_none(), "no switch began, so none is recorded: {models}");
    let (status, completion) = send(app, Method::POST, "/v1/chat/completions", Some(json!({ "messages": [{ "role": "user", "content": "hi" }], "max_tokens": 3 }))).await;
    assert_eq!(status, StatusCode::OK, "{completion}");
    assert_eq!(completion["model"], loaded);
}

/// ADR 0064 §3: `POST /v1/models/switch` by id to an entry above the card's
/// class is refused before anything stops — not even prepared, so the
/// status never leaves `serving` — and the loaded model serves on. A switch
/// naming its own artifact is the operator's word (ADR 0047), which only the
/// VRAM plan guards.
#[tokio::test]
async fn an_explicit_switch_to_an_entry_above_the_card_class_is_refused_before_anything_stops() {
    let loader = MockLoader::new();
    let server = server_on(&loader, SIXTEEN, Some(CardClass::Gb16), "explicit");
    let app = server.app();
    for model in [FLASH_NEXT, IMAGE, "qwen3.8-27b-abliterated"] {
        let (status, refused) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": model }))).await;
        refused_by_class(status, &refused, model);
    }
    still_serving(&app, &server, &loader, SIXTEEN).await;

    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": FLASH_NEXT, "artifact": "flash.ninfer" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    assert_eq!(until_serving(&app).await["data"][0]["id"], FLASH_NEXT);
}

/// ADR 0064 §3: a request naming an entry above the card's class is refused
/// before its implicit switch begins, on every route that switches — chat,
/// Responses, images, decisions (`422`, its own refusal shape) — and on the
/// counting route; the loaded model serves on.
#[tokio::test]
async fn a_request_naming_an_entry_above_the_card_class_is_refused_before_anything_stops() {
    let loader = MockLoader::new();
    let server = server_on(&loader, SIXTEEN, Some(CardClass::Gb16), "implicit");
    let app = server.app();
    for model in [FLASH_NEXT, &format!("{FLASH_NEXT}@agent")] {
        let (status, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(model))).await;
        refused_by_class(status, &refused, FLASH_NEXT);
    }
    let (status, refused) = send(&app, Method::POST, "/v1/responses", Some(response_body(FLASH_NEXT))).await;
    refused_by_class(status, &refused, FLASH_NEXT);
    let (status, refused) = send(&app, Method::POST, "/v1/tokenize", Some(json!({ "model": FLASH_NEXT, "messages": [{ "role": "user", "content": "count me" }] }))).await;
    refused_by_class(status, &refused, FLASH_NEXT);
    let (status, refused) = send(&app, Method::POST, "/v1/images/generations", Some(json!({ "model": IMAGE, "prompt": "a fox", "size": "1024x1024" }))).await;
    refused_by_class(status, &refused, IMAGE);
    let (status, refused) = send(&app, Method::POST, "/v1/decide", Some(decision(FLASH_NEXT))).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{refused}");
    assert_eq!(refused["error"]["code"], "model_above_card_class", "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("16 GB"), "{refused}");
    still_serving(&app, &server, &loader, SIXTEEN).await;
}

/// ADR 0064 §2: on a 16 GB card with the 16 GB 27B loaded, a request naming
/// the 27B's default id is served by it — no switch, no refusal — on every
/// route, lane tag and all; the answer names the model that served it.
#[tokio::test]
async fn on_a_16gb_card_the_default_id_is_served_by_the_entry_that_answers_to_it() {
    let loader = MockLoader::new();
    let server = server_on(&loader, SIXTEEN, Some(CardClass::Gb16), "alias");
    let app = server.app();
    // The lane tag is kept, an empty one included.
    for model in [DEFAULT_27B, &format!("{DEFAULT_27B}@agent"), &format!("{DEFAULT_27B}@")] {
        let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(model))).await;
        assert_eq!(status, StatusCode::OK, "{model}: {completion}");
        assert_eq!(completion["model"], SIXTEEN);
    }
    let (status, answered) = send(&app, Method::POST, "/v1/responses", Some(response_body(DEFAULT_27B))).await;
    assert_eq!(status, StatusCode::OK, "{answered}");
    assert_eq!(answered["model"], SIXTEEN);
    let (status, counted) = send(&app, Method::POST, "/v1/tokenize", Some(json!({ "model": DEFAULT_27B, "messages": [{ "role": "user", "content": "count me" }] }))).await;
    assert_eq!(status, StatusCode::OK, "{counted}");
    let (status, decided) = send(&app, Method::POST, "/v1/decide", Some(decision(DEFAULT_27B))).await;
    // The mock template has no answer labels: refused past the model checks.
    assert_ne!(decided["error"]["code"], "model_not_found", "{status} {decided}");
    assert_ne!(decided["error"]["code"], "model_above_card_class", "{status} {decided}");
    assert!(loader.loads().is_empty(), "served by the loaded model, no switch: {:?}", loader.loads());
    assert_eq!(server.active().model_id(), SIXTEEN);
}

/// ADR 0064 §2: from another model, the default id switches to the entry
/// that answers to it on this card, never to its own 32 GB one — and the
/// model loaded stays served by its own id whatever its class.
#[tokio::test]
async fn on_a_16gb_card_the_default_id_switches_to_the_entry_that_answers_to_it() {
    let loader = MockLoader::new();
    let server = server_on(&loader, FLASH_NEXT, Some(CardClass::Gb16), "alias-switch");
    let app = server.app();
    let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(FLASH_NEXT))).await;
    assert_eq!(status, StatusCode::OK, "the loaded model is served, whatever its class: {completion}");
    assert!(loader.loads().is_empty());

    let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(DEFAULT_27B))).await;
    assert_eq!(status, StatusCode::OK, "{completion}");
    assert_eq!(completion["model"], SIXTEEN);
    assert_eq!(loader.loads(), [SIXTEEN]);
    let dir = server.switcher.as_ref().unwrap().catalog().unwrap().dir.clone();
    assert_eq!(loader.prepared_artifacts(), [dir.join("q16.ninfer")], "the 16 GB entry's artifact, not the 32 GB one's");

    // And by id: the explicit switch resolves the alias the same way.
    let (status, _) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": FLASH_NEXT, "artifact": "flash.ninfer" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until_serving(&app).await;
    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": DEFAULT_27B }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    assert_eq!(accepted["to"], SIXTEEN, "{accepted}");
    assert_eq!(until_serving(&app).await["data"][0]["id"], SIXTEEN);
}

/// Spec 27b-16gb/04 §6: on a 32 GB card — or one of no known class — nothing
/// changes: the default id is its own entry's, and a request naming
/// Flash-Next switches to it.
#[tokio::test]
async fn on_a_32gb_card_or_an_unknown_one_nothing_changes() {
    for (card, tag) in [(Some(CardClass::Gb32), "32"), (None, "unknown")] {
        let loader = MockLoader::new();
        let server = server_on(&loader, DEFAULT_27B, card, tag);
        let app = server.app();
        let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(DEFAULT_27B))).await;
        assert_eq!((status, completion["model"].as_str()), (StatusCode::OK, Some(DEFAULT_27B)), "{completion}");
        assert!(loader.loads().is_empty());
        let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(FLASH_NEXT))).await;
        assert_eq!((status, completion["model"].as_str()), (StatusCode::OK, Some(FLASH_NEXT)), "{tag}: {completion}");
        assert_eq!(loader.loads(), [FLASH_NEXT], "{tag}");
        let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
        let data = models["data"].as_array().unwrap();
        assert!(data.iter().all(|entry| entry["available"] == true && entry.get("reason").is_none()), "{tag}: {models}");
    }
}

/// ADR 0064 §3: `GET /v1/models` lists an entry on disk the card cannot
/// load as unavailable, with the reason — the default id's own entry saying
/// which entry serves a request naming it — and `GET /v1/models/catalog`
/// says the same of every row, none of them switchable.
#[tokio::test]
async fn the_model_lists_mark_entries_above_the_card_class_unavailable_with_the_reason() {
    let loader = MockLoader::new();
    let server = server_on(&loader, SIXTEEN, Some(CardClass::Gb16), "lists");
    let app = server.app();
    let (status, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    let listed: Vec<(&str, bool)> =
        models["data"].as_array().unwrap().iter().map(|entry| (entry["id"].as_str().unwrap(), entry["available"].as_bool().unwrap())).collect();
    assert_eq!(
        listed,
        [(SIXTEEN, true), (DEFAULT_27B, false), ("qwen3.8-27b-abliterated", false), (FLASH_NEXT, false), (IMAGE, false)],
        "the entries it lists today, each marked: {models}"
    );
    let reason = |id: &str| -> String {
        let entry = models["data"].as_array().unwrap().iter().find(|entry| entry["id"] == id).unwrap();
        entry["reason"].as_str().unwrap_or_default().to_owned()
    };
    for id in [DEFAULT_27B, FLASH_NEXT, IMAGE] {
        let reason = reason(id);
        assert!(reason.contains(id) && reason.contains("32 GB") && reason.contains("16 GB"), "{id}: {reason}");
    }
    assert!(reason(DEFAULT_27B).contains(&format!("served by {SIXTEEN}")), "{}", reason(DEFAULT_27B));
    assert!(!reason(FLASH_NEXT).contains("served by"), "{}", reason(FLASH_NEXT));
    assert!(models["data"][0].get("reason").is_none(), "{models}");

    let (status, catalog) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!(status, StatusCode::OK);
    for row in catalog["data"].as_array().unwrap() {
        let id = row["id"].as_str().unwrap();
        let fits = id == SIXTEEN;
        assert_eq!(row["available"], fits, "{row}");
        assert_eq!(row.get("reason").is_some(), !fits, "{row}");
        if !fits {
            assert_eq!(row["switchable"], false, "{row}");
        }
    }

    // A model the operator listed by its path is their word (ADR 0047): not
    // fitted to the card, available and switchable on either list.
    let loader = MockLoader::new();
    let flash = switch_support::source(FLASH_NEXT).artifact;
    let listed = server_on(&loader, SIXTEEN, Some(CardClass::Gb16), "lists-listed");
    listed.switcher.as_ref().unwrap().set_listed_models([(FLASH_NEXT.to_owned(), flash)].into());
    let app = listed.app();
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    let entry = models["data"].as_array().unwrap().iter().find(|entry| entry["id"] == FLASH_NEXT).cloned().unwrap();
    assert_eq!((entry["available"].as_bool(), entry.get("reason")), (Some(true), None), "{models}");
    let (_, catalog) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    let row = catalog["data"].as_array().unwrap().iter().find(|row| row["id"] == FLASH_NEXT).cloned().unwrap();
    assert_eq!((row["available"].as_bool(), row["switchable"].as_bool(), row.get("reason")), (Some(true), Some(true), None), "{row}");
    let default_row = catalog["data"].as_array().unwrap().iter().find(|row| row["id"] == DEFAULT_27B).cloned().unwrap();
    assert_eq!(default_row["available"], false, "the entries not listed are still fitted: {default_row}");

    let thirty_two = server_on(&MockLoader::new(), SIXTEEN, Some(CardClass::Gb32), "lists-32").app();
    let (_, catalog) = send(&thirty_two, Method::GET, "/v1/models/catalog", None).await;
    for row in catalog["data"].as_array().unwrap() {
        assert_eq!((row["available"].as_bool(), row.get("reason")), (Some(true), None), "{row}");
        assert_eq!(row["switchable"], row["id"] != SIXTEEN, "{row}");
    }
}
