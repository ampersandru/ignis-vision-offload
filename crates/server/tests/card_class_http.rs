//! The card class (ADR 0064, spec 27b-16gb/04 §6, GitHub #336) over the
//! mock-backed router: on a 16 GB card a switch to a catalog entry above
//! the card's class — explicit, or a request's implicit one — is refused
//! `409 model_above_card_class` before anything is prepared, drained or torn
//! down, and the loaded model keeps serving; a request naming the 27B's
//! default id is served by the 16 GB entry that answers to it as an alias,
//! and only by an entry whose artifact is of the 27B's family;
//! `GET /v1/models` and `GET /v1/models/catalog` list such entries as
//! unavailable, with the reason. On a 32 GB card, or one of no known class,
//! nothing changes. The catalog is the built-in one with the 16 GB 27B as an
//! operator entry in place of the one GitHub #338 adds; every artifact is a
//! file under `download.path`, and the models are `support/switch.rs`'s
//! mocks, loaded by id.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_core::compute::ModelFamily;
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

#[path = "support/responses.rs"]
mod responses;

const PATIENT: Duration = Duration::from_secs(30);

/// The 16 GB 27B's catalog entry, the one the catalog's own tests read.
const SIXTEEN_CATALOG: &str = include_str!("fixtures/catalog/sixteen.yaml");

/// The 16 GB 27B's served id, and the id it answers to.
const SIXTEEN: &str = "qwen3.8-27b-16g";
const DEFAULT_27B: &str = "qwen3.8-27b";
const FLASH_NEXT: &str = "qwen3.8-flash-next";
const IMAGE: &str = "ignis-image-2.1-turbo";

/// A condition image an edit decodes.
const PNG: &[u8] = include_bytes!("../../artifact/tests/fixtures/vision/images/on_grid.png");

/// The built-in catalog and the operator's `yaml`, every entry's artifact a
/// file under a fresh `download.path`.
fn catalog_on_disk(tag: &str, yaml: &str) -> CatalogModels {
    let dir = std::env::temp_dir().join(format!("ignis-card-class-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let catalog = parse_operator(yaml, Format::Yaml, "test").expect("a valid catalog");
    for entry in catalog.entries() {
        std::fs::write(entry.artifact_path(&dir), b"mock").unwrap();
    }
    CatalogModels { catalog: Arc::new(catalog), dir }
}

/// A server on `start`, on a card of class `card`, whose requests may switch
/// to any catalog entry on disk — the built-in ones and the 16 GB 27B, whose
/// artifact's header names the 27B's family.
fn server_on(loader: &Arc<MockLoader>, start: &str, card: Option<CardClass>, tag: &str) -> Server {
    loader.header(SIXTEEN, ModelFamily::Qwen38_27b);
    server_over(loader, start, card, catalog_on_disk(tag, SIXTEEN_CATALOG))
}

fn server_over(loader: &Arc<MockLoader>, start: &str, card: Option<CardClass>, catalog: CatalogModels) -> Server {
    Server::from_active(loader.model(start))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT).with_catalog(catalog, true).with_card_class(card))
}

async fn send(app: &axum::Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    respond(app, request).await
}

async fn respond(app: &axum::Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, body)
}

/// `POST /v1/images/edits` naming `model`, with a prompt and one condition
/// image.
async fn edit(app: &axum::Router, model: &str) -> (StatusCode, Value) {
    const BOUNDARY: &str = "ignis-card-class-boundary";
    let mut body = Vec::new();
    for (name, text) in [("model", model), ("prompt", "a fox")] {
        body.extend(format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{text}\r\n").bytes());
    }
    body.extend(
        format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"image\"\r\nContent-Type: application/octet-stream\r\n\r\n")
            .bytes(),
    );
    body.extend_from_slice(PNG);
    body.extend(format!("\r\n--{BOUNDARY}--\r\n").bytes());
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/images/edits")
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={BOUNDARY}"))
        .body(Body::from(body))
        .unwrap();
    respond(app, request).await
}

/// One `response.create` over the Responses WebSocket naming `model`, and
/// the events up to its end or its `error` — with the listener, which the
/// caller keeps while it reads the server: once dropped, its serve stops,
/// and the model it shares with the router with it.
async fn over_the_socket(server: &Server, model: &str) -> (responses::Live, Vec<Value>) {
    let live = responses::live(server.clone()).await;
    let mut socket = responses::socket(&live).await;
    let mut create = response_body(model);
    create["type"] = json!("response.create");
    responses::send(&mut socket, create).await;
    let events = responses::events_until(&mut socket, responses::terminal).await;
    (live, events)
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
/// Responses over HTTP and over its WebSocket, image generations and edits,
/// decisions (`422`, its own refusal shape) — and on the counting route; the
/// loaded model serves on.
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
    let (_live, events) = over_the_socket(&server, FLASH_NEXT).await;
    let refused = events.last().expect("an event");
    assert_eq!((refused["type"].as_str(), refused["status"].as_u64()), (Some("error"), Some(409)), "{events:?}");
    refused_by_class(StatusCode::CONFLICT, refused, FLASH_NEXT);
    let (status, refused) = send(&app, Method::POST, "/v1/tokenize", Some(json!({ "model": FLASH_NEXT, "messages": [{ "role": "user", "content": "count me" }] }))).await;
    refused_by_class(status, &refused, FLASH_NEXT);
    let (status, refused) = send(&app, Method::POST, "/v1/images/generations", Some(json!({ "model": IMAGE, "prompt": "a fox", "size": "1024x1024" }))).await;
    refused_by_class(status, &refused, IMAGE);
    let (status, refused) = edit(&app, IMAGE).await;
    refused_by_class(status, &refused, IMAGE);
    let (status, refused) = send(&app, Method::POST, "/v1/decide", Some(decision(FLASH_NEXT))).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{refused}");
    assert_eq!(refused["error"]["code"], "model_above_card_class", "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("16 GB"), "{refused}");
    still_serving(&app, &server, &loader, SIXTEEN).await;
}

/// ADR 0064 §2: on a 16 GB card with the 16 GB 27B loaded, a request naming
/// the 27B's default id is served by it — no switch, no refusal — on every
/// route, lane tag and all; the answer names the model that served it. On
/// the image routes the id the kind check reads is the loaded 16 GB 27B's,
/// a text model, never the default id.
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
    let (_live, events) = over_the_socket(&server, DEFAULT_27B).await;
    let last = events.last().expect("a terminal event");
    assert!(matches!(last["type"].as_str(), Some("response.completed" | "response.incomplete")), "{events:?}");
    assert_eq!(last["response"]["model"], SIXTEEN, "{last}");
    let (status, counted) = send(&app, Method::POST, "/v1/tokenize", Some(json!({ "model": DEFAULT_27B, "messages": [{ "role": "user", "content": "count me" }] }))).await;
    assert_eq!(status, StatusCode::OK, "{counted}");
    let (status, decided) = send(&app, Method::POST, "/v1/decide", Some(decision(DEFAULT_27B))).await;
    // The mock template has no answer labels: refused past the model checks.
    assert_ne!(decided["error"]["code"], "model_not_found", "{status} {decided}");
    assert_ne!(decided["error"]["code"], "model_above_card_class", "{status} {decided}");
    let generation = send(&app, Method::POST, "/v1/images/generations", Some(json!({ "model": DEFAULT_27B, "prompt": "a fox", "size": "1024x1024" }))).await;
    for (status, refused) in [generation, edit(&app, DEFAULT_27B).await] {
        assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::CONFLICT, Some("model_kind_mismatch")), "{refused}");
        let message = refused["error"]["message"].as_str().unwrap();
        assert!(message.contains(&format!("the loaded model {SIXTEEN} is a text model")) && !message.contains(&format!("names {DEFAULT_27B},")), "{message}");
    }
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

/// ADR 0064 §2: an alias only ever resolves within one model family, and the
/// family is the artifact's header's (ADR 0047), not the alias's. An
/// operator entry that answers to the 27B's default id but holds a
/// Flash-Next artifact — on disk, so its header is read before anything
/// stops — does not answer to it: a request or a switch naming the default
/// id is refused naming both families, nothing is prepared, and the lists
/// say why. The same entry with a 27B header answers to it.
#[tokio::test]
async fn an_alias_never_resolves_to_an_artifact_of_another_family() {
    let acme = SIXTEEN_CATALOG.replace("id: qwen3.8-27b-16g", "id: acme-fn-16g");
    for (header, tag) in [(ModelFamily::FlashNext, "family-other"), (ModelFamily::Qwen38_27b, "family-same")] {
        let loader = MockLoader::new();
        loader.header("acme-fn-16g", header);
        let server = server_over(&loader, "mock-a", Some(CardClass::Gb16), catalog_on_disk(tag, &acme));
        let app = server.app();
        if header == ModelFamily::Qwen38_27b {
            let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(DEFAULT_27B))).await;
            assert_eq!((status, completion["model"].as_str()), (StatusCode::OK, Some("acme-fn-16g")), "{completion}");
            continue;
        }
        let (status, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(DEFAULT_27B))).await;
        refused_by_class(status, &refused, DEFAULT_27B);
        let message = refused["error"]["message"].as_str().unwrap();
        for named in ["acme-fn-16g lists it as an alias", "Qwen3.8-Flash-Next", "Qwen3.8-27B", "one model family"] {
            assert!(message.contains(named), "{named}: {message}");
        }
        let (status, refused) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": DEFAULT_27B }))).await;
        refused_by_class(status, &refused, DEFAULT_27B);
        still_serving(&app, &server, &loader, "mock-a").await;
        let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
        let default = models["data"].as_array().unwrap().iter().find(|entry| entry["id"] == DEFAULT_27B).cloned().unwrap();
        assert_eq!(default["available"], false, "{default}");
        assert!(default["reason"].as_str().unwrap().contains("one model family"), "{default}");
        // Named by its own id, the entry is the operator's word: served.
        let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("acme-fn-16g"))).await;
        assert_eq!((status, completion["model"].as_str()), (StatusCode::OK, Some("acme-fn-16g")), "{completion}");
    }
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

    let thirty_two = server_on(&MockLoader::new(), SIXTEEN, Some(CardClass::Gb32), "lists-32").app();
    let (_, catalog) = send(&thirty_two, Method::GET, "/v1/models/catalog", None).await;
    for row in catalog["data"].as_array().unwrap() {
        assert_eq!((row["available"].as_bool(), row.get("reason")), (Some(true), None), "{row}");
        assert_eq!(row["switchable"], row["id"] != SIXTEEN, "{row}");
    }
}

/// ADR 0064 §3: "A switch, explicit or implicit, and a download check the
/// card class first" — the class is the catalog entry's, so a catalog id the
/// operator also listed by path (`switch.known_models`) is fitted all the
/// same: unavailable on both lists, never switchable, and a request or a
/// switch naming it is refused before anything stops. An id that is no
/// catalog entry, the operator's own model, has no class and is never
/// refused by it.
#[tokio::test]
async fn a_catalog_id_the_operator_listed_is_fitted_to_the_card_all_the_same() {
    let loader = MockLoader::new();
    let server = server_on(&loader, SIXTEEN, Some(CardClass::Gb16), "listed");
    let switcher = server.switcher.as_ref().unwrap();
    let listed = [(FLASH_NEXT, switch_support::source(FLASH_NEXT).artifact), ("acme-own", switch_support::source("acme-own").artifact)];
    let listed: std::collections::BTreeMap<String, std::path::PathBuf> = listed.into_iter().map(|(id, artifact)| (id.to_owned(), artifact)).collect();
    switcher.set_knobs(PATIENT, listed.clone(), true);
    switcher.set_listed_models(listed);
    let app = server.app();
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    let entry = |id: &str| models["data"].as_array().unwrap().iter().find(|entry| entry["id"] == id).cloned().unwrap();
    assert_eq!(entry(FLASH_NEXT)["available"], false, "{models}");
    assert!(entry(FLASH_NEXT)["reason"].as_str().unwrap().contains("16 GB"), "{models}");
    assert_eq!((entry("acme-own")["available"].as_bool(), entry("acme-own").get("reason")), (Some(true), None), "{models}");
    let (_, catalog) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    let row = catalog["data"].as_array().unwrap().iter().find(|row| row["id"] == FLASH_NEXT).cloned().unwrap();
    assert_eq!((row["available"].as_bool(), row["switchable"].as_bool()), (Some(false), Some(false)), "{row}");
    let (status, refused) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": FLASH_NEXT }))).await;
    refused_by_class(status, &refused, FLASH_NEXT);
    let (status, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(FLASH_NEXT))).await;
    refused_by_class(status, &refused, FLASH_NEXT);
    still_serving(&app, &server, &loader, SIXTEEN).await;

    let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("acme-own"))).await;
    assert_eq!((status, completion["model"].as_str()), (StatusCode::OK, Some("acme-own")), "the operator's own model: {completion}");
}

/// ADR 0064 §2 for `image.prompt_enhancer`, `qwen3.8-27b` by default: it
/// takes the step a request's `model` takes, so on a 16 GB card the rewrite
/// runs on the entry that answers to the default id — the loaded one in
/// place, never a visit to itself, or the one a visit loads. With no such
/// entry it is an enhancer no visit reaches, which the start and a `PATCH`
/// say; on a 32 GB card it is its own entry's.
#[tokio::test]
async fn the_prompt_enhancer_is_served_by_the_entry_that_answers_to_it() {
    use ignis_server::images::enhance::{enhancer_on_this_card, unknown_enhancer};
    let none = std::collections::BTreeMap::new();
    let loaded = server_on(&MockLoader::new(), SIXTEEN, Some(CardClass::Gb16), "enhancer-loaded");
    assert_eq!(enhancer_on_this_card(&loaded, DEFAULT_27B), SIXTEEN, "the loaded model rewrites in place");
    assert_eq!(unknown_enhancer(&loaded, DEFAULT_27B, &none), None);

    let visited = server_on(&MockLoader::new(), FLASH_NEXT, Some(CardClass::Gb16), "enhancer-visit");
    assert_eq!(enhancer_on_this_card(&visited, DEFAULT_27B), SIXTEEN);
    let via = visited.switcher.as_ref().unwrap().text_model(SIXTEEN).expect("a visit loads the 16 GB 27B");
    assert_eq!(via.artifact.file_name().and_then(|name| name.to_str()), Some("q16.ninfer"));
    assert_eq!(unknown_enhancer(&visited, DEFAULT_27B, &none), None);

    // The built-in catalog alone: no entry answers to the default id here.
    let alone = server_over(&MockLoader::new(), FLASH_NEXT, Some(CardClass::Gb16), catalog_on_disk("enhancer-alone", "models: []"));
    assert_eq!(enhancer_on_this_card(&alone, DEFAULT_27B), DEFAULT_27B);
    assert!(unknown_enhancer(&alone, DEFAULT_27B, &none).is_some_and(|reason| reason.contains(DEFAULT_27B)));

    let thirty_two = server_on(&MockLoader::new(), FLASH_NEXT, Some(CardClass::Gb32), "enhancer-32");
    assert_eq!(enhancer_on_this_card(&thirty_two, DEFAULT_27B), DEFAULT_27B, "its own entry fits");
    assert_eq!(unknown_enhancer(&thirty_two, DEFAULT_27B, &none), None);
}
