//! A fake Hugging Face for the download routes' tests (GitHub #376), included
//! via `#[path]`: a loopback `axum` server answering the route a repo file
//! is fetched from, `{repo}/resolve/{revision}/{file}`, with `Range`, and an
//! operator catalog whose entries pin exactly the bytes it serves, so a
//! download the server starts really verifies.
//!
//! A test holds a download mid-transfer by holding one file's body after a
//! number of bytes until it releases it ([`Repo::hold`], [`Repo::release`]),
//! never by a timer; it makes the endpoint refuse every request
//! with a status ([`Repo::answer`]), as a gated repository does.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, Mutex};

use axum::body::{Body, Bytes};
use axum::extract::{Path as UrlPath, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use ignis_server::config::file::Format;
use ignis_server::download::catalog::parse_operator;
use ignis_server::model_switch::CatalogModels;

/// One request the endpoint was sent.
#[derive(Debug, Clone)]
pub struct Seen {
    pub file: String,
    pub range: Option<String>,
    pub authorization: Option<String>,
}

#[derive(Default)]
struct Served {
    /// Every file the endpoint holds, by name, whatever the repo.
    files: Mutex<BTreeMap<String, Vec<u8>>>,
    /// Every request, in order.
    seen: Mutex<Vec<Seen>>,
    /// Answer every request with this status instead (0 = serve the file).
    status: AtomicU16,
    /// Hold this file's body after this many bytes, until released.
    hold: Mutex<Option<(String, usize)>>,
    /// Lets one held body send the rest.
    release: tokio::sync::Notify,
}

/// The fake endpoint and the state a test steers it with.
#[derive(Clone)]
pub struct Repo {
    pub addr: SocketAddr,
    served: Arc<Served>,
}

impl Repo {
    /// A new endpoint on a loopback port, serving nothing yet.
    pub async fn start() -> Self {
        let served = Arc::new(Served::default());
        let app = Router::new().route("/{owner}/{name}/resolve/{revision}/{file}", get(resolve)).with_state(Arc::clone(&served));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Self { addr, served }
    }

    /// Its URL, as `download.endpoint` names it.
    pub fn endpoint(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Serve `body` as `name`.
    pub fn serve(&self, name: &str, body: Vec<u8>) {
        self.served.files.lock().unwrap().insert(name.to_owned(), body);
    }

    /// Answer every request `status` from now on; 0 serves the files again.
    pub fn answer(&self, status: u16) {
        self.served.status.store(status, Ordering::SeqCst);
    }

    /// Hold every response for `file` after `after` body bytes, until
    /// [`Repo::release`].
    pub fn hold(&self, file: &str, after: usize) {
        *self.served.hold.lock().unwrap() = Some((file.to_owned(), after));
    }

    /// Let one held response send the rest of its body, and hold no more.
    pub fn release(&self) {
        *self.served.hold.lock().unwrap() = None;
        self.served.release.notify_one();
    }

    /// Every request, in order.
    pub fn seen(&self) -> Vec<Seen> {
        self.served.seen.lock().unwrap().clone()
    }
}

async fn resolve(State(served): State<Arc<Served>>, UrlPath((_owner, _name, _revision, file)): UrlPath<(String, String, String, String)>, headers: HeaderMap) -> Response {
    let text = |name| headers.get(name).and_then(|value: &header::HeaderValue| value.to_str().ok()).map(str::to_owned);
    let range = text(header::RANGE);
    served.seen.lock().unwrap().push(Seen { file: file.clone(), range: range.clone(), authorization: text(header::AUTHORIZATION) });
    let status = served.status.load(Ordering::SeqCst);
    if status != 0 {
        return StatusCode::from_u16(status).expect("a status").into_response();
    }
    let Some(body) = served.files.lock().unwrap().get(&file).cloned() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let start = range.as_deref().and_then(|raw| raw.trim_start_matches("bytes=").trim_end_matches('-').parse::<usize>().ok()).unwrap_or(0);
    let status = if start > 0 { StatusCode::PARTIAL_CONTENT } else { StatusCode::OK };
    let mut body = body[start.min(body.len())..].to_vec();
    let hold = served.hold.lock().unwrap().clone().filter(|(held, after)| *held == file && *after < body.len());
    let Some((_, after)) = hold else {
        return (status, Body::from(body)).into_response();
    };
    let rest = body.split_off(after);
    let first = futures_util::stream::once(async move { Ok::<_, std::io::Error>(Bytes::from(body)) });
    let then = futures_util::stream::once(async move {
        served.release.notified().await;
        Ok(Bytes::from(rest))
    });
    (status, Body::from_stream(futures_util::StreamExt::chain(first, then))).into_response()
}

/// An entry's two files: a sidecar and an artifact.
pub struct Files {
    pub sidecar: (String, Vec<u8>),
    pub artifact: (String, Vec<u8>),
}

impl Files {
    /// Both files' bytes together: the entry's total.
    pub fn total(&self) -> u64 {
        (self.sidecar.1.len() + self.artifact.1.len()) as u64
    }
}

/// `id`'s files: a 1-byte sidecar and a 64 KiB `<id>.ninfer`, long enough to
/// hold mid-body.
pub fn files_of(id: &str) -> Files {
    Files {
        sidecar: (format!("{id}.ninfer.graft.json"), b"{".to_vec()),
        artifact: (format!("{id}.ninfer"), (0..64 * 1024u32).map(|i| (i % 251) as u8).collect()),
    }
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

/// An operator catalog of `entries` (`id`, `card_class`), each pinned to the
/// files [`files_of`] gives it — served by `repo` — under a fresh, empty
/// `download.path`.
pub fn catalog_served(repo: &Repo, tag: &str, entries: &[(&str, u32)]) -> CatalogModels {
    let dir = fresh_dir(tag);
    let mut yaml = String::from("models:\n");
    for (id, card_class) in entries {
        let files = files_of(id);
        yaml.push_str(&format!(
            "  - id: {id}\n    repo: acme/{id}\n    revision: v1\n    artifact: {id}.ninfer\n    card_class: {card_class}\n    files:\n      - {{ name: {}, bytes: {}, sha256: \"{}\" }}\n      - {{ name: {}, bytes: {}, sha256: \"{}\" }}\n",
            files.sidecar.0,
            files.sidecar.1.len(),
            sha256(&files.sidecar.1),
            files.artifact.0,
            files.artifact.1.len(),
            sha256(&files.artifact.1),
        ));
        repo.serve(&files.sidecar.0, files.sidecar.1);
        repo.serve(&files.artifact.0, files.artifact.1);
    }
    let catalog = parse_operator(&yaml, Format::Yaml, "test").expect("a valid catalog");
    CatalogModels { catalog: Arc::new(catalog), dir }
}

/// A directory of this test's own, empty.
pub fn fresh_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ignis-download-http-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
