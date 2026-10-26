//! GitHub #378 (spec #374, Testing Decisions, Seam 2): the start-time fetch
//! checks the room on the disk as the Server view's Download and `model
//! download` do. An entry its models folder has no room for — its bytes still
//! missing plus `download.free_margin` — is refused before a byte is written,
//! logging `ignis.model.download_failed` at ERROR with the reason, "refusing
//! to start"; `main` exits 1 on that refusal, as on a failed fetch. The short
//! disk is the margin itself, set larger than any disk.
//!
//! A binary of its own, with one test: `tracing` caches whether a callsite is
//! enabled, and a test of another thread of the same binary can settle that
//! cache before this one's scoped subscriber is consulted
//! (`model_download_log.rs` says the same).

use std::sync::Arc;

use serde_json::{json, Value};
use tracing_subscriber::layer::SubscriberExt;

#[path = "support/repo.rs"]
mod repo;
use repo::{catalog_served, Repo};

#[tokio::test]
async fn the_start_time_fetch_refuses_an_entry_the_folder_has_no_room_for_and_logs_it() {
    let repo = Repo::start().await;
    let models = catalog_served(&repo, "start-room", &[("acme-boot", 32)]);
    let (entry, dir) = (models.catalog.entry("acme-boot").expect("the entry").clone(), models.dir.clone());
    let sink = Arc::new(ignis_logging::MemorySink::new());
    let _guard = tracing::subscriber::set_default(tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())));
    let failed = || {
        sink.lines()
            .into_iter()
            .filter_map(|line| serde_json::from_str::<Value>(&line).ok())
            .filter(|event| event["event_name"] == "ignis.model.download_failed")
            .collect::<Vec<Value>>()
    };

    let margin = 1_000_000u64 << 30;
    let refused = ignis_server::download::fetch_at_start(&entry, &dir, &repo.endpoint(), None, margin).await.expect_err("no room");
    let reason = refused.to_string();
    assert!(reason.starts_with("not enough room for acme-boot under"), "{reason}");
    assert!(reason.contains(&format!("{} bytes needed", margin + entry.total_bytes())) && reason.contains("bytes free"), "both numbers: {reason}");
    assert!(repo.seen().is_empty(), "nothing fetched: {:?}", repo.seen());
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "nothing written, not even the entry's lock");

    let events = failed();
    assert_eq!(events.len(), 1, "{:?}", sink.lines());
    let event = &events[0];
    assert_eq!((event["severity_text"].clone(), event["body"].clone()), (json!("ERROR"), json!("refusing to start")), "{event}");
    let fields = &event["attributes"];
    assert_eq!((fields["model"].clone(), fields["repo"].clone(), fields["endpoint"].clone()), (json!("acme-boot"), json!("acme/acme-boot"), json!(repo.endpoint())), "{event}");
    assert_eq!(fields["error"], json!(reason), "{event}");

    // With room, the same start fetches it and goes on.
    let path = ignis_server::download::fetch_at_start(&entry, &dir, &repo.endpoint(), None, 0).await.expect("fetched");
    assert_eq!(path, entry.artifact_path(&dir));
    assert_eq!(failed().len(), 1, "no other failure: {:?}", sink.lines());
    let _ = std::fs::remove_dir_all(dir);
}
