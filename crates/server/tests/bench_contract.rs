//! What `ignis-bench small-card` copies from the server, held to the server
//! (#340, spec 27b-16gb/05 AC 7). The bench crate cannot depend on this one
//! (this one dev-depends on it), so the copies are checked from this side: a
//! field renamed or hidden, or the metrics listener's default moved, fails
//! here instead of a value silently dropping out of the record a user pastes.
//! CPU-only.

use ignis_bench::small_card::{CONFIG_KEYS, DEFAULT_METRICS_PORT};
use ignis_server::config::schema::GROUPS;
use ignis_server::config::DEFAULT_METRICS_BIND;

/// The fields the preset reads off `GET /v1/config` beside its whitelist:
/// where the metrics listener is (`small_card::metrics_url_from_config`).
const METRICS_KEYS: [&str; 2] = ["server.metrics", "server.metrics_bind"];

#[test]
fn every_config_key_the_small_card_reads_is_a_field_get_v1_config_shows() {
    for key in CONFIG_KEYS.iter().chain(&METRICS_KEYS) {
        let (group, name) = key.split_once('.').unwrap_or_else(|| panic!("`{key}` is not group.field"));
        let field = GROUPS
            .iter()
            .find(|(g, _)| *g == group)
            .and_then(|(_, fields)| fields.iter().find(|f| f.name == name))
            .unwrap_or_else(|| panic!("`{key}` is no config field"));
        assert!(field.visible, "`{key}` is hidden: GET /v1/config never shows it");
    }
}

#[test]
fn the_small_cards_default_metrics_port_is_the_servers() {
    let port = DEFAULT_METRICS_BIND.rsplit_once(':').and_then(|(_, port)| port.parse::<u16>().ok());
    assert_eq!(port, Some(DEFAULT_METRICS_PORT), "DEFAULT_METRICS_BIND is {DEFAULT_METRICS_BIND}");
}
