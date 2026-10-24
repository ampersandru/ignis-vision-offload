//! Response ids: `resp_` and 32 hex digits, 128 random bits per response,
//! so no two responses share one whatever load served them (why they
//! stopped being the request id: spec responses-api/01, Output items). The
//! `ignis.response.admitted` line pairs each with the request id the
//! engine's lines carry (ADR 0012).

use std::sync::atomic::{AtomicU64, Ordering};

/// Where a server's response ids come from ([`crate::Server::response_ids`]).
#[derive(Debug, Default)]
pub enum ResponseIds {
    /// 128 bits from the OS random source for each response.
    #[default]
    Random,
    /// 0, 1, 2… as 32 hex digits: a test's events the same on every run, as
    /// a fixed wall clock keeps their `created_at`.
    Counted(AtomicU64),
}

impl ResponseIds {
    /// Ids counted from 0.
    pub(crate) fn counted() -> Self {
        Self::Counted(AtomicU64::new(0))
    }

    /// The next response's suffix: what its `resp_` id, and the `msg_`,
    /// `rs_`, `fc_` and `call_` ids of its items, are made of.
    pub(crate) fn next_suffix(&self) -> String {
        match self {
            Self::Counted(next) => format!("{:032x}", next.fetch_add(1, Ordering::Relaxed)),
            Self::Random => random_suffix(),
        }
    }
}

/// 128 bits from the OS random source as 32 lowercase hex digits: a random
/// response's suffix, and a Realtime session's and its items' (spec
/// speech/03).
pub(crate) fn random_suffix() -> String {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).expect("the OS random source");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Write the `ignis.response.admitted` line: `response_id` is served by the
/// engine's `request_id`, the id every line of the engine's names it by.
pub(crate) fn log_admitted(response_id: &str, request_id: ignis_core::RequestId) {
    tracing::info!(name: "ignis.response.admitted", response_id, request_id, "response admitted: its id, and the request the engine serves it as");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_suffix(s: &str) -> bool {
        s.len() == 32 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }

    #[test]
    fn random_suffixes_are_32_lowercase_hex_digits_and_do_not_repeat() {
        let ids = ResponseIds::Random;
        let drawn: Vec<String> = (0..1000).map(|_| ids.next_suffix()).collect();
        assert!(drawn.iter().all(|s| is_suffix(s)), "{drawn:?}");
        assert_eq!(drawn.iter().collect::<std::collections::HashSet<_>>().len(), drawn.len());
    }

    #[test]
    fn counted_suffixes_count_from_zero() {
        let ids = ResponseIds::counted();
        assert_eq!(ids.next_suffix(), "0".repeat(32));
        assert_eq!(ids.next_suffix(), format!("{}1", "0".repeat(31)));
        assert!(is_suffix(&ids.next_suffix()));
    }
}
