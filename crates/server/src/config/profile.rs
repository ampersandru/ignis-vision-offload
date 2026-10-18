//! Hardware profiles (spec config-v2/02 §`--profile`): a named bundle of
//! field defaults for one kind of card, slotted between the config file and
//! the hardcoded defaults — a smarter default, never an override.
//!
//! The built-in profiles are written below in the config file's own
//! `profiles:` shape and read by the same [`super::file::read_document`] a
//! file's section is, so a profile an operator defines for a card the
//! maintainers do not own resolves exactly as a built-in one does: there is
//! one code path, and a built-in profile is only the default contents of it.

use std::collections::BTreeMap;

use super::file::read_document;
use super::source::Layer;
use super::ConfigError;

/// The profile used when nothing names one: the owner's own card, measured
/// (below). An unset `--profile` is this one, not "no profile" — for that,
/// name [`NONE_PROFILE`].
pub const DEFAULT_PROFILE: &str = "rtx5090";

/// `--profile none` / `IGNIS_PROFILE=none`: skip the profile layer
/// entirely, hardcoded defaults only. Not a name [`lookup`] resolves — a
/// config file's `profiles:` may not define one by this name either, since
/// it would never be reachable (`gather` checks for it first).
pub const NONE_PROFILE: &str = "none";

/// The built-in profiles, as a config file's `profiles:` section.
///
/// `rtx5090` is the owner's own card, measured end to end on it (2026-10-10):
/// the 27B's leg pushes context and turns vision on, found by raising
/// `model.max_context` under `--vision-enabled` until the VRAM plan refused,
/// then trading `vram.headroom_bytes` down from the 1536 MiB hardcoded
/// default against `spec.backend dflash2` staying on — the owner's own
/// choice over a `vram.headroom_bytes` trade that would have reached ~864K
/// tokens with speculation off instead. Flash-Next's leg holds its decode
/// lanes at their own default (3) and raises `model.max_context` to
/// 512K + 32K + 32K tokens, the floor the owner asked for on both the expert
/// cache (>= 15 GiB; it measured ~16.6 GiB) and the context: both legs need
/// `model.rope_scaling yarn:4` to reach past the checkpoint's trained
/// 262,144-position envelope. `vram.headroom_bytes`'s 1536 MiB hardcoded
/// default (finding 2026-10-09-vram-headroom-wddm-paging) still holds for
/// Flash-Next, which never needed to trade it down.
///
/// `rtxpro6000` is the RTX PRO 6000 Blackwell series (Workstation, Max-Q,
/// Server Edition: 96 GB each, sm_120, the 5090's architecture). Derived
/// first (2026-10-17), then measured on a Server Edition under Linux; the
/// numbers are in the finding 2026-10-18-rtxpro6000-profile-measured. The
/// measurement kept the derived choices and added one. Both families run at
/// 1,048,576 tokens of context, `yarn:4` over the 262,144 trained positions
/// and the 27B's whole hq-e8-2b attention envelope (the plan holds it; no
/// request that long was run). The 27B leaves its headroom at the hardcoded
/// default, since nothing has to be traded for vision any more, and drafts
/// with DFlash2 at 7 tokens, which beat 4 in every cell measured but one.
/// Flash-Next runs 8 decode lanes, where its aggregate peaks: its expert
/// cache holds nearly every expert, so a lane costs its state and its share
/// of the KV pool. It drafts with its MTP head, 3 tokens a lane, which the
/// 5090 leaves off because decode there is bound by the experts it copies
/// over PCIe (finding 2026-10-07-flash-next-mtp-speculation-is-pcie-bound);
/// here MTP pays at one to four active lanes and with simulated agents at
/// four and eight. The row budget stays the verify round's whole 8, so
/// 3 drafts at one or two lanes, 1 at three or four, none from five on. Its
/// n-gram hot rows are `auto`, the measured addition: the whole table in RAM
/// keeps the disk out of every decode round, and `auto` takes it wherever
/// the host plan leaves room, never refusing a start the 1 GiB default would
/// make. The VRAM plan sizes the rest, the KV pool among it. No other card
/// ships a profile; a config file's `profiles:` defines one for any other
/// card.
const BUILT_IN: &str = r#"{
    "rtx5090": {
        "vram": {
            "headroom_bytes": "1536M",
            "qwen38": { "headroom_bytes": "1200M" }
        },
        "model": {
            "rope_scaling": "yarn:4",
            "qwen38": { "max_context": 786432 },
            "qwen38flashnext": { "max_context": 589824 }
        },
        "vision": { "enabled": true },
        "spec": {
            "qwen38": { "backend": "dflash2", "draft_tokens": 7 },
            "decode_lanes": 3
        }
    },
    "rtxpro6000": {
        "model": {
            "rope_scaling": "yarn:4",
            "qwen38": { "max_context": 1048576 },
            "qwen38flashnext": { "max_context": 1048576 }
        },
        "vision": { "enabled": true },
        "spec": {
            "qwen38": { "backend": "dflash2", "draft_tokens": 7 },
            "qwen38flashnext": { "backend": "mtp", "draft_tokens": 3 },
            "decode_lanes": 8
        },
        "ngram": { "hot_bytes": "auto" }
    }
}"#;

/// Every built-in profile, by name.
pub fn built_in() -> BTreeMap<String, Layer> {
    let value = serde_json::from_str(&format!(r#"{{"profiles": {BUILT_IN}}}"#)).expect("the built-in profiles are JSON");
    read_document(&value, "the built-in profiles").expect("the built-in profiles name only declared fields").profiles
}

/// The profile called `name`: the config file's own first (an operator may
/// redefine a built-in one for their card), then the built-in ones.
pub fn lookup(name: &str, from_file: &BTreeMap<String, Layer>) -> Result<Layer, ConfigError> {
    if let Some(layer) = from_file.get(name) {
        return Ok(layer.clone());
    }
    let built_in = built_in();
    built_in.get(name).cloned().ok_or_else(|| {
        let mut known: Vec<&str> = built_in.keys().map(String::as_str).collect();
        known.extend(from_file.keys().map(String::as_str));
        known.sort_unstable();
        known.dedup();
        ConfigError(format!("no profile `{name}` (the profiles are {})", known.join(", ")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::field;

    #[test]
    fn the_default_profile_carries_the_rtx5090s_measured_numbers() {
        use ignis_core::compute::ModelFamily;

        let profiles = built_in();
        let rtx5090 = &profiles[DEFAULT_PROFILE];
        let headroom = field("vram", "headroom_bytes").unwrap();
        // Flash-Next never traded headroom down: the hardcoded default still
        // holds for it, and as the group's general (unscoped) value.
        assert_eq!((headroom.default)(), serde_json::Value::String("1536M".into()));
        assert_eq!(rtx5090.get(headroom, None).unwrap().raw, "1536M");
        assert!(rtx5090.get(headroom, Some(ModelFamily::FlashNext)).is_none(), "no override; it resolves to the general value");
        // The 27B traded it down to fit vision plus its own context.
        assert_eq!(rtx5090.get(headroom, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "1200M");

        let max_context = field("model", "max_context").unwrap();
        assert_eq!(rtx5090.get(max_context, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "786432");
        assert_eq!(rtx5090.get(max_context, Some(ModelFamily::FlashNext)).unwrap().raw, "589824");

        let rope = field("model", "rope_scaling").unwrap();
        assert_eq!(rtx5090.get(rope, None).unwrap().raw, "yarn:4", "one rope table, both families");

        let vision = field("vision", "enabled").unwrap();
        assert_eq!(rtx5090.get(vision, None).unwrap().raw, "true");

        let backend = field("spec", "backend").unwrap();
        assert_eq!(rtx5090.get(backend, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "dflash2");
        assert!(rtx5090.get(backend, Some(ModelFamily::FlashNext)).is_none(), "Flash-Next keeps speculation off here");

        let lanes = field("spec", "decode_lanes").unwrap();
        assert_eq!(rtx5090.get(lanes, None).unwrap().raw, "3", "Flash-Next-only; the 27B passes it over");
    }

    /// The RTX PRO 6000 Blackwell series (Workstation, Max-Q, Server Edition:
    /// 96 GB each, sm_120), measured on a Server Edition (2026-10-18).
    #[test]
    fn the_rtxpro6000_profile_gives_both_models_their_whole_context_and_flash_next_every_lane() {
        use ignis_core::compute::ModelFamily;

        let profiles = built_in();
        let pro = &profiles["rtxpro6000"];
        let max_context = field("model", "max_context").unwrap();
        // yarn:4 over the checkpoints' 262,144 trained positions: the most
        // either model can be asked for.
        assert_eq!(pro.get(max_context, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "1048576");
        assert_eq!(pro.get(max_context, Some(ModelFamily::FlashNext)).unwrap().raw, "1048576");
        let rope = field("model", "rope_scaling").unwrap();
        assert_eq!(pro.get(rope, None).unwrap().raw, "yarn:4");

        let lanes = field("spec", "decode_lanes").unwrap();
        assert_eq!(pro.get(lanes, None).unwrap().raw, "8", "every lane: on 96 GB a lane costs its state");

        // The 5090's choices the card's memory does not change.
        let vision = field("vision", "enabled").unwrap();
        assert_eq!(pro.get(vision, None).unwrap().raw, "true");
        let backend = field("spec", "backend").unwrap();
        assert_eq!(pro.get(backend, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "dflash2");
        let draft = field("spec", "draft_tokens").unwrap();
        assert_eq!(pro.get(draft, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "7");

        // Flash-Next drafts with its MTP head, 3 tokens a lane, cut only by
        // the verify round's own 8 rows: on 96 GB no expert crosses PCIe,
        // the cost that made MTP lose on the 5090.
        assert_eq!(pro.get(backend, Some(ModelFamily::FlashNext)).unwrap().raw, "mtp");
        assert_eq!(pro.get(draft, Some(ModelFamily::FlashNext)).unwrap().raw, "3");
        let rows = field("spec", "draft_rows").unwrap();
        assert!(pro.get(rows, None).is_none(), "unset: the whole row budget");

        // No headroom traded down: the hardcoded default, for both families.
        let headroom = field("vram", "headroom_bytes").unwrap();
        assert!(pro.get(headroom, None).is_none());
        assert!(pro.get(headroom, Some(ModelFamily::Qwen38_27b)).is_none());

        // The whole n-gram table when the host plan leaves room for it, so no
        // decode round reads the disk; Flash-Next-only, the 27B passes it over.
        let hot = field("ngram", "hot_bytes").unwrap();
        assert_eq!(pro.get(hot, None).unwrap().raw, "auto");
        assert!(profiles[DEFAULT_PROFILE].get(hot, None).is_none(), "the 5090 keeps the 1 GiB default");
    }

    /// The 27B started under `rtxpro6000` takes the profile's Flash-Next-only
    /// n-gram budget as a default it passes over, not a value to refuse.
    #[test]
    fn the_27b_starts_under_the_rtxpro6000_profile() {
        use super::super::source::{resolve_settings, Fit, Sources};
        use ignis_core::compute::ModelFamily;

        let sources = Sources { profile: built_in()["rtxpro6000"].clone(), ..Sources::default() };
        let qwen = resolve_settings(&sources, Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap();
        assert_eq!(qwen.settings.ngram.hot_bytes, None);
        let flash = resolve_settings(&sources, Some(ModelFamily::FlashNext), Fit::Start).unwrap();
        assert_eq!(flash.settings.ngram.hot_bytes.map(|b| b.to_string()).as_deref(), Some("auto"));
    }

    #[test]
    fn a_file_profile_shadows_a_built_in_one_and_an_unknown_name_lists_the_known() {
        let headroom = field("vram", "headroom_bytes").unwrap();
        let mut mine = Layer::default();
        mine.set(headroom, None, super::super::source::Candidate { raw: "2G".into(), spelling: "x".into() }).unwrap();
        let from_file = BTreeMap::from([("rtx5090".to_owned(), mine.clone()), ("a4000".to_owned(), mine.clone())]);
        assert_eq!(lookup("rtx5090", &from_file).unwrap(), mine);
        assert_eq!(lookup("rtx5090", &BTreeMap::new()).unwrap(), built_in()["rtx5090"]);
        let err = lookup("h100", &from_file).unwrap_err().0;
        assert!(err.contains("`h100`") && err.contains("a4000, rtx5090"), "{err}");
    }
}
