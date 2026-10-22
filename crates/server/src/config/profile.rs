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
/// tokens with speculation off instead. Re-measured the same way on
/// 2026-10-21 (GitHub #335), once the prompt planes were gone and the token
/// embedding had moved to host memory: 917,504 holds, 950,272 is refused by
/// 59 MB (finding 2026-10-21-prefill-attention-without-prompt-planes). The
/// 27B prefills 2,048 tokens a chunk (GitHub #341): with vision on the
/// encoder sizes the workspace arena, so the plan is the same byte for byte
/// as at 1,024, and against 1,024 it measured +1.9% / +2.2% cold prefill at
/// 46K / 105K, the same at 8K / 14K, a lower 4K KLD and decode unchanged; a
/// decoding lane's longest pause while a prompt prefills grows ~65% (finding
/// 2026-10-21-the-27b-prefill-chunk-at-2048). Flash-Next's leg holds its decode
/// lanes at their own default (3) and raises `model.max_context` to
/// 512K + 32K + 32K tokens, the floor the owner asked for on both the expert
/// cache (>= 15 GiB; it measured ~16.6 GiB) and the context. Past the
/// checkpoint's trained 262,144 positions the 27B's text rotary table needs
/// YaRN, which its leg does not spell out: `model.rope_scaling` defaults to
/// `auto`, the factor `max_context / 262,144` (3.5 at 917,504). Flash-Next
/// resolves the same way over its 589,824: 2.25 (ADR 0057, ADR 0059).
/// `vram.headroom_bytes`'s 1536 MiB hardcoded
/// default (finding 2026-10-09-vram-headroom-wddm-paging) still holds for
/// Flash-Next, which never needed to trade it down. Flash-Next also pins its
/// calibration-hottest 2 GiB of experts in the expert cache
/// (`vram.pinned_expert_bytes`, the owner's call, ADR 0055): measured on the
/// card (finding 2026-10-18-pinned-hot-set-measured-on-the-5090), that gives
/// back 2.149 GB of the host pool's RAM for 0.1-0.2% of one-lane and
/// 0.2-1.1% of three-lane decode (two sessions), greedy text unchanged; `4G`
/// would give back 4.296 GB for 4.3% and 5.9%. The field is
/// Flash-Next-only: the 27B passes it over.
///
/// `rtxpro6000` is the RTX PRO 6000 Blackwell series (Workstation, Max-Q,
/// Server Edition: 96 GB each, sm_120, the 5090's architecture). Derived
/// first (2026-10-17), then measured on a Server Edition under Linux; the
/// numbers are in the finding 2026-10-18-rtxpro6000-profile-measured. The
/// measurement kept the derived choices and added one. Both families run at
/// 1,048,576 tokens of context, the 27B's whole hq-e8-2b attention envelope
/// (the plan holds it; no request that long was run): `auto` rope scaling
/// makes both models' tables YaRN at 4 over the 262,144 trained positions
/// (ADR 0057, ADR 0059). The 27B leaves its headroom at the hardcoded
/// default, since nothing has to be traded for vision any more, and drafts
/// with DFlash2 at 7 tokens, which beat 4 in every cell measured but one,
/// and prefills 2,048 tokens a chunk, as on the 5090 (GitHub #341).
/// Flash-Next runs 8 decode lanes, where its aggregate peaks: its expert
/// cache holds nearly every expert, so a lane costs its state and its share
/// of the KV pool. It drafts with its MTP head, 3 tokens a lane, which the
/// 5090 leaves off because decode there is bound by the experts it copies
/// over PCIe (finding 2026-10-07-flash-next-mtp-speculation-is-pcie-bound);
/// here MTP pays at one to four active lanes and with simulated agents at
/// four and eight. The row budget stays the default 8, where every wider
/// one lost on this card (finding 2026-10-18-verify-rows-on-the-rtx-pro-6000),
/// so 3 drafts at one or two lanes, 1 at three or four, none from five on. Its
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
            "pinned_expert_bytes": "2G",
            "qwen38": { "headroom_bytes": "1200M" }
        },
        "model": {
            "qwen38": { "max_context": 917504, "prefill_chunk": 2048 },
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
            "qwen38": { "max_context": 1048576, "prefill_chunk": 2048 },
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

/// A **card class** (ADR 0063, ADR 0064, `CONTEXT.md`): the cards one
/// built-in profile serves, named by their memory, and the smallest a
/// catalog entry loads on. Ordered by size, so `entry > card` reads "the
/// entry needs a larger card than this one".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CardClass {
    Gb16,
    Gb32,
    Gb96,
}

/// The total memories each [`CardClass`] covers (spec 27b-16gb/04), as NVML
/// reports a card's total: below 20 GiB is the 16 GB class (a 16 GB card
/// reports ~15.9 GiB), 28-40 GiB the 32 GB one (the 5090 ~31.8 GiB), 88-104
/// GiB the 96 GB one (the RTX PRO 6000 ~95.6 GiB). Any other total is no
/// class: `--profile auto` refuses it, naming `--profile`.
pub const CARD_CLASS_16GB_BELOW_BYTES: u64 = 20 << 30;
pub const CARD_CLASS_32GB_BYTES: std::ops::RangeInclusive<u64> = (28 << 30)..=(40 << 30);
pub const CARD_CLASS_96GB_BYTES: std::ops::RangeInclusive<u64> = (88 << 30)..=(104 << 30);

impl CardClass {
    /// Every class, smallest first.
    pub const ALL: [CardClass; 3] = [Self::Gb16, Self::Gb32, Self::Gb96];

    /// The class a card whose total memory is `total_bytes` belongs to, by
    /// the boundaries above; `None` for a total between or past them.
    pub fn of_total(total_bytes: u64) -> Option<Self> {
        if total_bytes < CARD_CLASS_16GB_BELOW_BYTES {
            Some(Self::Gb16)
        } else if CARD_CLASS_32GB_BYTES.contains(&total_bytes) {
            Some(Self::Gb32)
        } else if CARD_CLASS_96GB_BYTES.contains(&total_bytes) {
            Some(Self::Gb96)
        } else {
            None
        }
    }

    /// The class's size in GB, as a catalog entry spells it (`card_class: 16`).
    pub fn gb(self) -> u32 {
        match self {
            Self::Gb16 => 16,
            Self::Gb32 => 32,
            Self::Gb96 => 96,
        }
    }

    /// The class a catalog entry's `card_class` names; `None` for any other
    /// number.
    pub fn from_gb(gb: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|class| class.gb() == gb)
    }

    /// The built-in profile `--profile auto` picks for a card of this class.
    pub fn profile(self) -> &'static str {
        match self {
            Self::Gb16 => "rtx50-16gb",
            Self::Gb32 => "rtx5090",
            Self::Gb96 => "rtxpro6000",
        }
    }
}

impl std::fmt::Display for CardClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} GB", self.gb())
    }
}

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
        assert_eq!(rtx5090.get(max_context, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "917504");
        assert_eq!(rtx5090.get(max_context, Some(ModelFamily::FlashNext)).unwrap().raw, "589824");

        let rope = field("model", "rope_scaling").unwrap();
        assert!(rtx5090.get(rope, None).is_none(), "unset: `auto` follows the 27B's own context");

        let vision = field("vision", "enabled").unwrap();
        assert_eq!(rtx5090.get(vision, None).unwrap().raw, "true");

        let backend = field("spec", "backend").unwrap();
        assert_eq!(rtx5090.get(backend, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "dflash2");
        assert!(rtx5090.get(backend, Some(ModelFamily::FlashNext)).is_none(), "Flash-Next keeps speculation off here");

        let lanes = field("spec", "decode_lanes").unwrap();
        assert_eq!(rtx5090.get(lanes, None).unwrap().raw, "3", "Flash-Next-only; the 27B passes it over");

        // Flash-Next pins its calibration-hottest 2 GiB of experts (the
        // owner's call, ADR 0055): the field's own default stays 0, and the
        // field is Flash-Next-only, so the 27B passes it over.
        let pinned = field("vram", "pinned_expert_bytes").unwrap();
        assert_eq!((pinned.default)(), serde_json::Value::from(0));
        assert_eq!(rtx5090.get(pinned, None).unwrap().raw, "2G");
        assert!(profiles["rtxpro6000"].get(pinned, None).is_none(), "a device-only load keeps no expert in RAM");
    }

    /// The 27B started under `rtx5090` takes the profile's Flash-Next-only
    /// pin as a default it passes over, and Flash-Next takes 2 GiB.
    #[test]
    fn under_the_rtx5090_profile_only_flash_next_pins_experts() {
        use super::super::source::{resolve_settings, Fit, Sources};
        use ignis_core::compute::ModelFamily;

        let sources = Sources { profile: built_in()[DEFAULT_PROFILE].clone(), ..Sources::default() };
        let qwen = resolve_settings(&sources, Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap();
        assert_eq!(qwen.settings.vram.pinned_expert_bytes, 0);
        assert!(qwen.dropped.is_empty(), "passed over, not dropped: {:?}", qwen.dropped);
        let flash = resolve_settings(&sources, Some(ModelFamily::FlashNext), Fit::Start).unwrap();
        assert_eq!(flash.settings.vram.pinned_expert_bytes, 2 << 30);
    }

    /// GitHub #341: the 27B prefills 2,048 tokens a chunk under both built-in
    /// profiles, free there because vision's encoder sizes the arena;
    /// Flash-Next and a start without a profile keep the hardcoded width.
    #[test]
    fn under_the_built_in_profiles_the_27b_prefills_2048_tokens_a_chunk() {
        use super::super::source::{resolve_settings, Fit, Sources};
        use crate::config::DEFAULT_PREFILL_CHUNK;
        use ignis_core::compute::ModelFamily;

        let sources = Sources { profile: built_in()[DEFAULT_PROFILE].clone(), ..Sources::default() };
        let qwen = resolve_settings(&sources, Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap();
        assert_eq!(qwen.settings.model.prefill_chunk, 2048);
        let flash = resolve_settings(&sources, Some(ModelFamily::FlashNext), Fit::Start).unwrap();
        assert_eq!(flash.settings.model.prefill_chunk, DEFAULT_PREFILL_CHUNK);
        let bare = resolve_settings(&Sources::default(), Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap();
        assert_eq!(bare.settings.model.prefill_chunk, DEFAULT_PREFILL_CHUNK);
        let pro = Sources { profile: built_in()["rtxpro6000"].clone(), ..Sources::default() };
        assert_eq!(resolve_settings(&pro, Some(ModelFamily::Qwen38_27b), Fit::Start).unwrap().settings.model.prefill_chunk, 2048);
    }

    /// `auto` rope scaling under each profile, as a load of each model
    /// resolves it: each model's factor is its own context over the trained
    /// position envelope (ADR 0057), Flash-Next's too (ADR 0059).
    #[test]
    fn the_profiles_rope_table_follows_each_models_context() {
        use ignis_core::compute::ModelFamily;

        let table = |profile: &str, family| {
            let args = ["--profile".to_owned(), profile.to_owned()];
            let crate::config::ConfigOutcome::Config(config) = crate::config::resolve(&args, |_| None).unwrap() else {
                panic!("a config")
            };
            let fitted = config.for_family(family).unwrap();
            assert!(fitted.settings().model.rope_scaling.is_auto());
            fitted.rope_scaling
        };
        assert_eq!(table(DEFAULT_PROFILE, ModelFamily::Qwen38_27b).factor(), 3.5, "917,504 / 262,144");
        assert_eq!(table(DEFAULT_PROFILE, ModelFamily::FlashNext).factor(), 2.25, "589,824 / 262,144");
        assert_eq!(table("rtxpro6000", ModelFamily::Qwen38_27b).factor(), 4.0);
        assert_eq!(table("rtxpro6000", ModelFamily::FlashNext).factor(), 4.0);
    }

    /// The RTX PRO 6000 Blackwell series (Workstation, Max-Q, Server Edition:
    /// 96 GB each, sm_120), measured on a Server Edition (2026-10-18).
    #[test]
    fn the_rtxpro6000_profile_gives_both_models_their_whole_context_and_flash_next_every_lane() {
        use ignis_core::compute::ModelFamily;

        let profiles = built_in();
        let pro = &profiles["rtxpro6000"];
        let max_context = field("model", "max_context").unwrap();
        // The 27B's whole hq-e8-2b envelope, YaRN at 4 under `auto`; so is
        // Flash-Next's (ADR 0057, ADR 0059).
        assert_eq!(pro.get(max_context, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "1048576");
        assert_eq!(pro.get(max_context, Some(ModelFamily::FlashNext)).unwrap().raw, "1048576");
        let rope = field("model", "rope_scaling").unwrap();
        assert!(pro.get(rope, None).is_none(), "unset: `auto` makes the 27B's 1,048,576 tokens YaRN at 4");

        let lanes = field("spec", "decode_lanes").unwrap();
        assert_eq!(pro.get(lanes, None).unwrap().raw, "8", "every lane: on 96 GB a lane costs its state");

        // The 5090's choices the card's memory does not change.
        let vision = field("vision", "enabled").unwrap();
        assert_eq!(pro.get(vision, None).unwrap().raw, "true");
        let backend = field("spec", "backend").unwrap();
        assert_eq!(pro.get(backend, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "dflash2");
        let draft = field("spec", "draft_tokens").unwrap();
        assert_eq!(pro.get(draft, Some(ModelFamily::Qwen38_27b)).unwrap().raw, "7");

        // Flash-Next drafts with its MTP head, 3 tokens a lane, cut by the
        // default row budget's 8 rows (GitHub #329 item 5: measured on this
        // card, every wider budget loses to 8): on 96 GB no expert crosses
        // PCIe, the cost that made MTP lose on the 5090.
        assert_eq!(pro.get(backend, Some(ModelFamily::FlashNext)).unwrap().raw, "mtp");
        assert_eq!(pro.get(draft, Some(ModelFamily::FlashNext)).unwrap().raw, "3");
        let rows = field("spec", "draft_rows").unwrap();
        assert!(pro.get(rows, None).is_none(), "unset: the default row budget, 8 of 32");

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

    /// Spec 27b-16gb/04: the class boundaries by NVML's total, at the totals
    /// the cards report and either side of every boundary.
    #[test]
    fn a_cards_total_memory_names_its_class() {
        const GIB: f64 = (1u64 << 30) as f64;
        let class = |gib: f64| CardClass::of_total((gib * GIB) as u64);
        assert_eq!(class(15.9), Some(CardClass::Gb16), "a 16 GB card");
        assert_eq!(class(11.9), Some(CardClass::Gb16), "a 12 GB card is the 16 GB class's");
        assert_eq!(class(31.8), Some(CardClass::Gb32), "the 5090");
        assert_eq!(class(95.5), Some(CardClass::Gb96), "the RTX PRO 6000");
        assert_eq!(CardClass::of_total(CARD_CLASS_16GB_BELOW_BYTES - 1), Some(CardClass::Gb16));
        assert_eq!(CardClass::of_total(CARD_CLASS_16GB_BELOW_BYTES), None, "20 GiB: no class");
        assert_eq!(class(24.0), None, "a 24 GB card");
        assert_eq!(CardClass::of_total(*CARD_CLASS_32GB_BYTES.start() - 1), None);
        assert_eq!(CardClass::of_total(*CARD_CLASS_32GB_BYTES.start()), Some(CardClass::Gb32));
        assert_eq!(CardClass::of_total(*CARD_CLASS_32GB_BYTES.end()), Some(CardClass::Gb32));
        assert_eq!(CardClass::of_total(*CARD_CLASS_32GB_BYTES.end() + 1), None);
        assert_eq!(class(48.0), None, "a 48 GB card");
        assert_eq!(CardClass::of_total(*CARD_CLASS_96GB_BYTES.start()), Some(CardClass::Gb96));
        assert_eq!(CardClass::of_total(*CARD_CLASS_96GB_BYTES.end() + 1), None, "past 104 GiB");
        assert_eq!(class(141.0), None, "an H200");
    }

    #[test]
    fn a_card_class_is_named_by_its_gb_and_orders_by_size() {
        for class in CardClass::ALL {
            assert_eq!(CardClass::from_gb(class.gb()), Some(class));
        }
        assert_eq!(CardClass::from_gb(24), None);
        assert!(CardClass::Gb16 < CardClass::Gb32 && CardClass::Gb32 < CardClass::Gb96);
        assert_eq!(CardClass::Gb32.to_string(), "32 GB");
        assert_eq!(CardClass::Gb32.profile(), "rtx5090", "the 5090's class keeps its profile");
        assert_eq!(CardClass::Gb96.profile(), "rtxpro6000");
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
