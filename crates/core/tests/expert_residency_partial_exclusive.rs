//! A partially exclusive expert pool, on the CPU policy model (GitHub #329
//! item 12; `docs/findings/2026-10-18-partial-exclusive-expert-pool.md`).
//!
//! Experts **pinned** in VRAM for a load's whole life need no host copy, so
//! the pinned host pool saves their bytes; the price is their slots, which
//! the LRU no longer has. The policy model has no pin, and it must not grow
//! one for a study (the GPU must match it exactly, spec flash-next/03
//! acceptance 4), so [`Pinned`] wraps it: each class's capacity shrinks by
//! its pinned slots, the warm start skips the pinned experts, and a step
//! hands the model its selection and its lookahead without them (a resident
//! candidate is skipped before the budget anyway). A pinned use is a hit.
//!
//! The synthetic tests run everywhere. The study is `#[ignore]` and
//! machine-local: it reads the converter's traces from
//! `IGNIS_FLASH_NEXT_TRACES` (the artifact directory; layout.md §9, §12),
//! as `expert_residency_study_replay.rs` does, whose loader it copies.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use ignis_core::residency::{
    DEFAULT_PREFETCH_WIDTH, ExpertCatalog, ExpertCacheRequest, ExpertTraffic, KBits, KClass, LayerStep,
    PolicyConfig, Projection, ProjectionId, ResidencyModel, StepError, StepOutcome, default_prefetch_budget,
    min_slots_per_class, plan_expert_cache, prefill_staging_ring_bytes, residency_table_bytes,
    warm_start_order,
};

const TOP_K: usize = 10;
const LOOKAHEAD: usize = 20;

/// Pinned experts (both projections), and the model of the LRU around them.
#[derive(Clone)]
struct Pinned {
    model: ResidencyModel,
    /// Per (layer, expert), layer-major.
    pinned: Vec<bool>,
    experts: usize,
    width: usize,
}

impl Pinned {
    /// `capacity` is the whole cache's slots per class; `pin` the experts
    /// to hold, whose slots it takes. The rest of `hottest_first` warm-starts
    /// the LRU.
    fn new(
        catalog: &ExpertCatalog,
        capacity: [u32; KClass::COUNT],
        config: PolicyConfig,
        pin: &[(u16, u16)],
        hottest_first: &[ProjectionId],
    ) -> Self {
        let experts = usize::from(catalog.experts());
        let mut pinned = vec![false; usize::from(catalog.layers()) * experts];
        let mut lru = capacity;
        for &(layer, expert) in pin {
            pinned[usize::from(layer) * experts + usize::from(expert)] = true;
            for projection in Projection::ALL {
                let class = catalog.class_of(ProjectionId::new(layer, expert, projection)).index();
                lru[class] -= 1;
            }
        }
        let mut model = ResidencyModel::new(catalog.clone(), PolicyConfig { capacity: lru, ..config });
        let rest: Vec<ProjectionId> = hottest_first
            .iter()
            .copied()
            .filter(|p| !pinned[usize::from(p.layer) * experts + usize::from(p.expert)])
            .collect();
        model.warm_start(&rest);
        Self {
            model,
            pinned,
            experts,
            width: config.prefetch_width,
        }
    }

    fn is_pinned(&self, layer: usize, expert: u16) -> bool {
        self.pinned[layer * self.experts + usize::from(expert)]
    }

    /// One decode step: the model's outcome over what is not pinned, and the
    /// selected experts the pin served.
    fn decode(&mut self, layer: u16, selected: &[u16], lookahead: &[Vec<u16>]) -> Result<(StepOutcome, Vec<u16>), StepError> {
        let l = usize::from(layer);
        let (pinned, rest): (Vec<u16>, Vec<u16>) = selected.iter().partition(|&&e| self.is_pinned(l, e));
        // The device takes the first `width` of each ranking, then skips what
        // is resident: truncate first, so a pinned expert does not let the
        // walk reach further down the rank. The last layer has no lookahead.
        let last = l + 1 == usize::from(self.model.catalog().layers());
        let ahead: Vec<Vec<u16>> = if last {
            Vec::new()
        } else {
            lookahead
                .iter()
                .map(|lane| lane.iter().take(self.width).copied().filter(|&e| !self.is_pinned(l + 1, e)).collect())
                .collect()
        };
        let refs: Vec<&[u16]> = ahead.iter().map(|v| v.as_slice()).collect();
        let out = self.model.step(&LayerStep::decode(layer, &rest).lookahead(&refs))?;
        Ok((out, pinned))
    }
}

/// Whole experts in `hottest_first` order (both projections at once), as
/// many as reach `bytes`, keeping every class at least `keep` LRU slots;
/// an expert a full class cannot take is skipped, as the warm start skips.
fn pin_prefix(
    catalog: &ExpertCatalog,
    capacity: [u32; KClass::COUNT],
    keep: u32,
    hottest_first: &[ProjectionId],
    bytes: u64,
) -> Vec<(u16, u16)> {
    let mut room: [u32; KClass::COUNT] = std::array::from_fn(|i| capacity[i].saturating_sub(keep));
    let mut out = Vec::new();
    let mut taken = 0u64;
    let mut seen = BTreeSet::new();
    for p in hottest_first {
        if taken >= bytes {
            break;
        }
        if !seen.insert((p.layer, p.expert)) {
            continue;
        }
        let ids = Projection::ALL.map(|projection| ProjectionId::new(p.layer, p.expert, projection));
        let classes = ids.map(|id| catalog.class_of(id).index());
        let mut need = [0u32; KClass::COUNT];
        for c in classes {
            need[c] += 1;
        }
        if (0..KClass::COUNT).any(|c| need[c] > room[c]) {
            continue;
        }
        for c in 0..KClass::COUNT {
            room[c] -= need[c];
        }
        taken += ids.iter().map(|&id| catalog.bytes(id)).sum::<u64>();
        out.push((p.layer, p.expert));
    }
    out
}

fn pinned_bytes(catalog: &ExpertCatalog, pin: &[(u16, u16)]) -> u64 {
    pin.iter()
        .flat_map(|&(l, e)| Projection::ALL.map(|projection| catalog.bytes(ProjectionId::new(l, e, projection))))
        .sum()
}

// ---------------------------------------------------------------------------
// Synthetic: the wrapper, without traces.

/// Two K classes per projection over 4 layers x 16 experts, slot bytes by class.
fn synthetic() -> (ExpertCatalog, ExpertTraffic) {
    let (layers, experts) = (4u16, 16u16);
    let map = (0..layers as usize * experts as usize)
        .map(|i| if i % 3 == 0 { (KBits::K2, KBits::K3) } else { (KBits::K3, KBits::K2) })
        .collect();
    let slot_bytes = std::array::from_fn(|i| 1000 + 100 * i as u64);
    let catalog = ExpertCatalog::new(layers, experts, map, slot_bytes).unwrap();
    let counts = (0..layers as u64 * experts as u64).map(|i| (i * 7919) % 97 + 1).collect();
    let traffic = ExpertTraffic::new(&catalog, counts).unwrap();
    (catalog, traffic)
}

/// A deterministic pseudo-random routing: `rounds` x layers of (top 3, ranked 6).
fn synthetic_routing(rounds: usize, layers: u16, experts: u16) -> Vec<(u16, Vec<u16>, Vec<u16>)> {
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut out = Vec::new();
    for _ in 0..rounds {
        for layer in 0..layers {
            // Skewed: low expert ids are hot.
            let mut pick = |n: usize| {
                let mut v = Vec::new();
                while v.len() < n {
                    let e = ((next() % u64::from(experts)) * (next() % u64::from(experts)) / u64::from(experts)) as u16;
                    if !v.contains(&e) {
                        v.push(e);
                    }
                }
                v
            };
            out.push((layer, pick(3), pick(6)));
        }
    }
    out
}

fn synthetic_config(capacity: [u32; KClass::COUNT]) -> PolicyConfig {
    PolicyConfig {
        capacity,
        prefetch_width: 4,
        prefill_prefetch_width: 4,
        prefetch_budget: Some(default_prefetch_budget()),
    }
}

/// With nothing pinned the wrapper is the model: the same outcome, step for
/// step.
#[test]
fn an_empty_pin_is_the_policy_model_itself() {
    let (catalog, traffic) = synthetic();
    let capacity = catalog.class_counts().map(|n| if n > 0 { 6 } else { 0 });
    let order = warm_start_order(&traffic);
    let mut plain = ResidencyModel::new(catalog.clone(), synthetic_config(capacity));
    plain.warm_start(&order);
    let mut wrapped = Pinned::new(&catalog, capacity, synthetic_config(capacity), &[], &order);
    for (layer, selected, ahead) in synthetic_routing(200, catalog.layers(), catalog.experts()) {
        let lanes = [ahead.clone()];
        let refs: Vec<&[u16]> = lanes.iter().map(|v| v.as_slice()).collect();
        let want = plain.step(&LayerStep::decode(layer, &selected).lookahead(&refs)).unwrap();
        let (got, served) = wrapped.decode(layer, &selected, &lanes).unwrap();
        assert_eq!(got, want);
        assert!(served.is_empty());
    }
    assert_eq!(wrapped.model.counters(), plain.counters());
}

/// Pinned experts take their slots from the LRU, are never copied or
/// evicted, and every use of one is served.
#[test]
fn a_pinned_expert_is_never_copied_and_its_slots_leave_the_lru() {
    let (catalog, traffic) = synthetic();
    // Only the populated classes have slots.
    let capacity = catalog.class_counts().map(|n| if n > 0 { 8 } else { 0 });
    let order = warm_start_order(&traffic);
    let pin = pin_prefix(&catalog, capacity, 4, &order, 6_000);
    assert!(!pin.is_empty());
    // Hottest first, whole experts, each class keeping its 4 LRU slots.
    assert_eq!(pin[0], (order[0].layer, order[0].expert));
    assert!(pinned_bytes(&catalog, &pin) >= 6_000);
    let mut model = Pinned::new(&catalog, capacity, synthetic_config(capacity), &pin, &order);
    let mut lru = capacity;
    for &(l, e) in &pin {
        for projection in Projection::ALL {
            lru[catalog.class_of(ProjectionId::new(l, e, projection)).index()] -= 1;
        }
    }
    assert!(lru.iter().zip(capacity).all(|(&slots, cap)| cap == 0 || slots >= 4), "{lru:?}");
    assert_eq!(model.model.config().capacity, lru);
    assert_eq!(model.model.occupancy(), lru, "the warm start fills what is left");
    let mut served = 0;
    for (layer, selected, ahead) in synthetic_routing(200, catalog.layers(), catalog.experts()) {
        let (out, pinned) = model.decode(layer, &selected, &[ahead]).unwrap();
        served += pinned.len();
        for p in pinned {
            assert!(selected.contains(&p) && pin.contains(&(layer, p)));
        }
        let touched = out
            .misses
            .iter()
            .map(|(p, _)| *p)
            .chain(out.evictions.iter().copied())
            .chain(out.prefetches.iter().map(|(p, _)| *p));
        for p in touched {
            assert!(!pin.contains(&(p.layer, p.expert)), "{p:?} is pinned");
        }
    }
    assert!(served > 0, "the routing selects pinned experts");
}

// ---------------------------------------------------------------------------
// The study over the converter's traces (machine-local, `--ignored`).

fn root() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("IGNIS_FLASH_NEXT_TRACES")?);
    (path.join("work/converter.json").is_file() && path.join("traces").is_dir()).then_some(path)
}

struct Chunk {
    id: u64,
    layers: usize,
    experts: Vec<i16>,
    lookahead: Vec<i16>,
    tokens: usize,
}

impl Chunk {
    fn selected(&self, token: usize, layer: usize) -> Vec<u16> {
        let at = (token * self.layers + layer) * TOP_K;
        self.experts[at..at + TOP_K].iter().map(|&e| e as u16).collect()
    }

    fn lookahead(&self, token: usize, layer: usize, width: usize) -> Vec<u16> {
        let at = (token * self.layers + layer) * LOOKAHEAD;
        self.lookahead[at..at + width.min(LOOKAHEAD)]
            .iter()
            .filter(|&&e| e >= 0)
            .map(|&e| e as u16)
            .collect()
    }
}

struct Study {
    catalog: ExpertCatalog,
    traffic: ExpertTraffic,
    chunks: Vec<Chunk>,
}

fn read_i16(path: &Path) -> Vec<i16> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect()
}

fn k_of(k2: &serde_json::Value) -> KBits {
    let k2 = k2.as_u64().expect("k2 is an integer");
    KBits::from_half_bits(k2 as u32).unwrap_or_else(|| panic!("k2 {k2} is not a K class"))
}

/// The replay test's loader: the test set's chunks, every domain, in index
/// order.
fn load(root: &Path) -> Study {
    let text = std::fs::read_to_string(root.join("work/converter.json")).expect("converter.json");
    let json: serde_json::Value = serde_json::from_str(&text).expect("json");
    let k_layers = json["k_map"]["layers"].as_array().expect("k_map");
    let mut map = Vec::new();
    for layer in k_layers {
        let gu = layer["gu"].as_array().expect("gu");
        let dn = layer["dn"].as_array().expect("dn");
        map.extend(gu.iter().zip(dn).map(|(g, d)| (k_of(g), k_of(d))));
    }
    let layers = k_layers.len();
    let experts = map.len() / layers;
    let mut slot_bytes = [0u64; KClass::COUNT];
    for class in json["k_classes"].as_array().expect("k_classes") {
        let name = class["class"].as_str().expect("class");
        let (proj, k) = name.split_once('-').expect("gu-K or dn-K");
        let projection = if proj == "gu" { Projection::GateUp } else { Projection::Down };
        let k: f64 = k.parse().expect("K");
        let k = KBits::from_half_bits((2.0 * k) as u32).expect("a K class");
        slot_bytes[KClass::new(projection, k).index()] = class["record_bytes"].as_u64().expect("record_bytes");
    }
    let catalog = ExpertCatalog::new(layers as u16, experts as u16, map, slot_bytes).expect("catalog");
    let counts = json["expert_traffic"]["layers"]
        .as_array()
        .expect("expert_traffic")
        .iter()
        .flat_map(|l| l["counts"].as_array().expect("counts").iter().map(|c| c.as_u64().expect("count")))
        .collect();
    let traffic = ExpertTraffic::new(&catalog, counts).expect("expert_traffic covers the K map");
    let mut chunks = Vec::new();
    for entry in std::fs::read_dir(root.join("traces")).expect("traces") {
        let dir = entry.expect("entry").path();
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).expect("manifest")).expect("manifest json");
        if manifest["set"].as_str() != Some("test") {
            continue;
        }
        let all_experts = read_i16(&dir.join("experts.i16"));
        let all_lookahead = read_i16(&dir.join("lookahead.i16"));
        let ids = manifest["chunks"].as_array().expect("chunks");
        let valid = manifest["valid"].as_array().expect("valid");
        let mut start = 0usize;
        for (id, valid) in ids.iter().zip(valid) {
            let tokens = valid.as_u64().expect("valid") as usize;
            let span = |per: usize, all: &[i16]| all[start * layers * per..(start + tokens) * layers * per].to_vec();
            chunks.push(Chunk {
                id: id.as_u64().expect("chunk index"),
                layers,
                experts: span(TOP_K, &all_experts),
                lookahead: span(LOOKAHEAD, &all_lookahead),
                tokens,
            });
            start += tokens;
        }
    }
    chunks.sort_by_key(|c| c.id);
    Study { catalog, traffic, chunks }
}

/// The served plan's slots for a `cache_bytes` expert cache at `lanes`,
/// as `flash_next::build_residency` asks for them.
fn capacity(study: &Study, cache_bytes: u64, lanes: u32) -> ([u32; KClass::COUNT], u64) {
    let catalog = &study.catalog;
    let ring = prefill_staging_ring_bytes(catalog);
    let tables = residency_table_bytes(u64::from(catalog.layers()), u64::from(catalog.experts()), 8192, DEFAULT_PREFETCH_WIDTH as u64);
    let plan = plan_expert_cache(&ExpertCacheRequest {
        budget_bytes: cache_bytes + ring + tables,
        planned_bytes: 0,
        staging_ring_bytes: ring,
        table_bytes: tables,
        floor_bytes: 0,
        allow_below_floor: false,
        catalog,
        traffic: &study.traffic,
        min_slots: min_slots_per_class(lanes, TOP_K as u32, DEFAULT_PREFETCH_WIDTH as u32),
    })
    .expect("plan");
    println!("plan at {cache_bytes} B, {lanes} lane(s): {plan}");
    (plan.capacity(), plan.pooled_bytes())
}

fn served_config(capacity: [u32; KClass::COUNT]) -> PolicyConfig {
    PolicyConfig {
        capacity,
        prefetch_width: DEFAULT_PREFETCH_WIDTH,
        prefill_prefetch_width: TOP_K,
        prefetch_budget: Some(default_prefetch_budget()),
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct Tally {
    uses: u64,
    hits: u64,
    pinned_hits: u64,
    demand_bytes: u64,
    prefetch_bytes: u64,
    rounds: u64,
    tokens: u64,
}

/// What a run saw of each warm-started projection: the decode step of its
/// first eviction, if any.
struct Survival {
    first_eviction: std::collections::HashMap<ProjectionId, u64>,
    warm: Vec<ProjectionId>,
    steps: u64,
}

/// Decode `lanes` for their common length, one step per layer per round.
/// Records first evictions into `survival` when given.
fn decode(
    model: &mut Pinned,
    lanes: &[(&Chunk, std::ops::Range<usize>)],
    measure: bool,
    tally: &mut Tally,
    mut survival: Option<&mut Survival>,
) {
    let rounds = lanes.iter().map(|(_, r)| r.len()).min().unwrap();
    let layers = usize::from(model.model.catalog().layers());
    let width = model.width;
    for round in 0..rounds {
        for layer in 0..layers {
            let picks: Vec<Vec<u16>> = lanes.iter().map(|(c, r)| c.selected(r.start + round, layer)).collect();
            let ahead: Vec<Vec<u16>> = if layer + 1 < layers {
                lanes.iter().map(|(c, r)| c.lookahead(r.start + round, layer, width)).collect()
            } else {
                Vec::new()
            };
            let union: Vec<u16> = picks.iter().flatten().copied().collect::<BTreeSet<u16>>().into_iter().collect();
            let (out, _) = model.decode(layer as u16, &union, &ahead).expect("step");
            if let Some(s) = survival.as_deref_mut() {
                s.steps += 1;
                for &p in &out.evictions {
                    s.first_eviction.entry(p).or_insert(s.steps);
                }
            }
            if !measure {
                continue;
            }
            let hit: BTreeSet<_> = out.hits.iter().map(|p| (p.expert, p.projection)).collect();
            let catalog = model.model.catalog();
            tally.demand_bytes += out.misses.iter().map(|(p, _)| catalog.bytes(*p)).sum::<u64>();
            tally.prefetch_bytes += out.prefetches.iter().map(|(p, _)| catalog.bytes(*p)).sum::<u64>();
            for pick in &picks {
                for e in pick.iter().copied().collect::<BTreeSet<u16>>() {
                    for projection in Projection::ALL {
                        tally.uses += 1;
                        if model.is_pinned(layer, e) {
                            tally.hits += 1;
                            tally.pinned_hits += 1;
                        } else {
                            tally.hits += u64::from(hit.contains(&(e, projection)));
                        }
                    }
                }
            }
        }
        if measure {
            tally.rounds += 1;
            tally.tokens += lanes.len() as u64;
        }
    }
}

/// The headroom finding's protocol (2026-10-08 §2): per chunk from the warm
/// start, 256 tokens to settle, 512 measured; three lanes run consecutive
/// chunks together.
fn per_chunk(study: &Study, warm: &Pinned, lanes: usize) -> Tally {
    let mut total = Tally::default();
    let groups: Vec<&[Chunk]> = study.chunks.chunks(lanes).filter(|g| g.len() == lanes).collect();
    for group in groups {
        if group.iter().any(|c| c.tokens < 768) {
            continue;
        }
        let mut model = warm.clone();
        let settle: Vec<_> = group.iter().map(|c| (c, 0..256)).collect();
        decode(&mut model, &settle, false, &mut total, None);
        let measured: Vec<_> = group.iter().map(|c| (c, 256..768)).collect();
        decode(&mut model, &measured, true, &mut total, None);
    }
    total
}

/// One serving life: a single model over every test chunk in id order,
/// every token (three lanes: consecutive chunks together, for the shortest's
/// length). Returns the tally and, per warm-started projection, its first
/// eviction.
fn continuous(study: &Study, warm: &Pinned, lanes: usize) -> (Tally, Survival) {
    let mut model = warm.clone();
    let mut survival = Survival {
        first_eviction: Default::default(),
        warm: Vec::new(),
        steps: 0,
    };
    let catalog = &study.catalog;
    for l in 0..catalog.layers() {
        for e in 0..catalog.experts() {
            for projection in Projection::ALL {
                let id = ProjectionId::new(l, e, projection);
                if model.model.is_resident(id) {
                    survival.warm.push(id);
                }
            }
        }
    }
    let mut total = Tally::default();
    for group in study.chunks.chunks(lanes).filter(|g| g.len() == lanes) {
        let len = group.iter().map(|c| c.tokens).min().unwrap();
        let lanes: Vec<_> = group.iter().map(|c| (c, 0..len)).collect();
        decode(&mut model, &lanes, true, &mut total, Some(&mut survival));
    }
    (total, survival)
}

fn row(label: &str, pinned: u64, t: &Tally, base: Option<&Tally>, stall_share: f64, round_ms: f64) -> String {
    let per_round = |b: u64| b as f64 / t.rounds as f64;
    let demand_ms = per_round(t.demand_bytes) / 12.0e9 * 1e3;
    let mut s = format!(
        "{label:28} pinned {:5.2} GB | hit {:6.2}% (pinned {:5.2}% of uses) | demand {:6.1} + prefetch {:6.1} MB/round ({:5.1} + {:5.1} MB/token) | demand at 12 GB/s {:5.2} ms/round",
        pinned as f64 / 1e9,
        t.hits as f64 / t.uses as f64 * 100.0,
        t.pinned_hits as f64 / t.uses as f64 * 100.0,
        per_round(t.demand_bytes) / 1e6,
        per_round(t.prefetch_bytes) / 1e6,
        t.demand_bytes as f64 / t.tokens as f64 / 1e6,
        t.prefetch_bytes as f64 / t.tokens as f64 / 1e6,
        demand_ms,
    );
    if let Some(b) = base {
        let base_demand = b.demand_bytes as f64 / b.rounds as f64;
        let ratio = per_round(t.demand_bytes) / base_demand;
        let delta_ms = (per_round(t.demand_bytes) - base_demand) / 12.0e9 * 1e3;
        // Ratio method (headroom finding): served stall x replayed demand ratio.
        let loss_ratio = stall_share * (ratio - 1.0);
        s += &format!(
            " | demand x{ratio:.3}, {delta_ms:+.2} ms/round at 12 GB/s ({:+.1}% of a {round_ms} ms round); served-stall ratio {:+.1}% decode time",
            delta_ms / round_ms * 100.0,
            loss_ratio * 100.0
        );
    }
    s
}

/// The rtx5090 profile's expert cache: the median of the seven served plans
/// at its shape (589,824 tokens, 3 lanes) in the owner's server logs,
/// 2026-10-12/13 (18.03-18.76 GB).
const RTX5090_CACHE_BYTES: u64 = 18_443_439_440;
/// The headroom finding's replay cache (2026-10-08 §2): the anchor.
const ANCHOR_CACHE_BYTES: u64 = 15_615_494_736;
/// Served demand stall's share of a round, 2026-10-08 §1: 1.84 of 9.85 ms
/// at one lane, 15.0 of 27.5 at three.
const STALL: [(f64, f64); 2] = [(1.84 / 9.85, 9.85), (15.0 / 27.5, 27.5)];

#[test]
#[ignore = "machine-local: IGNIS_FLASH_NEXT_TRACES, ~minutes of CPU"]
fn partial_exclusive_pool_study() {
    let Some(root) = root() else {
        eprintln!("SKIP: IGNIS_FLASH_NEXT_TRACES names no converter traces");
        return;
    };
    let study = load(&root);
    let catalog = &study.catalog;
    let order = warm_start_order(&study.traffic);
    let tokens: usize = study.chunks.iter().map(|c| c.tokens).sum();
    println!(
        "traces: {} ({} test chunks, {tokens} tokens), catalog {} B over {} layers x {} experts",
        root.display(),
        study.chunks.len(),
        catalog.total_bytes(),
        catalog.layers(),
        catalog.experts()
    );
    let min = min_slots_per_class(3, TOP_K as u32, DEFAULT_PREFETCH_WIDTH as u32);

    // The anchor: today's policy, the headroom finding's cache and protocol.
    let (anchor_cap, _) = capacity(&study, ANCHOR_CACHE_BYTES, 3);
    let anchor = Pinned::new(catalog, anchor_cap, served_config(anchor_cap), &[], &order);
    let (cap, pooled) = capacity(&study, RTX5090_CACHE_BYTES, 3);
    println!("rtx5090 pools hold {pooled} B of {} B ({:.1}%)", catalog.total_bytes(), pooled as f64 / catalog.total_bytes() as f64 * 100.0);

    // Pin sizes, GB of host RAM the pin saves (calibration order).
    let sizes_gb = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0];
    let pins: Vec<Vec<(u16, u16)>> = sizes_gb
        .iter()
        .map(|gb| pin_prefix(catalog, cap, min, &order, (gb * 1e9) as u64))
        .collect();

    // Every run on its own thread.
    let results = std::thread::scope(|s| {
        let anchor_runs: Vec<_> = [1usize, 3]
            .iter()
            .map(|&lanes| {
                let anchor = &anchor;
                let study = &study;
                s.spawn(move || per_chunk(study, anchor, lanes))
            })
            .collect();
        let mut sweep = Vec::new();
        for pin in &pins {
            for lanes in [1usize, 3] {
                let study = &study;
                let order = &order;
                sweep.push(s.spawn(move || {
                    let warm = Pinned::new(&study.catalog, cap, served_config(cap), pin, order);
                    let chunked = per_chunk(study, &warm, lanes);
                    let (life, survival) = continuous(study, &warm, lanes);
                    (lanes, chunked, life, survival)
                }));
            }
        }
        (
            anchor_runs.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>(),
            sweep.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>(),
        )
    });
    let (anchors, sweep) = results;
    println!("\n== anchor: 15.62 GB, per chunk 256 settle + 512 measured (headroom finding: 94.18% 1 lane, 90.55% 3 lanes)");
    for (lanes, t) in [1, 3].iter().zip(&anchors) {
        println!("{}", row(&format!("{lanes} lane(s)"), 0, t, None, 0.0, 0.0));
    }

    // Q1: the stable set of today's policy at the rtx5090 cache (pin 0).
    for (lanes, _, life, survival) in sweep.iter().take(2) {
        let steps = survival.steps;
        println!("\n== stable set, today's policy, rtx5090 cache, {lanes} lane(s), one serving life of {steps} layer steps ({} rounds)", life.rounds);
        let bytes_of = |ids: &mut dyn Iterator<Item = &ProjectionId>| ids.map(|&p| catalog.bytes(p)).sum::<u64>();
        let warm_bytes = bytes_of(&mut survival.warm.iter());
        println!("warm-started: {} projections, {:.2} GB", survival.warm.len(), warm_bytes as f64 / 1e9);
        for q in [0.1, 0.25, 0.5, 0.75, 1.0] {
            let at = (steps as f64 * q) as u64;
            let alive: Vec<&ProjectionId> = survival
                .warm
                .iter()
                .filter(|p| survival.first_eviction.get(p).is_none_or(|&s| s > at))
                .collect();
            println!(
                "  never evicted through {:3.0}% of the life: {:6} projections, {:6.2} GB",
                q * 100.0,
                alive.len(),
                bytes_of(&mut alive.iter().copied()) as f64 / 1e9
            );
        }
        let stable: BTreeSet<ProjectionId> = survival
            .warm
            .iter()
            .copied()
            .filter(|p| !survival.first_eviction.contains_key(p))
            .collect();
        let mut per_class = [0u64; KClass::COUNT];
        for &p in &stable {
            per_class[catalog.class_of(p).index()] += catalog.bytes(p);
        }
        println!(
            "  stable set by class (GB): {}",
            KClass::ALL
                .iter()
                .map(|c| format!("{} {:.2}", c.as_str(), per_class[c.index()] as f64 / 1e9))
                .collect::<Vec<_>>()
                .join(", ")
        );
        // How much of a calibration pin of each size the trace never evicted.
        for (gb, pin) in sizes_gb.iter().zip(&pins).skip(1) {
            let ids: Vec<ProjectionId> = pin
                .iter()
                .flat_map(|&(l, e)| Projection::ALL.map(|projection| ProjectionId::new(l, e, projection)))
                .collect();
            let kept = ids.iter().filter(|p| stable.contains(p)).count();
            let kept_bytes: u64 = ids.iter().filter(|p| stable.contains(p)).map(|&p| catalog.bytes(p)).sum();
            println!(
                "  calibration top {gb:.1} GB ({} experts): {kept}/{} projections ({:.1}% of bytes) never evicted",
                pin.len(),
                ids.len(),
                kept_bytes as f64 / pinned_bytes(catalog, pin) as f64 * 100.0
            );
        }
    }

    // Q2/Q3: the pin sweep.
    for lanes in [1usize, 3] {
        let (share, round_ms) = STALL[usize::from(lanes == 3)];
        let runs: Vec<_> = sweep.iter().filter(|r| r.0 == lanes).collect();
        for (label, pick) in [("per chunk (256+512)", 0usize), ("one serving life", 1)] {
            println!("\n== pin sweep, rtx5090 cache, {lanes} lane(s), {label}");
            let tally = |r: &(usize, Tally, Tally, Survival)| if pick == 0 { r.1 } else { r.2 };
            let base = tally(runs[0]);
            for (pin, run) in pins.iter().zip(&runs) {
                let t = tally(run);
                println!(
                    "{}",
                    row(&format!("pin {} experts", pin.len()), pinned_bytes(catalog, pin), &t, Some(&base), share, round_ms)
                );
            }
        }
    }
}
