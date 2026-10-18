//! The host expert pool filled from an artifact (spec flash-next/03,
//! acceptance 1, GitHub #301): on the Flash-Next binder's fixture artifact,
//! every expert projection lands in the pool at the offset the expert index
//! lays out, byte for byte what the file holds, and the residency catalog
//! read off the index carries each projection's K and its class's record
//! bytes.

use ignis_artifact::flash_next::{self, FlashNextGeometry, Projection as ArtifactProjection, fixture};
use ignis_artifact::Reader;
use ignis_core::residency::load::{catalog, fill_expert_pool, pool_layout, stream_expert_layers, stream_staging_bytes};
use ignis_core::residency::{KClass, Projection, ProjectionId};

#[test]
fn every_expert_projection_lands_in_the_pool_where_the_index_puts_it() {
    let artifact = fixture::build("residency-pool").expect("fixture artifact");
    let reader = Reader::open(&artifact.path).expect("open");
    let plan = flash_next::bind(&reader, &FlashNextGeometry::fixture()).expect("bind");
    let index = &plan.experts;

    let layout = pool_layout(index);
    let keys = index.layers() * index.experts_per_layer() * 2;
    assert_eq!(layout.k2.len(), keys);
    assert_eq!(layout.offsets.len(), keys);
    assert_eq!(layout.bytes, plan.plan.expert_pool_capacity_bytes);

    let mut pool = vec![0u8; layout.bytes as usize];
    // 2 layers < READ_WORKERS: exercises workers left with no layer too.
    let read = fill_expert_pool(&artifact.path, index, &mut pool).expect("fill");
    let file_bytes = std::fs::read(&artifact.path).expect("read");

    let cat = catalog(index).expect("catalog");
    let mut total = 0;
    for layer in 0..index.layers() {
        for expert in 0..index.experts_per_layer() {
            for (p, projection) in ArtifactProjection::ALL.into_iter().enumerate() {
                let r = index.get(layer, expert, projection).expect("indexed");
                let at = r.pool_offset as usize;
                let from = r.file_offset as usize;
                assert_eq!(
                    &pool[at..at + r.bytes as usize],
                    &file_bytes[from..from + r.bytes as usize],
                    "layer {layer} expert {expert} projection {p}"
                );
                let key = (layer * index.experts_per_layer() + expert) * 2 + p;
                assert_eq!(layout.k2[key], r.k.k2());
                assert_eq!(layout.offsets[key], r.pool_offset);
                let id = ProjectionId::new(
                    layer as u16,
                    expert as u16,
                    if p == 0 { Projection::GateUp } else { Projection::Down },
                );
                assert_eq!(u32::from(r.k.k2()), cat.class_of(id).k.half_bits());
                assert_eq!(cat.bytes(id), r.bytes);
                total += r.bytes;
            }
        }
    }
    assert_eq!(read, total);
    // The fixture's K map spans several classes, as the real one does.
    let used = cat.class_counts().iter().filter(|&&n| n > 0).count();
    assert!(used >= 3, "{used} of {} classes used", KClass::COUNT);
}

#[test]
fn a_pool_too_small_for_the_last_layer_is_refused_before_any_read() {
    let artifact = fixture::build("residency-pool-undersized").expect("fixture artifact");
    let reader = Reader::open(&artifact.path).expect("open");
    let plan = flash_next::bind(&reader, &FlashNextGeometry::fixture()).expect("bind");
    let index = &plan.experts;
    let layout = pool_layout(index);

    let mut pool = vec![0u8; layout.bytes as usize - 1];
    let err = fill_expert_pool(&artifact.path, index, &mut pool).expect_err("undersized pool");
    assert!(err.to_string().contains("past the"), "{err}");
}

/// A device-only residency's load (ADR 0054): every layer arrives once, its
/// records byte for byte the pool's, keys in key order, each record inside
/// its own layer's buffer; and the RAM held is the staging line's.
#[test]
fn every_layer_streams_once_with_the_pools_bytes() {
    let artifact = fixture::build("residency-stream").expect("fixture artifact");
    let reader = Reader::open(&artifact.path).expect("open");
    let plan = flash_next::bind(&reader, &FlashNextGeometry::fixture()).expect("bind");
    let index = &plan.experts;
    let layout = pool_layout(index);
    let mut pool = vec![0u8; layout.bytes as usize];
    let filled = fill_expert_pool(&artifact.path, index, &mut pool).expect("fill");

    let mut layers = Vec::new();
    let read = stream_expert_layers(&artifact.path, index, |layer| {
        layers.push(layer);
        Ok(())
    })
    .expect("stream");
    assert_eq!(read, filled, "the bytes the pool fill reads");
    layers.sort_by_key(|l| l.layer);
    assert_eq!(layers.iter().map(|l| l.layer).collect::<Vec<_>>(), (0..index.layers()).collect::<Vec<_>>());

    let experts = index.experts_per_layer();
    for records in &layers {
        let first = (records.layer * experts * 2) as u32;
        assert_eq!(records.keys, (first..first + experts as u32 * 2).collect::<Vec<_>>(), "layer {}", records.layer);
        assert_eq!(records.offsets.len(), records.keys.len());
        for (&key, &at) in records.keys.iter().zip(&records.offsets) {
            let (expert, p) = ((key as usize / 2) % experts, key as usize % 2);
            let r = index.get(records.layer, expert, ArtifactProjection::ALL[p]).expect("indexed");
            let (at, len, pool_at) = (at as usize, r.bytes as usize, r.pool_offset as usize);
            assert!(at + len <= records.bytes.len(), "key {key} runs past its layer's buffer");
            assert_eq!(&records.bytes[at..at + len], &pool[pool_at..pool_at + len], "key {key}");
        }
    }
    // Two layers, so two readers; one queued and one being admitted beside them.
    let heaviest = layers.iter().map(|l| l.bytes.len() as u64).max().expect("layers");
    assert_eq!(stream_staging_bytes(index).expect("staging"), 4 * heaviest);
}

/// The caller's admit refusing a layer ends the stream with its error, the
/// readers stopping rather than reading on.
#[test]
fn a_refused_layer_ends_the_stream_with_its_error() {
    let artifact = fixture::build("residency-stream-refused").expect("fixture artifact");
    let reader = Reader::open(&artifact.path).expect("open");
    let plan = flash_next::bind(&reader, &FlashNextGeometry::fixture()).expect("bind");
    let mut calls = 0;
    let err = stream_expert_layers(&artifact.path, &plan.experts, |_| {
        calls += 1;
        Err("no slot".to_owned())
    })
    .expect_err("refused");
    assert_eq!((err.as_str(), calls), ("no slot", 1));
}
