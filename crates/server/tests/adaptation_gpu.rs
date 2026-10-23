//! ADR 0063 §3, spec 27b-16gb/04 AC 4 (GitHub #336): a 27B load under
//! `rtx5090` whose explicit VRAM budget refuses the profile's 917,504 tokens
//! adapts the values the profile gave, logs each step in
//! `ignis.runtime.vram_plan`'s `adapted`, and serves at the values it kept --
//! `GET /v1/models` says the context it runs, and a request decodes. And AC
//! 1's leg part A left: the 27B at two and at one decode lane, named, inside
//! an explicit budget, loads and decodes.
//!
//! Every budget comes from the load's own plan. Refused at a 1 GiB budget,
//! the plan names exactly what it needs at the values it was given ("the VRAM
//! plan needs N bytes"), before anything is materialized. The spec's
//! 17,179,869,184-byte budget cannot be used: today's NVFP4 27B's weights
//! line alone is ~18.3 GB (the 16 GB body is #337/#338).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or an unavailable GPU is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::http::Request;
use tower::ServiceExt;

use ignis_artifact::{CudaDevice, FrontendSet, Reader};
use ignis_core::compute::ModelFamily;
use ignis_core::gpu_profile;
use ignis_core::{KvFormat, KvGeometry, VramMode};
use ignis_server::Server;
use ignis_server::adaptation::{Adaptable, Attempt};
use ignis_server::engine::Engine;
use ignis_server::runtime::{EngineShape, cuda_scheduler};
use ignis_server::telemetry::SystemClock;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";

/// The 27B's shape under `--profile rtx5090`, as a start resolves it: every
/// value the profile's or the default's, so every one adapts.
fn rtx5090() -> EngineShape {
    let args = ["--profile", "rtx5090"].map(String::from);
    let ignis_server::config::ConfigOutcome::Config(config) =
        ignis_server::config::resolve(&args, |_| None).expect("resolve")
    else {
        panic!("expected a runnable config");
    };
    EngineShape::from(&config.for_family(ModelFamily::Qwen38_27b).expect("the 27B"))
}

fn explicit(budget_bytes: u64) -> VramMode {
    VramMode::Explicit { budget_bytes, allow_oversubscription: false }
}

fn page_bytes() -> u64 {
    KvFormat::HqE8_2b.page_bytes(KvGeometry::qwen38_27b())
}

/// The artifact's EOS and frontend, or `None` where the profile allows a
/// skip.
fn artifact() -> Option<(u32, FrontendSet)> {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().expect("qwen3.8-27b generation config must carry eos_token_id");
    Some((eos, frontend))
}

/// What the 27B's plan needs at `shape`'s own values: its refusal at a
/// 1 GiB budget, with nothing allowed to adapt, names it.
fn needed_bytes(shape: EngineShape, eos: u32) -> u64 {
    let refused = EngineShape { vram: explicit(1 << 30), adaptable: Adaptable::default(), ..shape };
    let message = match cuda_scheduler(Path::new(ARTIFACT), MODEL.into(), eos, refused) {
        Ok(_) => panic!("a 1 GiB budget holds no 27B"),
        Err(message) => message,
    };
    let needed = message
        .split("the VRAM plan needs ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .and_then(|number| number.parse().ok());
    needed.unwrap_or_else(|| panic!("a plan refused below its minimum names what it needs: {message}"))
}

/// The card's free memory: an explicit budget above it is refused before
/// any plan, so a budget this test picks must fit under it.
fn free_bytes() -> u64 {
    CudaDevice::nvml_memory(0).expect("NVML reads the card").0
}

/// The GPU-backed router plus the model thread that owns its VRAM, and the
/// `ignis.runtime.vram_plan` event its load wrote. Dropping it disconnects
/// the engine and joins the model thread, which releases the weights before
/// the next load.
struct Harness {
    app: Option<axum::Router>,
    driver: Option<std::thread::JoinHandle<()>>,
    vram_plan: serde_json::Value,
}

impl Harness {
    fn app(&self) -> &axum::Router {
        self.app.as_ref().expect("app is only taken by Drop")
    }

    fn plan_field(&self, name: &str) -> &serde_json::Value {
        &self.vram_plan["attributes"][name]
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        drop(self.app.take());
        if let Some(driver) = self.driver.take() {
            let _ = driver.join();
        }
    }
}

/// Load `shape`, capturing the plan event its load logs.
fn harness(shape: EngineShape, eos: u32, frontend: FrontendSet) -> Harness {
    use tracing_subscriber::layer::SubscriberExt;
    let sink = Arc::new(ignis_logging::MemorySink::new());
    let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
    let loaded = tracing::subscriber::with_default(subscriber, || cuda_scheduler(Path::new(ARTIFACT), MODEL.into(), eos, shape));
    let scheduler = match loaded {
        Ok((scheduler, _reservations)) => scheduler,
        Err(e) => panic!("cuda_scheduler at {:?}: {e}", shape.adaptation_start()),
    };
    let vram_plan = sink
        .lines()
        .iter()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("json"))
        .find(|event| event["event_name"] == "ignis.runtime.vram_plan")
        .expect("the load logs its plan");
    println!("vram_plan: {vram_plan}");
    let (engine, driver) = Engine::with_clock_and_driver(Box::new(scheduler), Arc::new(SystemClock));
    let server = Server::with_artifact_template(engine, frontend).with_request_timeout(Duration::from_secs(300));
    Harness { app: Some(server.app()), driver: Some(driver), vram_plan }
}

async fn body_of(app: &axum::Router, request: Request<Body>) -> (u16, serde_json::Value) {
    let response = app.clone().oneshot(request).await.expect("route the request");
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.expect("read the body");
    let text = String::from_utf8(bytes.to_vec()).expect("the body is UTF-8");
    (status, serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}")))
}

/// The context `GET /v1/models` says the loaded model serves.
async fn max_model_len(app: &axum::Router) -> u64 {
    let request = Request::builder().uri("/v1/models").body(Body::empty()).expect("build the request");
    let (status, models) = body_of(app, request).await;
    assert_eq!(status, 200, "{models}");
    models["data"][0]["max_model_len"].as_u64().unwrap_or_else(|| panic!("{models}"))
}

/// One greedy completion of a short prompt: the load decodes.
async fn decodes(app: &axum::Router) -> String {
    let body = serde_json::json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "Name the capital of France in one word." }],
        "max_tokens": 24,
        "temperature": 0,
        "stream": false,
        "enable_thinking": false
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string().into_bytes()))
        .expect("build the request");
    let (status, value) = body_of(app, request).await;
    assert_eq!(status, 200, "{value}");
    let content = value["choices"][0]["message"]["content"].as_str().unwrap_or_else(|| panic!("{value}")).to_owned();
    println!("completion: {content:?}");
    assert!(content.contains("Paris"), "{value}");
    content
}

/// A budget at least 100 MB short of 917,504 tokens -- less when the card
/// has less free, an explicit budget above free memory being refused before
/// any plan: the load keeps the largest context whose pool holds a second
/// 65,536-token reservation beside it (786,432 for a deficit up to ~613 MB:
/// 851,968 fits, without that room), says so, and serves it.
#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn a_budget_that_refuses_917504_loads_at_the_adapted_context_and_decodes() {
    let Some((eos, frontend)) = artifact() else { return };
    let shape = rtx5090();
    assert_eq!((shape.max_context, shape.adaptable), (917_504, Adaptable::ALL));
    let needed = needed_bytes(shape, eos);
    let free = free_bytes();
    let budget = (needed - 100_000_000).min(free - (256 << 20));
    println!("needed at 917,504: {needed} bytes; budget {budget}; free {free}");

    let h = harness(EngineShape { vram: explicit(budget), ..shape }, eos, frontend);
    let adapted = h.plan_field("adapted").as_str().expect("adapted").to_owned();
    let context: u64 = adapted
        .strip_prefix("max_context 917504 -> ")
        .and_then(|context| context.parse().ok())
        .unwrap_or_else(|| panic!("one step, the context's: {adapted}"));
    // Above 262,144 tokens no line follows the context, so the pool is the
    // same at the next step up, where it must have fallen short.
    let pool = h.plan_field("kv_pool_tokens").as_u64().unwrap();
    assert!(pool >= context + 65_536, "a second reservation beside {context}: {pool}");
    assert!(context < 262_144 || pool < context + 2 * 65_536, "the largest such context: {context}, {pool}");
    assert_eq!(max_model_len(h.app()).await, context, "the context it runs, where clients read it");
    decodes(h.app()).await;
}

/// A budget where no context at eight lanes keeps a second 65,536-token
/// reservation: the lanes drop to one -- their host retained slots to two --
/// and the ladder runs again from 917,504.
#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn a_budget_no_eight_lane_context_fits_drops_the_lanes_to_one_and_decodes() {
    let Some((eos, frontend)) = artifact() else { return };
    let shape = rtx5090();
    let start = shape.adaptation_start();
    let floor = shape.at(Attempt { max_context: 65_536, ..start });
    let floor_pages = 65_536 / 64 + u64::from(floor.retained_slots());
    // One page short of 65,536 + 65,536 tokens at eight lanes.
    let budget = needed_bytes(floor, eos) + (2 * 1024 - floor_pages - 1) * page_bytes();
    println!("budget {budget}; free {}", free_bytes());

    let h = harness(EngineShape { vram: explicit(budget), ..shape }, eos, frontend);
    let adapted = h.plan_field("adapted").as_str().expect("adapted").to_owned();
    assert!(adapted.starts_with("decode_lanes 8 -> 1, max_context 917504 -> "), "{adapted}");
    let context: u64 = adapted.rsplit(' ').next().unwrap().parse().unwrap();
    assert!((65_536..917_504).contains(&context), "{adapted}");
    assert_eq!(h.plan_field("retained_host_slots"), 2, "two host retained slots a lane");
    assert_eq!(max_model_len(h.app()).await, context);
    decodes(h.app()).await;
}

/// AC 1's leg part A left: the 27B with two lanes named, then one, inside
/// an explicit budget that holds 262,144 tokens with 256 MB to spare, loads
/// at the values named -- nothing adapts -- and decodes.
#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn the_27b_at_two_and_at_one_named_lane_loads_and_decodes_in_an_explicit_budget() {
    for lanes in [2, 1] {
        let Some((eos, frontend)) = artifact() else { return };
        let base = rtx5090();
        let shape = EngineShape {
            adaptable: Adaptable { decode_lanes: false, max_context: false, ..Adaptable::ALL },
            ..base.at(Attempt { max_context: 262_144, decode_lanes: lanes, ..base.adaptation_start() })
        };
        let budget = needed_bytes(shape, eos) + (256 << 20);
        let h = harness(EngineShape { vram: explicit(budget), ..shape }, eos, frontend);
        assert_eq!(h.plan_field("adapted"), "", "{lanes} lanes");
        assert_eq!(h.plan_field("retained_host_slots"), 2 * lanes, "{lanes} lanes");
        assert_eq!(max_model_len(h.app()).await, 262_144);
        decodes(h.app()).await;
    }
}
