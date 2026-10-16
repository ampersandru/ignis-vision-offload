//! Serve exit must join the model thread (GitHub: Linux VRAM zombie).
//!
//! Dropping [`std::thread::JoinHandle`] without [`JoinHandle::join`] detaches
//! `ignis-model`. On a CUDA load that thread can hang in driver teardown and
//! leave the process a zombie still holding VRAM. `Server::serve_on_until`
//! therefore calls [`ignis_server::ActiveModel::shut_down`] after the HTTP
//! drain — these tests prove the driver handle is gone once serve returns,
//! a refused metrics listener included, and that a warm-up still running
//! then is cancelled, not ended by that teardown and reported after it.

use std::sync::Arc;

use ignis_core::{
    mock::MockCompute, ConcreteScheduler, RequestClass, RequestId, RequestInput, SchedEvent, Scheduler,
    SchedulerConfig, SubmitError,
};
use ignis_server::engine::Engine;
use ignis_server::telemetry::SystemClock;
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::{ActiveModel, Server};

fn scheduler() -> ConcreteScheduler {
    let compute = Arc::new(MockCompute::new());
    ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "qwen3.8-flash-next".into(),
            ..SchedulerConfig::default()
        },
        compute,
    )
}

/// A model on `scheduler` that keeps its model thread's handle, as a load does.
fn model_on(scheduler: Box<dyn Scheduler>) -> ActiveModel {
    let (engine, driver) = Engine::with_clock_and_driver(scheduler, Arc::new(SystemClock));
    ActiveModel::new(engine, Arc::new(SimpleTemplateProvider)).with_driver(driver)
}

fn model() -> ActiveModel {
    model_on(Box::new(scheduler()))
}

/// The mock's scheduler, except that it never runs what it takes: a warm-up
/// on it is still running whenever serving ends. Idle to the model thread,
/// so the thread waits on its next command instead of spinning.
struct Stalled(ConcreteScheduler);

impl Scheduler for Stalled {
    fn submit(&mut self, input: RequestInput, class: RequestClass) -> Result<RequestId, SubmitError> {
        self.0.submit(input, class)
    }
    fn cancel(&mut self, request: RequestId) -> bool {
        self.0.cancel(request)
    }
    fn advance(&mut self) -> Vec<SchedEvent> {
        Vec::new()
    }
    fn is_idle(&self) -> bool {
        true
    }
    fn model_id(&self) -> &str {
        self.0.model_id()
    }
    fn max_sequence_tokens(&self) -> u32 {
        self.0.max_sequence_tokens()
    }
    fn mode(&self) -> ignis_core::EngineMode {
        self.0.mode()
    }
    fn occupancy(&self) -> ignis_core::Occupancy {
        self.0.occupancy()
    }
}

async fn listener() -> tokio::net::TcpListener {
    tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind")
}

#[tokio::test]
async fn serve_on_until_joins_the_model_thread_before_returning() {
    let model = model();
    assert!(
        !model.driver_joined(),
        "a model built with_driver keeps its JoinHandle until shut_down"
    );

    let server = Server::from_active(model);
    let active = server.active();
    // Resolve immediately: we only care about the post-drain join.
    server
        .serve_on_until(listener().await, None, std::future::ready(()))
        .await
        .expect("serve");

    assert!(
        active.driver_joined(),
        "serve_on_until must shut_down/join the model thread before returning"
    );
}

#[tokio::test]
async fn a_metrics_listener_refused_still_joins_the_model_thread() {
    // Metrics off: a metrics listener is refused before anything is served.
    let server = Server::from_active(model());
    let active = server.active();
    let refused = server
        .serve_on_until(listener().await, Some(listener().await), std::future::ready(()))
        .await
        .expect_err("a metrics listener without metrics on is refused");

    assert_eq!(refused.kind(), std::io::ErrorKind::InvalidInput);
    assert!(
        active.driver_joined(),
        "the refusal must not detach the loaded model's thread"
    );
}

#[tokio::test]
async fn a_warm_up_still_running_when_serving_ends_is_cancelled_and_reports_nothing() {
    use tracing_subscriber::layer::SubscriberExt;
    let log = Arc::new(ignis_logging::MemorySink::new());
    let _guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(log.clone())),
    );

    let server = Server::from_active(model_on(Box::new(Stalled(scheduler())))).with_warm_up();
    server
        .serve_on_until(listener().await, None, std::future::ready(()))
        .await
        .expect("serve");
    // A warm-up left running is ended by the teardown and reports it on its
    // own task, after serve has returned: `ignis.process.ready` when its
    // request was ended in flight (an ended request still completes),
    // `warm_up_failed` when the thread was gone before it got there. Give it
    // the time to say either.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let reported: Vec<String> = log
        .lines()
        .iter()
        .map(|l| {
            let line: serde_json::Value = serde_json::from_str(l).unwrap();
            line["event_name"].as_str().unwrap_or_default().to_owned()
        })
        .filter(|name| name == "ignis.process.ready" || name == "ignis.process.warm_up_failed")
        .collect();
    assert!(
        reported.is_empty(),
        "a warm-up the serve left running must not report once serving ended: {reported:?}"
    );
}
