//! A warm-up still running when serving ends is cancelled before the serve
//! exit tears the model down (`Server::serve_on_until`): left running, the
//! teardown ends its request, and the warm-up reports that on its own task
//! after serve has returned — racing `ignis.process.stopped`, which must be
//! the last line (logging spec §28). A binary of its own with one test,
//! which installs the process's subscriber, so no other test can have
//! settled the warm-up lines' interest first.

use std::sync::Arc;

use ignis_core::{
    mock::MockCompute, ConcreteScheduler, RequestClass, RequestId, RequestInput, SchedEvent, Scheduler,
    SchedulerConfig, SubmitError,
};
use ignis_logging::{JsonLayer, MemorySink};
use ignis_server::engine::Engine;
use ignis_server::telemetry::SystemClock;
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::{ActiveModel, Server};
use tracing_subscriber::layer::SubscriberExt;

fn scheduler() -> ConcreteScheduler {
    ConcreteScheduler::with_config(
        SchedulerConfig { model: "qwen3.8-flash-next".into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    )
}

/// A model on `scheduler` that keeps its model thread's handle, as a load does.
fn model_on(scheduler: Box<dyn Scheduler>) -> ActiveModel {
    let (engine, driver) = Engine::with_clock_and_driver(scheduler, Arc::new(SystemClock));
    ActiveModel::new(engine, Arc::new(SimpleTemplateProvider)).with_driver(driver)
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

/// The warm-ups' own reports among `log`'s lines, either way, in order.
fn warm_up_reports(log: &MemorySink) -> Vec<String> {
    log.lines()
        .iter()
        .map(|l| {
            let line: serde_json::Value = serde_json::from_str(l).unwrap();
            line["event_name"].as_str().unwrap_or_default().to_owned()
        })
        .filter(|name| name == "ignis.process.ready" || name == "ignis.process.warm_up_failed")
        .collect()
}

#[tokio::test]
async fn a_warm_up_still_running_when_serving_ends_is_cancelled_and_reports_nothing() {
    let log = Arc::new(MemorySink::new());
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(JsonLayer::new(log.clone())))
        .expect("the only subscriber of this binary");

    // The control: a warm-up that runs reports here.
    let control = Server::from_active(model_on(Box::new(scheduler()))).with_warm_up();
    control.warm_up().await.expect("the mock warms up");
    assert_eq!(warm_up_reports(&log), ["ignis.process.ready"], "the control's warm-up is seen");

    let server = Server::from_active(model_on(Box::new(Stalled(scheduler())))).with_warm_up();
    server
        .serve_on_until(
            tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind"),
            None,
            std::future::ready(()),
        )
        .await
        .expect("serve");
    // A warm-up left running is ended by the teardown and reports its
    // failure on its own task, after serve has returned: give it the time.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    assert_eq!(
        warm_up_reports(&log),
        ["ignis.process.ready"],
        "only the control's warm-up reports: the one the serve left running must not, once serving ended"
    );
}
