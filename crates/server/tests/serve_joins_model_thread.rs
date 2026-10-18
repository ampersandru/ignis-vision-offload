//! Serve exit must join the model thread (GitHub: Linux VRAM zombie).
//!
//! Dropping [`std::thread::JoinHandle`] without [`JoinHandle::join`] detaches
//! `ignis-model`. On a CUDA load that thread can hang in driver teardown and
//! leave the process a zombie still holding VRAM. `Server::serve_on_until`
//! therefore calls [`ignis_server::ActiveModel::shut_down`] after the HTTP
//! drain — these tests prove the driver handle is gone once serve returns,
//! a refused metrics listener included. A switch still running then is not
//! waited for (GitHub #329): it is returned, nothing torn down, for `main` to
//! end the process under it. What a warm-up left running logs is
//! `serve_exit_warm_up_log.rs`'s; the requests waiting on a load,
//! `serve_exit_load.rs`'s.

use std::sync::Arc;
use std::time::Duration;

use ignis_core::{mock::MockCompute, ConcreteScheduler, SchedulerConfig};
use ignis_server::engine::Engine;
use ignis_server::model_switch::{begin, SwitchOutcome, SwitchStarted, Switcher};
use ignis_server::telemetry::SystemClock;
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::{ActiveModel, Served, Server};

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::{source, MockLoader};

/// A model on the CPU mock that keeps its model thread's handle, as a load does.
fn model() -> ActiveModel {
    let compute = Arc::new(MockCompute::new());
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "qwen3.8-flash-next".into(),
            ..SchedulerConfig::default()
        },
        compute,
    );
    let (engine, driver) =
        Engine::with_clock_and_driver(Box::new(scheduler), Arc::new(SystemClock));
    ActiveModel::new(engine, Arc::new(SimpleTemplateProvider)).with_driver(driver)
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
        model.holds_driver(),
        "a model built with_driver keeps its JoinHandle until shut_down"
    );

    let server = Server::from_active(model);
    let active = server.active();
    // Resolve immediately: only the join after the drain is looked at.
    let served = server
        .serve_on_until(listener().await, None, std::future::ready(()))
        .await
        .expect("serve");

    assert_eq!(served, Served::Drained);

    assert!(
        !active.holds_driver(),
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
        !active.holds_driver(),
        "the refusal must not detach the loaded model's thread"
    );
}

/// GitHub #329: a switch begun before serving ended — its `202` answered,
/// its load still running — is not waited for: a load has no cancel, and a
/// cold one outlasts a pod's grace period. The serve returns once HTTP has
/// drained, naming the switch, with nothing torn down: the model the switch
/// is tearing down or loading is the process end's to free.
#[tokio::test]
async fn a_switch_still_loading_when_serving_ends_is_returned_not_waited_for() {
    let loader = MockLoader::new();
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, Duration::from_secs(30)));
    let load = loader.hold_next_load();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let serving = tokio::spawn(server.clone().serve_on_until(listener().await, None, async {
        let _ = stopped.await;
    }));
    let (_, switch) = begin(&server, source("mock-b")).expect("begins");
    stop.send(()).expect("serving");

    let served = tokio::time::timeout(Duration::from_secs(5), serving)
        .await
        .expect("serve returned while the switch was still loading its target")
        .expect("the serve task")
        .expect("serve");
    assert_eq!(served, Served::LoadAbandoned(SwitchStarted { from: "mock-a".into(), to: "mock-b".into() }));
    assert!(loader.loads().is_empty(), "the load was not waited for");

    // Released only so the test's runtime is not left joining a held thread.
    load.send(()).expect("the load is held");
    assert_eq!(switch.await.expect("the switch task"), SwitchOutcome::Switched);
}
