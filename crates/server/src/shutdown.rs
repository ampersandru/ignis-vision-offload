//! The process's stop signal (GitHub #329 item 17): Ctrl+C, and SIGTERM on
//! unix, listened for from the start of `main` rather than from the serve,
//! so a stop that arrives while the model downloads or loads is acted on.
//!
//! The start and the serve stop differently. Until the server serves there
//! is nothing to drain, and the load has no clean abort (a synchronous run of
//! CUDA calls on its own thread), so a signal ends the process at once
//! ([`exit_now`]), which is what frees the GPU. Once it serves, the signal is
//! the graceful shutdown's ([`crate::Server::serve_on_until`], spec §28).
//!
//! The incident behind it (2026-10-18, a Knative pod on an RTX PRO 6000):
//! `ignis-server` was the container's PID 1, and the kernel drops a signal
//! PID 1 has no handler for. The handler was installed only once serving
//! began, so the kubelet's SIGTERM during a 118 s load did nothing, and the
//! pod held the GPU for the hour of its grace period.

use std::future::Future;
use std::sync::{Arc, Mutex, PoisonError};

/// Where the process is when a signal arrives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// Still starting, in the named phase (`starting`, `downloading`,
    /// `loading`): a signal ends the process.
    Starting(&'static str),
    /// Serving: a signal stops the serve, which drains and tears down.
    Serving,
}

/// The process's stop signal, listened for from [`StopSignal::install`] on.
/// A signal emits `ignis.process.stopping` once, then, before
/// [`StopSignal::serving`], ends the process (see the module docs); after
/// it, resolves the future that call returned.
pub struct StopSignal {
    stage: Arc<Mutex<Stage>>,
    stopped: tokio::sync::watch::Receiver<bool>,
}

impl StopSignal {
    /// Listen for Ctrl+C, and SIGTERM on unix: the handlers are registered
    /// before this returns, so no signal from here on is dropped, not even
    /// by a PID 1. A signal before [`StopSignal::serving`] logs
    /// `ignis.process.stopped` with the phase the start was in, then runs
    /// `before_serving`, which `main` makes flush the log and
    /// [`exit_now`]. Needs a tokio runtime.
    pub fn install(before_serving: impl FnOnce() + Send + 'static) -> std::io::Result<Self> {
        Ok(Self::on(process_signal()?, before_serving))
    }

    /// [`StopSignal::install`] over `signal` instead of the process's own
    /// (tests).
    pub fn on(
        signal: impl Future<Output = ()> + Send + 'static,
        before_serving: impl FnOnce() + Send + 'static,
    ) -> Self {
        Self::spawn(signal, Stage::Starting("starting"), before_serving)
    }

    /// For a caller that serves at once ([`crate::Server::serve_on`]): the
    /// signal is listened for from this call, and only ever stops the serve.
    pub(crate) fn install_serving() -> std::io::Result<Self> {
        Ok(Self::spawn(process_signal()?, Stage::Serving, || {}))
    }

    fn spawn(
        signal: impl Future<Output = ()> + Send + 'static,
        stage: Stage,
        before_serving: impl FnOnce() + Send + 'static,
    ) -> Self {
        let stage = Arc::new(Mutex::new(stage));
        let (stop, stopped) = tokio::sync::watch::channel(false);
        let at = Arc::clone(&stage);
        tokio::spawn(async move {
            signal.await;
            emit_stopping();
            // Read once: a serve that begins right after is stopped by the
            // exit, as one still starting is, with nothing admitted yet.
            let stage = *at.lock().unwrap_or_else(PoisonError::into_inner);
            match stage {
                Stage::Starting(phase) => {
                    emit_stopped_before_serving(phase);
                    before_serving();
                }
                Stage::Serving => {
                    let _ = stop.send(true);
                }
            }
        });
        Self { stage, stopped }
    }

    /// What the start is doing now: the `phase` a signal before serving
    /// names on its `ignis.process.stopped` line.
    pub fn starting(&self, phase: &'static str) {
        *self.stage.lock().unwrap_or_else(PoisonError::into_inner) = Stage::Starting(phase);
    }

    /// From now on a signal stops the serve, not the process: the future
    /// resolves once one arrives, also when it arrived between this call
    /// and the future's first poll — the shutdown
    /// [`crate::Server::serve_on_until`] takes.
    pub fn serving(&self) -> impl Future<Output = ()> + Send + 'static {
        *self.stage.lock().unwrap_or_else(PoisonError::into_inner) = Stage::Serving;
        let mut stopped = self.stopped.clone();
        async move {
            // An error is a watcher that has gone without signalling, which
            // only a signal before serving does (and that one exits).
            let _ = stopped.wait_for(|stopped| *stopped).await;
        }
    }
}

/// The process's own signal, its handlers registered before this returns:
/// SIGTERM or SIGINT on unix, Ctrl+C on Windows. Registration is the
/// constructor's, not the first poll's (`tokio::signal::ctrl_c()` registers
/// only when polled): the future may wait to be awaited.
fn process_signal() -> std::io::Result<impl Future<Output = ()> + Send + 'static> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut terminate = signal(SignalKind::terminate())?;
        let mut interrupt = signal(SignalKind::interrupt())?;
        Ok(async move {
            tokio::select! {
                _ = terminate.recv() => {}
                _ = interrupt.recv() => {}
            }
        })
    }
    #[cfg(windows)]
    {
        let mut ctrl_c = tokio::signal::windows::ctrl_c()?;
        Ok(async move {
            ctrl_c.recv().await;
        })
    }
}

/// End the process with `code` now. On unix through `_exit`, which runs no
/// exit handlers: `std::process::exit` runs them, and the CUDA runtime's and
/// the leaf's teardown among them, beside a load thread still inside a CUDA
/// call, is where a process hangs holding the card (the hang
/// [`crate::Server::serve_on_until`] joins the model thread to avoid). The
/// kernel frees the device memory as the process ends, as it did for the
/// SIGKILL in the incident. Flush the log first: nothing after this does.
pub fn exit_now(code: i32) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    #[cfg(unix)]
    // SAFETY: `_exit` takes any status and never returns; it runs no code of
    // this process's that could observe a broken invariant.
    unsafe {
        libc::_exit(code)
    }
    #[cfg(not(unix))]
    std::process::exit(code)
}

/// Emits `ignis.process.stopping` (spec §28), once per signal: split out so
/// the event's own shape is unit-testable without delivering Ctrl+C/SIGTERM
/// to a live process (not portably doable from a Rust test, and Windows has
/// no SIGTERM at all).
fn emit_stopping() {
    tracing::info!(name: "ignis.process.stopping", "shutdown signal received");
}

/// The last line of a process a signal stopped before it served (spec §28's
/// `ignis.process.stopped`), naming the phase the start was in.
fn emit_stopped_before_serving(phase: &'static str) {
    tracing::info!(
        name: "ignis.process.stopped",
        phase,
        "stopped before serving: the start is abandoned and the process exits, which frees the GPU"
    );
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use ignis_logging::{JsonLayer, MemorySink};
    use tracing_subscriber::layer::SubscriberExt;

    use super::{emit_stopping, StopSignal};

    fn records(sink: &MemorySink) -> Vec<serde_json::Value> {
        sink.lines().iter().map(|line| serde_json::from_str(line).expect("valid json")).collect()
    }

    /// The event a signal emits first has the exact name/severity spec §28
    /// asks for.
    #[test]
    fn emit_stopping_produces_the_spec_shaped_event() {
        let sink = Arc::new(MemorySink::new());
        let subscriber = tracing_subscriber::registry().with(JsonLayer::new(sink.clone()));
        tracing::subscriber::with_default(subscriber, emit_stopping);

        let lines = records(&sink);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["event_name"], "ignis.process.stopping");
        assert_eq!(lines[0]["severity_text"], "INFO");
    }

    /// A signal while the start is under way — the load, in the incident —
    /// is acted on at once: `stopping`, then `stopped` naming the phase as
    /// the last line, then the start's own stop (`main`'s exit).
    #[tokio::test]
    async fn a_signal_before_serving_stops_the_start_naming_its_phase() {
        let sink = Arc::new(MemorySink::new());
        let _logs = tracing::subscriber::set_default(tracing_subscriber::registry().with(JsonLayer::new(sink.clone())));
        let (signal, signalled) = tokio::sync::oneshot::channel::<()>();
        let (exited, exit) = tokio::sync::oneshot::channel::<()>();
        let stop = StopSignal::on(
            async move {
                let _ = signalled.await;
            },
            move || {
                let _ = exited.send(());
            },
        );
        stop.starting("loading");
        signal.send(()).expect("the watcher waits");
        tokio::time::timeout(Duration::from_secs(5), exit).await.expect("stopped at once").expect("the start's stop ran");

        let lines = records(&sink);
        let names: Vec<&str> = lines.iter().filter_map(|r| r["event_name"].as_str()).collect();
        assert_eq!(names, ["ignis.process.stopping", "ignis.process.stopped"], "{lines:?}");
        assert_eq!(lines[1]["attributes"]["phase"], "loading", "{lines:?}");
        assert_eq!(lines[1]["severity_text"], "INFO");
    }

    /// Once serving, a signal resolves the serve's shutdown and leaves the
    /// process alone: the serve's teardown runs, and `main` logs the end.
    #[tokio::test]
    async fn a_signal_while_serving_stops_the_serve_not_the_process() {
        let sink = Arc::new(MemorySink::new());
        let _logs = tracing::subscriber::set_default(tracing_subscriber::registry().with(JsonLayer::new(sink.clone())));
        let (signal, signalled) = tokio::sync::oneshot::channel::<()>();
        let (exited, exit) = tokio::sync::oneshot::channel::<()>();
        let stop = StopSignal::on(
            async move {
                let _ = signalled.await;
            },
            move || {
                let _ = exited.send(());
            },
        );
        let shutdown = stop.serving();
        signal.send(()).expect("the watcher waits");
        tokio::time::timeout(Duration::from_secs(5), shutdown).await.expect("the serve's shutdown resolved");
        // The watcher is done and dropped the start's stop without running it.
        assert!(exit.await.is_err(), "the process is not stopped from under the serve");

        let names: Vec<String> =
            records(&sink).iter().filter_map(|r| r["event_name"].as_str().map(str::to_owned)).collect();
        assert_eq!(names, ["ignis.process.stopping"]);
    }

    /// The incident's shape on the serve's side: a signal that lands before
    /// the serve first waits on it is kept, not missed.
    #[tokio::test]
    async fn a_signal_before_the_serve_waits_is_kept() {
        // Captured though unread: a test reaching these callsites with no
        // subscriber settles their interest for the tests above (tracing's
        // callsite cache), which then capture nothing.
        let sink = Arc::new(MemorySink::new());
        let _logs = tracing::subscriber::set_default(tracing_subscriber::registry().with(JsonLayer::new(sink)));
        let (signal, signalled) = tokio::sync::oneshot::channel::<()>();
        let stop = StopSignal::on(
            async move {
                let _ = signalled.await;
            },
            || panic!("serving: the process is not stopped"),
        );
        let shutdown = stop.serving();
        signal.send(()).expect("the watcher waits");
        // Let the watcher see the signal before the shutdown is first polled.
        tokio::time::sleep(Duration::from_millis(50)).await;
        tokio::time::timeout(Duration::from_secs(5), shutdown).await.expect("the signal was kept");
    }
}
