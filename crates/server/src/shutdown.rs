//! The process's stop signal (GitHub #329 item 17): Ctrl+C, and SIGTERM on
//! unix, listened for from the start of `main` rather than from the serve,
//! so a stop that arrives while the model downloads or loads is acted on.
//!
//! The start and the serve stop differently. Until the server serves there
//! is nothing to drain, and the load has no clean abort (a synchronous run of
//! CUDA calls on its own thread), so a signal ends the process at once
//! ([`exit_now`]), which is what frees the GPU. Once it serves, the signal is
//! the graceful shutdown's ([`crate::Server::serve_on_until`], spec §28),
//! and a second signal while that runs ends the process at once too: the
//! operator's or the orchestrator's way past a shutdown that takes too long.
//!
//! The incident behind it (2026-10-18, a Knative pod on an RTX PRO 6000):
//! `ignis-server` was the container's PID 1, and the kernel drops a signal
//! PID 1 has no handler for. The handler was installed only once serving
//! began, so the kubelet's SIGTERM during a 118 s load did nothing, and the
//! pod held the GPU for the hour of its grace period.

use std::future::Future;
use std::sync::{Arc, Mutex, PoisonError};

use tokio::sync::mpsc::UnboundedReceiver;

/// Where the process is when a signal arrives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// Still starting, in the named phase (`starting`, `downloading`,
    /// `loading`): a signal ends the process.
    Starting(&'static str),
    /// Serving: a signal stops the serve, which drains and tears down.
    Serving,
    /// A signal stopped the serve, whose shutdown runs: another one ends the
    /// process without waiting for it.
    ShuttingDown,
    /// The process is ending — `main` logs its last line, or a signal's exit
    /// does: a signal changes nothing.
    Ending,
}

/// The exit a signal runs: `main`'s flush and [`exit_now`].
type Exit = Box<dyn FnOnce() + Send>;

/// The process's stop signal, listened for from [`StopSignal::install`] on.
/// A first signal emits `ignis.process.stopping`, then, before
/// [`StopSignal::serving`], ends the process (see the module docs); after
/// it, resolves the future that call returned. A second one during that
/// shutdown ends the process without waiting for it.
pub struct StopSignal {
    stage: Arc<Mutex<Stage>>,
    stopped: tokio::sync::watch::Receiver<bool>,
}

impl StopSignal {
    /// Listen for Ctrl+C, and SIGTERM on unix: the handlers are registered
    /// before this returns, so no signal from here on is dropped, not even
    /// by a PID 1. A signal before [`StopSignal::serving`] logs
    /// `ignis.process.stopped` with the phase the start was in, then runs
    /// `exit`, which `main` makes flush the log and [`exit_now`]; so does a
    /// second signal while the serve's shutdown runs. Needs a tokio runtime.
    pub fn install(exit: impl FnOnce() + Send + 'static) -> std::io::Result<Self> {
        Ok(Self::on(process_signals()?, exit))
    }

    /// [`StopSignal::install`] over `signals` instead of the process's own
    /// (tests): each message one signal.
    pub fn on(signals: UnboundedReceiver<()>, exit: impl FnOnce() + Send + 'static) -> Self {
        Self::spawn(signals, Stage::Starting("starting"), Some(Box::new(exit)))
    }

    /// For a caller that serves at once ([`crate::Server::serve_on`]): the
    /// signal is listened for from this call, and only ever stops the serve.
    /// With no exit of the caller's to run, a second signal changes nothing.
    pub(crate) fn install_serving() -> std::io::Result<Self> {
        Ok(Self::spawn(process_signals()?, Stage::Serving, None))
    }

    fn spawn(mut signals: UnboundedReceiver<()>, stage: Stage, mut exit: Option<Exit>) -> Self {
        let stage = Arc::new(Mutex::new(stage));
        let (stop, stopped) = tokio::sync::watch::channel(false);
        let at = Arc::clone(&stage);
        tokio::spawn(async move {
            while signals.recv().await.is_some() {
                // One transition per signal, under the lock `main` takes to
                // end the process itself ([`StopSignal::ending`]): a serve
                // that begins right after a signal before serving is stopped
                // by the exit, as one still starting is, with nothing
                // admitted yet.
                let mut stage = at.lock().unwrap_or_else(PoisonError::into_inner);
                match *stage {
                    Stage::Starting(phase) => {
                        *stage = Stage::Ending;
                        drop(stage);
                        emit_stopping();
                        emit_stopped_before_serving(phase);
                        if let Some(exit) = exit.take() {
                            exit();
                        }
                        return;
                    }
                    Stage::Serving => {
                        *stage = Stage::ShuttingDown;
                        drop(stage);
                        emit_stopping();
                        let _ = stop.send(true);
                    }
                    Stage::ShuttingDown => {
                        // With no exit of the caller's there is nothing a
                        // second signal could do.
                        let Some(exit) = exit.take() else { continue };
                        *stage = Stage::Ending;
                        drop(stage);
                        emit_stopped_forced();
                        exit();
                        return;
                    }
                    Stage::Ending => {}
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

    /// The caller ends the process now, its last line its own: a signal from
    /// here on changes nothing. `false` when a second signal got there first
    /// — its exit is under way, its `ignis.process.stopped` the last line —
    /// and the caller logs nothing more.
    pub fn ending(&self) -> bool {
        let mut stage = self.stage.lock().unwrap_or_else(PoisonError::into_inner);
        let ours = *stage != Stage::Ending;
        *stage = Stage::Ending;
        ours
    }
}

/// The process's own signals, one message each, their handlers registered
/// before this returns: SIGTERM or SIGINT on unix, Ctrl+C on Windows.
/// Registration is the constructor's, not the first poll's
/// (`tokio::signal::ctrl_c()` registers only when polled): the receiver may
/// wait to be read. Needs a tokio runtime.
fn process_signals() -> std::io::Result<UnboundedReceiver<()>> {
    let (signalled, signals) = tokio::sync::mpsc::unbounded_channel();
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut terminate = signal(SignalKind::terminate())?;
        let mut interrupt = signal(SignalKind::interrupt())?;
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    Some(()) = terminate.recv() => {}
                    Some(()) = interrupt.recv() => {}
                    else => return,
                }
                if signalled.send(()).is_err() {
                    return;
                }
            }
        });
    }
    #[cfg(windows)]
    {
        let mut ctrl_c = tokio::signal::windows::ctrl_c()?;
        tokio::spawn(async move {
            while ctrl_c.recv().await.is_some() {
                if signalled.send(()).is_err() {
                    return;
                }
            }
        });
    }
    Ok(signals)
}

/// End the process with `code` now. On unix through `_exit`, which runs no
/// exit handlers: `std::process::exit` runs them, and the CUDA runtime's and
/// the leaf's teardown among them, beside a load thread still inside a CUDA
/// call, is where a process hangs holding the card (the hang
/// [`crate::Server::serve_on_until`] joins the model thread to avoid). The
/// kernel frees the device memory as the process ends, as it did for the
/// SIGKILL in the incident. It takes no lock: the caller's bounded log flush
/// is the last write, and stdout is line-buffered, so flushing it here would
/// only wait on the lock a writer stalled on a stuck pipe holds -- the
/// indefinite shutdown block spec §28 forbids.
pub fn exit_now(code: i32) -> ! {
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

/// The last line of a process whose serve stopped with a switch, a reload or
/// a load still running ([`crate::Served::LoadAbandoned`]): `phase` as `GET
/// /v1/models` reports the run (`switching`, or `loading` out of
/// `unloaded`), and the ids it moved between.
pub fn emit_stopped_abandoning(load: &crate::model_switch::SwitchStarted, phase: &'static str) {
    tracing::info!(
        name: "ignis.process.stopped",
        phase,
        from = %load.from,
        to = %load.to,
        "stopped while serving with a model loading: the load is abandoned and the process exits, which frees the GPU"
    );
}

/// The last line of a process a second signal ended while the serve's
/// shutdown ran (`phase` `shutting_down`): what that shutdown still waited
/// for is not waited for.
fn emit_stopped_forced() {
    tracing::info!(
        name: "ignis.process.stopped",
        phase = "shutting_down",
        "a second signal during the shutdown: the process exits without waiting for the rest of it"
    );
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use ignis_logging::{JsonLayer, MemorySink};
    use tracing_subscriber::layer::SubscriberExt;

    use tokio::sync::mpsc::UnboundedSender;
    use tokio::sync::oneshot;

    use super::{emit_stopped_abandoning, emit_stopping, Stage, StopSignal};
    use crate::model_switch::SwitchStarted;

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

    /// GitHub #329: the last line of a process whose serve stopped with a
    /// load running names the load it abandoned, in the phase the serve
    /// gave it up in.
    #[test]
    fn emit_stopped_abandoning_names_the_load() {
        for phase in ["switching", "loading"] {
            let sink = Arc::new(MemorySink::new());
            let subscriber = tracing_subscriber::registry().with(JsonLayer::new(sink.clone()));
            let load = SwitchStarted { from: "qwen3.8-27b".into(), to: "qwen3.8-flash-next".into() };
            tracing::subscriber::with_default(subscriber, || emit_stopped_abandoning(&load, phase));

            let lines = records(&sink);
            assert_eq!(lines.len(), 1, "{lines:?}");
            assert_eq!(lines[0]["event_name"], "ignis.process.stopped");
            assert_eq!(lines[0]["severity_text"], "INFO");
            let attributes = &lines[0]["attributes"];
            assert_eq!((attributes["phase"].as_str(), attributes["from"].as_str()), (Some(phase), Some("qwen3.8-27b")));
            assert_eq!(attributes["to"], "qwen3.8-flash-next");
        }
    }

    /// A stop signal over a test's own signals, each `send` one; the exit
    /// it runs reports on the returned receiver.
    fn stop_signal() -> (StopSignal, UnboundedSender<()>, oneshot::Receiver<()>) {
        let (signal, signals) = tokio::sync::mpsc::unbounded_channel();
        let (exited, exit) = oneshot::channel::<()>();
        let stop = StopSignal::on(signals, move || {
            let _ = exited.send(());
        });
        (stop, signal, exit)
    }

    fn names(sink: &MemorySink) -> Vec<String> {
        records(sink).iter().filter_map(|r| r["event_name"].as_str().map(str::to_owned)).collect()
    }

    /// A signal while the start is under way — the load, in the incident —
    /// is acted on at once: `stopping`, then `stopped` naming the phase as
    /// the last line, then the start's own stop (`main`'s exit).
    #[tokio::test]
    async fn a_signal_before_serving_stops_the_start_naming_its_phase() {
        let sink = Arc::new(MemorySink::new());
        let _logs = tracing::subscriber::set_default(tracing_subscriber::registry().with(JsonLayer::new(sink.clone())));
        let (stop, signal, exit) = stop_signal();
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
        let (stop, signal, exit) = stop_signal();
        let shutdown = stop.serving();
        signal.send(()).expect("the watcher waits");
        tokio::time::timeout(Duration::from_secs(5), shutdown).await.expect("the serve's shutdown resolved");
        // No more signals: the watcher ends, its exit dropped unrun.
        drop(signal);
        assert!(exit.await.is_err(), "the process is not stopped from under the serve");
        assert_eq!(names(&sink), ["ignis.process.stopping"]);
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
        let (stop, signal, _exit) = stop_signal();
        let shutdown = stop.serving();
        signal.send(()).expect("the watcher waits");
        // Let the watcher see the signal before the shutdown is first polled.
        tokio::time::sleep(Duration::from_millis(50)).await;
        tokio::time::timeout(Duration::from_secs(5), shutdown).await.expect("the signal was kept");
    }

    /// GitHub #329: a second signal while the serve's shutdown runs — one
    /// that a drain or a teardown holds — ends the process without waiting
    /// for it, `stopped` naming the shutdown it cut short as the last line.
    /// `main`, finding the end taken, logs nothing more.
    #[tokio::test]
    async fn a_second_signal_during_the_shutdown_ends_the_process() {
        let sink = Arc::new(MemorySink::new());
        let _logs = tracing::subscriber::set_default(tracing_subscriber::registry().with(JsonLayer::new(sink.clone())));
        let (stop, signal, exit) = stop_signal();
        let shutdown = stop.serving();
        signal.send(()).expect("the watcher waits");
        tokio::time::timeout(Duration::from_secs(5), shutdown).await.expect("the serve's shutdown resolved");
        signal.send(()).expect("the watcher waits again");
        tokio::time::timeout(Duration::from_secs(5), exit).await.expect("ended at once").expect("the exit ran");

        let lines = records(&sink);
        let names: Vec<&str> = lines.iter().filter_map(|r| r["event_name"].as_str()).collect();
        assert_eq!(names, ["ignis.process.stopping", "ignis.process.stopped"], "{lines:?}");
        assert_eq!(lines[1]["attributes"]["phase"], "shutting_down", "{lines:?}");
        assert!(!stop.ending(), "the end is the second signal's");
    }

    /// Once `main` ends the process itself, its last line its own, a signal
    /// changes nothing: no second `stopped`, no exit from under it.
    #[tokio::test]
    async fn once_main_ends_the_process_a_signal_changes_nothing() {
        let sink = Arc::new(MemorySink::new());
        let _logs = tracing::subscriber::set_default(tracing_subscriber::registry().with(JsonLayer::new(sink.clone())));
        let (stop, signal, exit) = stop_signal();
        let shutdown = stop.serving();
        signal.send(()).expect("the watcher waits");
        tokio::time::timeout(Duration::from_secs(5), shutdown).await.expect("the serve's shutdown resolved");
        assert!(stop.ending(), "the end is main's");
        signal.send(()).expect("the watcher waits again");
        drop(signal);
        assert!(exit.await.is_err(), "no exit from under main");
        assert_eq!(names(&sink), ["ignis.process.stopping"]);
    }

    /// A serve with no exit of its caller's ([`crate::Server::serve_on`])
    /// has nothing a second signal could do: it changes nothing.
    #[tokio::test]
    async fn without_an_exit_a_second_signal_changes_nothing() {
        let sink = Arc::new(MemorySink::new());
        let _logs = tracing::subscriber::set_default(tracing_subscriber::registry().with(JsonLayer::new(sink.clone())));
        let (signal, signals) = tokio::sync::mpsc::unbounded_channel();
        let stop = StopSignal::spawn(signals, Stage::Serving, None);
        let shutdown = stop.serving();
        signal.send(()).expect("the watcher waits");
        tokio::time::timeout(Duration::from_secs(5), shutdown).await.expect("the serve's shutdown resolved");
        signal.send(()).expect("the watcher waits again");
        // The watcher ends once it has read every signal, the second one
        // acted on: what is asserted below is after it, not before.
        drop(signal);
        let mut watcher = stop.stopped.clone();
        let ended = tokio::time::timeout(Duration::from_secs(5), watcher.wait_for(|_| false)).await.expect("the watcher ended");
        assert!(ended.is_err(), "the watcher is gone");
        assert_eq!(names(&sink), ["ignis.process.stopping"]);
        assert!(stop.ending(), "the end is still the caller's");
    }
}
