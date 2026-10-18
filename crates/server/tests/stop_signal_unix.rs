//! GitHub #329 item 17: the process's own signals reach the stop one by one,
//! which is what lets a second SIGTERM end a shutdown the first one began.
//! A test binary of its own: raising SIGTERM leaves tokio's handler
//! installed for the rest of the process, and a test binary that swallows
//! SIGTERM would outlive a harness that stops it so.
#![cfg(unix)]

use std::time::Duration;

use ignis_server::shutdown::StopSignal;

/// The first SIGTERM stops the serve; the second, during its shutdown, runs
/// the exit.
#[tokio::test]
async fn a_second_sigterm_during_the_shutdown_runs_the_exit() {
    let (exited, exit) = tokio::sync::oneshot::channel::<()>();
    let stop = StopSignal::install(move || {
        let _ = exited.send(());
    })
    .expect("handlers");
    let shutdown = stop.serving();
    // SAFETY: a signal to this process, whose handler is registered.
    assert_eq!(unsafe { libc::raise(libc::SIGTERM) }, 0);
    tokio::time::timeout(Duration::from_secs(5), shutdown).await.expect("the first stops the serve");
    // SAFETY: as above.
    assert_eq!(unsafe { libc::raise(libc::SIGTERM) }, 0);
    tokio::time::timeout(Duration::from_secs(5), exit).await.expect("the second ends the process").expect("the exit ran");
}
