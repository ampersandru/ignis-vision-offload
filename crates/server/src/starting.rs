//! The API listener while the server starts (GitHub #329 item 18): bound
//! before anything slow, it answers through the start-up download and load
//! — `GET /health` 200, every other path `503 server_not_ready` with
//! `Retry-After: 5` — and is handed to the server once it serves, so a
//! liveness probe never meets a refused connection on a healthy process.
//!
//! Before, the listener was bound only after the load: a Kubernetes liveness
//! probe on `/health` saw "connection refused" for a whole cold load (118 s
//! on the RTX PRO 6000) and had the kubelet kill a process that was fine.

use std::net::SocketAddr;

use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;

/// The start's answers on the API listener, until [`Starting::finish`]
/// takes the listener back for the server.
pub struct Starting {
    local_addr: SocketAddr,
    stop: Option<oneshot::Sender<()>>,
    back: oneshot::Receiver<TcpListener>,
}

impl Starting {
    /// Answer on `listener` from now on: `/health` 200, anything else 503
    /// `server_not_ready` ([`crate::api::starting_router`]). Needs a tokio
    /// runtime: it serves on its own task.
    pub fn answer(listener: TcpListener) -> std::io::Result<Self> {
        use std::future::IntoFuture;

        let local_addr = listener.local_addr()?;
        let (stop, stopped) = oneshot::channel::<()>();
        let (back, returned) = oneshot::channel();
        let lent = Lent { listener: Some(listener), back: Some(back) };
        let shutdown = async move {
            let _ = stopped.await;
        };
        tokio::spawn(axum::serve(lent, crate::api::starting_router()).with_graceful_shutdown(shutdown).into_future());
        Ok(Self { local_addr, stop: Some(stop), back: returned })
    }

    /// Where the listener is bound.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Stop answering and give the listener back, still bound: a connection
    /// that arrives before the server accepts waits in the listen backlog,
    /// never refused. The start's connections are closed gracefully on
    /// their own task (a client's idle keep-alive one included, so it does
    /// not go on reading 503 from the start after the server serves);
    /// this does not wait for them.
    pub async fn finish(mut self) -> TcpListener {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        // axum's graceful serve drops its listener as soon as it stops
        // accepting, before it drains its connections, and the drop hands
        // the listener back ([`Lent`]); were that to move after the drain,
        // this would wait for it, and still come back.
        self.back.await.expect("the start's serve hands its listener back when it drops it")
    }
}

/// The API listener, lent to the start's serve, which owns what it serves
/// on: given back through `back` when the serve drops it — as it stops
/// accepting, or as its task ends whatever the reason.
struct Lent {
    listener: Option<TcpListener>,
    back: Option<oneshot::Sender<TcpListener>>,
}

impl Lent {
    fn listener(&self) -> &TcpListener {
        self.listener.as_ref().expect("lent until dropped")
    }
}

impl axum::serve::Listener for Lent {
    type Io = TcpStream;
    type Addr = SocketAddr;

    fn accept(&mut self) -> impl std::future::Future<Output = (Self::Io, Self::Addr)> + Send {
        // `TcpListener`'s own, which retries a failed accept as axum does.
        axum::serve::Listener::accept(self.listener.as_mut().expect("lent until dropped"))
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        self.listener().local_addr()
    }
}

impl Drop for Lent {
    fn drop(&mut self) {
        if let (Some(listener), Some(back)) = (self.listener.take(), self.back.take()) {
            let _ = back.send(listener);
        }
    }
}
