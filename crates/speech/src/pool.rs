//! Our own thread pool (spec speech/01 §1, Q38, the spike): the calling
//! thread is worker 0, tasks are claimed from an atomic counter, and the
//! other workers spin, then park.
//!
//! rayon dispatched decode's 197 matmuls a token at 1.184 of ggml's time on
//! 4 cores; this shape ran them at 1.014 (ADR 0068). Between units of work
//! every worker parks, so an idle executor keeps no thread busy.

use std::cell::UnsafeCell;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::*};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle, Thread};

type Job = &'static (dyn Fn(usize) + Sync);

struct Shared {
    epoch: AtomicUsize,
    job: UnsafeCell<Option<Job>>,
    next: AtomicUsize,
    ntasks: AtomicUsize,
    done: AtomicUsize,
    sleepers: AtomicUsize,
    panicked: AtomicBool,
    quit: AtomicBool,
    threads: OnceLock<Vec<Thread>>,
}

// SAFETY: `job` is written only by the dispatching thread while every worker waits on
// `epoch`, and read by workers only after the epoch's release.
unsafe impl Sync for Shared {}

/// `pause` iterations an idle worker spins before it parks: a few milliseconds.
const SPIN_LIMIT: u32 = 100_000;

pub struct Pool {
    sh: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
    n: usize,
    /// One dispatch at a time: the job slot is shared.
    dispatch: Mutex<()>,
}

impl Pool {
    /// A pool of `n` workers, the caller of [`run`](Self::run) included.
    pub fn new(n: usize) -> Pool {
        let n = n.max(1);
        let sh = Arc::new(Shared {
            epoch: AtomicUsize::new(0),
            job: UnsafeCell::new(None),
            next: AtomicUsize::new(0),
            ntasks: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            quit: AtomicBool::new(false),
            threads: OnceLock::new(),
        });
        let handles: Vec<_> = (1..n)
            .map(|i| {
                let s = sh.clone();
                thread::Builder::new().name(format!("ignis-speech-{i}")).spawn(move || worker(&s)).expect("spawn a speech worker")
            })
            .collect();
        sh.threads.set(handles.iter().map(|h| h.thread().clone()).collect()).ok();
        Pool { sh, handles, n, dispatch: Mutex::new(()) }
    }

    pub fn threads(&self) -> usize {
        self.n
    }

    /// Workers currently parked (or committed to parking): `threads() − 1` once the pool is idle.
    pub fn parked(&self) -> usize {
        self.sh.sleepers.load(SeqCst)
    }

    /// Runs `f(0..ntasks)` across the pool, tasks claimed dynamically, and returns when all are done.
    /// A panic in any task is raised here once every worker has left the job.
    pub fn run(&self, ntasks: usize, f: &(dyn Fn(usize) + Sync)) {
        if ntasks == 0 {
            return;
        }
        if self.n == 1 || ntasks == 1 {
            (0..ntasks).for_each(f);
            return;
        }
        let _one = self.dispatch.lock().unwrap_or_else(|e| e.into_inner());
        let s = &*self.sh;
        // SAFETY: the 'static job never outlives this call: it returns (or unwinds) only after
        // every worker has counted itself done with it.
        unsafe { *s.job.get() = Some(std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(f)) };
        s.ntasks.store(ntasks, Relaxed);
        s.next.store(0, Relaxed);
        s.done.store(0, Relaxed);
        s.panicked.store(false, Relaxed);
        s.epoch.fetch_add(1, SeqCst);
        if s.sleepers.load(SeqCst) > 0 {
            for t in s.threads.get().unwrap() {
                t.unpark();
            }
        }
        let mine = catch_unwind(AssertUnwindSafe(|| claim(s, ntasks, f)));
        while s.done.load(Acquire) < self.n - 1 {
            std::hint::spin_loop();
        }
        // SAFETY: every worker is done with the job.
        unsafe { *s.job.get() = None };
        if let Err(p) = mine {
            resume_unwind(p);
        }
        if s.panicked.load(Relaxed) {
            panic!("a speech pool worker panicked");
        }
    }
}

fn claim(s: &Shared, ntasks: usize, f: &(dyn Fn(usize) + Sync)) {
    loop {
        let t = s.next.fetch_add(1, Relaxed);
        if t >= ntasks {
            break;
        }
        f(t);
    }
}

fn worker(s: &Shared) {
    let mut seen = 0usize;
    loop {
        let mut spins = 0u32;
        loop {
            let e = s.epoch.load(Acquire);
            if e != seen {
                seen = e;
                break;
            }
            if s.quit.load(Relaxed) {
                return;
            }
            if spins < SPIN_LIMIT {
                spins += 1;
                std::hint::spin_loop();
            } else {
                // Dekker with `run`: either it sees this sleeper and unparks, or this sees its epoch.
                s.sleepers.fetch_add(1, SeqCst);
                if s.epoch.load(SeqCst) == seen && !s.quit.load(SeqCst) {
                    thread::park();
                }
                s.sleepers.fetch_sub(1, SeqCst);
            }
        }
        // SAFETY: published before the epoch this worker just acquired.
        let f = unsafe { (*s.job.get()).expect("a job with the epoch") };
        let nt = s.ntasks.load(Relaxed);
        if catch_unwind(AssertUnwindSafe(|| claim(s, nt, f))).is_err() {
            s.panicked.store(true, Relaxed);
        }
        s.done.fetch_add(1, Release);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.sh.quit.store(true, SeqCst);
        for t in self.sh.threads.get().unwrap() {
            t.unpark();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn every_task_runs_once_across_the_workers() {
        let pool = Pool::new(4);
        for ntasks in [1, 3, 4, 17, 1000] {
            let hits: Vec<AtomicUsize> = (0..ntasks).map(|_| AtomicUsize::new(0)).collect();
            pool.run(ntasks, &|t| {
                hits[t].fetch_add(1, Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Relaxed) == 1), "{ntasks} tasks");
        }
    }

    /// Acceptance 8: between units of work every worker parks.
    #[test]
    fn an_idle_pool_parks_every_worker() {
        let pool = Pool::new(4);
        let sum = AtomicUsize::new(0);
        pool.run(64, &|t| {
            sum.fetch_add(t, Relaxed);
        });
        assert_eq!(sum.load(Relaxed), 64 * 63 / 2);
        let t0 = Instant::now();
        while pool.parked() < 3 {
            assert!(t0.elapsed() < Duration::from_secs(5), "only {} of 3 workers parked", pool.parked());
            thread::sleep(Duration::from_millis(5));
        }
        // A parked pool still runs the next unit.
        pool.run(8, &|_| {
            sum.fetch_add(1, Relaxed);
        });
        assert_eq!(sum.load(Relaxed), 64 * 63 / 2 + 8);
    }

    #[test]
    fn a_panic_in_a_task_reaches_the_caller_and_the_pool_survives() {
        let pool = Pool::new(3);
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.run(32, &|t| {
                if t == 31 {
                    panic!("task 31");
                }
            })
        }));
        assert!(r.is_err());
        let n = AtomicUsize::new(0);
        pool.run(10, &|_| {
            n.fetch_add(1, Relaxed);
        });
        assert_eq!(n.load(Relaxed), 10);
    }
}
