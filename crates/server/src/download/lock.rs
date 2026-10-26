//! One writer per entry (GitHub #379, spec #374): the server — its Server
//! view's Download and its start — and `ignis-server model download` never
//! fetch the same catalog entry into the same folder at once. Two writers on
//! one `.part` would truncate and append each other's bytes, and each would
//! check only the digest of what it sent itself.
//!
//! The lock is the operating system's, on a file of its own in the models
//! folder, `<artifact>.lock` beside the entry's files: `flock` on Linux and
//! macOS, `LockFileEx` on Windows, through [`std::fs::File::try_lock`]. The
//! kernel holds it for the open file and drops it when the file is closed —
//! when the fetch ends, whichever way, and when the process holding it dies,
//! killed or crashed — so a lock never outlives its holder. The file itself
//! stays, empty, for the next fetch of the entry to lock again: deleting it
//! on the way out would let a taker that opened it just before lock a file no
//! longer in the folder while a third locks a new one under the same name.

use std::fs::{File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};

use super::catalog::CatalogEntry;
use super::transfer::DownloadError;

/// An entry's lock, held until it is dropped.
#[derive(Debug)]
pub struct EntryLock {
    /// The open lock file: closing it releases the lock.
    _file: File,
}

/// Why [`EntryLock::take`] took no lock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotLocked {
    /// Another fetch of the entry into the folder holds it: the entry is
    /// being downloaded.
    Held(DownloadError),
    /// The folder or the lock file could not be opened, for this reason.
    Failed(DownloadError),
}

impl From<NotLocked> for DownloadError {
    fn from(refused: NotLocked) -> Self {
        match refused {
            NotLocked::Held(err) | NotLocked::Failed(err) => err,
        }
    }
}

impl EntryLock {
    /// The file `entry`'s lock is held on under `dir`.
    pub fn path(entry: &CatalogEntry, dir: &Path) -> PathBuf {
        let mut name = entry.artifact_path(dir).into_os_string();
        name.push(".lock");
        PathBuf::from(name)
    }

    /// Take `entry`'s lock under `dir`, creating the folder when it is not
    /// there yet; refused at once, never waited for, when another fetch
    /// holds it.
    pub fn take(entry: &CatalogEntry, dir: &Path) -> Result<Self, NotLocked> {
        let failed = |what: String| NotLocked::Failed(DownloadError(what));
        std::fs::create_dir_all(dir).map_err(|err| failed(format!("cannot create {}: {err}", dir.display())))?;
        let path = Self::path(entry, dir);
        // Read and write: `LockFileEx` locks only a handle opened for one of them.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|err| failed(format!("cannot open the lock {}: {err}", path.display())))?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file }),
            Err(TryLockError::WouldBlock) => Err(NotLocked::Held(DownloadError(format!(
                "{} is already being downloaded into {}, by a server or by `ignis-server model download`, which holds {}: one writer per entry, so this one fetched nothing; try again once that download ends",
                entry.id,
                dir.display(),
                path.display()
            )))),
            Err(TryLockError::Error(err)) => Err(failed(format!("cannot lock {}: {err}", path.display()))),
        }
    }
}
