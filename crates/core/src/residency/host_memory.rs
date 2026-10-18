//! The host plan's one measured input: physical memory available now
//! (spec flash-next/03, acceptance 1). Windows: `GlobalMemoryStatusEx`'s
//! `ullAvailPhys` — free plus standby pages, which Windows hands out before
//! it pages anything. Linux: `/proc/meminfo`'s `MemAvailable`, capped by
//! every cgroup v2 memory limit above the process (a container's
//! `memory.max`): `/proc/meminfo` is the whole machine's, and a load that
//! plans against it inside a smaller limit is OOM-killed instead of refused.
//! Elsewhere, or when the query fails, `None`: the caller decides what an
//! unmeasured machine means.

/// Bytes of physical memory available for a new allocation.
pub fn available_physical_bytes() -> Option<u64> {
    imp::available_physical_bytes()
}

#[cfg(windows)]
mod imp {
    /// `MEMORYSTATUSEX` (sysinfoapi.h).
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    pub fn available_physical_bytes() -> Option<u64> {
        let mut status = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: `status` is a properly sized, writable MEMORYSTATUSEX whose
        // `dwLength` is set, as the API requires.
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        (ok != 0).then_some(status.avail_phys)
    }
}

#[cfg(target_os = "linux")]
mod imp {
    pub fn available_physical_bytes() -> Option<u64> {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let machine = super::parse::mem_available(&meminfo)?;
        let own = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
        let headrooms = super::cgroup::headrooms(std::path::Path::new("/sys/fs/cgroup"), &own);
        Some(super::capped_by(machine, headrooms))
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod imp {
    pub fn available_physical_bytes() -> Option<u64> {
        None
    }
}

/// The machine's available memory, capped by every cgroup level's.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn capped_by(machine: u64, headrooms: impl IntoIterator<Item = u64>) -> u64 {
    headrooms.into_iter().fold(machine, u64::min)
}

/// The cgroup v2 walk, on plain files (compiled on every host, so a
/// temporary tree tests it everywhere; Linux runs it on `/sys/fs/cgroup`).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
mod cgroup {
    use std::path::{Component, Path, PathBuf};

    /// What each cgroup v2 level from the process's own up to `root` still
    /// lets it take, the process's own level first; a level with no limit
    /// (or no memory files: the host's root cgroup) gives none. `own` is
    /// `/proc/self/cgroup`. Inside a container with its own cgroup
    /// namespace that reads `0::/`, and `root` *is* the container's cgroup.
    /// cgroup v1 (no unified line), a path that leaves `root`, or one that
    /// does not exist under it walk `root` alone.
    pub fn headrooms(root: &Path, own: &str) -> Vec<u64> {
        let relative = Path::new(super::parse::cgroup_v2_path(own).unwrap_or("/"));
        let inside = relative.components().all(|c| matches!(c, Component::RootDir | Component::Normal(_)));
        let mut dir: PathBuf = root.join(relative.strip_prefix("/").unwrap_or(relative));
        if !inside || !dir.is_dir() {
            dir = root.to_path_buf();
        }
        let mut headrooms = Vec::new();
        loop {
            if let Some(headroom) = headroom_at(&dir) {
                headrooms.push(headroom);
            }
            if dir == root || !dir.pop() {
                break;
            }
        }
        headrooms
    }

    fn headroom_at(dir: &Path) -> Option<u64> {
        let max = std::fs::read_to_string(dir.join("memory.max")).ok()?;
        let current = std::fs::read_to_string(dir.join("memory.current")).ok()?;
        let stat = std::fs::read_to_string(dir.join("memory.stat")).unwrap_or_default();
        super::parse::cgroup_headroom(&max, &current, &stat)
    }

    #[cfg(test)]
    mod tests {
        use super::headrooms;
        use std::path::PathBuf;

        /// A cgroup tree under the temp dir: each level its memory files, or
        /// none for a level written as `None`.
        fn tree(name: &str, levels: &[(&str, Option<(&str, &str, &str)>)]) -> PathBuf {
            let root = std::env::temp_dir().join(format!("ignis-cgroup-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            for (dir, files) in levels {
                let at = root.join(dir);
                std::fs::create_dir_all(&at).expect("mkdir");
                if let Some((max, current, stat)) = files {
                    std::fs::write(at.join("memory.max"), max).expect("max");
                    std::fs::write(at.join("memory.current"), current).expect("current");
                    std::fs::write(at.join("memory.stat"), stat).expect("stat");
                }
            }
            root
        }

        #[test]
        fn a_namespaced_container_reads_its_own_root() {
            let root = tree("ns", &[("", Some(("1000\n", "400\n", "active_file 100\ninactive_file 100\n")))]);
            assert_eq!(headrooms(&root, "0::/\n"), vec![800]);
            std::fs::remove_dir_all(&root).ok();
        }

        #[test]
        fn every_level_up_to_the_root_caps_and_an_unlimited_one_does_not() {
            // A pod level limited at 2000, a container level at 1000, the
            // host root with no memory files, a slice unlimited.
            let root = tree(
                "walk",
                &[
                    ("", None),
                    ("kubepods.slice", Some(("max\n", "999\n", ""))),
                    ("kubepods.slice/pod", Some(("2000\n", "500\n", ""))),
                    ("kubepods.slice/pod/crio-1.scope", Some(("1000\n", "300\n", ""))),
                ],
            );
            assert_eq!(headrooms(&root, "0::/kubepods.slice/pod/crio-1.scope\n"), vec![700, 1500]);
            std::fs::remove_dir_all(&root).ok();
        }

        #[test]
        fn a_path_that_leaves_the_root_or_is_missing_or_v1_walks_the_root_alone() {
            let root = tree("odd", &[("", Some(("1000\n", "0\n", "")))]);
            assert_eq!(headrooms(&root, "0::/../../etc\n"), vec![1000]);
            assert_eq!(headrooms(&root, "0::/not/there\n"), vec![1000]);
            assert_eq!(headrooms(&root, "12:memory:/docker/abc\n"), vec![1000]);
            // A legal name with two dots in it is a name, not a parent.
            let dotted = tree("dots", &[("", None), ("a..b", Some(("500\n", "100\n", "")))]);
            assert_eq!(headrooms(&dotted, "0::/a..b\n"), vec![400]);
            std::fs::remove_dir_all(&root).ok();
            std::fs::remove_dir_all(&dotted).ok();
        }
    }
}

/// The Linux readers' parsing, kept off the filesystem (and compiled on
/// every host) so it is tested everywhere.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
mod parse {
    /// `MemAvailable:   12345678 kB` → bytes.
    pub fn mem_available(meminfo: &str) -> Option<u64> {
        let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        kib.checked_mul(1024)
    }

    /// `/proc/self/cgroup`'s unified (v2) line, `0::<path>` → `<path>`.
    pub fn cgroup_v2_path(proc_self_cgroup: &str) -> Option<&str> {
        proc_self_cgroup.lines().find_map(|l| l.strip_prefix("0::")).map(str::trim)
    }

    /// What one cgroup level still lets its processes take: `memory.max`
    /// less what the kernel cannot reclaim, `memory.current` less the page
    /// cache (`active_file` + `inactive_file`), which the kernel drops before
    /// it OOM-kills. The cache counts as available, as `MemAvailable` counts
    /// it: a container that just read a 70 GB artifact has most of its
    /// limit charged to that file's pages. `max` (no limit) or an unreadable
    /// value is `None`.
    pub fn cgroup_headroom(max: &str, current: &str, stat: &str) -> Option<u64> {
        let max = max.trim();
        if max == "max" {
            return None;
        }
        let max: u64 = max.parse().ok()?;
        let current: u64 = current.trim().parse().ok()?;
        let stat_value = |key: &str| {
            stat.lines()
                .find_map(|l| l.strip_prefix(key)?.strip_prefix(' '))
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(0)
        };
        let page_cache = stat_value("active_file").saturating_add(stat_value("inactive_file"));
        Some(max.saturating_sub(current.saturating_sub(page_cache)))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn mem_available_is_read_in_kib() {
            let text = "MemTotal:       65536000 kB\nMemAvailable:   52428800 kB\n";
            assert_eq!(mem_available(text), Some(52_428_800 * 1024));
            assert_eq!(mem_available("MemTotal: 1 kB\n"), None);
        }

        #[test]
        fn the_unified_line_names_the_cgroup() {
            // A Kubernetes pod without its own cgroup namespace, then one with.
            let v2 = "0::/kubepods.slice/kubepods-burstable.slice/crio-abc.scope\n";
            assert_eq!(cgroup_v2_path(v2), Some("/kubepods.slice/kubepods-burstable.slice/crio-abc.scope"));
            assert_eq!(cgroup_v2_path("0::/\n"), Some("/"));
            // cgroup v1 only: no unified line.
            assert_eq!(cgroup_v2_path("12:memory:/docker/abc\n11:cpu:/docker/abc\n"), None);
        }

        #[test]
        fn a_limit_less_what_cannot_be_reclaimed_is_the_headroom() {
            let gib = 1u64 << 30;
            // 112 GiB limit, 60 GiB charged of which 40 GiB is the artifact's
            // page cache: 20 GiB pinned/anon, 92 GiB still to take.
            let stat = format!("anon {}\nfile {}\nactive_file {}\ninactive_file {}\n", 20 * gib, 40 * gib, 15 * gib, 25 * gib);
            assert_eq!(cgroup_headroom(&format!("{}\n", 112 * gib), &format!("{}\n", 60 * gib), &stat), Some(92 * gib));
            // No limit at this level.
            assert_eq!(cgroup_headroom("max\n", "123\n", ""), None);
            // No memory.stat: everything charged counts as taken.
            assert_eq!(cgroup_headroom("1000", "400", ""), Some(600));
            // Over the limit (charged before the limit was lowered): nothing.
            assert_eq!(cgroup_headroom("100", "250", ""), Some(0));
            // A key that only starts like another is not it.
            assert_eq!(cgroup_headroom("1000", "400", "active_file_extra 300\n"), Some(600));
            assert_eq!(cgroup_headroom("garbage", "1", ""), None);
        }

        #[test]
        fn the_tightest_level_caps_the_machine() {
            use super::super::capped_by;
            let gib = 1u64 << 30;
            // anvil-serving's recipe: a 52 GiB container limit on a host
            // reporting ~93 GB available.
            assert_eq!(capped_by(93 * 1_000_000_000, [52 * gib]), 52 * gib);
            // Pod and container levels: the smaller wins.
            assert_eq!(capped_by(126 * gib, [112 * gib, 100 * gib]), 100 * gib);
            // No limits: the machine's figure.
            assert_eq!(capped_by(64 * gib, []), 64 * gib);
        }
    }
}
