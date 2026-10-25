# Testing

How changes to this repo are verified. One rule governs everything
below: every code change ships with a test, and a task is not complete
until the test suite is green.

## The gate

- `cargo test` from the workspace root — all five crates
  (`artifact`, `core`, `server`, `bench`, `vendor`). CI runs exactly this on
  Linux and on Windows, and the release builds do not start until both legs
  are green (`.github/workflows/release.yml`, ADR 0032). That is possible
  only because the default suite is CPU-only; everything below about the GPU
  profile stays on the machine with the 5090.
- Machine-local smoke tests skip gracefully when their fixture is
  absent (e.g. `crates/artifact/tests/real_artifact.rs` needs the
  artifact in `F:\ai\q38`). A skip counts as green; a failure does not.
- **Exception — compute work: a skip is never green.** For anything on the
  forward pass (the kernel leaf, the step ABI, the program), a test that
  self-skips on a busy GPU or on a kernel error proves nothing; that pattern
  is what let a broken forward stay green for two tickets
  (`docs/REVIEW-2026-09-05.md` §4.1). Those tests belong to the explicit
  **GPU profile** (below), which requires the GPU free and *fails* on any
  kernel error, busy GPU, or missing fixture. The default `cargo test` stays
  CPU-only (GitHub #38).
- Kernel work (`kernel/`, CUDA) is not exercised by `cargo test`: build and
  run the leaf's own op-test executable (CTest) with `kernel/build.ps1 -Test`,
  and verify end to end through the GPU profile. The one kernel-adjacent thing
  `cargo test` *does* check is the vendored subtree's integrity
  (`crates/vendor`): a vendored file edited without a recorded patch turns the
  workspace red (ADR 0010, `kernel/vendor/VENDOR.md`).
- Playground work (`web/`, ADR 0026) is not exercised by `cargo test`
  either: its logic lives in pure modules with vitest tests, so a change
  there also needs `npm --prefix web test` and `npm --prefix web run build`
  (the typecheck) green — `make test-web web-build`. `cargo test` covers only the server's `/ui` routes.
- A `--features cuda` build rebuilds the leaf for you.
  `crates/artifact/build.rs` watches every file under `kernel/src`,
  `kernel/include`, `kernel/tests`, `kernel/vendor/src` and
  `kernel/vendor/include` (plus `CMakeLists.txt` and `build.ps1`) and runs
  `kernel/build.ps1` whenever one changes; a build with nothing to do costs
  a few seconds. Do not trust a binary built any other way to carry your
  kernel change: until GitHub #94 that script rebuilt only when
  `kernel/build/*.lib` was *missing*, so an edited `.cu` was linked from
  the previous archive with every test around it still green. That is what
  invalidated the first G2 gate run (#93).

## The GPU test profile (ADR 0006, GitHub #38)

ignis and ninfer (the reference engine, also the coding agent's own model
runner) cannot share the RTX 5090 — a single engine's footprint already
takes ~28 GB of the 32 GB card (ADR 0006). So every GPU-touching Rust test
is `#[ignore]`d and runs only when asked for explicitly: `cargo test` never
touches the GPU on its own.

A GPU test decides what a busy GPU, a kernel error, or a missing fixture
means by asking [`ignis_core::gpu_profile`] — never by returning early on
its own:

- **Outside the profile** (`IGNIS_GPU_PROFILE` unset): those conditions
  print `SKIP: ...` and the test returns — a quick local check without
  stopping ninfer.
- **Under the profile** (`IGNIS_GPU_PROFILE=1`): the *same* conditions are
  hard failures (`panic!`), never a skip. This is the only mode whose green
  result counts for a gate.

Use it as `gpu_profile::check_rc(rc, "ignis_<op>")` for a kernel return
code, `check_compute_err(&e, "...")` for a `ComputeError`, and
`skip_or_fail("...")` for a missing fixture or an absent GPU. Its own unit
tests (in `crates/core/src/gpu_profile.rs`) pin the skip/fail decision on
CPU with an injected rc — no GPU needed, so they run in the default suite.

GPU-gated Rust tests use the `cuda` feature and `#[ignore]`: P1-21's
`gqa_layer_gpu` checks the BF16 exception and NVFP4 GQA layers against the
f64 oracle. They must use the helper above rather than reinventing a
self-skip. The GPU-side coverage also includes the kernel leaf's own op-test
executable (`kernel/build.ps1 -Test`), which the same runbook applies to.

### Running the profile: `scripts/gpu-profile.ps1`

ADR 0006 calls the guard "a preflight check in the `bench`/test harness",
not a script a developer must remember to run standalone. So the profile
takes **two** things, not one: `IGNIS_GPU_PROFILE=1` *and* a preflight pass
on record. `scripts/gpu-preflight.ps1` writes a marker file
(`$env:TEMP\ignis-gpu-preflight.ok`) when it passes and clears it when it
refuses; `active()` reads that marker and **panics** if the variable is set
without a recent one. Setting the variable by hand therefore fails loudly
instead of running un-preflighted while ninfer may hold the card.

`crates/core` has no FFI and no GPU access of its own (GitHub #39 removed
the flat C-ABI surface), so the GPU is inspected by the script and the
verdict is carried across to Rust by that marker.

`scripts/gpu-profile.ps1` is the entry point that ties it together: it runs
the preflight, and only on a pass sets `IGNIS_GPU_PROFILE=1` and runs the
GPU-gated work. It consumes the marker — both it and
the env var are cleared before the script exits, pass or fail — so one
preflight authorizes exactly one run. A pass also ages out after 30 minutes,
which only matters if a run was killed before it could clean up.
**This script is the normal, documented way to run the GPU profile.**

It runs three stages and reports each one's wall time at the end, so a slow
run says where the time went without anyone adding up per-test output
(GitHub #135):

| Stage | What it runs | Parallelism |
|-------|--------------|-------------|
| `kernel/build.ps1 -Test` | the leaf's own op tests (CTest) | CTest's own |
| `cpu f64 layer oracle` | `layer_reference_real`, `--ignored` | libtest default |
| `gpu tests (serialized)` | everything else `--ignored`, but `ignis-speech` and the server's two speech legs that touch no GPU (`$NotGpuLegs`) | `--test-threads=1` |

The middle stage exists because the f64 layer oracle touches no GPU: it
reads the stored weights through a memory map and evaluates them on the CPU.
It was paying for ADR 0006's exclusivity that it never needed, so it now
runs at full parallelism and the serialized stage skips it by name. It still
runs under `IGNIS_GPU_PROFILE=1`, so a missing artifact is a hard failure
there too. The server's two speech legs (the dictations through the route on
the CPU, which need the study disk, and the speech catalog's download) are
skipped by name as well; each binary tests that its leg is listed.

**The test profile is optimized** (`[profile.dev]` and `[profile.test]`,
`opt-level = 2`). The f64 oracles are CPU-bound on scalar `f64` with
per-element NVFP4/W8 dequantization, and at the default `opt-level = 0` they
dominated the serialized stage. Rust applies no fast-math, so every `f64`
result and every measured tolerance is bit-identical to an unoptimized
build; `debug-assertions` and the overflow checks that follow them stay on.
The oracle also spreads its output rows over threads, which for the same
reason cannot move a bit: a row's sum is computed by the same code in the
same order whatever the split, pinned by a unit test that compares schedules
bit-for-bit and by `layer_reference_real` still matching the committed
fixtures, which were recorded by the scalar path.

**"Free" means one process on the card, not just "ninfer is stopped."** The
preflight inspects the GPU once, before the run starts; nothing stops a
second GPU-touching process from entering while the sweep is under way. One
clean profile run peaks at **26.1 GiB of the 5090's 32.6 GiB** (measured
2026-09-13, sampled every 15s across two consecutive runs), so the headroom
left for anything else is about 6 GiB — less than one more artifact load. A
test binary that loses that race dies the way GitHub #145 recorded it: an
abrupt process exit, `exit code: 1`, no panic text and nothing printed
despite `--nocapture`, and the same test passing immediately when re-run
alone. Two unrelated tests died that way in one session, which is what a
contended card looks like from inside a test log — there is no diagnostic
that says so.

So before launching **any** GPU work — the profile, a bench run, a gate
leg, a single `--ignored` test — check that nobody has already launched
some. That includes another agent session in another worktree: the
worktrees are separate, the card is not. `nvidia-smi` (or
`scripts/gpu-preflight.ps1`) plus `tasklist` for stray `*_gpu-*.exe`,
`ignis-server`, `ignis-bench` and `ninfer*` processes answers it in one
step; `make gpu-status` prints both, and `make gpu-guard` turns them into a
pass/fail.

Runbook:

```powershell
# 0. Check nobody else is already running GPU work -- another agent session
#    in another worktree counts. nvidia-smi for held memory, tasklist for
#    stray *_gpu-*.exe / ignis-server / ignis-bench / ninfer processes.
# 1. Stop ninfer (frees the VRAM the GPU profile needs).
# 2. Preflight + profile in one step: refuses to proceed while the GPU is
#    held (and names the offending process); on a free GPU, runs the leaf's
#    op tests and the #[ignore]d Rust GPU tests under IGNIS_GPU_PROFILE=1,
#    then clears it.
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gpu-profile.ps1
# -ThresholdMiB <n>    forwarded to gpu-preflight.ps1
# -SkipKernelBuild     Rust tests only, skip kernel/build.ps1 -Test
# -SkipCargoTests      kernel leaf only, skip both cargo stages
# 3. Restart ninfer.
```

`make gpu-profile` runs the same script; `GPU_PROFILE_ARGS` forwards its
switches (`make gpu-profile GPU_PROFILE_ARGS=-SkipKernelBuild`).

`scripts/gpu-preflight.ps1` still exists as the standalone check
(`scripts/gpu-profile.ps1` calls it) for a quick "is the GPU free" query
that doesn't run anything.

## Testing one area

- Recording the canary oracle fixture or comparing an engine against it (G1),
  or reading the canary suite's thinking-only verdicts:
  [testing/canary-oracle.md](testing/canary-oracle.md).
- Recording or scoring the vision canary fixture (multimodal prompts against
  the reference's answers): [testing/vision-canary.md](testing/vision-canary.md).
- Testing `/v1/decide` (`locate` over a text state, the shortlist, the
  entailment fixture): [testing/decide.md](testing/decide.md).
- Testing the image model's media against Pillow (the resize fixtures, the
  WebP condition image): [testing/image-model.md](testing/image-model.md).
- Testing the speech model (the speech oracle, `server_vad`'s VAD oracle, their
  legs on the study disk): [testing/speech.md](testing/speech.md).
- Running a bench or a gate leg (TTFT and G2, the multimodal TTFT cell, the
  KV-format A/B, the harness's request deadline):
  [testing/bench.md](testing/bench.md).
- Writing a GPU test of a load option (the engine-shape flags, speculation,
  vision) or one that compares output or carries a tolerance (the loop guard
  off, BF16 as the oracle format):
  [testing/bench.md](testing/bench.md#the-g2-measurement-instrument-p2-05-github-87).

## Where tests live

- Unit tests: `#[cfg(test)]` modules, in the file they test.
- Integration tests: `crates/<crate>/tests/`, one file per behavior,
  named for what it verifies (cf. `real_artifact.rs`).

## Writing the tests

- New behavior: its test lands with the change, before the task is
  reported done — and it exercises the new path, not the old world
  (a test that already passed before the change proves nothing about
  it).
- Bug fix: the regression test goes first — reproduce the bug, watch
  the test go red, then fix.
- The standard loop is red-green, driven by the `/tdd` skill.
- Logging on the prefill/decode path: the hotpath lint
  (`crates/logging/src/hotpath_lint.rs`) fails `cargo test` on an
  INFO-or-above `tracing` call or a raw `println!`/`eprintln!`/`print!` in
  its hand-maintained list of hot-path files. It makes the hot-path logging
  constraint checkable in review, not only provable later at the G4 gate; a
  `hotpath-lint-allow:` comment marks a reviewed, intentional exception, never
  a silent one.

## Done means

1. every new or changed behavior has a test that covers it
2. `cargo test` passes workspace-wide (machine-local skips allowed)
3. `cargo check --workspace --features cuda,logits-capture --tests` is
   clean (`make check CUDA=1`). The
   cuda-gated test targets (`#![cfg(feature = "cuda")]`) are the ones
   `cargo test` never compiles, so a signature change can leave them
   uncompilable and nothing says so until someone runs the GPU profile
   — which needs a free 5090 and so runs rarely. GitHub #132 did exactly
   that: it added a third argument to `apply_chat_template`, missed two
   call sites under `crates/server/tests/`, and blocked the next GPU
   profile outright (found and fixed in #119). This check costs ~30s,
   needs no GPU, and reports *every* broken site rather than stopping at
   the first — which is how the second of those two was found at all.
   `logits-capture` adds the one GPU test the profile never builds
   (`flash_next_rope_logits_gpu.rs`, bit-exact on the card that recorded
   its capture, ADR 0059), which nothing else would compile.
4. kernel changes: `kernel/build.ps1` clean + the leaf's op tests green +
   the GPU profile green on a free 5090 (`scripts/gpu-profile.ps1` —
   preflight passed, `IGNIS_GPU_PROFILE=1` for the run — never a skip)
