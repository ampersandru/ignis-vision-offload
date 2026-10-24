//! ignis-speech: the speech model's CPU executor, ours in Rust (ADR 0067,
//! spec `docs/specs/speech/01-cpu-executor.md`).
//!
//! Qwen3-ASR on the CPU: our GGUF reader, the activation quantizers and the
//! Q4_K / Q6_K / Q8_0 / F16 / F32 matmul kernels (a scalar reference
//! everywhere, AVX2 kernels picked at run time on x86-64), our interleaved
//! Q4_K layout and thread pool, the AuT encoder and Qwen3 decoder graphs, the
//! log-mel front end, and the incremental API the server drives.
//!
//! Provenance: every kernel and op is our own (ADR 0051's first tier). The
//! GGUF block layouts and their quantization rules are the file format the
//! kernels read, not copied code; nothing from ggml or llama.cpp is in the
//! tree or the build.

#[cfg(target_arch = "x86_64")]
pub mod avx2;
pub mod cpu;
pub mod gguf;
pub mod math;
pub mod matmul;
pub mod mem;
pub mod ops;
pub mod pool;
pub mod quant;

#[cfg(test)]
pub(crate) mod testutil;
