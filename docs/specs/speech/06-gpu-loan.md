# 06 - phase 3: the GPU loan

GitHub: to be filed. Phase 3 (Q33, Q56). Feature: `spec.md`. Needs 01-03.

ADRs: 0066 (the loan), 0067 (one pipeline, two compute adapters), 0030
(nothing allocated on the device while serving), 0045 (KV pool policy,
eviction priority), 0049 (idle unload), 0051 (kernel provenance).

Decided 2026-10-23/24 in the speech grill. Details the grill left open were
decided by the owner on 2026-10-24, cited as `(owner, 2026-10-24)`.

## Problem Statement

- **The CPU path is the baseline, and it is slow next to the card.** On the
  i9, 1.7B Q4_K_M transcribes a 60 s clip whole in 12.3 s. Incremental
  transcription hides most of it, but each transcription holds all but one
  core. The 5090 could do the same work in well under a second.
- **The owner's rule** (2026-10-23): the speech model is never kept in VRAM.
  VRAM is lent to it only while it transcribes, then given back.
- **The KV pool already has the mechanism.** A sequence reserves pages
  (`kv_pool.reserve()`, `kernel/src/seq.cu`), and the pool reports its free
  pages (`ignis_seq_pool_stats`). A loan reserves like a sequence, so ADR
  0030 holds: nothing is allocated while serving.
- **Pages are not contiguous.** Under hq-e8-2b a page is 64 tokens of
  9,216 B, 576 KiB, and a loan is a scatter of them. The speech kernels must
  address their weights, cache and workspace through a page table. Kernel
  speed is not the bar; latency is (Q26).
- **No leaf kernel fits as it is** (the leaf inventory):
  - the audio encoder's heads are 64 wide, and no attention kernel takes
    D64;
  - the decoder's KV decode attention at 16/8/128 does not exist;
  - the Qwen3 dense layer of the image text encoder has its geometry fixed in
    `#define`s;
  - the text models' decode graphs capture on a blocking stream
    (`model.cu:1612`).

## Solution

### The loan (Q30, Q32, Q40)

- **When:** at the signal, which is a Realtime session opening, a mic press
  or a batch request (Q32).
- **What it takes:** free pages of the text model's KV pool, reserved as a
  sequence's are. Never a live reservation, and never pages held by
  retained state, which is not pushed down for it. When the free pages are
  not enough, the transcription runs on the CPU (Q30).
- **What lives on it:**
  - the weights: the same GGUF the CPU reads, whatever the variant, so a
    model is one download (Q46). They are uploaded from the mapped file at
    the signal, so the first window does not wait for them;
  - the transcription's own KV;
  - its workspace.
  Nothing else is allocated on the device (ADR 0030).
- **Without a CPU executor** (no AVX2), a transcription waits up to
  `speech.loan_wait_seconds` (5) for a loan to fit, then gets `503` with
  `Retry-After` (Q35; owner, 2026-10-24).
- **How long:**
  - held for `speech.loan_grace_seconds` (30) after its last transcription;
  - given back at once when the text model needs pages and no transcription
    is running;
  - a running transcription finishes first, bounded by its deadline (Q32).
- **Where not:** there is no loan without a text model's KV pool. With the
  image model loaded, during a switch, or with the text model unloaded,
  transcription runs on the CPU. A switch waits for a running loan, bounded
  by its deadline (Q40). An idle unload waits for it the same way, since it
  tears the model down as a switch does (ADR 0049; owner, 2026-10-24).
- **Priority:** transcription goes before text decode (Q14). The speech
  stream gets a higher CUDA stream priority (owner, 2026-10-24). Q26 bounds
  the cost.
- **The VRAM plan gains no line:** the loan lives inside the KV pool's. Lent
  pages count as used in `ignis_kv_pool_used_pages`, and also on a gauge of
  their own (owner, 2026-10-24).

### The CUDA adapter

One pipeline, two compute adapters (Q41, ADR 0067). This one runs 01's
session on the loan:
- the front end stays on the host (01's Rust mel);
- the program is the image model's pattern: a flat C ABI per stage, driven
  by Rust, with its own arena, here placed on the lent pages (the leaf
  inventory);
- the decoder (~350 launches a token) is captured as a CUDA graph on a
  non-blocking stream.

**The kernels read the GGUF block format** (Q46): Q4_K, Q6_K and Q8_0 as
the files hold them, uploaded from the mapped GGUF and not from the CPU
executor's interleaved Q4_K copy, with page-table addressing we write
ourselves. GPU decode is bound by memory bandwidth, so int8 and Q4_K do as
well as FP8 there, and there is no conversion step. The detail is decided in
this spec after measurement (Q46).

**The loan builds on our kernel leaf alone** (Q62). ggml-cuda is not a base:
nothing is copied or adapted from it.

Over PCIe Gen 3 (~12 GB/s on the owner's host), the 1.64 GB of
`qwen3-asr-1.7b` uploads in about 140 ms.

Starting points from the leaf inventory, to be confirmed by that
measurement:
- **The decoder layer:** the image text encoder's Qwen3 layer in
  `kernel/src/image/encoder.cu` (RMSNorm, q/k norm, interleaved M-RoPE
  [24, 20, 20] θ 1e6, causal GQA prefill, SwiGLU), adapted and parametrised:
  its geometry is `#define`d for the image model today (Q62).
- **New, ours** (Q62):
  - encoder attention at D64, group 1, per 8 s window (re-templated from
    `encoder_attn.cu`);
  - decode attention 16/8/128 over the transcription's contiguous cache
    (adapted from Flash-Next's dense QSA, or from `encoder_attn` with
    `q_len != kv_len`);
  - a conv stem variant, adapted from `ignis_image_vae_conv` (stride 2,
    pad 1, odd sizes, a 1-channel input padded to 4);
  - quantized GEMV and GEMM that read the GGUF's Q4_K, Q6_K and Q8_0 blocks
    through the page table.
- **As they are:** argmax sampling, embedding, RMSNorm, LayerNorm, exact
  GELU, residual add, bias add.

## Acceptance

1. A loan reserves only free pages. With a pool at a known count of free
   pages, it takes them when enough are free and falls back to the CPU when
   they are not. It never evicts, moves or pushes down anything (Q30).
2. A loan starts at the signal and is held for `speech.loan_grace_seconds`.
   It is returned at once when the text model needs pages with no
   transcription running. A running transcription finishes first, within its
   deadline (Q32).
3. Across a loaned transcription, the device allocation counter
   (`alloc_counter_baseline_gpu.rs`'s check) does not move (Q30).
4. With the image model loaded, during a switch, and with the text model
   unloaded, transcription runs on the CPU. A switch waits for a running
   loan (Q40).
5. Through the CUDA adapter, 01's oracle holds: the per-layer thresholds,
   teacher-forced top-1 agreement, and the 1-pt WER gate (Q52).
6. On the 5090, after the signal's upload, a 60 s dictation is transcribed
   in under 1 s (Q26).
7. On the 5090, the 27B's greedy decode is less than 10% slower during a
   loaned transcription. If 6 or 7 fails, the kernels are optimised until
   both hold, and not before (Q26).
8. Outside a loan the weights stay in the mapped file, and no VRAM is held
   (Q31, Q32).
9. With the 16 GB profile under its budget, the share of transcriptions the
   loan takes is measured. If it rarely fits and the CPU lacks AVX2, the
   speech model is reported unavailable. Without AVX2, a transcription waits
   up to `speech.loan_wait_seconds` (5) for a loan, then gets `503` with
   `Retry-After` (Q35; owner, 2026-10-24).
10. The loan reads the GGUF files the CPU executor reads, with no other
    download or conversion. Its kernels read the GGUF blocks through the page
    table (Q46). They build on our leaf alone: nothing is copied or adapted
    from ggml-cuda, and the new kernels (D64 encoder attention, decode
    attention at 16/8/128, the conv stem variant, the Q4_K/Q6_K/Q8_0 GEMV and
    GEMM) are ours (Q62; ADR 0051).
11. A loaned transcription is logged and counted with `executor = gpu_loan`
    on spec 02's log line and metrics (Q45). `/metrics` also counts loans
    granted and refused, and gauges the lent pages (owner, 2026-10-24).
12. GPU runs follow `spec.md` (gpu-status, lock, one run on the card), and
    `cargo test` passes workspace-wide (AGENTS.md, not a grill decision).

## Out of Scope

- Lending anything but free KV pages, for example the expert cache or a
  workspace (Q30).
- Keeping the speech model resident (ADR 0066).
- `server_vad`: spec 05, phase 2 (Q56).
