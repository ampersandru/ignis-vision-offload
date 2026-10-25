# 01 - the CPU executor: our Rust kernels and graphs, the pipeline, the oracle

GitHub: #351 (master #350). Phase 1. Feature: `spec.md`.

ADRs: 0068 (the CPU executor is ours, in Rust), 0067 (the speech model), 0051
(ours / adapted / vendored), 0043 (no reference engine: the checkpoint's own
modelling code is the oracle), 0032 (the default build is pure Rust), 0047
(catalog).

Decided 2026-10-23/24 in the speech grill. Details the grill left open were
decided by the owner on 2026-10-24, cited as `(owner, 2026-10-24)`. The
executor itself follows the owner's review of the vendoring (Q59-Q63) and the
spike it asked for (the finding
[Rust quantized matmul matches ggml on the CPU](../../findings/2026-10-24-rust-quantized-matmul-matches-ggml.md),
cited as "the spike"). The details that review left open are Q64-Q67.

## Problem Statement

- **ignis runs no model on the CPU.** The CPU tokenizes, renders templates,
  decodes images and preprocesses them. There is no CPU matmul, no mel front
  end and no audio code (`ignis-asr-seams` §1, §6).
- **llama.cpp runs the model, but not the way ignis needs it:**
  - its decoder sits on libllama;
  - its Realtime path does not support Qwen3-ASR;
  - it splits the last partial audio chunk differently from the official
    processor (d01, 10.34 s: 143 audio tokens against 135).
- **The measurements fix the path** (the finding). Quantized SIMD kernels are
  the only CPU path fast enough, and the spike showed that ours, in Rust,
  match ggml's on the matmuls. Q34 makes the CPU go only through incremental
  transcription, so the executor must encode, prefill and decode while the
  user is still speaking.

## Solution

### 1. The executor, in Rust (ADR 0068)

- **Pure Rust in the default build** (Q60; Q58 is moot). The executor is the
  crate `crates/speech` (`ignis-speech`, owner, 2026-10-24). Its own code
  has no C, no C++, no CMake, no build script and no cargo feature, and it
  adds no `cc` or `cmake` build dependency the workspace does not already
  have (`tokenizers` already brings `cc`). A bare `cargo build` builds it on
  both targets (`x86_64-pc-windows-msvc`, `x86_64-unknown-linux-gnu`), so the
  speech model is in every build and ADR 0032's default build needs no new
  toolchain.
- **Our own GGUF reader** (Q60): the header, the metadata, the tensor table
  and the tensor data, memory-mapped.
- **Activation quantizers** (the spike): f32 → Q8_K for the Q4_K and Q6_K
  weights, f32 → Q8_0 for the Q8_0 weights. Each matmul quantizes its own
  input.
- **Matmul kernels** (the spike; the GGUF headers): Q4_K × Q8_K, Q6_K × Q8_K
  and Q8_0 × Q8_0, plus F16 and F32 where the GGUF has them: the encoder's
  three convolutions and `conv_out`. They cover the catalog's three entries
  (Q49).
- **A scalar reference everywhere, AVX2 kernels on x86-64** (Q60, the
  spike). The AVX2 kernels are `#[target_feature(enable = "avx2,fma,f16c")]`
  functions, picked at run time after `is_x86_feature_detected!`. Nothing
  sets `target-cpu`, in `RUSTFLAGS` or in a cargo config, so one binary
  serves every x86-64 CPU. The AVX2 module is compiled only on x86-64.
- **Q4_K in our interleaved layout** (the spike). At load, Q4_K matmul
  weights are repacked into groups of 8 rows: one 32-byte load holds 4
  consecutive k of each row, and the 6-bit scales and mins are decoded once.
  That is 1,184 B per 8×256 weights against the file's 1,152, +2.8%. One
  layout serves decode and prefill, and it is the only copy of those tensors
  the executor keeps. Q6_K, Q8_0, F16 and F32 tensors are read from the
  mapped file (Q31).
- **Our own thread pool** (Q60, the spike): the calling thread is worker 0,
  tasks are claimed from an atomic counter, and workers spin, then park.
  rayon ran decode's 197 dispatches a token at 1.184 of ggml's time on 4
  cores; this pool ran them at 1.014. It runs `speech.threads` workers
  (Q38).
- **Provenance** (ADR 0051; Q59, Q60). Every kernel is our own. The GGUF
  block layouts and their quantization rules are the file format the kernels
  read, not copied code. No file of the executor is copied from ggml or
  llama.cpp, and neither is in the tree or the build. llama.cpp b11840 is an
  out-of-tree comparison and benchmark baseline on `H:\speech-study\`, and
  the offline converter of §6.
- **AVX2** (Q28, Q35). The executor detects AVX2, FMA and F16C at run time.
  Without them it is unavailable, with the reason "the CPU lacks AVX2", and
  no AVX2 kernel runs. The scalar kernels are the reference the tests compare
  against, and they are not served on x86-64 (Q28, Q35). No setting serves
  them either, not even for debugging (Q64).

### 2. The graphs, ours in Rust (ADR 0068)

- **Encoder (AuT):**
  - the convolutional stem: three 3×3 convolutions of stride 2 (1 → 480 →
    480 → 480 channels) with GELU, F16 weights;
  - `conv_out` (F16, 7,680 → d) and the position embedding;
  - per layer, LayerNorm with bias, attention with heads of 64,
    bidirectional inside each 8 s window of 104 tokens, and the FFN: 18
    layers at d 896 (0.6B), 24 at d 1024 (1.7B);
  - the final norm and the two-layer projector to the decoder's hidden size.
- **Decoder (Qwen3):** 28 layers of RMSNorm, QK-norm, GQA (16 query and 8 KV
  heads of 128), M-RoPE as the GGUF declares it (sections 24/20/20, θ 1e6)
  and SwiGLU, then the final norm and the LM head. The checkpoint ties the
  head to the embedding; the Q4_K_M file holds it as its own Q6_K
  `output.weight`. The KV cache is ours. It is one contiguous
  per-transcription buffer of 112 KiB per token. It is sized for the context
  cap, `speech.max_clip_seconds` of audio tokens and the decode. Decoding
  stops at EOS (151643, 151645) or at the cache's end.
- **Their other ops are ours too:** attention, softmax, LayerNorm, RMSNorm,
  RoPE, GELU, SiLU and the convolutions. Each is tested against an fp64
  reference at the model's real geometry, with a negative control, as ADR
  0043 asks of an op of our own.

### 3. The pipeline, in Rust (Q41)

- **Input:** mono f32 PCM at 16 kHz. Decoding and resampling belong to 02 and
  03.
- **Front end:** the official log-mel (128 bins, n_fft 400, hop 160, Hann,
  reflect padding, Whisper's normalisation), in Rust.
- **Chunking follows the official processor** (Q52): 100-frame chunks of
  13 tokens, encoder windows of 800 frames (8 s, 104 tokens), and the last
  partial chunk counted as the official processor counts it, not as
  llama.cpp does.
- **Prompt:** the official template, with the system turn carrying the
  context, empty when there is none (Q15).
- **Tokens:** the `tokenizers` crate, with the Qwen3 vocabulary the GGUF
  carries (Q41).
- **Decoding:** greedy (Q41).
- **Model output** (Q51):
  - the leading `language X<asr_text>` is stripped from the text and X kept as
    the label;
  - `None` followed by text is speech, and the text is kept;
  - `None` with nothing after it is no speech, an empty transcript;
  - the language is never forced by default. A forced language (Q6) is the
    official assistant prefill, `language <Name><asr_text>`.
- **Context** (Q15, Q50):
  - the system turn is capped at `speech.context_max_tokens`;
  - the executor keeps a one-entry prefix cache of the prefilled prompt head.
    A transcription whose context equals the previous one's prefills none of
    it; a different context replaces the entry. On the study's 1,496-character
    README context, an uncached context cost 1.7B Q4_K_M a median 2.0 s and a
    cached one 0.1 s. It lowered 1.7B Q4_K_M's Italian WER by 1.2 pt (3.7% to
    2.4%), and no context term was inserted where it was not spoken.
- **Threads:** `speech.threads`, `auto` = physical cores − 1 (Q38), counted
  with `num_cpus::get_physical`, which pulls no `windows-sys`.
- **One compute worker** (Q44) drives the executor's pool and does one unit
  of work at a time: a session's window, partial or commit, or a window of a
  batch transcription. Every unit is a window or less, so sessions can
  interleave and batch work can fill the gaps. The order is 02's.
- **Weights:** memory-mapped from the GGUF by default; `speech.pinned_weights`
  pins them (Q31).
- **Loaded at first use, unloaded when idle** (Q53). The Q4_K repack copies
  those tensors into private memory at load (on llama.cpp, 1.7B Q4_K_M held
  2.27 GB resident, against 2.75 GB with the mapped pages counted twice). So
  nothing is prepared at start:
  - the executor maps and repacks the model at the first signal: a Realtime
    session opening, a mic press or a batch request. It is prefetched on that
    signal, as the loan is (1.4-1.8 s for 1.7B on llama.cpp; the executor's
    own time goes in "As built");
  - it is unloaded after `speech.idle_unload` seconds with no transcription
    (default 300, `model.idle_unload`'s);
  - nothing is held while it is unused.

### 4. The incremental API (Q37)

One session per transcription, for 03 to drive, and used whole by 02:

1. `open(context)`: prefills the system turn, the user turn's head and
   `<|audio_start|>`. That is the prefix cache's entry.
2. `push(samples)`: encodes each 8 s window as soon as it is complete and
   prefills its 104 audio tokens. That is the **audio-prefix prefill**. Each
   window is encoded once; the encoder attends only inside a window, so the
   result is the whole-clip encode's.
3. `partial()`, every `speech.partial_interval_ms` while audio arrives:
   - encodes the incomplete window;
   - appends its tokens, the template's tail and the **stable text** so far;
   - decodes greedily;
   - the last `speech.rollback_tokens` of what it decoded are the
     **provisional tail**, and the rest joins the stable text;
   - then rolls the cache back to the audio prefix.
   - A partial its caller stops (Q79, a Realtime commit: spec 03) ends at its
     next yield, before each step above and where its decode yields the
     compute unit, and leaves the session as it was before it began: nothing
     joins the stable text, and the cache is rolled back.
4. `commit()`: encodes the last window, prefills it, and decodes from the
   stable text to EOS. The completed transcript is the stable text plus what
   this decode adds. So **it begins with every stable text ever reported**,
   which is what 03's `concat(deltas) == completed` rests on (Q43).

The stable text is fed as the decode's prefix, as the official streaming
does ("previous text minus its last 5 tokens"). It is never revised.

### 5. The oracle (Q52, which amends Q25)

Same method as the image model (Q25), with Q52's changes:
- **Per-layer, teacher-forced**, against Transformers FP32 on the CPU
  (`Qwen/Qwen3-ASR-*-hf`), on the same audio tokens:
  - the log-mel;
  - every encoder layer;
  - the projector;
  - every decoder layer;
  - each within `thresholds.json`, with a negative control that must fail.
- **Teacher-forced top-1 agreement** of the decoder against Transformers on
  the same audio tokens.
- **A WER gate:** pooled WER not worse than the reference's by more than
  1 pt, on FLEURS (the finding's 20 Italian + 20 English clips) and on the
  owner's 11 dictations, with the official prompt (empty system turn).
- **No free-running transcript equality** (Q52). The engine follows the
  official chunking, so the audio-token count is the official one.
- **Fixtures** come from Transformers, never from llama.cpp, whose chunking
  differs. In the repo they come from FLEURS clips only. The owner's
  dictations stay on `H:\speech-study\` and feed only the WER gate.

### 6. The files and the catalog (Q47, Q49, Q39)

- Converted by us from the official Qwen weights at a pinned revision, with
  llama.cpp b11840's converter and quantizer:
  - both sizes, decoder and audio encoder (mmproj);
  - bf16, Q8_0 and Q4_K_M;
  - the Q4_K_M decoder paired with the Q8_0 encoder, as measured.
- Our Q8_0 is compared with ggml-org's as a check: tensor data
  byte-identical, metadata free to differ (owner, 2026-10-24).
- Three built-in catalog entries (Q49), pinned like every other (ADR 0047):
  `qwen3-asr-1.7b`, `qwen3-asr-1.7b-q8` and `qwen3-asr-0.6b`, from
  `gpillon/Qwen3-ASR-1.7B-GGUF-ignis` and `gpillon/Qwen3-ASR-0.6B-GGUF-ignis`
  (owner, 2026-10-24). `ignis-server model download <id>` fetches them (Q39).
  - **Two files, one entry** (owner, 2026-10-24). `artifact` names the
    decoder GGUF. The encoder GGUF is found beside it under a fixed name and
    listed in `files`, as Flash-Next's MTP companion is.
  - **As built** (#351): the entry states `kind: speech`, the one kind a
    catalog says. It takes the speech rules in place of the sidecar rule: the
    artifact is a `.gguf`, and an `mmproj-*.gguf` file is among the `files`.
    It states no `card_class` (the CPU runs it; no card refuses it). The two
    1.7B entries share one encoder file, `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf`;
    the 1.7B Q4_K_M decoder is the `auto` default.
  - The family is read from the GGUF pair's metadata (a `qwen3vl` decoder
    with no deepstack layers, a `qwen3a` projector), never from a file name
    (ADR 0047; owner, 2026-10-24).
- The HF upload waits for the owner's OK and a privacy scan (Q47).

## Testing Decisions

- CPU tests, run by `cargo test` in the default build, so in CI's CPU leg
  (§1; AGENTS.md):
  - the GGUF reader;
  - the quantizers, scalar against AVX2;
  - every matmul kernel and tile shape against the fp64 reference, on
    generated blocks of every format, with the negative controls;
  - the AVX2 gate with feature detection injected;
  - the graphs' other ops against fp64 at the real geometry;
  - the front end and the per-layer checks against recorded fixtures;
  - the output parser;
  - the prefix cache;
  - the incremental session against a whole-clip run.
- A recording script under `tools/speech-oracle/`, its Python pinned (the
  study venv's torch 2.14.1+cpu and transformers 5.20.2), writes the fixtures
  and `thresholds.json`.
- The WER gate, the timings and the kernel check on the catalog files' own
  tensors are `--ignored` CPU legs: they read `H:\speech-study\` and run
  alone on the machine.

## Acceptance

1. The executor is pure Rust in `crates/speech`: no build script, no cargo
   feature for speech, and no `cc` or `cmake` build dependency the workspace
   does not already have. A bare `cargo build` builds it on
   `x86_64-pc-windows-msvc` and `x86_64-unknown-linux-gnu`. No `target-cpu`
   is set in `RUSTFLAGS` or a cargo config (Q60; Q58 is moot; ADR 0032;
   owner, 2026-10-24 for the crate).
2. Our GGUF reader reads the header, metadata and tensor table of every
   catalog file and maps its tensor data; the per-layer checks (12) run on
   what it reads (Q60).
3. The f32 → Q8_K and f32 → Q8_0 quantizers give bit-identical blocks on the
   scalar and the AVX2 path, for every test input (the spike).
4. Every matmul kernel (Q4_K × Q8_K on both layouts, Q6_K × Q8_K,
   Q8_0 × Q8_0, F16, F32), scalar and AVX2, at every register-tile shape it
   can pick and at M = 1, a remainder, 104 and 512, is within 1e-5 of an fp64
   reference: the dot product of the dequantized weights with the dequantized
   activations, the error normalised per element as |y − ref| / Σ|w·a|.
   `cargo test` checks it on generated blocks, and an `--ignored` leg on the
   catalog files' tensors (the spike; AGENTS.md).
5. Doubling one block scale, in each format and each Q4_K layout, fails 4's
   check (the spike).
6. AVX2 kernels run only in functions selected after
   `is_x86_feature_detected!` reports AVX2, FMA and F16C. With detection
   reporting none of them (injected), the executor is unavailable with the
   reason "the CPU lacks AVX2", and no AVX2 kernel has run (Q28, Q35, Q60).
7. Q4_K matmul weights are repacked at load into the interleaved 8-row
   layout, 1,184 B per 8×256 weights. The executor keeps no other copy of
   them and does not read their file pages again after the repack; the
   phase 3 loan uploads from the file, not from this copy (the spike, Q53,
   Q46).
8. The executor runs on its own spin-then-park pool of `speech.threads`
   workers. Between units of work the workers park, so an idle executor
   keeps no thread busy (the spike, Q38).
9. Both graphs are written in Rust, and their ops (attention, softmax,
   LayerNorm, RMSNorm, RoPE, GELU, SiLU, the convolutions) are each within a
   threshold of an fp64 reference at the model's real geometry, with a
   negative control that must fail (Q60; ADR 0043, 0051).
10. No file of the executor is copied from ggml or llama.cpp: every kernel is
    our own (ADR 0051), neither project is in the tree or the build, and
    `NOTICE` gains no line for the executor (Q59, Q60).
11. On every FLEURS clip and every dictation, the audio-token count equals the
    official processor's (d01: 135) (Q52).
12. Per catalog variant, the mel, every encoder layer, the projector and
    every decoder layer are within `thresholds.json` against Transformers
    FP32, teacher-forced on the same audio tokens. A perturbed layer fails
    (Q52, Q25).
13. Teacher-forced top-1 agreement against Transformers is at least 95% per
    catalog variant, the G1 floor's number (Q52; owner, 2026-10-24).
14. Per catalog variant, through the batch path: pooled WER on FLEURS 40 and
    on the 11 dictations, official prompt, is not worse than Transformers
    FP32's by more than 1 pt. No transcript is compared for equality (Q52).
15. Through the incremental path at real-time pace, at the default tail of 10
    (`speech.rollback_tokens`, Q75), pooled WER is not worse than the batch
    path's by more than 1 pt, for the default variant and `qwen3-asr-0.6b`.
    `qwen3-asr-1.7b-q8` is documented at +1.29 pt on the dictations, the
    same explicit-choice treatment as AC 24 (Q73). The paper's streaming mode
    lost 0.6-1.1 pt against offline (Q37, Q52; owner, 2026-10-24). At the
    former tail of 5 the 1.7B lost 1.12 pt and the 0.6B 1.94 pt, which is why
    the default moved.
16. The parser strips `language X<asr_text>` and keeps X. `None` plus text
    keeps the text. `None` alone gives an empty transcript. No language is
    forced unless asked, and a forced one is the official prefill (Q51, Q6).
17. The context is the system turn. The text tokens equal the official
    processor's: 15 with no context, 420 with the study's README context.
    The turn is capped at `speech.context_max_tokens`. A second transcription
    with the same context prefills no context token; a new context replaces
    the cache's one entry (Q15, Q50).
18. Fed each dictation at real-time pace, a session encodes each window once
    and prefills the audio prefix once. It reports a partial at most every
    `speech.partial_interval_ms`, skipping one while the previous partial or a
    window is still in work (Q74; owner, 2026-10-24), and its completed
    transcript begins with every stable text it reported (Q37, Q43).
19. One compute worker does one unit of work, a window or less, at a time
    (Q44).
20. `speech.threads: auto` resolves to physical cores − 1, an explicit value
    is used as given, and both are logged (Q38).
21. Weights are memory-mapped by default and pinned under
    `speech.pinned_weights` (Q31). Nothing is mapped or repacked before the
    first signal. The model is unloaded after `speech.idle_unload` seconds
    with no transcription, and the process then holds none of its memory
    (Q53).
22. bf16, Q8_0 and Q4_K_M of both sizes are converted from the official
    weights with b11840's tools and published in
    `gpillon/Qwen3-ASR-1.7B-GGUF-ignis` and
    `gpillon/Qwen3-ASR-0.6B-GGUF-ignis`. Our Q8_0 is checked against
    ggml-org's. The catalog entries `qwen3-asr-1.7b`, `qwen3-asr-1.7b-q8` and
    `qwen3-asr-0.6b` download and verify by sha256 through
    `ignis-server model download`. Nothing is uploaded before the owner's OK and a
    privacy scan (Q47, Q49, Q39; owner, 2026-10-24).
23. The executor's own times, warm, at `speech.threads: auto`, on the i9 and
    on the 4-core container, for a 10 s and a 60 s clip per catalog variant,
    and its load at first use, are written in "As built" and update
    `spec.md`'s requirements (Q34, Q53).
24. With 23's times, the default variant (`auto`) is inside Q34's go band
    (at least 2x real time) on the i9 and on the 4-core container, and every
    catalog variant is inside it on the i9; `qwen3-asr-1.7b-q8` is an
    explicit choice, documented as under 2x on four cores (Q73; owner,
    2026-10-24). Every variant's pooled WER
    on the finding's FLEURS clips and the 11 dictations is within 1 pt of
    llama.cpp b11840's on the same clips. The spike timed the matmuls alone
    (ADR 0068), so this is the whole executor's gate. The ratio to llama.cpp
    b11840's end-to-end time on the same clips is reported beside it and does
    not gate (Q65).
25. `cargo test` passes workspace-wide (AGENTS.md, not a grill decision).

## Out of Scope

- VAD: spec 05, phase 2 (Q56).
- The GPU loan and its CUDA adapter: spec 06, phase 3 (Q56).
- Word timestamps (Qwen3-ForcedAligner), later, on demand (Q23).
- Audio container decoding and resampling: specs 02 and 03.
- NEON kernels for aarch64: a later ticket after phase 1 (Q66; DGX Spark is
  not a priority, owner). ignis builds no aarch64 target today.
- AVX-VNNI and AVX-512 kernels: not in phase 1. A later ticket, if a
  measurement on such a CPU shows a gap to ggml's variants for it (Q67,
  ADR 0068).

## As built (GitHub #351, 2026-10-24)

Built on `speech-351`; measured on the i9-10900K (Windows 11) and on the
4-core container (podman on WSL, `--cpuset-cpus=0-3`, the Linux build).
Where the build decided something the spec left open, or differs from it:

- **The crate.** `crates/speech` (`ignis-speech`): `gguf.rs` (the reader),
  `quant.rs` (formats, quantizers, scalar dots, the fp64 reference),
  `avx2.rs` (the kernels, compiled on x86-64 only), `cpu.rs` (detection and
  the `Isa` an AVX2 kernel needs), `matmul.rs` (weights, the GEMV/GEMM driver),
  `pool.rs`, `ops.rs` and `math.rs` (the other ops), `mel.rs`, `text.rs` (the
  tokenizer, the prompt, the parser), `model.rs` (the GGUF pair),
  `encoder.rs`, `decoder.rs`, `session.rs` (§4) and `executor.rs`. About
  5,700 lines with the kernels; no build script, no feature, no C.
- **The server serves it.** #352 left the adapter open
  (`crates/server/src/speech/cpu.rs`): a load builds the crate's executor from
  the catalog entry's decoder and the `mmproj-` file beside it, with the
  load's threads and pinned weights; the crate's idle unload stays off (the
  worker keeps the idle clock); `open` takes the request's context cap.
- **`speech.threads: auto`** is the physical cores less one *counted within
  the CPUs the process may run on*: on Linux `num_cpus::get_physical` reads
  `/proc/cpuinfo`, which a cpuset does not narrow (the container saw 10 cores
  on 4 CPUs), so it is capped by the affinity mask and the cgroup quota. The i9
  runs 9 threads, the container 3.
- **The mel floor** (max − 8) is taken over the frames computed so far: a
  stream cannot know the clip's maximum, and the server's batch path pushes 8 s
  at a time. Only bins 80 dB under the loudest so far move. The batch WER legs
  push 8 s at a time, as the server does.
- **Partials.** The first two fix nothing (`unfixed_partials`, the official
  streaming's `unfixed_chunk_num`): fixing from the first partial decided the
  language on 2 s of audio and labelled an Italian dictation Portuguese (the
  incremental path then lost 1.45 pt on the dictations). The stable text ends
  on a whole word and a whole character. `unfixed_partials` and
  `rollback_tokens` are `speech` fields (the owner's rule on thresholds), read
  when a session opens: the server passes them in the session-open settings
  (`SessionOpen`, which the adapter hands `Executor::open_tuned`), so a live
  change holds from the next session. §4's operations are unchanged.
  `partial_interval_ms` stays the server's: it decides when `partial()` is
  called.
- **The KV cache** is sized for `speech.max_clip_seconds` (the executor's
  default, 1,200 s: the seam carries no clip length) plus half as many text
  tokens as audio tokens, but grows to it on demand, doubling from the head, two
  windows and the tail; decoding stops at its end. A decode yields the compute
  unit every 16 steps, about a window's prefill.
- **A partial can be stopped** (Q79, #353): `Session::partial_unless(stop)`
  asks `stop` before the encode, between the encode and the incomplete
  window's prefill, before the decode's prefill and after it, and at each
  16-step yield. Stopped, it returns `None`, the cache truncated to the audio
  prefix, the stable text and the partials counted (`Stats::partials`, which
  `unfixed_partials` reads) unchanged. `partial()` is the one nothing stops.
- **Pinned weights** copy every tensor but the repacked Q4_K ones into
  anonymous memory (locked with `mlock` on Linux; memmap2 offers no lock on
  Windows, where the private copy is what pins). The files are not held open.
- **A context's special tokens are encoded as text**, never as control tokens;
  for ordinary text the tokens are the official processor's (AC 17).
- **No repetition fix.** The checkpoint's parser runs
  `_detect_and_fix_repetitions` on the text; ours does not.
- **Testing decisions.** The per-layer checks need the catalog files, so they
  are an `--ignored` leg with the WER gate and the timings; the default suite
  holds the front end against the fixture and the executor on a synthetic GGUF
  pair. The legs read `IGNIS_SPEECH_STUDY` and fail without it.
- **The files** (AC 22): `tools/speech-gguf/convert.sh` with b11840's
  converter and quantizer, from Qwen's *original* repos (`Qwen/Qwen3-ASR-1.7B`
  @ `7278e1e7`, `Qwen/Qwen3-ASR-0.6B` @ `5eb14417`): the converter's mmproj
  class reads their `thinker_config`, which the `-hf` twins (the oracle's
  checkpoints) lack. The converter needs `sentencepiece` in its venv. Our Q8_0
  tensors are ggml-org's byte for byte except `a.position_embd.weight`, a
  sinusoid table the converter computes in f32: at late positions an angle near
  1,500 rad has an f32 ulp of 1.2e-4; the 13 positions a chunk uses differ by
  9.4e-7. Our Q4_K_M is the study's self-quantized file byte for byte, so the
  finding's llama.cpp runs read the same weights. llama-quantize also made an
  mmproj Q4_K_M (rehost only). sha256 in `SHA256SUMS` beside the files.
  **Open:** the HF upload (the owner's OK and a privacy scan) and the three
  catalog entries, which need its revision.
- **The oracle** (ACs 11-13). `tools/speech-oracle/record.py` records
  Transformers FP32 on FLEURS `it_it` 1703 (9.06 s: two windows, 104 + 14
  tokens, and a partial last chunk): the log-mel and six rows of every tapped
  tensor, per size; and the 40 clips' greedy outputs. The bound of each tensor
  is twice its error under the sim yardstick (the GGUF pair dequantized with
  b11840's `gguf-py`, each matmul's input quantized as ours), at least 1e-5;
  the log-mel's is twice torch's fp32 front end against fp64 (2.0e-5). Our
  error is the yardstick's within a few percent at every layer of every
  variant (1.7B Q4_K_M decoder layer 0: 9.73e-2 against 9.69e-2). The decoder
  bounds are wide because Q4_K's own error is 12-17% of each row of the
  residual stream: swapping a layer's Q and K norms stayed inside them, so the
  decoder's negative control swaps two layers' down projections (an off-by-one
  load); the encoder's drops a layer's attention output bias. Teacher-forced
  top-1 over the 40 clips: 99.27% (1.7B), 99.82% (1.7B Q8), 99.45% (0.6B).
  Every FLEURS clip and dictation has the official token count (d01: 135).
- **WER** (ACs 14, 15, 24), pooled, the official prompt. Batch pushed 8 s at
  a time as the server does; incremental pushed 100 ms at a time with a partial
  every 2 s.

  | variant | FLEURS 40: batch / incremental (5) / incremental (10) / Transformers / llama.cpp | 11 dictations: the same |
  |---|---|---|
  | `qwen3-asr-1.7b` | 4.05 / 4.25 / 3.94 / 3.63 / 4.36 | 3.23 / **4.35** / 4.03 / 3.06 / 3.39 |
  | `qwen3-asr-1.7b-q8` | 3.73 / 3.73 / 3.63 / 3.63 / 4.15 | 2.90 / 3.71 / **4.19** / 3.06 / 3.06 |
  | `qwen3-asr-0.6b` | 5.91 / 6.64 / 6.43 / 6.64 / 6.33 | 7.74 / **9.68** / 8.71 / 8.23 / 8.55 |

  (%, the incremental path with a provisional tail of 5 tokens, the default,
  and of 10.) Pushing the batch path 8 s at a time or whole gives the same
  WER. **AC 15 is not met at the default tail of 5:** the
  incremental path loses 1.12 pt (1.7B) and 1.94 pt (0.6B) on the dictations,
  0.0-0.7 pt on FLEURS. A tail of 10 brings the 1.7B and the 0.6B inside the
  point and moves the 1.7B Q8 out (1.29 pt): the eleven dictations are 620
  words, a point is six errors, and their technical vocabulary (identifiers,
  numbers, English terms in Italian) is where a word fixed with one or two
  seconds of right context differs from the batch's spelling ("api.rs" as
  "api. rs", "4" as "quattro"). The paper's own streaming loses 0.6-1.1 pt on
  its benchmarks. Which default (or which partial design) is the owner's call.

  llama.cpp's figures are the finding's runs (10 threads; its FLEURS runs had no
  system turn, its dictation arm A the official one).
- **Times** (ACs 23-24), warm, the median of the last three of four runs
  (`examples/transcribe.rs`, batch path, one gated run at a time after 20 s
  with no build; logs in `H:\speech-study\results\executor-351\`).
  `speech.threads: auto`: 9 threads on the i9, 3 in the container.

  | variant | i9, 10.26 s | i9, 60.8 s | 4 cores, 10.26 s | 4 cores, 60.8 s |
  |---|---|---|---|---|
  | `qwen3-asr-1.7b` | 2.47 s, 4.1x | 13.05 s, 4.7x | 3.58 s, 2.9x | 20.42 s, 3.0x |
  | `qwen3-asr-1.7b-q8` | 3.81 s, 2.7x | 19.46 s, 3.1x | 6.54 s, **1.6x** | 34.89 s, **1.7x** |
  | `qwen3-asr-0.6b` | 1.70 s, 6.0x | 9.25 s, 6.6x | 2.84 s, 3.6x | 16.31 s, 3.7x |

  Against llama.cpp b11840's end-to-end time on the same clips, at its thread
  count (10 on the i9, 4 in the container) and ours at the same count
  (reported, not gated, Q65): 1.7B 1.04 / 1.05 / 0.82 / 0.83, 1.7B Q8 1.01 /
  1.06 / 1.03 / 1.06, 0.6B 0.96 / 1.03 / 1.00 / 1.09 (i9 10 s, i9 60 s,
  container 10 s, container 60 s). **AC 24 misses on `qwen3-asr-1.7b-q8` in
  the container:** 1.6x and 1.7x at `auto`, 1.9x and 2.1x at 4 threads, where
  llama.cpp does 1.9x and 2.2x: on four cores that variant is under the go
  band whichever executor runs it. The other two pass everywhere.
- **Load at first use** (the files in the OS cache): 0.54 s for 1.7B Q4_K_M on
  the i9 (645 MB repacked), 0.13 s for the two Q8_0 variants (mapped: their
  pages fault in on the first transcription, which takes 0.5-1 s longer than
  the warm ones). In the container, whose files cross WSL's `/mnt` mount, the
  Q4_K_M repack takes 3.9-4.2 s.
- **Partials at real-time pace** (spec 03's concern, measured here): a 60.8 s
  clip with a partial every 2 s, run flat out, takes 46.6 s of compute on the
  i9 for 1.7B (0.77 of real time; commit 1.6 s), 76 s for 1.7B Q8 (1.25),
  35.2 s for 0.6B (0.58); in the container 83 s, 159 s and 71 s. Each partial
  re-encodes the incomplete window and prefills the prompt's tail and the whole
  stable text again (it sits after the growing audio, so its cache cannot be
  kept), so on four cores no variant keeps up with a partial every 2 s.
