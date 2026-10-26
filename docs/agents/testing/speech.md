# Testing the speech model

## The speech oracle (GitHub #351)

The speech executor (`crates/speech`) is checked against Transformers FP32
on the CPU, never against llama.cpp (spec speech/01 §5, ADR 0043). The
default suite holds the kernels and ops against fp64, the log-mel against
the recorded fixture, and the executor on a synthetic GGUF pair. The checks
on the real catalog files are `--ignored` CPU legs in
`crates/speech/tests/oracle.rs`: every layer within `thresholds.json`, the
teacher-forced top-1, the token counts, the context's tokens, the batch and
incremental WER gates, the real-time session, the reader and the Q8_0 check,
the kernels on the files' own tensors. They read the study directory
(`IGNIS_SPEECH_STUDY`, `H:/speech-study` on the owner's machine: the GGUF
files in `models/ignis-conv/`, the FLEURS clips, the private dictations and
their records in `results/oracle/`) and fail without it. They are CPU-heavy
(the WER legs take about 15 minutes per variant): run them alone, as a GPU
run, and never inside the GPU profile, whose stage 3 excludes the crate.

```sh
cargo test --release -p ignis-speech --test oracle -- --ignored --test-threads=1
```

The fixtures under `crates/speech/tests/fixtures/oracle/` come from
`tools/speech-oracle/record.py` (its `requirements.txt` pins the venv), in
two steps per size and variant: `record` (Transformers on the FLEURS fixture
clip and every clip; the dictations' records go to the study directory, never
the repo) and `yardstick` (the GGUF pair's weights dequantized with
llama.cpp b11840's `gguf-py`, each matmul's input quantized as the executor
does). Each tapped tensor's bound is twice the yardstick's error, at least
1e-5; the log-mel's is twice torch's fp32 front end against fp64. The GGUF
files themselves come from `tools/speech-gguf/convert.sh`.

## server_vad's VAD oracle (GitHub #355)

`server_vad`'s VAD (`crates/speech/src/vad.rs`) is checked against Silero
VAD 6.2.3's published model, `silero_vad.jit`, on the CPU. The default suite
holds the port within `thresholds.json`'s bound of the `.jit`'s per-frame
probabilities on the FLEURS fixture clip, with a negative control, and the
turn detector (`Turns`) on scripted probabilities. The `--ignored` legs in
`crates/speech/tests/vad.rs` read the study directory: every frame of the 40
FLEURS clips and the 11 dictations within the bound (seconds), and AC 7's
WER gate, `server_vad`'s items against a manual commit on the dictations with
the default variant, each item fed after the server's leading silence (about
15 minutes): run that alone, as the oracle's WER legs. `IGNIS_SPEECH_LEAD_MS`
(`0,500,1000,1500`) adds arms of other leading silences, each reported and each
some 5 minutes more; the default is the one asserted.

```sh
cargo test --release -p ignis-speech --test vad -- --ignored --test-threads=1
```

`tools/speech-vad/record.py` (its `requirements.txt` pins the venv; the
silero-vad 6.2.3 wheel is passed as a file, its sha256 checked) has two
steps: `convert` writes the `.jit`'s 16 kHz weights to
`crates/speech/vad/silero-vad-16k.safetensors`, the same bytes on every run;
`record` writes the `.jit`'s probabilities to
`crates/speech/tests/fixtures/vad/` (the dictations' to the study directory,
never the repo) and the bound: ten times the `.jit`'s distance from the same
graph in fp64, at least 1e-5.

## The diarizer's oracle (GitHub #361)

The diarizer (`crates/speech/src/diarizer/`, spec speech/08) is checked
against the checkpoint's Transformers code in FP32 on the CPU, recorded by
`tools/speech-diarizer/record.py` (its `requirements.txt` pins the venv) on
the committed fixture clip (`crates/speech/tests/fixtures/diarizer/`, 16 s of
one of the twenty synthetic conversations, VoxPopuli audio, CC0) and on three
conversations of the study directory. Each bound is ten times the
reference's distance from fp64, at least 1e-5, in `thresholds.json`.
`crates/speech/tests/diarizer.rs` holds the legs. The default suite runs the
two that need no weights: `the_log_mel_matches_transformers_fp32` and
`the_speaker_cache_keeps_the_frames_the_reference_keeps` (the speaker cache
replayed on the recorded probabilities, at a small geometry that compresses
it and at the live one). The four `--ignored` legs read the F32 GGUF
(`models/diarizer/` of `IGNIS_SPEECH_STUDY`, from
`tools/speech-diarizer/convert.py`) and the records in
`nemotron-spike/oracle-361/`, and fail without them:
`the_first_step_matches_the_reference_tensor_by_tensor`,
`the_port_matches_the_reference_on_the_fixture_clip`,
`the_port_matches_the_reference_on_the_conversations` (the live and file
geometries, a few minutes) and
`the_speaker_cache_keeps_the_frames_the_reference_keeps_on_the_conversations`.
The GPU profile excludes the crate.

```sh
cargo test --release -p ignis-speech --test diarizer -- --ignored --test-threads=1
```

## Speakers in a Realtime session (GitHub #361)

The server's side of the diarizer (spec speech/08) is tested in the default
suite against a diarizer scripted frame by frame (`FakeDiarizer`, beside
`FakeExecutor`): `crates/server/tests/speech_realtime_speakers.rs` runs the
real VAD on the FLEURS fixture clip, so the cuts at a speaker change, the
length guard's cuts and their times are exact, and checks the queue limit and
that the diarizer's steps, on a thread of their own, neither wait for the
refreshes nor hold them back (slow fakes, timed spans). `ignis_speech::Session::cut`
is checked against the session of the audio before the cut on the synthetic
pair (`crates/speech/tests/executor.rs`). The real adapter, `CpuDiarizer`, has
one `--ignored` leg on the study directory's F32 GGUF, on the CPU: its frames,
on its own pool, against the crate's stream, step by step, its tail and its
`auto` constants.
`scripts/gpu-profile.ps1` skips it by name (`$NotGpuLegs`).

Its catalog side is `crates/server/tests/speech_diarizer_models.rs`: a diarizer
entry of the test's own (`kind: diarizer`; the built-in catalog has none until
its repo exists), `diarizer.model` resolved by the real `CatalogDiarizer` on
the host's CPU (skipped without AVX2), the `speech.diarizer` member of `GET
/v1/models`, the `PATCH` refusals, and the load at the first session that asks
and the unload with the speech model, on a `FakeDiarizer`. The session's end
line is `speech_realtime_speakers_log.rs`, a binary of its own: tracing caches
a callsite's interest process-wide.

```sh
CUDA_VISIBLE_DEVICES= cargo test -p ignis-server --lib speech::cpu::tests::the_diarizer_adapter -- --ignored
```

The real executor and the real diarizer in one session are
`crates/server/tests/speech_diarizer_study.rs`, two `--ignored` CPU legs on the
study directory (Qwen3-ASR 1.7B's GGUF pair in `models/ignis-conv/`, the F16
diarizer GGUF or `IGNIS_DIARIZER_GGUF`, the twenty conversations), both listed
in `$NotGpuLegs`. Run each alone, with nothing else on the CPU, and the
`CUDA_VISIBLE_DEVICES` empty:

- `the_diarizer_keeps_up_with_a_conversation_at_real_time_pace` (AC 11): one
  conversation (`IGNIS_DIARIZER_CONV`, default `conv07`, 5 minutes) at real
  time, speakers `live`; the session's `lag_max_ms` within the live geometry's
  latency plus 2 s. It prints each item's time from `committed` to
  `completed`, with their median and max. A timing run: `--release`. `IGNIS_SPEECH_PARTIAL_MS` and
  `IGNIS_SPEECH_PARTIAL_FACTOR` move the refreshes.
- `every_item_of_the_conversations_sends_its_text_whole_at_its_end` (AC 7):
  every conversation (`IGNIS_DIARIZER_CONVS`, names separated by commas, for
  fewer) in a session of its own, fed `IGNIS_DIARIZER_PACE` (2) times faster
  than real time: no item's delta before its `committed`, its deltas joined
  its `completed.transcript`, cut items included. It prints each
  conversation's items, cuts and speakers; about 95 minutes at 2x in the test
  profile on the i9, where the worker falls behind the feed, so changes found
  after their item ended split nothing (9 cuts in 347 items).

```sh
CUDA_VISIBLE_DEVICES= cargo test --release -p ignis-server --test speech_diarizer_study -- --ignored --nocapture the_diarizer_keeps_up
CUDA_VISIBLE_DEVICES= cargo test -p ignis-server --test speech_diarizer_study -- --ignored --nocapture every_item_of_the_conversations
```

`crates/server/tests/speech_diarizer_refreshes.rs`'s `--ignored` leg
`a_long_monologue_keeps_its_live_text_with_speakers_on` (in `$NotGpuLegs`)
feeds one long item of `tools/speech-diarizer/long_items.py` (`IGNIS_LONG_ITEM`,
default `125192-240`, under `length-guard-361/` on the study disk) at real time
with the provisional tail, without speakers and then with them, and prints per
item the refreshes the client saw and the longest the live text stood still: the
measure of how the diarizer's steps leave room for the refreshes on a long
monologue (decision D6). A timing run, `--release`, alone on the CPU.
