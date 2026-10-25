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
