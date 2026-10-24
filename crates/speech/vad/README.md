# Silero VAD's weights

`silero-vad-16k.safetensors` is the 16 kHz model of Silero VAD 6.2.3, the
weights `server_vad` runs with (spec speech/05, `crates/speech/src/vad.rs`).
It is built into the binary, so a bare `cargo build` needs nothing fetched.

- **Source:** `silero_vad/data/silero_vad.jit` (sha256 `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`)
  in the PyPI wheel `silero_vad-6.2.3-py3-none-any.whl` (sha256
  `7b7f5436cfcb02fae583a05b512ea96467fd449fe54cb49a5e4f06c51a1e43b8`), the
  model `load_silero_vad()` returns. The same file is in
  <https://github.com/snakers4/silero-vad> at tag v6.2.3 (commit 5cd79456),
  `src/silero_vad/data/`.
- **Conversion:** `tools/speech-vad/record.py convert`. It takes the `.jit`'s
  16 kHz model (`_model`): every parameter and the STFT basis, f32 and
  unchanged, renamed to the names of the wheel's `tinygrad_model.py`. The
  wheel also ships a `silero_vad_16k.safetensors`, but it holds other
  weights than the `.jit`, so it is not used.
- **This file:** 1,239,884 bytes, sha256
  `8597e1914787562b099ca34f24a288930247aa6f30c65756061febe92a5164b2`. The
  conversion writes it byte for byte the same on every run.
- **Licence:** MIT, Copyright (c) 2020-present Silero Team: `LICENSE` beside
  this file, and the root `NOTICE`.

The oracle (`crates/speech/tests/vad.rs`) checks the port against the `.jit`'s
per-frame probabilities, recorded by `tools/speech-vad/record.py record`. Its
records are in `crates/speech/tests/fixtures/vad/`, and its bound is in
`thresholds.json` there.
