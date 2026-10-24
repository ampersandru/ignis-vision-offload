# server_vad's oracle fixtures

Recorded by `tools/speech-vad/record.py record` from Silero VAD's published
model, `silero_vad.jit` of the silero-vad 6.2.3 wheel, on the CPU with torch
2.14.1+cpu (spec speech/05, acceptance 1). The model reads each clip as the
wheel's own `utils_vad.py` drives it: `reset_states()`, then `model(chunk,
16000)` for every 512 samples, each chunk with the 64 samples before it, and
the last chunk zero-padded.

- `fixture.{json,bin}`: the per-frame speech probabilities on
  `../oracle/it_it-1703.wav`, 284 frames.
- `fleurs.{json,bin}`: the same on the finding's 20 Italian and 20 English
  FLEURS clips, read from the study disk.
- `thresholds.json`: the bound. The script runs the same graph, written out
  in torch from the converted weights, in fp32 and in fp64. In fp32 it
  reproduces the `.jit` (`plain_fp32_vs_jit`), which checks the weights file
  and the graph. The `.jit`'s own distance from fp64 (`jit_fp32_vs_fp64`)
  times `factor`, and at least `floor`, is `bound`, the most any frame of
  the port may differ from the `.jit`.

Each `.bin` is little-endian f32: each clip's probabilities at the offset its
`.json` gives. The owner's dictations are recorded the same way to the study
disk (`results/vad/dictations.{json,bin}`); nothing from them is in the repo.

The probabilities are computed from FLEURS audio, which is © Google, from the
FLEURS dataset (Conneau et al., 2022,
<https://huggingface.co/datasets/google/fleurs>), licensed CC BY 4.0
(<https://creativecommons.org/licenses/by/4.0/>).
