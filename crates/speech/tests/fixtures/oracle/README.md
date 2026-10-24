# The speech oracle's fixtures

Recorded by `tools/speech-oracle/record.py` from Transformers FP32 on the CPU
(`Qwen/Qwen3-ASR-1.7B-hf` @ `bcd2b5b7`, `Qwen/Qwen3-ASR-0.6B-hf` @ `7f1569a4`;
torch 2.14.1+cpu, transformers 5.20.2), never from llama.cpp (spec
speech/01 §5, ADR 0043).

- `it_it-1703.wav`: the fixture clip, 9.06 s of Italian (FLEURS `it_it` id
  1703, file `13686425341231579548.wav`), 16 kHz mono 16-bit. Two encoder
  windows (104 + 14 tokens) and a partial last chunk.
- `<size>/layers.{json,bin}`: on that clip, the log-mel (128 × 906) and, at six
  rows each, the encoder's input and every layer, `ln_post`, the projector,
  every decoder layer and the final norm, teacher-forced on the official
  prompt with the reference's own greedy transcript; `top1` at every position.
  The `.bin` is little-endian f32 at the index's offsets.
- `<size>/fleurs.json`: for the finding's 20 Italian and 20 English FLEURS
  clips, the audio-token count, the greedy output's token ids and the parsed
  transcript; and the no-context prompt's text tokens.
- `thresholds.json`: per catalog variant, each tapped tensor's error under the
  sim yardstick (the GGUF pair's weights dequantized, each matmul's input
  quantized as the executor quantizes it) and its bound, and the log-mel's
  bound (torch's fp32 front end against fp64).

The owner's dictations and the README context stay on the study disk; nothing
from them is in the repo.

The FLEURS audio and transcriptions are © Google, from the FLEURS dataset
(Conneau et al., 2022, <https://huggingface.co/datasets/google/fleurs>),
licensed CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>). The clip
is unmodified.
