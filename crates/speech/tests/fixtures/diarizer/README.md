# The diarizer's oracle fixtures

Recorded by `tools/speech-diarizer/record.py record` from the checkpoint's
Transformers code (`nemotron3_diarization` in transformers 5.20.2, FP32, on
the CPU; torch 2.14.1+cpu) with `nvidia/Nemotron-3-Diarization` @ `f667ed73`,
streamed as the model card streams it: one forward per chunk, the speaker
cache passed along (spec speech/08, acceptance 1).

- `conv04-28-44.wav`: the fixture clip, 16 s, 16 kHz mono 16-bit: seconds
  28-44 of `conv04`, one of the twenty synthetic conversations of the finding
  `docs/findings/2026-10-25-nemotron-on-the-cpu.md`. Two speakers, a change
  with no pause at 6.42 s. Its audio is VoxPopuli's (Italian): the end of
  `20141021-0900-PLENARY-16-it_20141021-22:29:50_8` (speaker 124868), then
  `20151014-0900-PLENARY-15-it_20151014-21:56:25_2` (speaker 28429), as the
  conversation placed them (`conv04.json` on the study disk).
- `fixture.{json,bin}`, at the live geometry (264 / 264 / 25 / 4 / 222 frames
  of 80 ms) and at a small one (32 / 16 / 6 / 2 / 12) that overflows and
  compresses the speaker cache many times in 16 s:
  - `<geometry>.probs`: every 10 ms row's 8 speaker probabilities;
  - `<geometry>.step.<i>.*`: each step's probabilities of every row it read
    (80 ms), and its state after it: the frames the speaker cache and the FIFO
    hold (−1 a silence slot) and the probabilities the cache stored;
  - `mel.frames` and `mel.rows`: the log-mel of the clip's first 400 and last
    400 frames, as the streaming chunks carried them.
  Each `.bin` is little-endian, at the offsets and dtypes its `.json` lists.
- `thresholds.json`: the bounds. The script runs the model again in fp64,
  its state forced at every step to the fp32 run's (the same frames in the
  cache and the FIFO), so that the distance measures the arithmetic alone;
  that distance times `factor`, and at least `floor`, is `bound`. `ties`
  counts the boundary ties of the cache's top-k selections, whose order torch
  leaves undefined. `mel` is the front end's bound, from torch's fp32 front
  end against an fp64 one.

The forward pass needs the weights, which are not in the repository, so its
legs are `--ignored` and read the converted GGUF and the oracle's other
records (the first step's taps, three whole conversations) from the study
disk. Every `cargo test` checks the log-mel and replays the speaker cache on
these records.

VoxPopuli (Wang et al., 2021, <https://github.com/facebookresearch/voxpopuli>)
transcribed data is licensed CC0.
