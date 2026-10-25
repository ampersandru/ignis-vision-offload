# Testing against the vision canary

## The vision canary fixture (GitHub #178)

`crates/server/tests/fixtures/vision_canary/` is ADR 0014's floor applied to
multimodal prompts: four generated images (a rendered number, a coloured
square, three circles, a line of screenshot text), one unambiguous question
each, and the reference's greedy answers.
`crates/server/tests/vision_canary_gpu.rs` scores ignis against them by
teacher-forced agreement over the first 32 answer positions, each prompt
prefilled twice — whole, and in 48-token spans so the placeholder run crosses
span boundaries — and **each way clears the 95% floor on its own**, pooled
over the four canaries the way ADR 0014 pools its own suite.

The questions ask for a sentence, not a word, so the score has resolution:
the four answers are ~50 scored positions per prefill mode, and one near-tie
flip is two percent. At one token per answer it was a seventh of the suite,
which is noise rather than a floor. Measured 2026-09-16: whole 48/50 = 96.0%,
spanning 49/50 = 98.0%, and decode continuing the prefill path's own greedy
chain 46/46.

Re-record it (needs the GPU and the reference, ADR 0006 — stop
`ignis-server` first; the images are regenerated deterministically, so a
re-record with the same PIL/font is a no-op on them):

```powershell
F:\ai\q38\ninfer\build-ninja\apps\ninfer-serve.exe `
  F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer `
  --host 127.0.0.1 --port 8080 --vision --greedy --no-thinking `
  --max-context 8192 --max-concurrency 1 --kv-capacity auto
python tools/vision-canary/record.py crates/server/tests/fixtures/vision_canary
# then stop ninfer-serve and commit the fixture
```

`crates/runtime/tests/cuda_leaf_vision_gpu.rs` is the same load's serving-shape
check: an image prompt chunked at 64 tokens while three text lanes decode, each
lane still answering its own question, and 100 image requests leaving no media
embedding live and the leaf's footprint where it was. Its second test evicts an
image request to KV-RAM mid-decode and checks the restored one continues with
the never-evicted tokens (GitHub #194).
