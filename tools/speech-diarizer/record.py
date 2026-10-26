"""The diarizer's oracle (spec speech/08, acceptance 1): Nemotron 3 Diarization's Transformers code on the CPU.

The reference is the checkpoint's own Transformers code (`nemotron3_diarization` in transformers 5.20.2,
which restates NeMo's Sortformer), with the checkpoint's FP32 weights, driven in streaming mode as the model
card drives it: one forward per chunk, the speaker cache passed along, the first chunk centred, the later
ones not, the last one with `is_last_audio_chunk=True`. Each geometry (speaker cache, FIFO, chunk, right
context, update period, in 80 ms frames) is given to the processor as a streaming mode and to the cache it
starts with. Three subcommands:

excerpt    the committable fixture clip: a span of one of the twenty synthetic conversations (VoxPopuli
           audio, CC0), 16 kHz mono 16-bit.
record     per clip and geometry, the reference's per-frame speaker probabilities (sigmoid of its logits,
           one row of 8 per 10 ms) and each step's speaker-cache state, and the same in fp64 (the model
           and its input features cast) with the state forced, step by step, to the fp32 run's: the
           reference's distance from fp64 with the same state, times --factor and at least --floor, is
           the bound in thresholds.json. Unforced, the twin could keep other frames where two scores
           are near equal, and its distance would measure that choice, not the arithmetic. On the fixture clip (into --out, the repo) also: the log-mel at the frames
           the streaming chunks carry, each step's speaker-cache state (which frames the cache and the
           FIFO hold, and the probabilities it scores them with), and every boundary tie of the cache's
           top-k selections, which torch leaves in no defined order. The taps of one step (the embedder,
           every layer at a few rows, the head) and the conversations' records go to --private-out (the
           study disk): they need the weights to be read, so only the `--ignored` legs read them.
mel64      the front end in fp64 (numpy), to bound the log-mel: torch's fp32 front end's distance from it.

Pinned: torch 2.14.1+cpu, numpy 2.5.4, transformers 5.20.2, safetensors 0.8.0, soundfile 0.14.0, librosa
1.0.0 (requirements.txt); the checkpoint nvidia/Nemotron-3-Diarization at f667ed73 (model.safetensors sha256
below).
"""
import argparse
import copy
import hashlib
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import librosa
import numpy as np
import safetensors
import soundfile as sf
import torch
import transformers
from transformers import AutoModelForAudioFrameClassification, AutoProcessor
from transformers.models.nemotron3_diarization import modeling_nemotron3_diarization as M

PINS = {"torch": "2.14.1+cpu", "numpy": "2.5.4", "transformers": "5.20.2", "safetensors": "0.8.0",
        "soundfile": "0.14.0", "librosa": "1.0.0"}
REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
WEIGHTS_SHA256 = "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6"
RATE = 16000
# speaker cache, FIFO, chunk, right context, update period, in 80 ms encoder frames
GEOMETRIES = {
    # spec 08's live default and the card's offline geometry (Q106)
    "live": (264, 264, 25, 4, 222),
    "file": (264, 40, 340, 40, 300),
    # a cache small enough to fill, overflow and compress many times on the fixture clip
    "small": (32, 16, 6, 2, 12),
}
FIXTURE_GEOMETRIES = ("live", "small")
CONV_GEOMETRIES = ("live", "file")
# the log-mel frames of the fixture recorded: its first and its last
MEL_HEAD = 400
MEL_TAIL = 400
# the encoder rows tapped in the taps' step
TAP_ROWS = (0, 7, 14, 21, 28)


def check_pins():
    got = {"torch": torch.__version__, "numpy": np.__version__, "transformers": transformers.__version__,
           "safetensors": safetensors.__version__, "soundfile": sf.__version__, "librosa": librosa.__version__}
    if got != PINS:
        sys.exit(f"pinned {PINS}, running {got}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_wav(path):
    a, sr = sf.read(path, dtype="int16")
    assert sr == RATE and a.ndim == 1, path
    return a.astype(np.float32) / 32768.0


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


class Blob:
    """Named little-endian arrays in one `.bin`, indexed by a `.json`: name, dtype, shape, offset."""

    def __init__(self):
        self.parts, self.index, self.off = [], [], 0

    def add(self, name, a, dtype="<f4"):
        a = np.ascontiguousarray(a, dtype=dtype)
        self.parts.append(a.tobytes())
        self.index.append({"name": name, "dtype": {"<f4": "f32", "<i4": "i32"}[dtype], "shape": list(a.shape),
                           "offset": self.off})
        self.off += a.nbytes

    def write(self, path, meta):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".bin", "wb") as f:
            for p in self.parts:
                f.write(p)
        write_json(path + ".json", {"tensors": self.index, **meta})


def load(model_dir):
    if sha256(os.path.join(model_dir, "model.safetensors")) != WEIGHTS_SHA256:
        sys.exit(f"{model_dir}/model.safetensors is not nvidia/Nemotron-3-Diarization @ {REVISION[:8]}")
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForAudioFrameClassification.from_pretrained(model_dir, dtype=torch.float32).eval()
    model64 = copy.deepcopy(model).double().eval()
    return processor, model, model64


def new_cache(model, geometry):
    cache_len, fifo, _, _, update = geometry
    cfg = copy.deepcopy(model.config.streaming_config)
    cfg.speaker_cache_length = cache_len
    return M.Nemotron3DiarizationSpeakerCache(cfg, fifo_length=fifo, speaker_cache_update_period=update)


def chunks(processor, audio, geometry):
    """The processor's streaming inputs, as the card's example feeds them, with each chunk's first mel frame."""
    _, _, chunk, rc, _ = geometry
    processor.streaming_modes = dict(processor.streaming_modes, oracle=(chunk, rc))
    processor.set_streaming_mode("oracle")
    first = processor.num_samples_first_audio_chunk
    kw = dict(sampling_rate=RATE, is_streaming=True, return_tensors="pt")
    if audio.shape[0] < first:
        yield 0, processor(audio, is_first_audio_chunk=True, is_last_audio_chunk=True, **kw)
        return
    yield 0, processor(audio[:first], is_first_audio_chunk=True, **kw)
    mel = processor.num_mel_frames_per_step
    start = processor.audio_chunk_start(mel)
    while (end := start + processor.num_samples_per_audio_chunk) <= audio.shape[0]:
        yield mel, processor(audio[start:end], is_first_audio_chunk=False, **kw)
        mel += processor.num_mel_frames_per_step
        start = processor.audio_chunk_start(mel)
    yield mel, processor(audio[start:], is_first_audio_chunk=False, is_last_audio_chunk=True, **kw)


class Probe:
    """Records each step's speaker-cache state, and every boundary tie of its top-k selections.

    With `force` (a run's recorded steps), the state after each step is set to that run's: the
    same frames in the cache and the FIFO, as this run computed them. A twin of a run so forced
    differs from it by its arithmetic alone, never by a selection made otherwise.
    """

    def __init__(self, silence, force=None):
        self.silence = silence.detach()
        self.rows_by_frame = {}
        # the frame each slot of the cache and of the FIFO holds, -1 a silence slot
        self.cache_src = []
        self.fifo_src = []
        # the rows a compression kept, as indices of its input (-1 its silence row)
        self.picked = None
        self.steps = []
        self.ties = 0
        self.topk_calls = 0
        self.base = 0
        self.force = force

    def stack(self, frames, like):
        rows = [self.silence.to(like) if f < 0 else self.rows_by_frame[f] for f in frames]
        return torch.stack(rows) if rows else like.new_zeros((0, like.shape[-1]))


def tie_at_boundary(scores, k):
    vals, _ = torch.topk(scores, k, dim=1, sorted=True)
    kth = vals[:, k - 1:k]
    return bool(((scores >= kth).sum(dim=1, keepdim=True) > k).logical_and(torch.isfinite(kth)).any())


def instrument(probe):
    """Wraps the cache's update and top-k selections to feed `probe`; returns the undo."""
    orig_update = M.Nemotron3DiarizationSpeakerCache.update
    orig_boost = M.Nemotron3DiarizationSpeakerCache._boost_scores
    orig_compress = M.Nemotron3DiarizationSpeakerCache._compress
    orig_topk = torch.topk

    def update(self, chunk_input_embeds, chunk_logits, silence_embeds, num_chunk_frames, mask=None):
        cached = self.num_cache_frames + self.num_fifo_frames
        chunk = chunk_input_embeds[:, cached:cached + num_chunk_frames]
        for i in range(num_chunk_frames):
            probe.rows_by_frame[probe.base + i] = chunk[0, i].clone()
        probs = self._pool_probs(chunk_logits, mask)
        fifo_all = probe.fifo_src + list(range(probe.base, probe.base + num_chunk_frames))
        probe.picked = None
        orig_update(self, chunk_input_embeds, chunk_logits, silence_embeds, num_chunk_frames, mask)
        # Which frames the cache and the FIFO now hold, from the frames they held: digital silence
        # embeds frames identically, so a row's bytes do not tell its frame.
        popped = len(fifo_all) - self.num_fifo_frames
        probe.fifo_src = fifo_all[popped:]
        if popped:
            cache_all = probe.cache_src + fifo_all[:popped]
            probe.cache_src = cache_all if probe.picked is None else [cache_all[i] if i >= 0 else -1 for i in probe.picked]
        assert len(probe.cache_src) == self.num_cache_frames, (len(probe.cache_src), self.num_cache_frames)
        if probe.force is not None:
            rec = probe.force[len(probe.steps)]
            for buf, frames, attr in ((self.embeds, rec["cache"], "num_cache_frames"), (self.fifo, rec["fifo"], "num_fifo_frames")):
                buf[0, :len(frames)] = probe.stack(frames, buf)
                setattr(self, attr, len(frames))
            self.is_compressed = rec["compressed"]
            probe.cache_src, probe.fifo_src = list(rec["cache"]), list(rec["fifo"])
            step = dict(rec)
        else:
            step = {
                "cache": list(probe.cache_src),
                # copies: the cache's buffers are written again by every later step
                "cache_probs": self.probs[0, :self.num_cache_frames].detach().to(torch.float32).numpy().copy(),
                "fifo": list(probe.fifo_src),
                "compressed": bool(self.is_compressed),
            }
        step.update({"first_frame": probe.base, "inputs": int(chunk_input_embeds.shape[1]), "cached": int(cached),
                     "chunk": int(num_chunk_frames), "probs": probs[0].detach().to(torch.float32).numpy().copy()})
        probe.steps.append(step)
        probe.base += num_chunk_frames

    def compress(self, embeds, probs, silence_embeds):
        # The checkpoint's own compression, each row carrying its index in one more column: the
        # rows it keeps say which they are.
        n = embeds.shape[1]
        idx = torch.arange(n, dtype=embeds.dtype, device=embeds.device).view(1, n, 1).expand(embeds.shape[0], n, 1)
        tagged_silence = torch.cat([silence_embeds.to(embeds), silence_embeds.new_full((1,), -1.0).to(embeds)])
        kept, kept_probs = orig_compress(self, torch.cat([embeds, idx], dim=-1), probs, tagged_silence)
        probe.picked = [int(round(v)) for v in kept[0, :, -1].tolist()]
        return kept[..., :-1], kept_probs

    def boost(self, scores, num_boosted, boost):
        probe.topk_calls += 1
        probe.ties += tie_at_boundary(scores, num_boosted)
        return orig_boost(self, scores, num_boosted, boost)

    def topk(x, k, dim=-1, largest=True, sorted=True, **kw):
        # the compression's own selection: a flat row per batch, the cache's length
        if x.dim() == 2 and dim == 1 and not sorted:
            probe.topk_calls += 1
            probe.ties += tie_at_boundary(x, k)
        return orig_topk(x, k, dim=dim, largest=largest, sorted=sorted, **kw)

    M.Nemotron3DiarizationSpeakerCache.update = update
    M.Nemotron3DiarizationSpeakerCache._boost_scores = boost
    M.Nemotron3DiarizationSpeakerCache._compress = compress
    M.torch.topk = topk

    def undo():
        M.Nemotron3DiarizationSpeakerCache.update = orig_update
        M.Nemotron3DiarizationSpeakerCache._boost_scores = orig_boost
        M.Nemotron3DiarizationSpeakerCache._compress = orig_compress
        M.torch.topk = orig_topk

    return undo


@torch.inference_mode()
def run(processor, model, audio, geometry, dtype=torch.float32, probe=None, mel=None, taps=None):
    """Per-frame speaker probabilities (rows of 8 per 10 ms) of a streamed clip."""
    cache = new_cache(model, geometry)
    undo = instrument(probe) if probe else (lambda: None)
    hooks = []
    try:
        out = []
        for step, (first_mel, inputs) in enumerate(chunks(processor, audio, geometry)):
            feats = inputs["input_features"]
            if mel is not None:
                for i, row in enumerate(feats[0].numpy()):
                    mel.setdefault(first_mel + i, row.copy())
            if taps is not None and step == 0:
                hooks = tap(model, taps)
            kw = {k: v for k, v in inputs.items() if k not in ("input_features",)}
            o = model(input_features=feats.to(dtype), speaker_cache=cache, **kw)
            for h in hooks:
                h.remove()
            hooks = []
            out.append(torch.sigmoid(o.logits[0].to(torch.float64)).numpy())
            cache = o.speaker_cache
        return np.concatenate(out)
    finally:
        undo()
        for h in hooks:
            h.remove()


def twin(processor, model64, audio, geometry, probe32):
    """The fp64 twin of a recorded fp32 run, forced to its state at every step."""
    p = Probe(model64.silence_embeds, force=probe32.steps)
    return run(processor, model64, audio, geometry, dtype=torch.float64, probe=p)


def add_steps(blob, g, steps):
    meta = []
    for i, s in enumerate(steps):
        blob.add(f"{g}.step.{i}.probs", s["probs"])
        blob.add(f"{g}.step.{i}.cache", np.array(s["cache"], dtype=np.int32), dtype="<i4")
        blob.add(f"{g}.step.{i}.fifo", np.array(s["fifo"], dtype=np.int32), dtype="<i4")
        blob.add(f"{g}.step.{i}.cache_probs", s["cache_probs"].reshape(-1, 8))
        meta.append({k: s[k] for k in ("first_frame", "inputs", "cached", "chunk", "compressed")})
    return meta


def tap(model, taps):
    rows = list(TAP_ROWS)

    def hook(name, all_rows=False):
        def f(_module, _inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            t = t[0].detach().to(torch.float32).numpy()
            taps[name] = t if all_rows else t[rows]
        return f

    tower = model.model.audio_tower
    hs = [tower.embedder.register_forward_hook(hook("embedder")),
          tower.input_layer_norm.register_forward_hook(hook("input_ln")),
          tower.layer_norm.register_forward_hook(hook("final_ln")),
          model.model.proj.register_forward_hook(hook("proj", True)),
          model.model.upsampler.register_forward_hook(hook("upsampled", True)),
          model.classifier.register_forward_hook(hook("logits", True))]
    for i, layer in enumerate(tower.layers):
        hs.append(layer.register_forward_hook(hook(f"layer.{i}")))
    return hs


def worst(a, b):
    n = min(len(a), len(b))
    assert len(a) == len(b), (len(a), len(b))
    return float(np.abs(a[:n] - b[:n]).max())


def cmd_excerpt(a):
    check_pins()
    x = read_wav(a.conv)
    piece = x[int(a.start * RATE):int(a.end * RATE)]
    sf.write(a.out, (np.round(piece * 32768.0)).clip(-32768, 32767).astype(np.int16), RATE, subtype="PCM_16")
    print(json.dumps({"out": a.out, "seconds": len(piece) / RATE, "sha256": sha256(a.out)}))


def cmd_record(a):
    check_pins()
    torch.set_num_threads(a.threads)
    processor, model, model64 = load(a.model)
    meta = {"reference": f"transformers {transformers.__version__} nemotron3_diarization, FP32, streaming as the "
                         "model card drives it", "revision": REVISION, "weights_sha256": WEIGHTS_SHA256,
            "geometry_fields": ["speaker_cache", "fifo", "chunk", "right_context", "update_period"],
            "probs": "sigmoid of the logits, one row of 8 speakers per 10 ms mel frame"}
    worst_by = {g: 0.0 for g in GEOMETRIES}
    ties = {}

    # the fixture clip: every committed record
    audio = read_wav(a.fixture_clip)
    fx = Blob()
    private = Blob()
    mel = {}
    steps_meta = {}
    for g in FIXTURE_GEOMETRIES:
        probe = Probe(model.silence_embeds)
        taps = {} if g == "live" else None
        p32 = run(processor, model, audio, GEOMETRIES[g], probe=probe, mel=mel if g == "live" else None, taps=taps)
        p64 = twin(processor, model64, audio, GEOMETRIES[g], probe)
        worst_by[g] = max(worst_by[g], worst(p32, p64))
        ties[f"fixture.{g}"] = {"topk_calls": probe.topk_calls, "boundary_ties": probe.ties}
        fx.add(f"{g}.probs", p32)
        steps_meta[g] = add_steps(fx, g, probe.steps)
        if taps:
            for k, v in taps.items():
                private.add(f"taps.{k}", v)
        print(f"fixture {g}: {len(p32)} frames, {len(probe.steps)} steps, fp32-fp64 {worst(p32, p64):.3e}, "
              f"ties {probe.ties}/{probe.topk_calls}", flush=True)
    frames = sorted(mel)
    keep = frames[:MEL_HEAD] + frames[-MEL_TAIL:]
    fx.add("mel.frames", np.array(keep, dtype=np.int32), dtype="<i4")
    fx.add("mel.rows", np.stack([mel[f] for f in keep]))
    fx.write(os.path.join(a.out, "fixture"), {**meta, "clip": os.path.basename(a.fixture_clip),
                                              "mel_frames": len(frames), "geometries": {g: GEOMETRIES[g] for g in FIXTURE_GEOMETRIES},
                                              "steps": steps_meta, "tap_rows": list(TAP_ROWS)})
    private.write(os.path.join(a.private_out, "fixture-taps"), {**meta, "tap_rows": list(TAP_ROWS),
                                                               "taps_step": "the first step of the live geometry"})

    # the conversations: the --ignored legs' records, on the study disk
    for c in [c for c in a.convs.split(",") if c]:
        x = read_wav(os.path.join(a.conv_dir, f"{c}.wav"))
        blob = Blob()
        conv_steps = {}
        for g in CONV_GEOMETRIES:
            probe = Probe(model.silence_embeds)
            p32 = run(processor, model, x, GEOMETRIES[g], probe=probe)
            p64 = twin(processor, model64, x, GEOMETRIES[g], probe)
            worst_by[g] = max(worst_by[g], worst(p32, p64))
            ties[f"{c}.{g}"] = {"topk_calls": probe.topk_calls, "boundary_ties": probe.ties}
            blob.add(f"{g}.probs", p32)
            conv_steps[g] = add_steps(blob, g, probe.steps)
            print(f"{c} {g}: {len(p32)} frames, fp32-fp64 {worst(p32, p64):.3e}, ties {probe.ties}/{probe.topk_calls}",
                  flush=True)
        blob.write(os.path.join(a.private_out, c), {**meta, "clip": f"{c}.wav", "steps": conv_steps,
                                                    "geometries": {g: GEOMETRIES[g] for g in CONV_GEOMETRIES}})

    write_json(os.path.join(a.out, "thresholds.json"), {
        "schema": 1,
        "metric": "max over frames and speakers of |p - the reference's p|, p a probability",
        "fp32_vs_fp64": worst_by,
        "factor": a.factor,
        "floor": a.floor,
        "bound": {g: max(a.factor * w, a.floor) for g, w in worst_by.items()},
        "ties": ties,
        "mel": json.load(open(a.mel_bound, encoding="utf-8")) if a.mel_bound else None,
    })
    print(json.dumps({"fp32_vs_fp64": worst_by}), flush=True)


def mel64(audio, center=True, extra_pad=0):
    """The front end in fp64: preemphasis 0.97, a 400-sample Hann window (symmetric) centred in a 512-point
    FFT every 160 samples, the power spectrum, 128 Slaney mel filters (0-8 kHz), ln(x + 2^-24)."""
    x = audio.astype(np.float64)
    x = np.concatenate([x[:1], x[1:] - 0.97 * x[:-1]])
    if extra_pad:
        x = np.concatenate([x, np.zeros(extra_pad)])
    if center:
        x = np.concatenate([np.zeros(256), x, np.zeros(256)])
    n = (len(x) - 512) // 160 + 1
    win = np.zeros(512)
    win[56:456] = np.hanning(400)
    frames = np.stack([x[i * 160:i * 160 + 512] * win for i in range(n)])
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    fb = librosa.filters.mel(sr=RATE, n_fft=512, n_mels=128, fmin=0.0, fmax=RATE / 2, norm="slaney", dtype=np.float64)
    return np.log(power @ fb.T + 2.0 ** -24)


def cmd_mel64(a):
    check_pins()
    processor = AutoProcessor.from_pretrained(a.model)
    audio = read_wav(a.fixture_clip)
    fe = processor.feature_extractor
    out = fe(audio, sampling_rate=RATE, return_tensors="np")
    feats = out["input_features"][0].astype(np.float64)
    ref = mel64(audio)
    # the frames past the audio's own are zeroed by the attention mask, not computed
    n = int(out["attention_mask"][0].sum())
    assert n <= min(len(feats), len(ref)), (n, len(feats), len(ref))
    err = float(np.abs(feats[:n] - ref[:n]).max())
    out = {"metric": "max over frames and bins of |log-mel - fp64|, natural log", "fp32_vs_fp64": err,
           "factor": a.factor, "bound": max(a.factor * err, a.floor), "frames": n}
    write_json(a.out, out)
    print(json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("excerpt")
    e.add_argument("--conv", required=True, help="a conversation's .wav (H:/speech-study/nemotron-spike/data/conv)")
    e.add_argument("--start", type=float, required=True)
    e.add_argument("--end", type=float, required=True)
    e.add_argument("--out", required=True, help="crates/speech/tests/fixtures/diarizer/<clip>.wav")
    r = sub.add_parser("record")
    m = sub.add_parser("mel64")
    for p in (r, m):
        p.add_argument("--model", required=True, help="nvidia/Nemotron-3-Diarization @ f667ed73: config, processor, weights")
        p.add_argument("--fixture-clip", required=True)
        p.add_argument("--factor", type=float, default=10.0)
        p.add_argument("--floor", type=float, default=1e-5)
    r.add_argument("--out", required=True, help="crates/speech/tests/fixtures/diarizer")
    r.add_argument("--private-out", required=True, help="the study disk's records: taps and conversations")
    r.add_argument("--conv-dir", default="H:/speech-study/nemotron-spike/data/conv")
    r.add_argument("--convs", default="conv03,conv08,conv15")
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--mel-bound", help="mel64's output, folded into thresholds.json")
    m.add_argument("--out", required=True, help="where mel64 writes its bound (JSON)")
    a = ap.parse_args()
    {"excerpt": cmd_excerpt, "record": cmd_record, "mel64": cmd_mel64}[a.cmd](a)


if __name__ == "__main__":
    main()
