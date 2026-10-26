"""RTTMs from the diarizer's output, for spec speech/08's measurements (acceptance 3 and 5).

Reads what `cargo run --release -p ignis-speech --example diarize` writes per file: `<name>.<geometry>.probs`
(every 10 ms row's 8 probabilities, f32) and `<name>.<geometry>.speakers.json` (the 80 ms frames' speakers
after spec 08's rules, per ghost-rule length). Two subcommands, each writing `<name>.rttm` per file into --out:

nsc        the probabilities segmented as NeMo-Speech.cpp segments its own (src/asr/diar/diar_pipeline.cpp,
           diar_segments_from_probs, at caa3f47, with its defaults: onset 0.641, offset 0.561, pads 0.229 /
           0.079 s, gaps under 0.296 s filled, segments under 0.511 s dropped), so that the port's DER is
           scored like for like with that runtime's at the same geometry (acceptance 3).
speakers   the speakers' runs at one ghost-rule length, one segment per run of 80 ms frames (acceptance 5).

The DER itself: the finding's scorer, H:/speech-study/nemotron-spike/scripts/score_der.py --hyp-dir DIR.

usage: python -I der.py nsc|speakers --probs DIR --geometry live|file [--ghost S] --out DIR
"""
import argparse
import glob
import json
import os

import numpy as np

SPEAKERS = 8
ROW_S = 0.01
FRAME_S = 0.08
# NeMo-Speech.cpp's DiarSegmentationCfg defaults (NeMo's callhome-tuned values for this checkpoint)
ONSET, OFFSET, PAD_ONSET, PAD_OFFSET, MIN_ON, MIN_OFF = 0.641, 0.561, 0.229, 0.079, 0.511, 0.296


def nsc_segments(probs):
    """diar_segments_from_probs: per speaker, onset/offset hysteresis and padding, short gaps filled, short
    segments dropped."""
    n = len(probs)
    total = n * ROW_S
    out = []
    for s in range(SPEAKERS):
        segs, active, start = [], False, 0
        for f in range(n):
            p = probs[f, s]
            if not active and p > ONSET:
                active, start = True, f
            elif active and p < OFFSET:
                active = False
                segs.append((max(0.0, start * ROW_S - PAD_ONSET), min(total, f * ROW_S + PAD_OFFSET)))
        if active:
            segs.append((max(0.0, start * ROW_S - PAD_ONSET), total))
        merged = []
        for a, b in segs:
            if merged and a - merged[-1][1] < MIN_OFF:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        out += [(a, b, s) for a, b in merged if b - a >= MIN_ON]
    return sorted(out)


def write_rttm(path, name, segs):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for a, b, s in segs:
            f.write(f"SPEAKER {name} 1 {a:.3f} {b - a:.3f} <NA> <NA> speaker_{s} <NA> <NA>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["nsc", "speakers"])
    ap.add_argument("--probs", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--ghost", default="0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    suffix = f".{a.geometry}.probs" if a.cmd == "nsc" else f".{a.geometry}.speakers.json"
    files = sorted(glob.glob(os.path.join(a.probs, "*" + suffix)))
    assert files, f"no *{suffix} in {a.probs}"
    for path in files:
        name = os.path.basename(path)[: -len(suffix)]
        if a.cmd == "nsc":
            probs = np.fromfile(path, dtype="<f4").reshape(-1, SPEAKERS)
            segs = nsc_segments(probs)
        else:
            runs = json.load(open(path, encoding="utf-8"))["speakers"][a.ghost]
            segs = [(r[0] * FRAME_S, r[1] * FRAME_S, r[2]) for r in runs]
        write_rttm(os.path.join(a.out, name + ".rttm"), name, segs)
    print(json.dumps({"files": len(files), "out": a.out}))


if __name__ == "__main__":
    main()
