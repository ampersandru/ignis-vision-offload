"""Long items with no pause for spec speech/08's length guard (acceptance 12): continuous Italian speech of 60, 120
and 240 s, built from VoxPopuli, which `server_vad` would never end at a silence.

Two subcommands:

concat     per speaker, their VoxPopuli segments (test and validation, in the pool's order) concatenated into
           one WAV, with each segment's span and normalized text in a JSON beside it.
squeeze    from that WAV and `server_vad`'s per-frame probabilities of it (`cargo run --release -p
           ignis-speech --example vad_probs`), every run of frames under the VAD's threshold longer than
           --max-pause shortened to it, then the first segments reaching each length cut out as an item:
           `<speaker>-<length>.wav` and its reference (the segments' normalized texts) in `items.json`.

The items' audio is VoxPopuli's (CC0); they stay on the study disk.

usage: python -I long_items.py concat --pool DIR --speakers A,B --out DIR
       python -I long_items.py squeeze --dir DIR --speakers A,B [--lengths 60,120,240] [--max-pause 0.3]
"""
import argparse
import csv
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

RATE = 16000
VAD_FRAME = 512
THRESHOLD = 0.5


def cmd_concat(a):
    rows = list(csv.DictReader(open(os.path.join(a.pool, "pool.csv"), encoding="utf-8")))
    tables = {s: pq.read_table(os.path.join(a.pool, f"{s}.parquet")).to_pydict() for s in ("test", "validation")}
    os.makedirs(a.out, exist_ok=True)
    for spk in a.speakers.split(","):
        pieces, segs, at = [], [], 0
        for r in rows:
            if r["speaker_id"] != spk:
                continue
            x, sr = sf.read(io.BytesIO(tables[r["split"]]["audio"][int(r["idx"])]["bytes"]), dtype="float32")
            assert sr == RATE and x.ndim == 1
            pieces.append(x)
            segs.append({"audio_id": r["audio_id"], "start": at, "end": at + len(x), "text": r["normalized_text"]})
            at += len(x)
        audio = np.concatenate(pieces)
        sf.write(os.path.join(a.out, f"{spk}.wav"), audio, RATE, subtype="PCM_16")
        json.dump(segs, open(os.path.join(a.out, f"{spk}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{spk}: {len(segs)} segments, {at / RATE:.1f} s")


def cmd_squeeze(a):
    max_pause = int(round(a.max_pause * RATE / VAD_FRAME))
    lengths = [int(v) for v in a.lengths.split(",")]
    items = {}
    for spk in a.speakers.split(","):
        x, _ = sf.read(os.path.join(a.dir, f"{spk}.wav"), dtype="float32")
        segs = json.load(open(os.path.join(a.dir, f"{spk}.json"), encoding="utf-8"))
        p = np.fromfile(os.path.join(a.dir, f"{spk}.vad"), dtype="<f4")
        # sample ranges to drop: the middle of every quiet run longer than max_pause frames
        drop = []
        f = 0
        while f < len(p):
            if p[f] < THRESHOLD:
                g = f
                while g < len(p) and p[g] < THRESHOLD:
                    g += 1
                if g - f > max_pause:
                    keep = max_pause // 2
                    drop.append(((f + keep) * VAD_FRAME, (g - (max_pause - keep)) * VAD_FRAME))
                f = g
            else:
                f += 1
        keep = np.ones(len(x), dtype=bool)
        for s, e in drop:
            keep[s:min(e, len(x))] = False
        new_at = np.cumsum(keep) - keep  # each old sample's position in the squeezed audio
        y = x[keep]
        for seg in segs:
            seg["new_end"] = int(new_at[min(seg["end"], len(x)) - 1]) + 1
        for n in lengths:
            k = next(i for i, s in enumerate(segs) if s["new_end"] >= n * RATE)
            end = segs[k]["new_end"]
            name = f"{spk}-{n}"
            sf.write(os.path.join(a.dir, f"{name}.wav"), y[:end], RATE, subtype="PCM_16")
            items[name] = {"seconds": end / RATE, "segments": k + 1, "text": " ".join(s["text"] for s in segs[:k + 1])}
            print(f"{name}: {end / RATE:.1f} s, {k + 1} segments, {len(drop)} pauses shortened in the speaker's audio")
    json.dump(items, open(os.path.join(a.dir, "items.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("concat")
    c.add_argument("--pool", required=True, help="H:/speech-study/nemotron-spike/data/voxpopuli_it")
    c.add_argument("--speakers", required=True)
    c.add_argument("--out", required=True)
    s = sub.add_parser("squeeze")
    s.add_argument("--dir", required=True)
    s.add_argument("--speakers", required=True)
    s.add_argument("--lengths", default="60,120,240")
    s.add_argument("--max-pause", type=float, default=0.3, help="seconds")
    a = ap.parse_args()
    {"concat": cmd_concat, "squeeze": cmd_squeeze}[a.cmd](a)


if __name__ == "__main__":
    main()
