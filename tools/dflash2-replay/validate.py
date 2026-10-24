"""Replay vs the live ninfer rounds of the same turns (GitHub #267).

python validate.py OUT_DIR MANIFEST

The recorded text determines a live run's acceptance exactly (a one-hot draft is accepted
iff the sampled token equals it), so replaying it with the same drafter and target gives
the live rounds back. Per turn, ninfer's request log has what its live run did: rounds and
accepted-per-position. Any gap is engine numerics (ignis vs ninfer target and drafter), the
context the replay could rebuild (qwen-code transcripts lack the system prompt and tool
schemas), or a reconstructed text that is not what was generated.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from records import RECORD  # noqa: E402


def replay_acceptance(out, manifest):
    """P(accepted >= i), i = 1..7, of the replay's rounds per source."""
    import os
    src = {}
    for j in manifest["jobs"]:
        for t in j["turns"]:
            src[t["turn"]] = t["source"]
    acc = defaultdict(list)
    for j in manifest["jobs"]:
        path = f"{out}/{j['job']}.rounds"
        if os.path.exists(f"{out}/{j['job']}.done"):
            r = np.fromfile(path, dtype=RECORD)
            for turn, c in zip(r["turn"], r["committed"]):
                acc[src[int(turn)]].append(int(c) - 1)
    return {s: np.array([(np.array(v) >= i).mean() for i in range(1, 8)]) for s, v in acc.items()}


def main():
    out = sys.argv[1]
    manifest = json.load(open(sys.argv[2]))
    ours_app = replay_acceptance(out, manifest)
    lines = [json.loads(l) for l in open(f"{out}/turns.jsonl")]
    seen = {}
    for l in lines:
        seen[(l["job"], l["turn"])] = l  # a resumed job rewrites its lines; keep the last
    by = defaultdict(list)
    for l in seen.values():
        nf = l["ninfer"]
        by[l["source"]].append((l["output"], l["rounds"], nf["completion_tokens"], nf["rounds"],
                                l["kind"], nf["accepted_per_position"]))
    print("source  turns  replay tok/round  ninfer tok/round  ratio  per-turn |log ratio| median  within 5%")
    for s in sorted(by):
        r = np.array([x[:4] for x in by[s]], float)
        ours = r[:, 0].sum() / r[:, 1].sum()
        theirs = r[:, 2].sum() / r[:, 3].sum()
        per = np.log((r[:, 0] / r[:, 1]) / (r[:, 2] / r[:, 3]))
        print(f"{s:6} {len(r):6} {ours:17.3f} {theirs:17.3f} {ours / theirs:6.3f} {np.median(np.abs(per)):12.3f} "
              f"{np.mean(np.abs(per) < np.log(1.05)):10.2f}")
        for kind in ("think", "content"):
            sel = [x for x in by[s] if x[4] == kind]
            if not sel:
                continue
            q = np.array([x[:4] for x in sel], float)
            print(f"   {kind:8} {len(q):4} {q[:, 0].sum() / q[:, 1].sum():14.3f} {q[:, 2].sum() / q[:, 3].sum():17.3f}")
        # per-position acceptance rate, ninfer: accepted_per_position[i] / rounds
        app = np.array([x[5] for x in by[s]], float).sum(0) / r[:, 3].sum()
        print("   ninfer P(accept >= i):", " ".join(f"{v:.3f}" for v in app))
        print("   replay P(accept >= i):", " ".join(f"{v:.3f}" for v in ours_app.get(s, [])))


if __name__ == "__main__":
    main()
