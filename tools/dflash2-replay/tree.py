"""Lever 3 ceiling (GitHub #267): a token tree verified in one round, built from the drafter's
own lattice, at a node budget.

python tree.py OUT_DIR MANIFEST [TEMPERATURES [BUDGETS]]

The tree grows best-first by the drafter's path probability: a node's children are the 16
candidates of the next column, each with q(child | node) = softmax over the selector's
transition scores after the node's candidate (the anchor's row at column 0), at drafter
temperature 1. The top `budget` nodes (the root excluded) make the tree; a chain of 7 is
budget 7 with the greedy walk's path. Acceptance: multi-draft rejection over a node's
one-hot children accepts one of them with the target's mass on the set, so along the
recorded text

    E[accepted] = sum_i [r_1..r_{i-1} is a path of the tree] * p_rec(children at depth i)

(the same coupling as analyze.py). Every branch needs its own GDN state and the verify a
tree mask: this prices the idea, it does not build it.
"""
import heapq
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from records import K, TOPK, load, read_manifest  # noqa: E402


def tree_value(scores, cand_p, rec_index, budget, temperature=1.0):
    """E[accepted] of one round's best-first tree of `budget` nodes."""
    x = scores / temperature
    q = np.exp(x - x.max(-1, keepdims=True))
    q /= q.sum(-1, keepdims=True)  # [7, pred, succ]
    # node: (depth, candidate index) path tuple; children keyed by parent path
    heap = [(-q[0, 0, i], (i,)) for i in range(TOPK)]
    heapq.heapify(heap)
    tree = set()
    while heap and len(tree) < budget:
        neg, path = heapq.heappop(heap)
        tree.add(path)
        depth = len(path)
        if depth < K:
            for i in range(TOPK):
                heapq.heappush(heap, (neg * q[depth, path[-1], i], path + (i,)))
    value, prefix = 0.0, ()
    for depth in range(K):
        children = [i for i in range(TOPK) if prefix + (i,) in tree]
        if not children:
            break
        value += cand_p[depth, children].sum()
        r = rec_index[depth]
        if r < 0 or (prefix + (int(r),)) not in tree:
            break
        prefix = prefix + (int(r),)
    return value


def main():
    out_dir = sys.argv[1]
    manifest = read_manifest(sys.argv[2])
    R, _ = load(out_dir, manifest)
    d = np.load(os.path.join(out_dir, "per_round.npz"))
    assert len(d["cand_p"]) == len(R), "per_round.npz is from another set of rounds: rerun analyze.py"
    cand_p, rec_index = d["cand_p"].astype(np.float64), d["rec_index"]
    cell_of = {t["turn"]: f"{t['source']}-{t['kind']}" for j in manifest["jobs"] for t in j["turns"]}
    cells = np.array([cell_of[int(t)] for t in R["turn"]])
    weights = {"A-content": 28586, "A-think": 58665, "B-content": 1524816, "B-think": 8088074,
               "C-content": 2085240, "C-think": 6471440}
    rng = np.random.default_rng(267)
    pick = rng.choice(len(R), size=min(len(R), 12000), replace=False)
    base = {"chain (walked)": 1 + np.array([d["exact"][n] - 1 for n in pick])}
    results = {}
    temperatures = [float(t) for t in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1.0]
    budgets = [int(b) for b in sys.argv[4].split(",")] if len(sys.argv) > 4 else [7, 15, 31, 63]
    for t in temperatures:
        for budget in budgets:
            results[f"tree {budget} nodes T{t}"] = 1 + np.array([
                tree_value(R["scores"][n].astype(np.float64), cand_p[n], rec_index[n], budget, t) for n in pick])
    for name, v in list(base.items()) + list(results.items()):
        pooled = sum(weights[c] for c in weights if (cells[pick] == c).any()) / sum(
            weights[c] / v[cells[pick] == c].mean() for c in weights if (cells[pick] == c).any())
        print(f"{name:18} mean {v.mean():.3f}  traffic-weighted {pooled:.3f}")


if __name__ == "__main__":
    main()
