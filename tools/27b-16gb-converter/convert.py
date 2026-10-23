"""The 16 GB artifact's converter (spec docs/specs/27b-16gb/01-the-16gb-artifact.md, GitHub #338).

`run` writes the packer's work tree (docs/specs/flash-next/layout.md §2: per unit a
`tensors.json`, one payload file per tensor and a `DONE` written last; `frontend/` holds the
resources) and `converter.json`, the converter's half of the sidecar. The packer
(`ignis-artifact-pack --family qwen3.8-27b-trellis`) assembles the container from it; Python never
writes the container.

- The body and the output head are the public quant's trellis records, repacked bit for bit
  (`exl3.py`); everything else is the served artifact's object, byte for byte (`plan.py`).
- The source is pinned: the quant's repository, branch and revision (from its Hugging Face
  download metadata), every file the converter reads by sha256, each shard's checked against its
  LFS oid; the served artifact and its graft sidecar by sha256.
- Every trellis matrix's reconstruction error against the BF16 checkpoint is recorded, and one
  above `--error-bound` fails the conversion (`errors.py`, the GPU step). `--skip-errors` writes a
  `dry-run` tree instead (the packer packs it only when asked).

`verify` reads a packed artifact back: its identity, its object table against the plan, every
trellis object against its source tensors bit for bit, every copied object's sha256 against the
served one's. It prints VERIFY PASS or VERIFY FAILED (exit 3).

  python convert.py run --served S.ninfer --quant DIR --bf16 DIR --work DIR [--skip-errors]
  python convert.py verify --artifact A.ninfer --served S.ninfer --quant DIR [--report R.json]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import container
import exl3
import plan
import work

SCHEMA = "qwen3.8-27b-trellis-converter-v1"
IDENTITY = {"model_id": "qwen3.8-27b", "weights_id": "trellis-sc30"}
SERVED_IDENTITY = {"model_id": "qwen3.8-27b", "weights_id": "nvfp4full"}
REPO = "turboderp/Qwen3.8-27B-exl3"
BRANCH = "SC_3.00bpw_H4"
# The quant's recipe as its quantization_config.json states it (ADR 0060 §1, §4).
QUANT_CONFIG = {"quant_method": "exl3", "codebook": "mul1", "out_scales": "always", "bits": 3.0, "head_bits": 4}
# The quant's files the converter reads, beside its shards.
QUANT_META = ("model.safetensors.index.json", "quantization_config.json", "config.json")


class Refused(Exception):
    """A conversion that must not produce a work tree."""


def sha256_file(path, block=64 << 20):
    return work.file_digest(path, block)


def hf_metadata(root, name):
    """(revision, etag) of a file in a Hugging Face download (`.cache/huggingface/download`)."""
    path = os.path.join(root, ".cache", "huggingface", "download", name + ".metadata")
    if not os.path.isfile(path):
        raise Refused(f"{root} has no Hugging Face download record for {name}: the revision cannot be pinned")
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    return lines[0].strip(), lines[1].strip()


def check_quant_config(config):
    for k, v in QUANT_CONFIG.items():
        if config.get(k) != v:
            raise Refused(f"the quant's quantization_config.json has {k} = {config.get(k)!r}, not {v!r}")


def text_matrices(quant):
    """(matrices, unused): the quant's text body and head, keyed; the other quantized keys."""
    matrices, unused = {}, []
    for key in quant.keys():
        if key.startswith("model.language_model.") or key == "lm_head":
            matrices[key] = quant.matrix(key)
        else:
            unused.append(key)
    return matrices, unused


def file_name(p):
    """The work file of object `p` inside its unit: the name past the unit, '/' -> '.'."""
    rest = p.name[len(p.unit) + 1:]
    return rest if p.kind == "resource" else rest.replace("/", ".") + ".bin"


def tool_commit():
    """{tool, commit, dirty}: the commit the converter ran from, and whether its files differed from it."""
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*args):
        return subprocess.run(["git", *args], cwd=here, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"tool": "tools/27b-16gb-converter", "commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--", "."))}
    except (OSError, subprocess.CalledProcessError):
        return {"tool": "tools/27b-16gb-converter", "commit": None, "dirty": None}


def versions():
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for dist in ("torch", "exllamav3", "safetensors"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def convert(served_path, quant_root, bf16_root, out_work, skip_errors=False, error_bound=None,
            errors_fn=None, log=print):
    """Writes the work tree `out_work`; returns the converter.json record. Raises Refused."""
    import errors as errors_mod
    error_bound = errors_mod.ERROR_BOUND if error_bound is None else error_bound
    t0 = time.time()
    if os.path.isdir(out_work) and os.listdir(out_work):
        raise Refused(f"{out_work} is not empty: remove it to convert again")
    served = container.Container(served_path)
    if served.identity != SERVED_IDENTITY:
        raise Refused(f"{served_path} is {served.identity}, not the served artifact {SERVED_IDENTITY}")
    graft_path = served_path + ".graft.json"
    if not os.path.isfile(graft_path):
        raise Refused(f"{graft_path} is missing: the DFlash2 graft's provenance comes from it")
    with open(graft_path, encoding="utf-8") as f:
        graft = json.load(f)
    quant = exl3.Quant(quant_root)
    check_quant_config(quant.config)

    matrices, unused = text_matrices(quant)
    objects, dropped = plan.plan(served.objects, matrices)
    log(f"plan: {len(objects)} objects ({sum(p.key is not None for p in objects)} trellis), "
        f"{len(dropped)} served objects dropped, {len(unused)} quant matrices unused")

    # The pins hash ~33 GB; they run beside the GPU step.
    revisions = {}
    pinned = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {name: pool.submit(sha256_file, os.path.join(quant_root, name))
                   for name in list(quant.shards) + list(QUANT_META)}
        served_digest = pool.submit(sha256_file, served_path)
        if skip_errors:
            rel = None
        else:
            rel = (errors_fn or errors_mod.rel_errors)(quant, bf16_root, objects, log=log)
        for name, fut in futures.items():
            revision, etag = hf_metadata(quant_root, name)
            revisions[name] = revision
            digest = fut.result()
            if name in quant.shards and digest != etag:
                raise Refused(f"{name} hashes to {digest}, its LFS oid is {etag}")
            pinned[name] = {"bytes": os.path.getsize(os.path.join(quant_root, name)), "sha256": digest}
        served_sha256 = served_digest.result()
    if len(set(revisions.values())) != 1:
        raise Refused(f"the quant's files come from different revisions: {sorted(set(revisions.values()))}")
    if rel is not None:
        over = {n: e for n, e in rel.items() if not e <= error_bound}
        if over:
            raise Refused(f"reconstruction error above {error_bound} (a layout error): {over}")

    trellis, copied = {}, {}
    for unit in plan.UNITS:
        d = os.path.join(out_work, unit)
        os.makedirs(d)
        entries, digests = [], {}
        started = time.time()
        for p in (p for p in objects if p.unit == unit):
            if p.key is not None:
                m = matrices[p.key]
                payload = quant.record(m)
            else:
                payload = served.read(p.copy)
            digest = hashlib.sha256(payload).hexdigest()
            fname = file_name(p)
            if fname in digests:
                raise ValueError(f"two objects of {unit} share the file name {fname}")
            with open(os.path.join(d, fname), "wb") as f:
                f.write(payload)
            digests[fname] = digest
            if p.key is not None:
                trellis[p.name] = {"source": p.key, "k2": m.k2, "format": p.format,
                                   "shape": p.shape, "bytes": len(payload), "sha256": digest,
                                   "rel_error": None if rel is None else round(rel[p.name], 6)}
            else:
                if served.by_name[p.copy]["bytes"] != len(payload):
                    raise ValueError(f"{p.copy}: read {len(payload)} bytes")
                copied[p.name] = {"bytes": len(payload), "sha256": digest}
            if p.kind == "tensor":
                entries.append({"name": p.name, "file": fname, "format": p.format, "layout": p.layout,
                                "shape": p.shape, "bytes": len(payload)})
        if unit != "frontend":
            work.write_json_atomic(os.path.join(d, "tensors.json"), {"tensors": entries})
        work.mark_done(d, digests)
        log(f"{unit}: {len(digests)} objects, {time.time() - started:.1f} s")

    nvfp4_copied = {p.name for p in objects if p.copy is not None and p.format == "NVFP4"}
    local_nvfp4 = dict(graft.get("local_nvfp4") or {})
    local_nvfp4["parents"] = {n: v for n, v in (local_nvfp4.get("parents") or {}).items() if n in nvfp4_copied}
    errors_list = [t["rel_error"] for t in trellis.values() if t["rel_error"] is not None]
    k_hist = {}
    for t in trellis.values():
        k_hist[t["format"]] = k_hist.get(t["format"], 0) + 1
    record = {
        "schema": SCHEMA,
        "status": "dry-run" if skip_errors else "complete",
        "identity": IDENTITY,
        "source": {
            "repo": REPO,
            "branch": BRANCH,
            "revision": next(iter(revisions.values())),
            "license": "apache-2.0",
            "quantization_config": {k: quant.config.get(k) for k in QUANT_CONFIG},
            "files": pinned,
        },
        "served": {
            "file": os.path.basename(served_path),
            "bytes": os.path.getsize(served_path),
            "sha256": served_sha256,
            "identity": served.identity,
            "graft_sidecar": {"file": os.path.basename(graft_path), "sha256": sha256_file(graft_path)},
        },
        "graft": {
            "how": "the 66 dflash2/* objects copied byte for byte from the served artifact, which grafted them",
            "recipe_id": graft.get("recipe_id"),
            "grafted_from": graft.get("grafted_from"),
            "source": graft.get("source"),
            "objects": graft.get("objects"),
        },
        # The served graft's per-object NVFP4 records of the objects copied here, so
        # `checksum::verify` covers the copied drafter in this artifact as in the served one.
        "local_nvfp4": local_nvfp4,
        "objects": {"tensors": sum(p.kind == "tensor" for p in objects),
                    "resources": sum(p.kind == "resource" for p in objects),
                    "trellis": len(trellis), "copied": len(copied)},
        "trellis": {
            "bytes": sum(t["bytes"] for t in trellis.values()),
            "formats": dict(sorted(k_hist.items())),
            "error_bound": error_bound,
            "rel_error_max": max(errors_list) if errors_list else None,
            "rel_error_mean": round(sum(errors_list) / len(errors_list), 6) if errors_list else None,
            "matrices": trellis,
        },
        "copied": {"bytes": sum(c["bytes"] for c in copied.values()), "objects": copied},
        "dropped": dropped,
        "quant_unused": unused,
        "converter": tool_commit(),
        "versions": versions(),
    }
    work.write_json_atomic(os.path.join(out_work, "converter.json"), record)
    log(f"converter.json: {record['status']}, {len(trellis)} trellis + {len(copied)} copied, "
        f"{time.time() - t0:.0f} s")
    return record


def verify(artifact_path, served_path, quant_root, log=print):
    """(ok, report) for a packed artifact against the plan and its sources."""
    art = container.Container(artifact_path)
    served = container.Container(served_path)
    quant = exl3.Quant(quant_root)
    matrices, _ = text_matrices(quant)
    objects, _ = plan.plan(served.objects, matrices)
    problems = []
    if art.identity != IDENTITY:
        problems.append(f"identity {art.identity}, not {IDENTITY}")
    want = {p.name: p for p in objects}
    extra = sorted(set(art.by_name) - set(want))
    missing = sorted(set(want) - set(art.by_name))
    if extra:
        problems.append(f"objects the plan does not have: {extra[:5]}")
    if missing:
        problems.append(f"objects missing: {missing[:5]}")
    checked = {"trellis": 0, "copied": 0}
    for name, p in want.items():
        o = art.by_name.get(name)
        if o is None:
            continue
        if p.kind == "resource":
            if o["kind"] != "resource" or o["encoding"] != p.encoding:
                problems.append(f"{name}: {o['kind']} {o.get('encoding')}, not a {p.encoding} resource")
        elif (o["kind"], o["format"], o["layout"], o["shape"]) != ("tensor", p.format, p.layout, p.shape):
            problems.append(f"{name}: {o.get('format')} {o.get('layout')} {o.get('shape')}, "
                            f"not {p.format} {p.layout} {p.shape}")
        data = art.read(name)
        if p.key is not None:
            if data != quant.record(matrices[p.key]):
                problems.append(f"{name}: not {p.key}'s trellis, suh and svh bit for bit")
            checked["trellis"] += 1
        else:
            if hashlib.sha256(data).digest() != hashlib.sha256(served.read(p.copy)).digest():
                problems.append(f"{name}: sha256 differs from the served object's")
            checked["copied"] += 1
    report = {"artifact": os.path.basename(artifact_path), "identity": art.identity,
              "objects": len(art.objects), "checked": checked, "problems": problems}
    log(f"verify: {checked['trellis']} trellis objects, {checked['copied']} copied, {len(problems)} problems")
    return not problems, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--served", required=True)
    r.add_argument("--quant", required=True)
    r.add_argument("--bf16", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--skip-errors", action="store_true")
    r.add_argument("--error-bound", type=float, default=None)
    v = sub.add_parser("verify")
    v.add_argument("--artifact", required=True)
    v.add_argument("--served", required=True)
    v.add_argument("--quant", required=True)
    v.add_argument("--report")
    a = ap.parse_args(argv)
    log = lambda line: print(line, flush=True)
    if a.command == "run":
        try:
            convert(a.served, a.quant, a.bf16, a.work, skip_errors=a.skip_errors, error_bound=a.error_bound, log=log)
        except Refused as e:
            print(f"REFUSED: {e}", flush=True)
            return 2
        return 0
    ok, report = verify(a.artifact, a.served, a.quant, log=log)
    if a.report:
        work.write_json_atomic(a.report, report)
    for p in report["problems"]:
        print(f"  {p}", flush=True)
    print("VERIFY PASS" if ok else "VERIFY FAILED", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
