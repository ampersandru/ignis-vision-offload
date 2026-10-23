import hashlib
import json
import os

import pytest

import container
import convert
import exl3
from conftest import BF16_REVISION, REVISION, make_sources, write_container


def _errors(value=0.25):
    def fn(quant, bf16_root, objects, log=print):
        return {p.name: value for p in objects if p.key is not None}
    return fn


def _run(src, out, **kw):
    kw.setdefault("errors_fn", _errors())
    return convert.convert(src["served"], src["quant"], src["bf16"], out, log=lambda _: None, **kw)


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _never(*args, **kw):
    raise AssertionError("the GPU step ran before a refusal that needs none")


def pack(work, artifact, as_resource=()):
    """What the packer does with a work tree, for `verify`'s tests: the units in order, frontend
    files by DONE's (sorted) names, tensors in tensors.json order; the tensors named in
    `as_resource` written as resources instead (a broken artifact)."""
    objects = []
    for unit in ("frontend", "text", "mtp", "vision", "dflash2"):
        d = os.path.join(work, unit)
        if unit == "frontend":
            for name in sorted(_json(os.path.join(d, "DONE"))["files"]):
                with open(os.path.join(d, name), "rb") as f:
                    objects.append(({"name": f"frontend/{name}", "kind": "resource", "encoding": "raw-bytes-v1"},
                                    f.read()))
            continue
        for t in _json(os.path.join(d, "tensors.json"))["tensors"]:
            if t["name"] in as_resource:
                desc = {"name": t["name"], "kind": "resource", "encoding": "raw-bytes-v1"}
            else:
                desc = {k: t[k] for k in ("name", "format", "layout", "shape")} | {"kind": "tensor"}
            with open(os.path.join(d, t["file"]), "rb") as f:
                objects.append((desc, f.read()))
    write_container(artifact, convert.IDENTITY, objects)


def test_run_writes_a_work_tree_the_packer_takes(tmp_path):
    src = make_sources(str(tmp_path))
    work = str(tmp_path / "work")
    record = _run(src, work)
    assert sorted(os.listdir(work)) == ["converter.json", "dflash2", "frontend", "mtp", "text", "vision"]
    served = container.Container(src["served"])
    quant = exl3.Quant(src["quant"])
    for unit in ("frontend", "text", "mtp", "vision", "dflash2"):
        d = os.path.join(work, unit)
        done = _json(os.path.join(d, "DONE"))["files"]
        for name, entry in done.items():
            assert entry == {"bytes": os.path.getsize(os.path.join(d, name)), "sha256": _sha(os.path.join(d, name))}
        if unit == "frontend":
            assert sorted(done) == ["chat_template.jinja", "tokenizer.json"]
            continue
        for t in _json(os.path.join(d, "tensors.json"))["tensors"]:
            assert t["name"].startswith(unit + "/") and t["bytes"] == done[t["file"]]["bytes"]
            with open(os.path.join(d, t["file"]), "rb") as f:
                data = f.read()
            source = record["trellis"]["matrices"].get(t["name"])
            if source:
                assert t["layout"] == "trellis-tile16-v1"
                assert data == quant.record(quant.matrix(source["source"]))
            else:
                assert data == served.read(t["name"])
    assert record == _json(os.path.join(work, "converter.json"))


def test_converter_json_carries_the_provenance(tmp_path):
    src = make_sources(str(tmp_path))
    record = _run(src, str(tmp_path / "work"))
    assert record["status"] == "complete"
    assert record["identity"] == {"model_id": "qwen3.8-27b", "weights_id": "trellis-sc30"}
    s = record["source"]
    assert (s["repo"], s["branch"], s["revision"]) == ("turboderp/Qwen3.8-27B-exl3", "SC_3.00bpw_H4", REVISION)
    for name, pin in s["files"].items():
        assert pin == {"bytes": os.path.getsize(os.path.join(src["quant"], name)),
                       "sha256": _sha(os.path.join(src["quant"], name))}
    assert sorted(s["files"]) == ["config.json", "model-00001-of-00002.safetensors",
                                  "model-00002-of-00002.safetensors", "model.safetensors.index.json",
                                  "quantization_config.json"]
    assert record["served"]["sha256"] == _sha(src["served"])
    ref = record["reference"]
    assert (ref["repo"], ref["revision"]) == ("Qwen/Qwen3.8-27B", BF16_REVISION)
    for name, pin in ref["files"].items():
        assert pin == {"bytes": os.path.getsize(os.path.join(src["bf16"], name)),
                       "sha256": _sha(os.path.join(src["bf16"], name))}
    assert sorted(ref["files"]) == ["model-00001-of-00001.safetensors", "model.safetensors.index.json"]
    assert record["graft"]["how"].startswith("the 2 dflash2/* objects")
    assert record["served"]["graft_sidecar"]["sha256"] == _sha(src["served"] + ".graft.json")
    assert record["graft"]["grafted_from"]["repository"] == "someone/base"
    # only the copied drafter's NVFP4 record survives: the body plane it named is gone
    assert list(record["local_nvfp4"]["parents"]) == ["dflash2/feature_projection"]
    m = record["trellis"]["matrices"]
    assert len(m) == 3 * 6 + 1 * 7 + 1 and record["objects"]["trellis"] == len(m)
    assert m["text/layers/3/attention/v_proj"]["k2"] == 10
    assert m["text/output_head"] == m["text/output_head"] | {"source": "lm_head", "k2": 8, "rel_error": 0.25}
    assert all(c["sha256"] for c in record["copied"]["objects"].values())
    assert record["quant_unused"] == ["mtp.fc"]
    assert all(n.endswith("input_scale_divisor") for n in record["dropped"])
    assert record["trellis"]["rel_error_max"] == 0.25
    assert sorted(record["converter"]) == ["commit", "dirty", "tool"]
    assert record["converter"]["tool"] == "tools/27b-16gb-converter"
    assert record["versions"]["safetensors"]


def test_the_same_inputs_give_the_same_work_tree(tmp_path):
    src = make_sources(str(tmp_path))
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _run(src, a)
    _run(src, b)
    for unit in ("frontend", "text", "mtp", "vision", "dflash2"):
        assert _sha(os.path.join(a, unit, "DONE")) == _sha(os.path.join(b, unit, "DONE"))
    assert _sha(os.path.join(a, "converter.json")) == _sha(os.path.join(b, "converter.json"))


def test_a_reconstruction_error_above_the_bound_fails_the_conversion(tmp_path):
    src = make_sources(str(tmp_path))
    work = tmp_path / "work"
    with pytest.raises(convert.Refused, match="layout error"):
        _run(src, str(work), errors_fn=_errors(0.61))
    assert not (work / "converter.json").exists() and not work.exists()


def test_skip_errors_writes_a_dry_run(tmp_path):
    src = make_sources(str(tmp_path))
    record = _run(src, str(tmp_path / "work"), skip_errors=True, errors_fn=None)
    assert record["status"] == "dry-run"
    assert all(t["rel_error"] is None for t in record["trellis"]["matrices"].values())
    assert record["reference"] is None, "a dry run reads no BF16 checkpoint"


@pytest.mark.parametrize("root,shard", [("quant", "model-00002-of-00002.safetensors"),
                                        ("bf16", "model-00001-of-00001.safetensors")])
def test_a_shard_that_is_not_its_lfs_oid_is_refused(tmp_path, root, shard):
    src = make_sources(str(tmp_path))
    path = os.path.join(src[root], shard)
    with open(path, "r+b") as f:
        f.seek(-1, 2)
        last = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([last[0] ^ 1]))
    with pytest.raises(convert.Refused, match="LFS oid"):
        _run(src, str(tmp_path / "work"))


@pytest.mark.parametrize("root,name,content,message", [
    ("quant", "config.json", None, "no Hugging Face download record"),
    ("quant", "model-00001-of-00002.safetensors", "86b95530f9d365d90dd24d01a997a8de0cdf6c55", "not a Hugging Face"),
    ("bf16", "model.safetensors.index.json", "f" * 40 + "\n" + "0" * 40, "different revisions"),
])
def test_a_source_that_cannot_be_pinned_is_refused_before_the_gpu_step(tmp_path, root, name, content, message):
    src = make_sources(str(tmp_path))
    record = os.path.join(src[root], ".cache", "huggingface", "download", name + ".metadata")
    if content is None:
        os.remove(record)
    else:
        with open(record, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
    with pytest.raises(convert.Refused, match=message):
        _run(src, str(tmp_path / "work"), errors_fn=_never)


def test_a_nonempty_work_dir_is_refused(tmp_path):
    src = make_sources(str(tmp_path))
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "x").write_text("x")
    with pytest.raises(convert.Refused, match="not empty"):
        _run(src, str(tmp_path / "work"))


def test_another_quant_recipe_is_refused(tmp_path):
    src = make_sources(str(tmp_path))
    path = os.path.join(src["quant"], "quantization_config.json")
    config = _json(path) | {"codebook": "mcg"}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(config, f)
    with pytest.raises(convert.Refused, match="codebook"):
        _run(src, str(tmp_path / "work"))


def test_verify_passes_the_packed_artifact_and_names_a_changed_byte(tmp_path):
    src = make_sources(str(tmp_path))
    work = str(tmp_path / "work")
    _run(src, work)
    artifact = str(tmp_path / "a.ninfer")
    pack(work, artifact)
    ok, report = convert.verify(artifact, src["served"], src["quant"], log=lambda _: None)
    assert ok, report["problems"]
    assert report["checked"] == {"trellis": 26, "copied": len(container.Container(artifact).objects) - 26}

    art = container.Container(artifact)
    for name in ("text/layers/2/mlp/gate_proj", "vision/patch_embedding"):
        offset, _ = art.span(name)
        with open(artifact, "r+b") as f:
            f.seek(offset + 3)
            b = f.read(1)
            f.seek(offset + 3)
            f.write(bytes([b[0] ^ 0x40]))
    ok, report = convert.verify(artifact, src["served"], src["quant"], log=lambda _: None)
    assert not ok
    assert any(p.startswith("text/layers/2/mlp/gate_proj: not") for p in report["problems"])
    assert any(p.startswith("vision/patch_embedding: sha256") for p in report["problems"])


def test_verify_names_an_object_of_the_wrong_kind(tmp_path):
    src = make_sources(str(tmp_path))
    work = str(tmp_path / "work")
    _run(src, work)
    artifact = str(tmp_path / "a.ninfer")
    pack(work, artifact, as_resource={"text/layers/0/mlp/down"})
    ok, report = convert.verify(artifact, src["served"], src["quant"], log=lambda _: None)
    assert not ok
    assert any(p.startswith("text/layers/0/mlp/down: None None None, not TRELLIS_MUL1_K3") for p in report["problems"])
