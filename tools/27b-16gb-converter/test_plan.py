import random

import pytest

import plan
from conftest import served_objects


class M:
    def __init__(self, rows, columns, fmt="TRELLIS_MUL1_K3"):
        self.shape, self.format = [rows, columns], fmt


def _served():
    return [d for d, _ in served_objects(random.Random(1))]


def _matrices():
    from conftest import GDN, GQA, MLP, LAYERS, VOCAB, H
    out = {}
    for i in range(LAYERS):
        for module, rows, columns, _ in (GQA if (i + 1) % 4 == 0 else GDN) + MLP:
            out[f"model.language_model.layers.{i}.{module}"] = M(rows, columns)
    out["lm_head"] = M(VOCAB, H, "TRELLIS_MUL1_K4")
    return out


def test_the_body_planes_become_the_quants_matrices_in_the_served_order():
    objects, dropped = plan.plan(_served(), _matrices())
    names = [p.name for p in objects]
    gdn = [n for n in names if n.startswith("text/layers/0/")]
    assert gdn == ["text/layers/0/input_norm", "text/layers/0/gdn/a_log", "text/layers/0/gdn/in_proj_qkv",
                   "text/layers/0/gdn/in_proj_z", "text/layers/0/gdn/norm", "text/layers/0/gdn/output",
                   "text/layers/0/post_attention_norm", "text/layers/0/mlp/gate_proj", "text/layers/0/mlp/up_proj",
                   "text/layers/0/mlp/down"]
    gqa = [n for n in names if n.startswith("text/layers/3/")]
    assert gqa == ["text/layers/3/input_norm", "text/layers/3/attention/q_proj", "text/layers/3/attention/k_proj",
                   "text/layers/3/attention/v_proj", "text/layers/3/attention/query_norm",
                   "text/layers/3/attention/output", "text/layers/3/post_attention_norm",
                   "text/layers/3/mlp/gate_proj", "text/layers/3/mlp/up_proj", "text/layers/3/mlp/down"]
    by = {p.name: p for p in objects}
    assert by["text/layers/0/gdn/in_proj_qkv"].key == "model.language_model.layers.0.linear_attn.in_proj_qkv"
    assert by["text/layers/3/attention/q_proj"].shape == [256, 128]
    assert by["text/layers/1/gdn/output"].key == "model.language_model.layers.1.linear_attn.out_proj"
    head = by["text/output_head"]
    assert (head.key, head.format, head.layout, head.shape) == ("lm_head", "TRELLIS_MUL1_K4", "trellis-tile16-v1",
                                                                [256, 128])
    assert all(n.endswith("input_scale_divisor") for n in dropped) and len(dropped) == 14


def test_everything_else_is_the_served_object():
    served = _served()
    objects, dropped = plan.plan(served, _matrices())
    copies = [p for p in objects if p.copy is not None]
    replaced = {"gdn/query_key_value_z", "gdn/output", "attention/query_key_gate_value", "attention/output",
                "mlp/gate_up", "mlp/down"}
    want = [o for o in served if o["name"] not in dropped and o["name"] != "text/output_head"
            and not any(o["name"].startswith("text/layers/") and o["name"].endswith(s) for s in replaced)]
    assert [p.name for p in copies] == [o["name"] for o in want]
    for p, o in zip(copies, want):
        assert p.copy == o["name"] and p.key is None
        if o["kind"] == "tensor":
            assert (p.format, p.layout, p.shape) == (o["format"], o["layout"], o["shape"])
        else:
            assert p.encoding == o["encoding"]
    # the MTP plane shares a body suffix and stays the served W8 object
    mtp = next(p for p in objects if p.name == "mtp/layer/mlp/gate_up")
    assert (mtp.copy, mtp.format) == ("mtp/layer/mlp/gate_up", "W8G32_F16S")
    assert [p.unit for p in objects] == sorted((p.unit for p in objects), key=plan.UNITS.index)


def test_a_quant_matrix_left_over_is_refused():
    matrices = _matrices()
    matrices["model.language_model.layers.9.mlp.down_proj"] = M(128, 256)
    with pytest.raises(ValueError, match="no served plane"):
        plan.plan(_served(), matrices)


def test_a_missing_quant_matrix_is_refused():
    matrices = _matrices()
    del matrices["model.language_model.layers.2.linear_attn.in_proj_z"]
    with pytest.raises(ValueError, match="in_proj_z"):
        plan.plan(_served(), matrices)


@pytest.mark.parametrize("key,shape,message", [
    ("model.language_model.layers.0.mlp.up_proj", [128, 128], "give 384 rows"),
    ("model.language_model.layers.0.mlp.down_proj", [128, 128], "takes 128 inputs"),
])
def test_the_split_parts_must_tile_the_served_plane(key, shape, message):
    matrices = _matrices()
    matrices[key] = M(*shape)
    with pytest.raises(ValueError, match=message):
        plan.plan(_served(), matrices)


def test_an_object_outside_the_units_is_refused():
    served = _served() + [{"name": "extra/thing", "kind": "tensor", "format": "BF16",
                           "layout": "contiguous-le-v1", "shape": [1]}]
    with pytest.raises(ValueError, match="outside the units"):
        plan.plan(served, _matrices())
