import os
import struct

import pytest

import exl3
from conftest import write_safetensors


def _quant(tmp_path, tensors):
    write_safetensors(os.path.join(tmp_path, "model-00001-of-00001.safetensors"), tensors)
    import json
    with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {n: "model-00001-of-00001.safetensors" for n in tensors}}, f)
    with open(os.path.join(tmp_path, "quantization_config.json"), "w") as f:
        json.dump({"codebook": "mul1"}, f)
    return exl3.Quant(tmp_path)


def _linear(key, rows, columns, words, mul1=exl3.MUL1, suh_len=None, trellis_dtype="I16"):
    return {
        f"{key}.trellis": (trellis_dtype, [columns // 16, rows // 16, words],
                           bytes(range(256)) * (rows * columns // 128 * words // 256)),
        f"{key}.suh": ("F16", [suh_len or columns], b"\x11" * 2 * (suh_len or columns)),
        f"{key}.svh": ("F16", [rows], b"\x22" * 2 * rows),
        f"{key}.mul1": ("I32", [], struct.pack("<I", mul1)),
    }


def test_record_bytes_match_the_containers_class_table():
    # docs/specs/flash-next/layout.md §3's class table (the reader's trellis_geometry).
    assert exl3.record_bytes(4, 1280, 2560) == 827_392
    assert exl3.record_bytes(5, 1280, 2560) == 1_032_192
    assert exl3.record_bytes(6, 2560, 640) == 622_592
    assert exl3.record_bytes(8, 2560, 640) == 827_392


def test_a_matrix_is_its_trellis_suh_svh_and_padding(tmp_path):
    q = _quant(tmp_path, _linear("m", 256, 128, 48))
    m = q.matrix("m")
    assert (m.k2, m.rows, m.columns, m.format, m.shape) == (6, 256, 128, "TRELLIS_MUL1_K3", [256, 128])
    record = q.record(m)
    trellis = 256 * 128 // 16 * 6
    assert len(record) == exl3.record_bytes(6, 256, 128) == 16_384
    assert record[:trellis] == q.read(m.trellis)
    assert record[trellis:trellis + 256] == b"\x11" * 256
    assert record[trellis + 256:trellis + 256 + 512] == b"\x22" * 512
    assert record[trellis + 768:] == bytes(len(record) - trellis - 768)


@pytest.mark.parametrize("words,k2,fmt", [(32, 4, "TRELLIS_MUL1_K2"), (40, 5, "TRELLIS_MUL1_K2P5"),
                                          (64, 8, "TRELLIS_MUL1_K4"), (80, 10, "TRELLIS_MUL1_K5"),
                                          (96, 12, "TRELLIS_MUL1_K6")])
def test_k_is_the_trellis_words_per_tile(tmp_path, words, k2, fmt):
    m = _quant(tmp_path, _linear("m", 128, 128, words)).matrix("m")
    assert (m.k2, m.format) == (k2, fmt)


@pytest.mark.parametrize("tensors,message", [
    (_linear("m", 128, 128, 64, mul1=0x12345678), "mul1"),
    (_linear("m", 128, 128, 64, suh_len=256), "suh"),
    (_linear("m", 128, 128, 64, trellis_dtype="I32"), "not I16"),
    (_linear("m", 128, 128, 56), "not a K"),
    ({k: v for k, v in _linear("m", 128, 128, 64).items() if not k.endswith("svh")}, "no m.svh"),
])
def test_a_matrix_the_container_cannot_store_is_refused(tmp_path, tensors, message):
    with pytest.raises(ValueError, match=message):
        _quant(tmp_path, tensors).matrix("m")


def test_keys_are_the_quantized_linears(tmp_path):
    tensors = dict(_linear("a.b", 128, 128, 64), **_linear("c", 128, 128, 32))
    tensors["norm.weight"] = ("BF16", [128], b"\0" * 256)
    assert _quant(tmp_path, tensors).keys() == ["a.b", "c"]
