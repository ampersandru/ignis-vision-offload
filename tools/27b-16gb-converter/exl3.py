"""The public EXL3 quant's matrices, read from its safetensors headers without torch.

An EXL3 linear `key` is four tensors: `key.trellis` (int16, `(in/16, out/16, 16·K)`), `key.suh`
(fp16, `(in,)`), `key.svh` (fp16, `(out,)`) and `key.mul1` (int32 scalar, the codebook flag). Its
container record (`trellis-tile16-v1`, docs/formats/flash-next-layout.md §3) is the trellis bytes,
then suh, then svh, as stored, zero-padded to the next multiple of 4096: the repack copies bytes
and never decodes a weight. K is part of the format (`TRELLIS_MUL1_K{2,2P5,3,4,5,6}`): `2·K` is
the trellis's last dimension over 8.
"""
import json
import os
import struct

MUL1 = 0x83DCD12D
RECORD_ALIGNMENT = 4096
DTYPE_BYTES = {"I16": 2, "F16": 2, "BF16": 2, "I32": 4, "F32": 4}
FORMATS = {4: "TRELLIS_MUL1_K2", 5: "TRELLIS_MUL1_K2P5", 6: "TRELLIS_MUL1_K3", 8: "TRELLIS_MUL1_K4",
           10: "TRELLIS_MUL1_K5", 12: "TRELLIS_MUL1_K6"}
LAYOUT = "trellis-tile16-v1"


class Tensor:
    """Where a tensor's bytes are: its shard, the absolute offset in it, the byte count."""

    def __init__(self, name, shard, offset, size, dtype, shape):
        self.name, self.shard, self.offset, self.size = name, shard, offset, size
        self.dtype, self.shape = dtype, list(shape)


class Matrix:
    """One EXL3 linear: its `[out, in]` shape, `2·K` and the three tensors of its record."""

    def __init__(self, key, k2, rows, columns, trellis, suh, svh):
        self.key, self.k2, self.rows, self.columns = key, k2, rows, columns
        self.trellis, self.suh, self.svh = trellis, suh, svh

    @property
    def format(self):
        return FORMATS[self.k2]

    @property
    def shape(self):
        return [self.rows, self.columns]


def record_bytes(k2, rows, columns):
    """The stored size of a `trellis-tile16-v1` record (the reader's `trellis_geometry`)."""
    data = rows * columns // 16 * k2 + 2 * columns + 2 * rows
    return (data + RECORD_ALIGNMENT - 1) // RECORD_ALIGNMENT * RECORD_ALIGNMENT


def read_header(path):
    """A safetensors file's tensors: name -> Tensor (offsets absolute in the file)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    base = 8 + n
    shard = os.path.basename(path)
    out = {}
    for name, v in header.items():
        start, end = v["data_offsets"]
        out[name] = Tensor(name, shard, base + start, end - start, v["dtype"], v["shape"])
    return out


class Quant:
    """The quant's directory: its index, its shards' headers and its quantization config."""

    def __init__(self, root):
        self.root = str(root)
        with open(os.path.join(self.root, "model.safetensors.index.json"), encoding="utf-8") as f:
            self.weight_map = json.load(f)["weight_map"]
        self.shards = sorted(set(self.weight_map.values()))
        self.tensors = {}
        for shard in self.shards:
            for name, t in read_header(os.path.join(self.root, shard)).items():
                if self.weight_map.get(name) != shard:
                    raise ValueError(f"{shard} holds {name}, which the index puts elsewhere")
                self.tensors[name] = t
        missing = set(self.weight_map) - set(self.tensors)
        if missing:
            raise ValueError(f"the index lists tensors no shard holds: {sorted(missing)[:5]}")
        with open(os.path.join(self.root, "quantization_config.json"), encoding="utf-8") as f:
            self.config = json.load(f)

    def read(self, t):
        with open(os.path.join(self.root, t.shard), "rb") as f:
            f.seek(t.offset)
            data = f.read(t.size)
        if len(data) != t.size:
            raise ValueError(f"{t.shard}: {t.name} extends beyond the file")
        return data

    def keys(self):
        """Every quantized linear's key (a `.trellis` tensor's name without the suffix)."""
        return sorted(n[: -len(".trellis")] for n in self.tensors if n.endswith(".trellis"))

    def matrix(self, key):
        """The linear `key`, its four tensors checked: dtypes, shapes, the mul1 codebook flag."""
        t = {s: self.tensors.get(f"{key}.{s}") for s in ("trellis", "suh", "svh", "mul1")}
        for s, v in t.items():
            if v is None:
                raise ValueError(f"the quant has no {key}.{s}")
        want = {"trellis": "I16", "suh": "F16", "svh": "F16", "mul1": "I32"}
        for s, dtype in want.items():
            if t[s].dtype != dtype:
                raise ValueError(f"{key}.{s} is {t[s].dtype}, not {dtype}")
        shape = t["trellis"].shape
        if len(shape) != 3 or shape[2] % 8 or shape[2] // 8 not in FORMATS:
            raise ValueError(f"{key}.trellis has shape {shape}: not a K the container stores")
        k2 = shape[2] // 8
        columns, rows = shape[0] * 16, shape[1] * 16
        if t["suh"].shape != [columns] or t["svh"].shape != [rows]:
            raise ValueError(f"{key}: suh {t['suh'].shape} / svh {t['svh'].shape} do not match "
                             f"the trellis's {columns} inputs and {rows} outputs")
        if t["trellis"].size != rows * columns // 16 * k2:
            raise ValueError(f"{key}.trellis holds {t['trellis'].size} bytes, not K = {k2 / 2:g}'s")
        if rows % 128 or columns % 128:
            raise ValueError(f"{key}: [{rows}, {columns}] is not a multiple of 128 both ways")
        (flag,) = struct.unpack("<I", self.read(t["mul1"]))
        if flag != MUL1:
            raise ValueError(f"{key}.mul1 is {flag:#010x}, not the mul1 codebook's {MUL1:#010x}")
        return Matrix(key, k2, rows, columns, t["trellis"], t["suh"], t["svh"])

    def record(self, m):
        """The container record of matrix `m`: trellis, suh, svh, zero padding."""
        data = self.read(m.trellis) + self.read(m.suh) + self.read(m.svh)
        size = record_bytes(m.k2, m.rows, m.columns)
        return data + bytes(size - len(data))
