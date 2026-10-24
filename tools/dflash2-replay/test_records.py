"""CPU tests of the replay's record reader (records.py): the 9,520-byte record the
driver writes (crates/runtime/examples/dflash2_replay.rs, whose Record::write asserts
the same size), field by field in its order, and the sampler's distribution.

python -m pytest tools/dflash2-replay
"""
import json

import numpy as np

import records

# Record::write's order: the field, its dtype, its count.
DRIVER_LAYOUT = [
    ("job", "<u4", 1), ("turn", "<u4", 1), ("anchor", "<u4", 1), ("committed", "<u4", 1), ("extent", "<u4", 1),
    ("forced", "<i4", 8),
    ("drafts", "<i4", 7),
    ("cand_ids", "<i4", 7 * 16),
    ("cand_logits", "<f4", 7 * 16),
    ("scores", "<f4", 7 * 16 * 16),
    ("drafter_lse", "<f4", 7),
    ("top_ids", "<i4", 8 * 20),
    ("top_logits", "<f4", 8 * 20),
    ("target_lse", "<f4", 8),
    ("forced_logit", "<f4", 8),
    ("agreed", "<u4", 1),
]


def driver_record(seed):
    """One record's bytes laid out the way Record::write lays them, and its fields."""
    rng = np.random.default_rng(seed)
    fields, parts = {}, []
    for name, dtype, count in DRIVER_LAYOUT:
        values = (rng.integers(-1, 248_320, count) if dtype == "<i4"
                  else rng.integers(0, 1 << 20, count) if dtype == "<u4"
                  else rng.standard_normal(count)).astype(dtype)
        fields[name] = values
        parts.append(values.tobytes())
    return b"".join(parts), fields


def test_the_record_is_the_drivers_9520_bytes_in_its_order():
    assert records.RECORD.itemsize == 9520
    assert sum(np.dtype(dtype).itemsize * count for _, dtype, count in DRIVER_LAYOUT) == 9520
    assert list(records.RECORD.names) == [name for name, _, _ in DRIVER_LAYOUT]
    offset = 0
    for name, dtype, count in DRIVER_LAYOUT:
        field, at = records.RECORD.fields[name]
        assert at == offset, name
        assert field.base == np.dtype(dtype), name
        assert int(np.prod(field.shape)) == count, name
        offset += np.dtype(dtype).itemsize * count


def test_a_driver_record_reads_back_field_by_field():
    raw, fields = driver_record(267)
    row = np.frombuffer(raw, dtype=records.RECORD)[0]
    for name, _, _ in DRIVER_LAYOUT:
        np.testing.assert_array_equal(np.ravel(row[name]), fields[name], err_msg=name)
    # The tables keep the driver's row-major nesting.
    assert row["scores"].shape == (7, 16, 16)
    assert row["scores"][2, 3, 4] == fields["scores"][(2 * 16 + 3) * 16 + 4]
    assert row["top_ids"][5, 7] == fields["top_ids"][5 * 20 + 7]


def test_load_reads_the_finished_jobs_only(tmp_path):
    inputs, out = tmp_path / "input", tmp_path / "out"
    inputs.mkdir()
    out.mkdir()
    manifest = {"jobs": [{"job": "j000"}, {"job": "j001"}]}
    (inputs / "manifest.json").write_text(json.dumps(manifest))
    for job, rounds in (("j000", 3), ("j001", 2)):
        (out / f"{job}.rounds").write_bytes(b"".join(driver_record(rounds + i)[0] for i in range(rounds)))
        np.full(10, 1, dtype="i1").tofile(inputs / f"{job}.cat")
    (out / "j000.done").write_bytes(b"")
    rows, cats = records.load(str(out), records.read_manifest(str(inputs / "manifest.json")))
    assert len(rows) == 3, "j001 has no .done: a job the run did not finish is not read"
    assert sorted(cats) == [0]


def test_the_sampler_keeps_the_smallest_prefix_reaching_top_p_and_renormalizes():
    p = np.array([0.5, 0.46] + [0.04 / 18] * 18)
    probs = records.sampling_probs(np.log(p)[None, :])[0]
    np.testing.assert_allclose(probs[:2], [0.5 / 0.96, 0.46 / 0.96])
    assert (probs[2:] == 0).all()
    ids = np.arange(20)[None, :] + 100
    assert records.prob_of(ids, probs[None, :], 101)[0] == probs[1]
    assert records.prob_of(ids, probs[None, :], 7)[0] == 0
