import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scan2scope.cache import CacheMiss, OutputCache, array_sha256, file_sha256, key_hash


class Counter:
    def __init__(self, out):
        self.calls = 0
        self.out = out

    def __call__(self):
        self.calls += 1
        return self.out


KEY = {"model": "mapanything", "revision": "abc", "inputs": ["f1", "f2"], "params": {"res": 518, "fps": 2.0}}


def test_key_hash_matches_contract_and_ignores_order():
    expected = hashlib.sha256(json.dumps(KEY, sort_keys=True).encode()).hexdigest()
    assert key_hash(KEY) == expected
    reordered = {"params": {"fps": 2.0, "res": 518}, "inputs": ["f1", "f2"], "revision": "abc", "model": "mapanything"}
    assert key_hash(reordered) == expected
    assert key_hash({**KEY, "revision": "abd"}) != expected


def test_key_hash_accepts_numpy_and_paths():
    a = key_hash({"n": np.int64(5), "x": np.float32(0.5), "p": Path("/tmp/a.jpg"), "v": np.arange(3)})
    b = key_hash({"n": 5, "x": 0.5, "p": "/tmp/a.jpg", "v": [0, 1, 2]})
    assert a == b
    with pytest.raises(TypeError):
        key_hash({"bad": object()})


def test_live_computes_once_then_loads(tmp_path):
    fn = Counter({"depth": np.arange(12, dtype=np.float32).reshape(3, 4), "scale": 1.25, "name": "kitchen"})
    cache = OutputCache("live", tmp_path)
    assert cache.effective_mode == "none"
    first = cache.compute(KEY, fn)
    assert fn.calls == 1 and cache.effective_mode == "live"
    h = key_hash(KEY)
    npz = tmp_path / h[:2] / f"{h}.npz"
    assert npz.is_file() and KEY in cache
    sidecar = npz.with_suffix(".json")
    assert json.loads(sidecar.read_text()) == KEY
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == h

    second = cache.compute(KEY, fn)
    assert fn.calls == 1 and cache.effective_mode == "mixed"
    assert set(second) == {"depth", "scale", "name"}
    np.testing.assert_array_equal(second["depth"], first["depth"])
    assert second["depth"].dtype == np.float32
    assert second["scale"].shape == () and float(second["scale"]) == 1.25
    assert str(second["name"]) == "kitchen"


def test_replay_loads_or_raises(tmp_path):
    OutputCache("live", tmp_path).compute(KEY, Counter({"a": np.ones(3)}))
    replay = OutputCache("replay", tmp_path)
    fn = Counter({"a": np.zeros(3)})
    np.testing.assert_array_equal(replay.compute(KEY, fn)["a"], np.ones(3))
    assert fn.calls == 0 and replay.effective_mode == "replay"
    with pytest.raises(CacheMiss):
        replay.compute({**KEY, "revision": "other"}, fn)
    assert fn.calls == 0 and replay.effective_mode == "replay"


def test_off_always_computes_and_stores_nothing(tmp_path):
    cache = OutputCache("off", tmp_path)
    fn = Counter({"a": np.ones(2)})
    cache.compute(KEY, fn)
    cache.compute(KEY, fn)
    assert fn.calls == 2 and cache.effective_mode == "live"
    assert not any(tmp_path.rglob("*.npz"))


def test_corrupt_entry_recomputed_live_and_miss_in_replay(tmp_path):
    cache = OutputCache("live", tmp_path)
    cache.compute(KEY, Counter({"a": np.ones(2)}))
    cache.path_for(KEY).write_bytes(b"not a zip")
    with pytest.raises(CacheMiss):
        OutputCache("replay", tmp_path).compute(KEY, Counter({"a": np.ones(2)}))
    fn = Counter({"a": np.full(2, 7.0)})
    out = OutputCache("live", tmp_path).compute(KEY, fn)
    assert fn.calls == 1 and out["a"][0] == 7.0
    np.testing.assert_array_equal(OutputCache("replay", tmp_path).compute(KEY, fn)["a"], [7.0, 7.0])


def test_bad_outputs_rejected(tmp_path):
    cache = OutputCache("live", tmp_path)
    with pytest.raises(TypeError):
        cache.compute(KEY, lambda: [np.ones(2)])
    with pytest.raises(TypeError):
        cache.compute({"k": 2}, lambda: {"a": np.array([{"x": 1}], dtype=object)})
    with pytest.raises(ValueError):
        cache.compute({"k": 3}, lambda: {"file": np.ones(1)})
    with pytest.raises(ValueError):
        OutputCache("record", tmp_path)


def test_file_sha256_streams_and_tracks_changes(tmp_path):
    p = tmp_path / "a.bin"
    data = np.random.default_rng(0).integers(0, 256, 3_000_000, dtype=np.uint8).tobytes()
    p.write_bytes(data)
    assert file_sha256(p) == hashlib.sha256(data).hexdigest()
    p.write_bytes(data + b"x")
    assert file_sha256(p) == hashlib.sha256(data + b"x").hexdigest()


def test_array_sha256_depends_on_dtype_and_shape():
    a = np.arange(6, dtype=np.int32)
    assert array_sha256(a) == array_sha256(a.copy())
    assert array_sha256(a) != array_sha256(a.reshape(2, 3))
    assert array_sha256(a) != array_sha256(a.view(np.float32))
    assert array_sha256(np.asfortranarray(a.reshape(2, 3))) == array_sha256(a.reshape(2, 3))


def test_device_label_honours_override(monkeypatch):
    monkeypatch.setenv("SCAN2SCOPE_DEVICE", "cpu")
    assert OutputCache.device_label() == "cpu"
