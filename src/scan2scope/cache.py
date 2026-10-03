"""Content-addressed cache of model outputs, so reported numbers can be replayed without running the models.

Entries live under root/<h[:2]>/<h>.npz with a <h>.json sidecar holding the key, where h is the SHA-256 of
json.dumps(key, sort_keys=True). Keys hold input file hashes, model revisions and preprocessing parameters,
never the device.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("scan2scope.cache")

MODES = ("live", "replay", "off")
# np.savez_compressed takes these as its own arguments, so they cannot be array names.
_RESERVED_NAMES = frozenset({"file", "allow_pickle"})

_file_hash_memo: dict[tuple[str, int, int], str] = {}


class CacheMiss(KeyError):
    """Replay mode found no stored output for a key."""


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cache key value of type {type(obj).__name__} is not JSON serialisable")


def key_json(key: Mapping[str, Any]) -> str:
    """Canonical JSON of a key. For plain JSON values this equals json.dumps(key, sort_keys=True)."""
    return json.dumps(key, sort_keys=True, default=_json_default)


def key_hash(key: Mapping[str, Any]) -> str:
    return hashlib.sha256(key_json(key).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Streaming SHA-256 of a file, memoised on (path, size, mtime) so repeated keys do not rehash big videos."""
    p = Path(path)
    st = p.stat()
    memo = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    digest = _file_hash_memo.get(memo)
    if digest is None:
        with open(p, "rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        _file_hash_memo[memo] = digest
    return digest


def array_sha256(arr: np.ndarray) -> str:
    """SHA-256 over dtype, shape and C-order bytes, so equal bytes with a different layout hash differently."""
    a = np.ascontiguousarray(arr)
    if a.dtype.hasobject:
        raise TypeError("array_sha256 does not support object arrays")
    h = hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _as_outputs(out: Any) -> dict[str, np.ndarray]:
    if not isinstance(out, Mapping):
        raise TypeError(f"cached function must return dict[str, np.ndarray], got {type(out).__name__}")
    arrays: dict[str, np.ndarray] = {}
    for name, value in out.items():
        if not isinstance(name, str) or not name:
            raise TypeError(f"cache output names must be non-empty strings, got {name!r}")
        if name in _RESERVED_NAMES:
            raise ValueError(f"cache output name {name!r} is reserved by np.savez_compressed")
        a = np.asarray(value)
        if a.dtype.hasobject:
            raise TypeError(f"cache output {name!r} has dtype object; store numeric or string arrays")
        arrays[name] = a
    return arrays


def _atomic_write(path: Path, write: Callable[[Any], None]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class OutputCache:
    """mode live: compute on a miss and store; replay: load or raise CacheMiss; off: always compute, store nothing."""

    def __init__(self, mode: str = "live", root: str | Path | None = None) -> None:
        if mode not in MODES:
            raise ValueError(f"cache mode must be one of {', '.join(MODES)}, got {mode!r}")
        if root is None:
            from scan2scope.config import OUTPUT_CACHE_DIR

            root = OUTPUT_CACHE_DIR
        self.mode = mode
        self.root = Path(root)
        self.n_computed = 0
        self.n_loaded = 0

    def path_for(self, key: Mapping[str, Any]) -> Path:
        h = key_hash(key)
        return self.root / h[:2] / f"{h}.npz"

    def __contains__(self, key: Mapping[str, Any]) -> bool:
        return self.path_for(key).is_file()

    def compute(self, key: Mapping[str, Any], fn: Callable[[], Mapping[str, Any]]) -> dict[str, np.ndarray]:
        if "device" in key:
            log.warning("cache key has a 'device' entry; outputs would not replay on another machine")
        if self.mode == "off":
            out = _as_outputs(fn())
            self.n_computed += 1
            return out
        path = self.path_for(key)
        if path.is_file():
            try:
                out = self._load(path)
                self.n_loaded += 1
                return out
            except Exception as exc:  # truncated or corrupt entry: recompute in live mode
                log.warning("cache entry %s is unreadable (%s)", path.name, exc)
                if self.mode == "replay":
                    raise CacheMiss(f"cache entry {path} is unreadable: {exc}") from exc
        elif self.mode == "replay":
            raise CacheMiss(f"replay mode: no cached output for key {key_json(key)[:300]}")
        out = _as_outputs(fn())
        self.n_computed += 1
        try:
            self._store(path, key, out)
        except OSError as exc:  # a full disk must not lose the computed result
            log.warning("could not store cache entry %s: %s", path, exc)
        return out

    @staticmethod
    def _load(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as z:
            return {name: z[name] for name in z.files}

    @staticmethod
    def _store(path: Path, key: Mapping[str, Any], out: dict[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, lambda f: np.savez_compressed(f, **out))
        # Same canonical JSON that was hashed, so sha256(sidecar bytes) equals the file name.
        _atomic_write(path.with_suffix(".json"), lambda f: f.write(key_json(key).encode("utf-8")))

    @property
    def effective_mode(self) -> str:
        if self.n_computed and self.n_loaded:
            return "mixed"
        if self.n_computed:
            return "live"
        if self.n_loaded:
            return "replay"
        return "none"

    @staticmethod
    def device_label() -> str:
        try:
            from scan2scope.config import torch_device

            device = torch_device()
        except ImportError:
            return "cpu (torch not installed)"
        if device.startswith("cuda"):
            try:
                import torch

                return f"{device}:{torch.cuda.get_device_name(0)}"
            except (RuntimeError, AssertionError):
                return device
        return device
