import hashlib
import http.server
import io
import json
import math
import re
import threading
import zipfile

import numpy as np
import pytest
import requests

from scan2scope import weights
from scan2scope.config import ModelSpec
from scan2scope.weights import RemoteFile, download_file, plan_parallel, split_ranges


class Store:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.api: dict[str, dict] = {}
        self.requests: list[tuple[str, str | None]] = []
        self.auth: list[str | None] = []
        self.fail: dict[tuple[str, str | None], int] = {}
        self.cut: dict[tuple[str, str | None], int] = {}  # send only this many body bytes, then drop
        self.ranges = True
        self.lock = threading.Lock()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _send(self, code: int, body: bytes = b"", headers: dict | None = None) -> None:
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        st: Store = self.server.store
        path, rng = self.path.split("?")[0], self.headers.get("Range")
        with st.lock:
            st.requests.append((path, rng))
            st.auth.append(self.headers.get("Authorization"))
            failing = st.fail.get((path, rng), 0)
            if failing:
                st.fail[(path, rng)] = failing - 1
            cut = st.cut.pop((path, rng), None)
        if failing:
            return self._send(500, b"boom")
        if cut is not None:
            data = st.files[path]
            a, b = (int(x) for x in re.fullmatch(r"bytes=(\d+)-(\d+)", rng).groups())
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
            self.send_header("Content-Length", str(b - a + 1))
            self.end_headers()
            self.wfile.write(data[a:a + cut])
            self.wfile.flush()
            self.close_connection = True
            return None
        if path in st.api:
            return self._send(200, json.dumps(st.api[path]).encode(), {"Content-Type": "application/json"})
        if path.startswith("/redirect/"):
            return self._send(302, b"", {"Location": path.replace("/redirect/", "/files/", 1)})
        data = st.files.get(path)
        if data is None:
            return self._send(404, b"missing")
        m = re.fullmatch(r"bytes=(\d+)-(\d+)", rng or "")
        if m and st.ranges:
            a, b = int(m[1]), min(int(m[2]), len(data) - 1)
            return self._send(206, data[a:b + 1], {"Content-Range": f"bytes {a}-{b}/{len(data)}"})
        return self._send(200, data)


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    srv.store = Store()
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    yield srv.store, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(weights.time, "sleep", lambda s: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)


def blob(n: int, seed: int = 0) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


def lfs_file(store: Store, base: str, name: str, data: bytes, prefix: str = "/files/") -> RemoteFile:
    store.files["/files/" + name] = data
    return RemoteFile(name, base + prefix + name, len(data), sha256=hashlib.sha256(data).hexdigest())


CHUNK = 256 << 10


def test_split_ranges_cover_exactly():
    for size in (0, 1, 5, 10, 11, 1_000_003):
        for chunk in (1, 3, 10, 4096):
            if size // chunk > 100_000:
                continue
            r = split_ranges(size, chunk)
            assert len(r) == math.ceil(size / chunk)
            assert all(b - a + 1 <= chunk and b >= a for a, b in r)
            if r:
                assert [a for a, _ in r] == [0] + [b + 1 for _, b in r[:-1]]
                assert r[-1][1] == size - 1
    with pytest.raises(ValueError):
        split_ranges(10, 0)


def test_plan_parallel():
    assert plan_parallel(184_305_280) == (8, math.ceil(184_305_280 / 8))
    assert plan_parallel(4_900_000_000) == (16, 64 << 20)
    assert plan_parallel(101_000_000) == (8, math.ceil(101_000_000 / 8))
    workers, chunk = plan_parallel(689_000_000)
    assert workers == 8 and 8 << 20 <= chunk <= 64 << 20


def test_parallel_download_follows_redirect_and_sends_token(server, tmp_path, monkeypatch):
    store, base = server
    monkeypatch.setattr(weights, "PARALLEL_MIN_BYTES", 1 << 20)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    data = blob(3 * (1 << 20) + 123)
    rf = lfs_file(store, base, "model.safetensors", data, prefix="/redirect/")
    target = tmp_path / "m" / "model.safetensors"
    download_file(rf, target, workers=4, chunk=CHUNK)
    assert target.read_bytes() == data
    assert (tmp_path / "m" / "model.safetensors.verified").read_text().strip() == rf.sha256
    assert not list((tmp_path / "m").glob("*.part*"))
    ranged = [r for p, r in store.requests if p.startswith("/files/") and r]
    assert len(ranged) == len(split_ranges(len(data), CHUNK))
    assert set(store.auth) == {"Bearer hf_test"}


def test_resume_fetches_only_missing_chunks(server, tmp_path, monkeypatch):
    store, base = server
    monkeypatch.setattr(weights, "PARALLEL_MIN_BYTES", 1 << 20)
    data = blob(2 * (1 << 20) + 7, seed=1)
    rf = lfs_file(store, base, "w.bin", data)
    ranges = split_ranges(len(data), CHUNK)
    target = tmp_path / "w.bin"
    part = tmp_path / "w.bin.part"
    buf = bytearray(len(data))
    done, partial = [0, 1, 5], {3: 100_000}
    for i in done:
        a, b = ranges[i]
        buf[a:b + 1] = data[a:b + 1]
    a3 = ranges[3][0]
    buf[a3:a3 + partial[3]] = data[a3:a3 + partial[3]]
    part.write_bytes(bytes(buf))
    (tmp_path / "w.bin.part.json").write_text(json.dumps({"size": len(data), "hash": rf.sha256, "chunk": CHUNK,
                                                          "done": done, "partial": {"3": partial[3]}}))
    download_file(rf, target, workers=3, chunk=CHUNK)
    assert target.read_bytes() == data
    fetched = sorted(r for _, r in store.requests if r)
    expected = sorted(f"bytes={a + partial.get(i, 0)}-{b}" for i, (a, b) in enumerate(ranges) if i not in done)
    assert fetched == expected


def test_dropped_connection_resumes_mid_range(server, tmp_path, monkeypatch):
    store, base = server
    monkeypatch.setattr(weights, "PARALLEL_MIN_BYTES", 1 << 20)
    data = blob(1 << 20, seed=7)
    rf = lfs_file(store, base, "d.bin", data)
    a, b = split_ranges(len(data), CHUNK)[1]
    store.cut[("/files/d.bin", f"bytes={a}-{b}")] = 100_000
    download_file(rf, tmp_path / "d.bin", workers=2, chunk=CHUNK)
    assert (tmp_path / "d.bin").read_bytes() == data
    retried = [r for p, r in store.requests if r and r.endswith(f"-{b}")]
    assert retried[0] == f"bytes={a}-{b}" and len(retried) == 2
    assert int(retried[1][6:].split("-")[0]) > a  # the retry asked only for the missing tail


def test_corrupt_resumed_chunk_is_caught_by_the_hash(server, tmp_path, monkeypatch):
    store, base = server
    monkeypatch.setattr(weights, "PARALLEL_MIN_BYTES", 1 << 20)
    data = blob(1 << 20 | 5, seed=2)
    rf = lfs_file(store, base, "w.bin", data)
    part = tmp_path / "w.bin.part"
    part.write_bytes(b"\x01" * len(data))  # claims chunk 0 is done, but holds garbage
    (tmp_path / "w.bin.part.json").write_text(json.dumps({"size": len(data), "hash": rf.sha256, "chunk": CHUNK,
                                                          "done": [0]}))
    with pytest.raises(OSError, match="does not match"):
        download_file(rf, tmp_path / "w.bin", workers=2, chunk=CHUNK)
    assert not part.exists() and not (tmp_path / "w.bin").exists()
    download_file(rf, tmp_path / "w.bin", workers=2, chunk=CHUNK)  # the rerun starts over and succeeds
    assert (tmp_path / "w.bin").read_bytes() == data


def test_complete_files_are_skipped_and_markers_reused(server, tmp_path):
    store, base = server
    data = blob(50_000, seed=3)
    rf = lfs_file(store, base, "c.bin", data)
    target = tmp_path / "c.bin"
    download_file(rf, target)
    n = len(store.requests)
    download_file(rf, target)
    assert len(store.requests) == n  # marker present: no request, no hashing
    (tmp_path / "c.bin.verified").unlink()
    download_file(rf, target)
    assert len(store.requests) == n and (tmp_path / "c.bin.verified").is_file()  # hashed once, marker rewritten
    target.write_bytes(b"\0" * len(data))  # same size, wrong content, newer than the marker
    (tmp_path / "c.bin.verified").unlink()
    download_file(rf, target)
    assert target.read_bytes() == data and len(store.requests) == n + 1
    assert download_file(rf, target, verify=False) == target and len(store.requests) == n + 1


def test_server_without_range_support_falls_back_to_one_stream(server, tmp_path, monkeypatch):
    store, base = server
    store.ranges = False
    monkeypatch.setattr(weights, "PARALLEL_MIN_BYTES", 1 << 20)
    data = blob(1 << 20 | 99, seed=4)
    rf = lfs_file(store, base, "n.bin", data)
    download_file(rf, tmp_path / "n.bin", workers=2, chunk=CHUNK)
    assert (tmp_path / "n.bin").read_bytes() == data
    assert not (tmp_path / "n.bin.part.json").exists()


def test_failed_range_is_retried(server, tmp_path, monkeypatch):
    store, base = server
    monkeypatch.setattr(weights, "PARALLEL_MIN_BYTES", 1 << 20)
    data = blob(1 << 20, seed=5)
    rf = lfs_file(store, base, "r.bin", data)
    a, b = split_ranges(len(data), CHUNK)[2]
    store.fail[("/files/r.bin", f"bytes={a}-{b}")] = 2
    download_file(rf, tmp_path / "r.bin", workers=2, chunk=CHUNK)
    assert (tmp_path / "r.bin").read_bytes() == data
    assert sum(1 for p, r in store.requests if r == f"bytes={a}-{b}") == 3


def test_missing_file_is_not_retried(server, tmp_path):
    _, base = server
    with pytest.raises(PermissionError, match="404"):
        download_file(RemoteFile("x", base + "/files/x", 10, sha256="0" * 64), tmp_path / "x")
    assert not (tmp_path / "x").exists()


def test_small_file_checked_against_git_blob_id(server, tmp_path):
    store, base = server
    data = b'{"model_type": "sam2"}\n'
    store.files["/files/config.json"] = data
    sha1 = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    rf = RemoteFile("config.json", base + "/files/config.json", len(data), git_sha1=sha1)
    download_file(rf, tmp_path / "config.json")
    assert (tmp_path / "config.json").read_bytes() == data
    bad = RemoteFile("config.json", rf.url, len(data), git_sha1="0" * 40)
    with pytest.raises(OSError, match="does not match"):
        download_file(bad, tmp_path / "other" / "config.json")
    assert not (tmp_path / "other" / "config.json").exists()


def _serve_model(store: Store, data: dict[str, bytes], repo: str = "org/tiny", rev: str = "abc123"):
    siblings = []
    for name, content in data.items():
        store.files[f"/{repo}/resolve/{rev}/{name}"] = content
        entry = {"rfilename": name, "size": len(content),
                 "blobId": hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()}
        if name.endswith(".safetensors"):
            entry["lfs"] = {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content), "pointerSize": 134}
        siblings.append(entry)
    store.api[f"/api/models/{repo}/revision/{rev}"] = {"sha": rev, "siblings": siblings}
    return ModelSpec("tiny", repo, rev, "Apache-2.0", tuple(data))


def test_fetch_model_from_api_listing(server, tmp_path, monkeypatch):
    store, base = server
    monkeypatch.setenv("HF_ENDPOINT", base)
    monkeypatch.setattr(weights, "PARALLEL_MIN_BYTES", 1 << 20)
    files = {"config.json": b'{"a": 1}', "model.safetensors": blob(1 << 20 | 3, seed=6)}
    spec = _serve_model(store, files)
    paths = weights.fetch_model(spec, dest=tmp_path / "tiny")
    assert [p.read_bytes() for p in paths] == list(files.values())
    assert ("/api/models/org/tiny/revision/abc123", None) in store.requests
    assert all((tmp_path / "tiny" / f"{n}.verified").is_file() for n in files)
    wrong = ModelSpec("tiny", spec.repo, spec.revision, spec.license, ("config.json", "nope.bin"))
    with pytest.raises(RuntimeError, match="nope.bin"):
        weights.fetch_model(wrong, dest=tmp_path / "tiny")


def test_fetch_model_offline_keeps_local_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")  # nothing listens on the discard port
    spec = ModelSpec("tiny", "org/tiny", "abc", "MIT", ("config.json",))
    with pytest.raises(requests.RequestException):
        weights.fetch_model(spec, dest=tmp_path)
    (tmp_path / "config.json").write_text("{}")
    assert weights.fetch_model(spec, dest=tmp_path) == [tmp_path / "config.json"]


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_install_dinov2_hub(server, tmp_path):
    store, base = server
    store.files["/files/main.zip"] = _zip_bytes({"dinov2-main/hubconf.py": b"dependencies = ['torch']\n",
                                                 "dinov2-main/dinov2/__init__.py": b""})
    (tmp_path / "hub").mkdir()
    (tmp_path / "hub" / "trusted_list").write_text("someone_else")
    dest = weights.install_dinov2_hub(tmp_path, url=base + "/files/main.zip")
    assert dest == tmp_path / "hub" / "facebookresearch_dinov2_main"
    assert (dest / "hubconf.py").is_file() and (dest / "dinov2" / "__init__.py").is_file()
    n = len(store.requests)
    weights.install_dinov2_hub(tmp_path, url=base + "/files/main.zip")
    assert len(store.requests) == n
    assert (tmp_path / "hub" / "trusted_list").read_text().split() == ["someone_else", "facebookresearch_dinov2"]
    assert not [p for p in (tmp_path / "hub").iterdir() if p.name.startswith(".dinov2-")]


def test_install_dinov2_hub_rejects_unsafe_archive(server, tmp_path):
    store, base = server
    store.files["/files/evil.zip"] = _zip_bytes({"../escape.py": b"x", "dinov2-main/hubconf.py": b""})
    with pytest.raises(ValueError, match="unsafe"):
        weights.install_dinov2_hub(tmp_path, url=base + "/files/evil.zip")
    assert not (tmp_path / "escape.py").exists() and not (tmp_path / "hub" / "facebookresearch_dinov2_main").exists()


def test_doctor_flags_missing_weights(tmp_path, capsys):
    spec = ModelSpec("ghost", "org/scan2scope-test-missing-weights", "abc", "MIT", ("model.safetensors",))
    code = weights.doctor(models={"ghost": spec}, torch_home=tmp_path)
    out = capsys.readouterr().out
    assert code == 1
    assert "[FAIL] weights ghost: missing model.safetensors" in out and "[FAIL] DINOv2 hub code" in out
    assert "PyAV HEVC decoder" in out and "pillow-heif" in out and "free disk" in out
