"""Pinned model weights: parallel ranged downloads with resume and hash checks, DINOv2 hub code, doctor."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import requests

from scan2scope import __version__
from scan2scope.config import CACHE_DIR, DINOV2_HUB_ZIP, MODELS, TORCH_HOME, ModelSpec, setup_env

log = logging.getLogger("scan2scope.weights")

PARALLEL_MIN_BYTES = 100_000_000  # files above this use parallel range requests
MIN_CHUNK, MAX_CHUNK = 8 << 20, 64 << 20
RETRIES = 5
TIMEOUT = (15, 60)  # connect, read seconds
DINOV2_DIR = "facebookresearch_dinov2_main"
DINOV2_TRUSTED = "facebookresearch_dinov2"


class RangeNotSupported(RuntimeError):
    """The server answered a Range request with the whole file."""


def _raise_for_status(r: requests.Response) -> None:
    """Client errors other than timeouts and rate limits are final; everything else may be retried."""
    if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
        raise PermissionError(f"{r.status_code} {r.reason} for {r.url}")
    r.raise_for_status()


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str
    size: int  # 0 when the server did not say
    sha256: str | None = None  # LFS files
    git_sha1: str | None = None  # git blob id of small (non-LFS) files

    @property
    def expected_hash(self) -> str | None:
        return self.sha256 or self.git_sha1


def hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


_local = threading.local()


def _session() -> requests.Session:
    """One session per thread: requests sessions are not documented as thread-safe."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = f"scan2scope/{__version__}"
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _local.session = s
    return s


# ---------------------------------------------------------------------------------------------------------
# Planning


def split_ranges(size: int, chunk: int) -> list[tuple[int, int]]:
    """Inclusive (start, end) byte ranges of at most chunk bytes covering [0, size)."""
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    return [(start, min(start + chunk, size) - 1) for start in range(0, size, chunk)]


def plan_parallel(size: int) -> tuple[int, int]:
    """(workers, chunk bytes) for a large file: 16 workers from 1 GiB, else 8; chunks of 8 to 64 MiB."""
    workers = 16 if size >= 1 << 30 else 8
    chunk = min(MAX_CHUNK, max(MIN_CHUNK, math.ceil(size / workers)))
    return workers, chunk


# ---------------------------------------------------------------------------------------------------------
# Hashing and markers


def _digest(path: Path, rf: RemoteFile) -> str | None:
    if rf.sha256:
        h = hashlib.sha256()
    elif rf.git_sha1:
        h = hashlib.sha1(usedforsecurity=False)
        h.update(b"blob %d\0" % path.stat().st_size)
    else:
        return None
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _marker(target: Path) -> Path:
    return target.with_name(target.name + ".verified")


def _part(target: Path) -> Path:
    return target.with_name(target.name + ".part")


def _state_path(part: Path) -> Path:
    return part.with_name(part.name + ".json")


def is_complete(target: Path, rf: RemoteFile, verify: bool = True) -> bool:
    """Size matches and, with verify, the hash matches. A FILE.verified marker saves rehashing later."""
    if not target.is_file() or (rf.size and target.stat().st_size != rf.size):
        return False
    expected = rf.expected_hash
    if not verify or expected is None:
        return True
    marker = _marker(target)
    try:
        if marker.read_text().strip() == expected and marker.stat().st_mtime_ns >= target.stat().st_mtime_ns:
            return True
    except OSError:
        pass
    log.info("checking %s against its pinned hash", target.name)
    if _digest(target, rf) == expected:
        marker.write_text(expected + "\n")
        return True
    log.warning("%s does not match its pinned hash; downloading it again", target)
    return False


# ---------------------------------------------------------------------------------------------------------
# Transfers


def _get(url: str, headers: dict[str, str]) -> requests.Response:
    return _session().get(url, headers={**_auth_headers(), "Accept-Encoding": "identity", **headers},
                          stream=True, timeout=TIMEOUT)


def _fetch_range(url: str, start: int, end: int, part: Path, written: int = 0,
                 on_progress: Callable[[int], None] | None = None) -> None:
    """Write bytes start..end (inclusive) of url into part at the same offset.

    The first `written` bytes are already in place; retries continue from the last byte written.
    on_progress(written) runs after each block reaches the OS (unbuffered writes).
    """
    total = end - start + 1
    failures = 0
    while written < total:
        first = start + written
        try:
            with _get(url, {"Range": f"bytes={first}-{end}"}) as r:
                if r.status_code == 200:
                    raise RangeNotSupported(url)
                _raise_for_status(r)
                content_range = r.headers.get("Content-Range", "")
                if r.status_code != 206 or not content_range.startswith(f"bytes {first}-{end}/"):
                    raise OSError(f"unexpected answer {r.status_code} {content_range!r} for bytes {first}-{end}")
                with open(part, "r+b", buffering=0) as f:
                    f.seek(first)
                    for block in r.iter_content(64 << 10):  # urllib3 drops a partial block if the link breaks
                        view = memoryview(block)[: total - written]
                        while view:
                            n = f.write(view)
                            view = view[n:]
                            written += n
                        if on_progress is not None:
                            on_progress(written)
                        if written >= total:
                            break
            if written < total:
                raise OSError(f"connection closed after {written} of {total} bytes of range {start}-{end}")
        except (RangeNotSupported, PermissionError):
            raise
        except (requests.RequestException, OSError) as exc:
            failures += 1
            if failures >= RETRIES:
                raise
            delay = min(20.0, 0.5 * 2**failures)
            log.debug("range %d-%d failed at byte %d (%s); retrying in %.1fs", start, end, written, exc, delay)
            time.sleep(delay)


def _download_stream(rf: RemoteFile, part: Path) -> None:
    for attempt in range(RETRIES):
        try:
            with _get(rf.url, {}) as r:
                _raise_for_status(r)
                with open(part, "wb") as f:
                    f.writelines(r.iter_content(1 << 20))
            if rf.size and part.stat().st_size != rf.size:
                raise OSError(f"got {part.stat().st_size} of {rf.size} bytes")
            return
        except PermissionError:
            raise
        except (requests.RequestException, OSError) as exc:
            if attempt == RETRIES - 1:
                raise
            log.debug("download of %s failed (%s); retrying", rf.name, exc)
            time.sleep(min(20.0, 0.5 * 2**attempt))


@dataclass
class _Progress:
    """Resume state of a parallel download, kept in FILE.part.json next to FILE.part."""

    done: set[int]
    partial: dict[int, int]  # chunk index -> bytes already written from the chunk start


def _load_state(path: Path, rf: RemoteFile, chunk: int) -> _Progress | None:
    try:
        state = json.loads(path.read_text())
        if state.get("size") != rf.size or state.get("hash") != rf.expected_hash or state.get("chunk") != chunk:
            return None
        return _Progress({int(i) for i in state.get("done", [])},
                         {int(k): int(v) for k, v in state.get("partial", {}).items()})
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _save_state(path: Path, rf: RemoteFile, chunk: int, prog: _Progress) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"size": rf.size, "hash": rf.expected_hash, "chunk": chunk, "done": sorted(prog.done),
                               "partial": {str(k): v for k, v in sorted(prog.partial.items())}}))
    os.replace(tmp, path)


def _download_parallel(rf: RemoteFile, part: Path, workers: int | None = None, chunk: int | None = None) -> None:
    """Parallel range requests into one preallocated .part. FILE.part.json records finished chunks and the
    bytes written in unfinished ones, so an interrupted download resumes where each chunk stopped."""
    plan_workers, plan_chunk = plan_parallel(rf.size)
    workers, chunk = workers or plan_workers, chunk or plan_chunk
    ranges = split_ranges(rf.size, chunk)
    state_file = _state_path(part)
    prog = _load_state(state_file, rf, chunk) if part.is_file() and part.stat().st_size == rf.size else None
    if prog is None:
        with open(part, "wb") as f:
            f.truncate(rf.size)
        prog = _Progress(set(), {})
        _save_state(state_file, rf, chunk, prog)
    else:
        lengths = {i: b - a + 1 for i, (a, b) in enumerate(ranges)}
        prog.done &= set(lengths)
        prog.partial = {i: n for i, n in prog.partial.items() if i in lengths and 0 < n <= lengths[i]}
        have = sum(lengths[i] for i in prog.done) + sum(prog.partial.values())
        if have:
            log.info("resuming %s: %.1f of %.1f MB already downloaded", rf.name, have / 1e6, rf.size / 1e6)
    todo = [i for i in range(len(ranges)) if i not in prog.done]
    lock = threading.Lock()
    last_save = [time.monotonic()]
    from tqdm import tqdm

    initial = sum(ranges[i][1] - ranges[i][0] + 1 for i in prog.done) + sum(prog.partial.values())
    bar = tqdm(total=rf.size, initial=initial, unit="B", unit_scale=True, unit_divisor=1024, desc=rf.name,
               disable=None)

    def work(i: int) -> None:
        start, end = ranges[i]
        with lock:
            seen = [prog.partial.get(i, 0)]

        def progress(written: int) -> None:
            with lock:
                bar.update(written - seen[0])
                seen[0] = prog.partial[i] = written
                if time.monotonic() - last_save[0] > 0.5:
                    _save_state(state_file, rf, chunk, prog)
                    last_save[0] = time.monotonic()

        _fetch_range(rf.url, start, end, part, written=seen[0], on_progress=progress)
        with lock:
            prog.done.add(i)
            prog.partial.pop(i, None)
            _save_state(state_file, rf, chunk, prog)

    try:
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(todo)))) as pool:
            futures = [pool.submit(work, i) for i in todo]
            finished, _ = wait(futures, return_when=FIRST_EXCEPTION)
            failed = [f for f in finished if f.exception() is not None]
            if failed:
                for f in futures:
                    f.cancel()
                raise failed[0].exception()
    finally:
        bar.close()


def download_file(rf: RemoteFile, target: Path, verify: bool = True, *, workers: int | None = None,
                  chunk: int | None = None) -> Path:
    """Download rf to target unless it is already complete. The target only changes once the data checks out."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = _part(target)
    if is_complete(target, rf, verify):
        log.info("%s: present", target.name)
        part.unlink(missing_ok=True)  # leftovers of an interrupted earlier download
        _state_path(part).unlink(missing_ok=True)
        return target
    if rf.size >= PARALLEL_MIN_BYTES:
        try:
            _download_parallel(rf, part, workers, chunk)
        except RangeNotSupported:
            log.warning("server ignores range requests for %s; downloading it in one stream", rf.name)
            _state_path(part).unlink(missing_ok=True)
            _download_stream(rf, part)
    else:
        _download_stream(rf, part)
    if rf.size and part.stat().st_size != rf.size:
        raise OSError(f"{rf.name}: downloaded {part.stat().st_size} bytes, expected {rf.size}")
    expected = rf.expected_hash
    if verify and expected is not None:
        got = _digest(part, rf)
        if got != expected:
            part.unlink(missing_ok=True)
            _state_path(part).unlink(missing_ok=True)
            raise OSError(f"{rf.name}: hash {got} does not match the pinned {expected}; the partial file was removed")
    os.replace(part, target)
    _state_path(part).unlink(missing_ok=True)
    if verify and expected is not None:
        _marker(target).write_text(expected + "\n")
    else:
        _marker(target).unlink(missing_ok=True)
    log.info("%s: downloaded %.1f MB", target.name, target.stat().st_size / 1e6)
    return target


# ---------------------------------------------------------------------------------------------------------
# Models


def remote_files(spec: ModelSpec) -> dict[str, RemoteFile]:
    """Files of spec.repo at the pinned revision with sizes and hashes, from the Hugging Face API."""
    base = hf_endpoint()
    r = _session().get(f"{base}/api/models/{spec.repo}/revision/{spec.revision}", params={"blobs": "true"},
                       headers=_auth_headers(), timeout=TIMEOUT)
    r.raise_for_status()
    out = {}
    for s in r.json().get("siblings", []):
        lfs = s.get("lfs") or {}
        name = s["rfilename"]
        out[name] = RemoteFile(
            name=name,
            url=f"{base}/{spec.repo}/resolve/{spec.revision}/{name}",
            size=int(lfs.get("size") or s.get("size") or 0),
            sha256=lfs.get("sha256"),
            git_sha1=None if lfs else s.get("blobId"),
        )
    return out


def fetch_model(spec: ModelSpec, dest: Path | None = None, verify: bool = True) -> list[Path]:
    dest = Path(dest) if dest is not None else spec.local_dir
    try:
        remote = remote_files(spec)
    except requests.RequestException as exc:
        if all((dest / f).is_file() for f in spec.files):
            log.warning("cannot reach Hugging Face for %s (%s); keeping the local files unchecked", spec.repo, exc)
            return [dest / f for f in spec.files]
        raise
    missing = [f for f in spec.files if f not in remote]
    if missing:
        raise RuntimeError(f"{spec.repo} at {spec.revision[:10]} has no {', '.join(missing)}")
    return [download_file(remote[f], dest / f, verify) for f in spec.files]


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    root = dest.resolve()
    for info in zf.infolist():
        parts = PurePosixPath(info.filename.replace("\\", "/")).parts
        if info.filename.startswith("/") or ".." in parts:
            raise ValueError(f"unsafe path in archive: {info.filename}")
        target = dest.joinpath(*parts)
        if not target.resolve().is_relative_to(root):
            raise ValueError(f"unsafe path in archive: {info.filename}")
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)


def _trust(hub: Path) -> None:
    """Add the repo to torch.hub's trusted_list so loading it never prompts."""
    trusted = hub / "trusted_list"
    text = trusted.read_text() if trusted.is_file() else ""
    if DINOV2_TRUSTED not in text.split():
        sep = "" if not text or text.endswith("\n") else "\n"
        trusted.write_text(f"{text}{sep}{DINOV2_TRUSTED}\n")


def install_dinov2_hub(torch_home: str | Path | None = None, url: str = DINOV2_HUB_ZIP) -> Path:
    """Unpack the DINOv2 repository where torch.hub looks for it (TORCH_HOME/hub/facebookresearch_dinov2_main)."""
    hub = Path(torch_home or os.environ.get("TORCH_HOME") or TORCH_HOME) / "hub"
    dest = hub / DINOV2_DIR
    hub.mkdir(parents=True, exist_ok=True)
    if not (dest / "hubconf.py").is_file():
        log.info("installing DINOv2 hub code into %s", dest)
        with tempfile.TemporaryDirectory(dir=hub, prefix=".dinov2-") as tmp:
            archive = Path(tmp) / "dinov2.zip"
            _download_stream(RemoteFile("dinov2.zip", url, 0), archive)
            unpacked = Path(tmp) / "unpacked"
            with zipfile.ZipFile(archive) as zf:
                _safe_extract(zf, unpacked)
            tops = [p for p in unpacked.iterdir() if not p.name.startswith(".")]
            src = tops[0] if len(tops) == 1 and tops[0].is_dir() else unpacked  # GitHub zips hold dinov2-main/
            if not (src / "hubconf.py").is_file():
                raise RuntimeError(f"{url} does not contain hubconf.py")
            if dest.exists():
                shutil.rmtree(dest)
            os.replace(src, dest)
    _trust(hub)
    return dest


def fetch_all(verify: bool = True) -> dict[str, list[Path]]:
    """Download every config.MODELS entry at its pinned revision and install the DINOv2 hub code."""
    setup_env()
    results: dict[str, list[Path]] = {}
    errors = []
    for key, spec in MODELS.items():
        try:
            results[key] = fetch_model(spec, verify=verify)
        except (requests.RequestException, OSError, RuntimeError, ValueError) as exc:
            log.error("%s: %s", spec.repo, exc)
            errors.append(f"{spec.repo}: {exc}")
    try:
        install_dinov2_hub()
    except (requests.RequestException, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        log.error("DINOv2 hub code: %s", exc)
        errors.append(f"DINOv2 hub code: {exc}")
    if errors:
        raise RuntimeError("fetch-weights failed; rerun to resume:\n  " + "\n  ".join(errors))
    return results


# ---------------------------------------------------------------------------------------------------------
# Doctor


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} GB"


def doctor(models: dict[str, ModelSpec] | None = None, torch_home: str | Path | None = None,
           min_free_gb: float = 2.0) -> int:
    """Print a readiness checklist; returns 0 when every required check passes, else 1."""
    setup_env()
    models = MODELS if models is None else models
    failed = 0

    def report(status: str, label: str, detail: str) -> None:
        nonlocal failed
        failed += status == "FAIL"
        print(f"  [{status:4}] {label}: {detail}")

    print("scan2scope doctor")
    for spec in models.values():
        d = spec.local_dir
        missing = [f for f in spec.files if not (d / f).is_file()]
        if missing:
            report("FAIL", f"weights {spec.name}", f"missing {', '.join(missing)} in {d} (run: scan2scope fetch-weights)")
            continue
        size = sum((d / f).stat().st_size for f in spec.files)
        verified = all(_marker(d / f).is_file() for f in spec.files)
        report("ok" if verified else "warn", f"weights {spec.name}",
               f"{_fmt_bytes(size)} in {d}" + ("" if verified else ", hashes not checked yet (run: scan2scope fetch-weights)"))

    hub = Path(torch_home or os.environ.get("TORCH_HOME") or TORCH_HOME) / "hub" / DINOV2_DIR
    report("ok" if (hub / "hubconf.py").is_file() else "FAIL", "DINOv2 hub code", str(hub))

    try:
        import torch

        from scan2scope.config import torch_device

        mps = "available" if torch.backends.mps.is_available() else "not available"
        report("ok", "torch", f"{torch.__version__}, device {torch_device()}, MPS {mps}")
    except ImportError:
        report("FAIL", "torch", "not installed; the photo and video tiers and damage detection need it")

    import importlib.util

    missing_ml = [m for m in ("mapanything", "transformers") if importlib.util.find_spec(m) is None]
    report("FAIL" if missing_ml else "ok", "ml packages",
           f"missing {', '.join(missing_ml)}" if missing_ml else "mapanything, transformers")

    probe_dir = CACHE_DIR if CACHE_DIR.exists() else Path.home()
    free = shutil.disk_usage(probe_dir).free
    status = "FAIL" if free < min_free_gb * 1e9 else "warn" if free < 2.5 * min_free_gb * 1e9 else "ok"
    report(status, "free disk", f"{_fmt_bytes(free)} at {probe_dir}")

    try:
        import av

        av.codec.Codec("hevc", "r")
        report("ok", "PyAV HEVC decoder", f"PyAV {av.__version__}")
    except (ImportError, ValueError) as exc:
        report("FAIL", "PyAV HEVC decoder", f"unavailable ({exc}); iPhone videos cannot be decoded")

    try:
        import pillow_heif

        report("ok", "pillow-heif", f"{pillow_heif.__version__}, libheif {pillow_heif.libheif_version()}")
    except ImportError:
        report("FAIL", "pillow-heif", "not installed; HEIC photos cannot be read")

    print("all required checks passed" if not failed else f"{failed} required check(s) failed")
    return 0 if not failed else 1
