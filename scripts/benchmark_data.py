"""Package raw benchmark captures for a GitHub release, and fetch them back.

    python scripts/benchmark_data.py package bench/data/home      # zips raw/ into dist/, writes raw_manifest.json
    python scripts/benchmark_data.py fetch bench/data/home        # downloads and verifies the zips listed there

Raw captures are too large for git; the manifest (committed) pins every zip by size and SHA-256 and records
which release asset holds it. GitHub release assets are capped at 2 GiB each, so captures are split by folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "abhayraihal/scan2scope"
TAG = "benchmark-data-v1"
MAX_ASSET = 1_900_000_000
GPS_TAGS = (0x8825,)  # EXIF GPSInfo IFD pointer


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gps_warnings(raw: Path) -> list[str]:
    try:
        from PIL import Image
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        return []
    found = []
    for p in sorted(raw.rglob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".heic", ".heif"}:
            try:
                if any(t in Image.open(p).getexif() for t in GPS_TAGS):
                    found.append(str(p.relative_to(raw)))
            except Exception:  # unreadable files are reported by the pipeline, not here
                continue
    return found


def package(prop: Path, dist: Path) -> None:
    raw = prop / "raw"
    if not raw.is_dir():
        sys.exit(f"no raw/ folder in {prop}")
    gps = gps_warnings(raw)
    if gps:
        print(f"warning: {len(gps)} photos carry GPS location, e.g. {gps[0]}; strip it before publishing")
    dist.mkdir(parents=True, exist_ok=True)
    assets = []
    for item in sorted(raw.iterdir()):
        if item.name.startswith("."):
            continue
        name = f"{prop.name}__{item.name.replace(' ', '_')}.zip"
        out = dist / name
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            files = [item] if item.is_file() else sorted(p for p in item.rglob("*") if p.is_file())
            for f in files:
                z.write(f, f.relative_to(raw).as_posix())
        size = out.stat().st_size
        if size > MAX_ASSET:
            sys.exit(f"{name} is {size / 1e9:.2f} GB, over the release asset limit; split the capture")
        assets.append({"asset": name, "size": size, "sha256": sha256(out), "extracts_to": "raw/"})
        print(f"{name}: {size / 1e6:.1f} MB")
    manifest = {"repo": REPO, "tag": TAG, "property": prop.name, "assets": assets}
    (prop / "raw_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {prop / 'raw_manifest.json'}; upload with:\n  gh release upload {TAG} {dist}/{prop.name}__*.zip "
          f"--repo {REPO}  (create it first with: gh release create {TAG} --repo {REPO} --notes 'Raw benchmark data')")


def fetch(prop: Path) -> None:
    manifest = json.loads((prop / "raw_manifest.json").read_text())
    for a in manifest["assets"]:
        dest = prop / ".download" / a["asset"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not (dest.exists() and dest.stat().st_size == a["size"] and sha256(dest) == a["sha256"]):
            url = f"https://github.com/{manifest['repo']}/releases/download/{manifest['tag']}/{a['asset']}"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest)
            if sha256(dest) != a["sha256"]:
                sys.exit(f"checksum mismatch for {a['asset']}")
        with zipfile.ZipFile(dest) as z:
            z.extractall(prop / a["extracts_to"])
        print(f"ok {a['asset']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["package", "fetch"])
    p.add_argument("property_dir", type=Path)
    p.add_argument("--dist", type=Path, default=Path("dist"))
    a = p.parse_args()
    package(a.property_dir, a.dist) if a.action == "package" else fetch(a.property_dir)


if __name__ == "__main__":
    main()
