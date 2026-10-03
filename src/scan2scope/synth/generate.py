"""Synthetic LiDAR benchmark: apartments with three Stray Scanner captures each and exact ground truth.

Layout per property (same as real benchmark data):
    synth_K/ground_truth.yaml   bench/templates/ground_truth.yaml format plus `polygon`, `adjacency`
    synth_K/plan.png            ground-truth plan with the lidar_1 route
    synth_K/raw/lidar_1         normal drift
    synth_K/raw/lidar_2         repeat: different route, noise and drift seeds
    synth_K/raw/lidar_drift     lidar_1 frames with strong pose drift (drift ablation); sensor files hard-linked
    synth_K/truth/<capture>.yaml, <capture>_poses.csv   ARKit-from-property transform, drift, true poses
"""

from __future__ import annotations

import datetime as _dt
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scan2scope.synth.apartment import TEMPLATES, Apartment, adjacency, ground_truth_rooms, random_apartment
from scan2scope.synth.capture import CaptureConfig, CaptureResult, redrift_capture, write_capture
from scan2scope.synth.render import RenderScene

log = logging.getLogger("scan2scope.synth")

CAPTURE_NOTES = {"lidar_1": "normal drift",
                 "lidar_2": "repeat: different route, noise and drift seeds",
                 "lidar_drift": "lidar_1 frames with strong pose drift, for the drift ablation"}


def generate_benchmark(out: str | Path, n_properties: int = 4, seed: int = 0, *, fps: float = 10.0,
                       max_frames: int | None = None, rgb_scale: float = 0.5, workers: int = 1,
                       overwrite: bool = True) -> list[Path]:
    """Write `n_properties` synthetic properties under `out` (synth_0, synth_1, ...) and return their folders.

    Property K uses template TEMPLATES[K % 3]; odd K get an L-shaped room and landscape holding, even K a
    passage between rooms and portrait holding, so a 4-property run covers every variant.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    states = np.random.SeedSequence(seed).generate_state(max(n_properties, 0))
    jobs = [(out / f"synth_{k}", k, int(states[k]), fps, max_frames, rgb_scale, overwrite)
            for k in range(n_properties)]
    summaries = None
    if workers > 1 and len(jobs) > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                     initargs=(log.getEffectiveLevel(),)) as pool:
                summaries = list(pool.map(_property_job, jobs))
        except BrokenProcessPool as exc:  # e.g. a calling script without a __main__ guard under spawn
            log.warning("worker pool failed (%s); generating sequentially", exc)
    if summaries is None:
        summaries = [_property_job(job) for job in jobs]
    done = []
    for job, s in zip(jobs, summaries):
        if "error" in s:
            log.error("%s failed: %s", s["property"], s["error"])
            continue
        log.info("%s: %s, %d rooms, frames %s", s["property"], s["template"], s["rooms"], s["captures"])
        done.append(job[0])
    return done


def _init_worker(level: int) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


def _property_job(job: tuple) -> dict:
    path, index, seed, fps, max_frames, rgb_scale, overwrite = job
    try:
        return generate_property(path, index, seed, fps=fps, max_frames=max_frames, rgb_scale=rgb_scale,
                                 overwrite=overwrite)
    except Exception as exc:  # one bad property must not stop the others
        log.exception("property %s", path.name)
        return {"property": path.name, "error": f"{type(exc).__name__}: {exc}"}


def generate_property(path: str | Path, index: int, seed: int, *, fps: float = 10.0, max_frames: int | None = None,
                      rgb_scale: float = 0.5, overwrite: bool = True) -> dict:
    path = Path(path)
    if path.exists() and overwrite:
        shutil.rmtree(path)
    ss = np.random.SeedSequence(seed).generate_state(4)
    apt = random_apartment(int(ss[0]), template=TEMPLATES[index % len(TEMPLATES)], l_room=index % 2 == 1,
                           passage=True if index % 2 == 0 else None)
    scene = RenderScene(apt, seed=int(ss[0]))
    orientation = "portrait" if index % 2 == 0 else "landscape"
    cfg = CaptureConfig(fps=fps, orientation=orientation, drift="normal", rgb_scale=rgb_scale, max_frames=max_frames)
    raw = path / "raw"
    caps = {"lidar_1": write_capture(apt, raw / "lidar_1", seed=int(ss[1]), config=cfg, scene=scene)}
    caps["lidar_2"] = write_capture(apt, raw / "lidar_2", seed=int(ss[2]), config=cfg, scene=scene)
    caps["lidar_drift"] = redrift_capture(caps["lidar_1"], raw / "lidar_drift", seed=int(ss[3]), drift="strong")
    write_ground_truth(apt, path / "ground_truth.yaml", path.name, list(caps))
    for cid, res in caps.items():
        write_truth(res, path / "truth", cid, path.name, same_frames_as="lidar_1" if cid == "lidar_drift" else None)
    save_plan_png(apt, path / "plan.png", title=path.name, trajectory=caps["lidar_1"].true_pos)
    return {"property": path.name, "template": apt.template, "rooms": len(apt.rooms) - 1,
            "captures": {cid: res.n_frames for cid, res in caps.items()}}


def _f3(v: float) -> str:
    return f"{v:.3f}"


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def ground_truth_text(apt: Apartment, property_id: str, capture_ids: list[str],
                      date: str | None = None) -> str:
    """ground_truth.yaml text in the bench/templates/ground_truth.yaml layout, plus polygon and adjacency."""
    rooms = ground_truth_rooms(apt)
    lines = [
        f"# Synthetic ground truth from scan2scope.synth (apartment seed {apt.seed}, template {apt.template}).",
        "# Definitions: docs/ground_truth_protocol.md. Lengths in metres, exact to the millimetre.",
        "# Walls start at the wall holding the room's entry door (hallway: the entrance wall where every capture",
        "# starts and ends) and go clockwise seen from above. polygon: interior corners in property coordinates",
        "# (x east, y north, metres) in wall order; vertex k is the left end of wall W(k+1) seen from inside.",
        "# diagonal (non-rectangular rooms): distance from the start of W1 to the farthest corner.",
        f"property: {property_id}",
        "measured_by: synthetic",
        'instrument: "exact (generated geometry)"',
        f"date: {_q(date or _dt.datetime.now(_dt.UTC).date().isoformat())}",
        "",
        "rooms:",
    ]
    for room in rooms:
        lines.append(f"  - id: {_q(room['id'])}")
        lines.append("    walls:")
        for w in room["walls"]:
            lines.append(f"      - {{id: {w['id']}, length: {_f3(w['length'])}}}")
        lines.append(f"    diagonal: {'null' if room['diagonal'] is None else _f3(room['diagonal'])}")
        lines.append(f"    ceiling_height: [{', '.join(_f3(h) for h in room['ceiling_height'])}]")
        if room["openings"]:
            lines.append("    openings:")
            for o in room["openings"]:
                item = (f"{{id: {o['id']}, type: {o['type']}, wall: {o['wall']}, offset: {_f3(o['offset'])}, "
                        f"width: {_f3(o['width'])}, height: {_f3(o['height'])}")
                if o["type"] == "window":
                    item += f", sill: {_f3(o['sill'])}}}"
                else:
                    item += f", leads_to: {_q(o['leads_to'])}, wall_thickness: {_f3(o['wall_thickness'])}}}"
                lines.append(f"      - {item}")
        else:
            lines.append("    openings: []")
        lines.append("    damage: []")
        lines.append("    polygon: [" + ", ".join(f"[{_f3(x)}, {_f3(y)}]" for x, y in room["polygon"]) + "]")
        lines.append("")
    lines.append("# Rooms joined by a door or an open passage, with the opening id on each side.")
    lines.append("adjacency:")
    for a in adjacency(apt, rooms):
        lines.append(f"  - {{rooms: [{_q(a['rooms'][0])}, {_q(a['rooms'][1])}], type: {a['type']}, "
                     f"openings: [{a['openings'][0]}, {a['openings'][1]}]}}")
    lines.append("")
    lines.append("# Which captures belong to this property (paths relative to this file).")
    lines.append("captures:")
    for cid in capture_ids:
        note = CAPTURE_NOTES.get(cid)
        lines.append(f"  - {{id: {cid}, tier: lidar, path: raw/{cid}}}" + (f"   # {note}" if note else ""))
    return "\n".join(lines) + "\n"


def write_ground_truth(apt: Apartment, path: Path, property_id: str, capture_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ground_truth_text(apt, property_id, capture_ids))


def write_truth(res: CaptureResult, folder: Path, capture_id: str, property_id: str,
                same_frames_as: str | None = None) -> None:
    """Per-capture truth: the transform into the capture's ARKit world, drift record and true poses."""
    import yaml

    folder.mkdir(parents=True, exist_ok=True)
    cfg = res.config
    doc = {
        "capture": capture_id,
        "property": property_id,
        "synthetic": True,
        "frames": int(res.n_frames),
        "duration_s": round(float(res.frame_times[-1]), 3),
        "fps": float(cfg.fps),
        "dropped_frames": int(res.slots[-1] + 1 - res.n_frames),
        "walking_speed_factor": round(float(res.trajectory.speed), 3),
        "orientation": cfg.orientation,
        "rgb_size": list(cfg.rgb_size),
        "depth_size": list(cfg.depth_size),
        "same_frames_as": same_frames_as,
        "drift": res.drift.summary(),
        "depth_model": {"sigma_m": "0.004 + 0.006 * depth", "max_range_m": float(cfg.max_range),
                        "confidence": "2: depth < 3 m and incidence < 60 deg; 1: depth < 4.5 m; else 0",
                        "noise": bool(cfg.noise)},
        "T_arkit_from_property": [[round(float(v), 9) for v in row] for row in res.T_arkit_property],
        "poses": f"{capture_id}_poses.csv",
        "flags": list(res.flags),
    }
    (folder / f"{capture_id}.yaml").write_text(
        "# Synthetic capture truth. T_arkit_from_property maps property coordinates (metres, z up) to this\n"
        "# capture's ARKit world (y up). The poses file holds the true OpenCV camera-to-property pose per frame.\n"
        + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=120))
    q = Rotation.from_matrix(res.true_R).as_quat()
    rows = ["frame, timestamp, x, y, z, qx, qy, qz, qw"]
    for i in range(res.n_frames):
        p = res.true_pos[i]
        rows.append(f"{i:06d}, {res.timestamps[i]:.6f}, {p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}, "
                    f"{q[i, 0]:.8f}, {q[i, 1]:.8f}, {q[i, 2]:.8f}, {q[i, 3]:.8f}")
    (folder / f"{capture_id}_poses.csv").write_text("\n".join(rows) + "\n")


def save_plan_png(apt: Apartment, path: Path, *, title: str = "", trajectory: np.ndarray | None = None) -> None:
    """Ground-truth plan: walls cut at 1.0 m, openings, furniture, wall ids and lengths, optional route."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rooms = ground_truth_rooms(apt)
    fp = apt.footprint
    w, h = (fp.x1 - fp.x0) / 1000.0, (fp.y1 - fp.y0) / 1000.0
    fig, ax = plt.subplots(figsize=(min(16.0, 2.0 + 0.9 * w), min(12.0, 1.5 + 0.9 * h)))
    for b in apt.boxes:
        if b.kind == "wall" and not (b.lo[2] <= 1.0 <= b.hi[2]):
            continue
        style = {"wall": ("#2b2b2b", 1.0), "furniture": ("#c9955c", 0.55), "door_leaf": ("#3a9d3a", 0.8)}[b.kind]
        ax.add_patch(plt.Rectangle(b.lo[:2], b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], color=style[0], alpha=style[1],
                                   lw=0))
    colours = {"door": "#d62728", "window": "#17becf", "opening": "#9467bd"}
    for o in apt.openings:
        r = o.plan_rect()
        ax.add_patch(plt.Rectangle((r.x0 / 1000.0, r.y0 / 1000.0), r.w / 1000.0, r.h / 1000.0, fill=True,
                                   color=colours[o.kind], alpha=0.6, lw=0))
    for room, gt in zip(apt.rooms, rooms):
        P = room.polygon
        c = room.shape_m().representative_point()
        ax.text(c.x, c.y, f"{room.name}\n{room.area:.2f} m²", ha="center", va="center", fontsize=8,
                fontweight="bold")
        for k, wall in enumerate(gt["walls"]):
            a, b = P[k], P[(k + 1) % len(P)]
            mid = 0.5 * (a + b)
            d = (b - a) / np.linalg.norm(b - a)
            inward = np.array([d[1], -d[0]])
            pos = mid + inward * 0.22
            ax.text(pos[0], pos[1], f"{wall['id']} {wall['length']:.3f}", ha="center", va="center", fontsize=5.5,
                    rotation=0 if abs(d[0]) > 0.5 else 90, color="#333366")
    if trajectory is not None and len(trajectory):
        ax.plot(trajectory[:, 0], trajectory[:, 1], "-", color="#1f77b4", lw=0.6, alpha=0.7)
        ax.plot(trajectory[0, 0], trajectory[0, 1], "o", color="#1f77b4", ms=6)
    ax.set_xlim(fp.x0 / 1000.0 - 0.3, fp.x1 / 1000.0 + 0.3)
    ax.set_ylim(fp.y0 / 1000.0 - 0.3, fp.y1 / 1000.0 + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    area = sum(r.area for r in apt.rooms)
    ax.set_title(f"{title}: {apt.template}, {len(apt.rooms) - 1} rooms + hallway, ceiling {apt.ceiling:.3f} m, "
                 f"floor area {area:.2f} m²   (red door, purple passage, cyan window, green door leaf)", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
