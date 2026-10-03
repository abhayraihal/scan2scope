"""Ray casting for synthetic apartments: z-depth, surface ids and a simple shaded RGB image.

Rays are tested against axis-aligned boxes with the slab method, vectorised over rays and boxes. The floor and
the ceiling are planes clipped to the footprint. Cameras use OpenCV axes (x right, y down, z forward) and
camera-to-world poses in the property frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from scan2scope.synth.apartment import Apartment

log = logging.getLogger("scan2scope.synth")

_CHUNK = 4096
_WALL_PAINT = [(0.93, 0.91, 0.86), (0.86, 0.90, 0.93), (0.90, 0.93, 0.86), (0.95, 0.89, 0.84),
               (0.88, 0.86, 0.92), (0.94, 0.94, 0.94), (0.92, 0.87, 0.80), (0.83, 0.89, 0.88)]
_FLOOR_TILE = [(0.80, 0.80, 0.78), (0.62, 0.64, 0.66), (0.85, 0.82, 0.76)]
_FLOOR_WOOD = [(0.66, 0.50, 0.34), (0.74, 0.60, 0.44), (0.55, 0.40, 0.28)]
_FURNITURE_COLOURS = {"wardrobe": (0.62, 0.45, 0.30), "dresser": (0.70, 0.55, 0.38),
                      "desk": (0.55, 0.40, 0.28), "shelf": (0.75, 0.65, 0.50), "sofa": (0.35, 0.40, 0.55),
                      "tv_unit": (0.25, 0.25, 0.27), "cabinet": (0.80, 0.78, 0.74),
                      "counter": (0.88, 0.86, 0.82), "tall_unit": (0.90, 0.90, 0.90),
                      "vanity": (0.92, 0.92, 0.90), "bathtub": (0.97, 0.97, 0.97),
                      "table": (0.58, 0.42, 0.28), "door_leaf": (0.93, 0.92, 0.89)}
TEXTURES = ("paint", "wood", "tile", "ceiling", "grain", "fabric", "trim", "sky")


@dataclass
class Surface:
    name: str  # "02 bedroom/W3", "02 bedroom/floor", "02 bedroom/wardrobe", "trim"
    kind: str  # wall | floor | ceiling | furniture | door_leaf | trim
    room: int  # -1 when not inside a room
    style: int


@dataclass
class Hits:
    t: np.ndarray  # (N,) distance along the unit ray, inf for no hit
    point: np.ndarray  # (N, 3)
    normal: np.ndarray  # (N, 3) unit, facing the camera
    box: np.ndarray  # (N,) index into RenderScene boxes, -1 for floor, ceiling or nothing
    style: np.ndarray  # (N,) index into RenderScene styles, -1 for nothing
    sid: np.ndarray  # (N,) index into RenderScene surfaces, -1 for nothing
    room: np.ndarray  # (N,) room on the visible side of the surface, -1 if none


@dataclass
class DepthRender:
    depth: np.ndarray  # (h, w) z-depth in metres, inf where nothing is hit
    cos_incidence: np.ndarray  # (h, w) |cos| of the angle between the ray and the surface normal
    sid: np.ndarray  # (h, w) surface ids, -1 for nothing
    boxes_seen: np.ndarray  # box indices hit by at least one ray


class RenderScene:
    """Arrays for casting rays into one apartment."""

    def __init__(self, apt: Apartment, seed: int = 0) -> None:
        rng = np.random.default_rng(seed + 7919 * apt.seed)
        self.apt = apt
        self.floor_z = 0.0
        self.ceil_z = apt.ceiling
        fp = apt.footprint
        self.footprint = np.array([fp.x0, fp.y0, fp.x1, fp.y1], float) / 1000.0
        self.lo = np.array([b.lo for b in apt.boxes], np.float64).reshape(-1, 3)
        self.hi = np.array([b.hi for b in apt.boxes], np.float64).reshape(-1, 3)
        self.is_wall = np.array([b.kind == "wall" for b in apt.boxes], bool)

        self.styles_color: list[tuple[float, float, float]] = []
        self.styles_texture: list[int] = []
        self.styles_seed: list[float] = []

        def style(color, texture: str) -> int:
            self.styles_color.append(tuple(float(c) for c in color))
            self.styles_texture.append(TEXTURES.index(texture))
            self.styles_seed.append(float(rng.uniform(0, 100)))
            return len(self.styles_color) - 1

        self.surfaces: list[Surface] = []

        def surface(name: str, kind: str, room: int, st: int) -> int:
            self.surfaces.append(Surface(name, kind, room, st))
            return len(self.surfaces) - 1

        trim_style = style((0.96, 0.95, 0.93), "trim")
        self.trim_sid = surface("trim", "trim", -1, trim_style)
        self.threshold_style = style((0.55, 0.42, 0.30), "grain")
        self.threshold_sid = surface("threshold", "floor", -1, self.threshold_style)
        self.room_paint, self.room_floor, self.room_ceil = [], [], []
        self.floor_sid, self.ceil_sid, self.wall_sid = [], [], []
        palette = rng.permutation(len(_WALL_PAINT))
        for room in apt.rooms:
            paint = style(_WALL_PAINT[palette[room.index % len(palette)]], "paint")
            if room.kind in ("bathroom", "kitchen"):
                floor = style(rng.choice(_FLOOR_TILE), "tile")
            else:
                floor = style(rng.choice(_FLOOR_WOOD), "wood")
            ceil = style((0.97, 0.97, 0.96), "ceiling")
            self.room_paint.append(paint)
            self.room_floor.append(floor)
            self.room_ceil.append(ceil)
            self.floor_sid.append(surface(f"{room.name}/floor", "floor", room.index, floor))
            self.ceil_sid.append(surface(f"{room.name}/ceiling", "ceiling", room.index, ceil))
            self.wall_sid.append([surface(f"{room.name}/W{k + 1}", "wall", room.index, paint)
                                  for k in range(len(room.polygon_mm))])
        self.box_style = np.full(len(apt.boxes), -1, np.int32)
        self.box_sid = np.full(len(apt.boxes), -1, np.int32)
        counts: dict[tuple[int, str], int] = {}
        for i, b in enumerate(apt.boxes):
            if b.kind == "wall":
                continue
            base = np.array(_FURNITURE_COLOURS.get(b.label, (0.6, 0.5, 0.4))) * rng.uniform(0.9, 1.1)
            tex = "fabric" if b.label == "sofa" else ("trim" if b.label in ("bathtub", "vanity", "door_leaf")
                                                      else "grain")
            st = style(np.clip(base, 0, 1), tex)
            counts[b.room, b.label] = counts.get((b.room, b.label), 0) + 1
            name = f"{apt.rooms[b.room].name}/{b.label}{counts[b.room, b.label]}" if b.room >= 0 else b.label
            self.box_style[i] = st
            self.box_sid[i] = surface(name, b.kind, b.room, st)
        self.sky_style = style((0.75, 0.85, 0.97), "sky")
        self.colors = np.array(self.styles_color, np.float32)
        self.textures = np.array(self.styles_texture, np.int32)
        self.seeds = np.array(self.styles_seed, np.float64)

        rects, rect_room = [], []
        for room in apt.rooms:
            for r in room.rects:
                rects.append((r.x0 / 1000.0, r.y0 / 1000.0, r.x1 / 1000.0, r.y1 / 1000.0))
                rect_room.append(room.index)
        self.rects = np.array(rects, np.float64)
        self.rect_room = np.array(rect_room, np.int32)
        self.lights = np.array([[*room.shape_m().representative_point().coords[0], apt.ceiling - 0.25]
                                for room in apt.rooms], np.float64)
        # Wall edges per room in ground-truth order: plane axis, inward sign, plane coordinate, along range.
        self.edges = []
        for room in apt.rooms:
            P = room.polygon
            rows = []
            for k in range(len(P)):
                a, b = P[k], P[(k + 1) % len(P)]
                d = b - a
                inward = np.array([d[1], -d[0]]) / np.linalg.norm(d)
                axis = 0 if abs(d[0]) < 1e-9 else 1
                lo_s, hi_s = sorted((a[1 - axis], b[1 - axis]))
                rows.append((axis, np.sign(inward[axis]), a[axis], lo_s, hi_s, k))
            self.edges.append(np.array(rows, np.float64))

    @property
    def n_boxes(self) -> int:
        return len(self.lo)

    def room_of(self, xy: np.ndarray) -> np.ndarray:
        """Room index for plan points (N, 2), -1 outside every room."""
        x, y = xy[:, 0:1], xy[:, 1:2]
        r = self.rects
        inside = (x > r[:, 0]) & (x < r[:, 2]) & (y > r[:, 1]) & (y < r[:, 3])
        any_in = inside.any(1)
        return np.where(any_in, self.rect_room[inside.argmax(1)], -1).astype(np.int32)

    def cast(self, origin: np.ndarray, dirs: np.ndarray, boxes: np.ndarray | None = None,
             resolve_walls: bool = True) -> Hits:
        """Cast unit rays (N, 3) from one origin. `boxes` restricts the box test to a subset.

        With resolve_walls=False, wall hits inside rooms keep sid -1 (styles are still set); this saves the
        per-wall lookup when only colour is needed.
        """
        o = np.asarray(origin, np.float64)
        d = np.asarray(dirs, np.float64)
        n = len(d)
        idx = np.arange(self.n_boxes) if boxes is None else np.asarray(boxes, np.int64)
        t_box = np.full(n, np.inf)
        k_box = np.full(n, -1, np.int64)
        axis = np.full(n, -1, np.int64)
        dd = np.where(np.abs(d) < 1e-12, 1e-12, d)
        if len(idx):
            inv = (1.0 / dd).astype(np.float32)
            a = (self.lo[idx] - o).astype(np.float32)
            b = (self.hi[idx] - o).astype(np.float32)
            for s in range(0, n, _CHUNK):
                iv = inv[s:s + _CHUNK]
                tn = tf = None
                for ax in range(3):
                    ta = iv[:, ax:ax + 1] * a[None, :, ax]
                    tb = iv[:, ax:ax + 1] * b[None, :, ax]
                    lo_t = np.minimum(ta, tb)
                    hi_t = np.maximum(ta, tb, out=tb)
                    if tn is None:
                        tn, tf = lo_t, hi_t
                    else:
                        np.maximum(tn, lo_t, out=tn)
                        np.minimum(tf, hi_t, out=tf)
                tn[~((tn <= tf) & (tn > 1e-6))] = np.inf
                k = tn.argmin(1)
                t_box[s:s + _CHUNK] = tn[np.arange(len(k)), k]
                k_box[s:s + _CHUNK] = idx[k]
            k_box[~np.isfinite(t_box)] = -1
            # Recompute the winning box in float64 (depth exact well below 1 mm) and find the hit face.
            hit = np.nonzero(k_box >= 0)[0]
            if len(hit):
                kk = k_box[hit]
                dh = dd[hit]
                t_ax = (np.where(dh > 0, self.lo[kk], self.hi[kk]) - o) / dh
                axis[hit] = t_ax.argmax(1)
                t_box[hit] = t_ax[np.arange(len(hit)), axis[hit]]
        fx0, fy0, fx1, fy1 = self.footprint
        t_plane = np.full(n, np.inf)
        plane_z = np.zeros(n)
        down = d[:, 2] < -1e-12
        up = d[:, 2] > 1e-12
        t_f = np.where(down, (self.floor_z - o[2]) / np.where(down, d[:, 2], -1.0), np.inf)
        t_c = np.where(up, (self.ceil_z - o[2]) / np.where(up, d[:, 2], 1.0), np.inf)
        for tp, zsign in ((t_f, 1.0), (t_c, -1.0)):
            fin = np.isfinite(tp)
            p = o[:2] + np.where(fin, tp, 0.0)[:, None] * d[:, :2]
            ok = fin & (p[:, 0] > fx0) & (p[:, 0] < fx1) & (p[:, 1] > fy0) & (p[:, 1] < fy1)
            better = ok & (tp < t_plane)
            t_plane[better] = tp[better]
            plane_z[better] = zsign
        use_plane = t_plane < t_box
        t = np.where(use_plane, t_plane, t_box)
        box = np.where(use_plane, -1, k_box).astype(np.int32)
        finite = np.isfinite(t)
        point = o + np.where(finite, t, 0.0)[:, None] * d
        normal = np.zeros((n, 3), np.float32)
        normal[use_plane, 2] = plane_z[use_plane]
        on_box = (~use_plane) & (k_box >= 0)
        axis[~on_box] = -1
        if on_box.any():
            sel = np.nonzero(on_box)[0]
            normal[sel, axis[sel]] = -np.sign(dd[sel, axis[sel]])
        style = np.full(n, -1, np.int32)
        sid = np.full(n, -1, np.int32)
        room = np.full(n, -1, np.int32)

        plane = use_plane & finite
        if plane.any():
            r = self.room_of(point[plane, :2])
            room[plane] = r
            is_floor = normal[plane, 2] > 0
            st = np.where(r >= 0, np.where(is_floor, np.take(self.room_floor, r.clip(0)),
                                           np.take(self.room_ceil, r.clip(0))), self.threshold_style)
            si = np.where(r >= 0, np.where(is_floor, np.take(self.floor_sid, r.clip(0)),
                                           np.take(self.ceil_sid, r.clip(0))), self.threshold_sid)
            style[plane] = st
            sid[plane] = si
        furn = on_box & ~self.is_wall[k_box.clip(0)]
        if furn.any():
            style[furn] = self.box_style[k_box[furn]]
            sid[furn] = self.box_sid[k_box[furn]]
            room[furn] = self.room_of(point[furn, :2] + 0.02 * normal[furn, :2])
        wall = on_box & self.is_wall[k_box.clip(0)]
        if wall.any():
            vertical = wall & (axis != 2)
            r = np.full(n, -1, np.int32)
            r[vertical] = self.room_of(point[vertical, :2] + 0.02 * normal[vertical, :2])
            room[wall] = r[wall]
            in_room = vertical & (r >= 0)
            style[wall & ~in_room] = self.trim_style_id
            sid[wall & ~in_room] = self.trim_sid
            style[in_room] = np.take(self.room_paint, r[in_room])
            if resolve_walls and in_room.any():
                sid[in_room] = self._wall_ids(point[in_room], axis[in_room], normal[in_room], r[in_room])
        sky = ~finite
        style[sky] = self.sky_style
        return Hits(t=t, point=point, normal=normal, box=box, style=style, sid=sid, room=room)

    @property
    def trim_style_id(self) -> int:
        return self.surfaces[self.trim_sid].style

    def _wall_ids(self, p: np.ndarray, axis: np.ndarray, normal: np.ndarray, room: np.ndarray) -> np.ndarray:
        out = np.full(len(p), self.trim_sid, np.int32)
        for ri in np.unique(room):
            sel = np.nonzero(room == ri)[0]
            E = self.edges[ri]
            ax = axis[sel]
            sgn = normal[sel, ax] if len(sel) else np.zeros(0)
            coord = p[sel, ax]
            along = p[sel, 1 - ax]
            match = ((ax[:, None] == E[:, 0]) & (np.sign(sgn)[:, None] == E[:, 1])
                     & (np.abs(coord[:, None] - E[:, 2]) < 3e-3)
                     & (along[:, None] > E[:, 3] - 3e-3) & (along[:, None] < E[:, 4] + 3e-3))
            found = match.any(1)
            k = match.argmax(1)
            ids = np.asarray(self.wall_sid[ri], np.int32)
            out[sel[found]] = ids[E[k[found], 5].astype(int)]
        return out


def camera_rays(K: np.ndarray, rgb_size: tuple[int, int], out_size: tuple[int, int]) -> np.ndarray:
    """Camera-frame ray directions (h*w, 3) with z = 1 for an out_size grid covering the RGB field of view.

    Pixel (i, j) of the grid covers RGB pixel ((j + 0.5) * W / w - 0.5, (i + 0.5) * H / h - 0.5), the same
    convention as scan2scope.types.CameraView point maps.
    """
    W, Hh = rgb_size
    w, h = out_size
    u = (np.arange(w) + 0.5) * (W / w) - 0.5
    v = (np.arange(h) + 0.5) * (Hh / h) - 0.5
    uu, vv = np.meshgrid(u, v)
    x = (uu - K[0, 2]) / K[0, 0]
    y = (vv - K[1, 2]) / K[1, 1]
    return np.stack([x, y, np.ones_like(x)], -1).reshape(-1, 3)


def render_depth(scene: RenderScene, K: np.ndarray, rgb_size: tuple[int, int], depth_size: tuple[int, int],
                 R_wc: np.ndarray, t_wc: np.ndarray) -> DepthRender:
    """Noise-free z-depth (metres) and surface ids on the depth grid, aligned with the RGB image."""
    rays = camera_rays(K, rgb_size, depth_size)
    norm = np.linalg.norm(rays, axis=1)
    d = (rays / norm[:, None]) @ R_wc.T
    hits = scene.cast(t_wc, d)
    w, h = depth_size
    depth = (hits.t / norm).reshape(h, w)
    cos_inc = np.abs((d * hits.normal).sum(1)).reshape(h, w)
    seen = np.unique(hits.box[hits.box >= 0])
    return DepthRender(depth=depth, cos_incidence=cos_inc, sid=hits.sid.reshape(h, w), boxes_seen=seen)


def render_rgb(scene: RenderScene, K: np.ndarray, rgb_size: tuple[int, int], R_wc: np.ndarray,
               t_wc: np.ndarray, boxes: np.ndarray | None = None, scale: float = 0.5,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Shaded RGB image (H, W, 3) uint8. Rendered at `scale` of the RGB size and resized up."""
    W, Hh = rgb_size
    w, h = max(1, round(W * scale)), max(1, round(Hh * scale))
    rays = camera_rays(K, rgb_size, (w, h))
    d = (rays / np.linalg.norm(rays, axis=1)[:, None]) @ R_wc.T
    hits = scene.cast(t_wc, d, boxes=boxes, resolve_walls=False)
    col = shade(scene, hits, d)
    img = col.reshape(h, w, 3)
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    img = img * (1.0 - 0.18 * r2)[..., None]
    if rng is not None:
        img = img + rng.normal(0.0, 0.006, img.shape)
    img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if (w, h) != (W, Hh):
        img = cv2.resize(img, (W, Hh), interpolation=cv2.INTER_LINEAR)
    return img


def _hash2(i: np.ndarray, j: np.ndarray, seed: np.ndarray | float) -> np.ndarray:
    x = np.sin(i * 127.1 + j * 311.7 + seed * 17.13) * 43758.5453
    return x - np.floor(x)


def _noise(u: np.ndarray, v: np.ndarray, seed: np.ndarray | float) -> np.ndarray:
    """Smooth value noise in [0, 1]."""
    i, j = np.floor(u), np.floor(v)
    fu, fv = u - i, v - j
    fu = fu * fu * (3.0 - 2.0 * fu)
    fv = fv * fv * (3.0 - 2.0 * fv)
    a, b = _hash2(i, j, seed), _hash2(i + 1, j, seed)
    c, e = _hash2(i, j + 1, seed), _hash2(i + 1, j + 1, seed)
    return a + (b - a) * fu + (c - a) * fv + (a - b - c + e) * fu * fv


def _texture(tex: np.ndarray, u: np.ndarray, v: np.ndarray, seed: np.ndarray) -> np.ndarray:
    out = np.ones(len(u))
    m = tex == TEXTURES.index("paint")
    if m.any():
        out[m] = 1.0 + 0.07 * (_noise(u[m] * 2.5, v[m] * 2.5, seed[m]) - 0.5) \
            + 0.04 * (_noise(u[m] * 18, v[m] * 18, seed[m] + 3) - 0.5)
    m = tex == TEXTURES.index("ceiling")
    if m.any():
        out[m] = 1.0 + 0.04 * (_noise(u[m] * 3, v[m] * 3, seed[m]) - 0.5)
    m = tex == TEXTURES.index("wood")
    if m.any():
        uu, vv, sd = u[m], v[m], seed[m]
        row = np.floor(vv / 0.16)
        shift = _hash2(row, row * 0.0, sd) * 1.2
        col = np.floor((uu + shift) / 1.2)
        tint = 0.82 + 0.3 * _hash2(row, col, sd + 1)
        grain = 0.9 + 0.2 * _noise(uu * 1.5, vv * 45, sd + 2)
        seam = ((vv / 0.16 - row) < 0.035) | (((uu + shift) / 1.2 - col) < 0.005)
        out[m] = tint * grain * np.where(seam, 0.55, 1.0)
    m = tex == TEXTURES.index("tile")
    if m.any():
        uu, vv, sd = u[m] / 0.3, v[m] / 0.3, seed[m]
        grout = ((uu - np.floor(uu)) < 0.03) | ((vv - np.floor(vv)) < 0.03)
        tint = 0.95 + 0.08 * _hash2(np.floor(uu), np.floor(vv), sd)
        out[m] = np.where(grout, 0.68, tint)
    m = tex == TEXTURES.index("grain")
    if m.any():
        out[m] = 0.85 + 0.25 * _noise(u[m] * 3, v[m] * 20, seed[m])
    m = tex == TEXTURES.index("fabric")
    if m.any():
        out[m] = 0.88 + 0.16 * _noise(u[m] * 35, v[m] * 35, seed[m])
    m = tex == TEXTURES.index("trim")
    if m.any():
        out[m] = 0.97 + 0.03 * _noise(u[m] * 4, v[m] * 4, seed[m])
    return out


def shade(scene: RenderScene, hits: Hits, dirs: np.ndarray) -> np.ndarray:
    """Linear-ish RGB in [0, 1] for each ray: per-surface colour, procedural texture, one light per room."""
    n = len(dirs)
    out = np.zeros((n, 3))
    sky = hits.style == scene.sky_style
    if sky.any():
        z = dirs[sky, 2]
        horizon = np.array([0.86, 0.90, 0.95])
        zenith = np.array([0.45, 0.63, 0.92])
        ground = np.array([0.42, 0.46, 0.38])
        a = np.clip(z, 0, 1)[:, None]
        out[sky] = np.where((z >= 0)[:, None], horizon * (1 - a) + zenith * a, ground)
    m = ~sky & (hits.style >= 0)
    if not m.any():
        return out
    st = hits.style[m]
    p = hits.point[m]
    nrm = hits.normal[m].astype(np.float64)
    ax = np.abs(nrm).argmax(1)
    uv_axes = np.array([[1, 2], [0, 2], [0, 1]])[ax]
    u = p[np.arange(len(p)), uv_axes[:, 0]]
    v = p[np.arange(len(p)), uv_axes[:, 1]]
    tex = _texture(scene.textures[st], u, v, scene.seeds[st])
    base = scene.colors[st].astype(np.float64)
    room = hits.room[m]
    if (room < 0).any():
        dist = np.linalg.norm(p[room < 0, None, :] - scene.lights[None], axis=2)
        room = room.copy()
        room[room < 0] = dist.argmin(1)
    light = scene.lights[room]
    l_vec = light - p
    dist = np.linalg.norm(l_vec, axis=1)
    ndl = np.clip((nrm * l_vec).sum(1) / np.maximum(dist, 1e-6), 0.0, 1.0)
    intensity = 0.46 + 0.75 * ndl / (1.0 + 0.08 * dist ** 2)
    out[m] = base * (tex * intensity)[:, None]
    return out
