"""Mobs from AI-made 3D models (2026-10-26): a GLB (the user's ComfyUI multiview workflow, Pixal3D +
TRELLIS.2, decimated to a few thousand triangles) turned into a Bedrock geometry of polygon meshes
(`poly_mesh`, marked experimental in Bedrock's format docs; the in-game test decides) under the vanilla
id it replaces, cut into the bones the game's animations move (head, body, leg0-3), with the GLB's
base colour as the texture.

Placement: glTF is y up; Pixal3D's animal faces +z. Bedrock mobs face -z and the game draws model x
mirrored (render.py), so a point goes to (x, y, -z): the same animal, its head to -z, not mirrored.
"""
from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from mcreskin import imaging as I

ROOT = Path(__file__).resolve().parents[2]


def load_glb(path: str | Path) -> dict:
    """positions (n, 3), normals (n, 3), uvs (n, 2; glTF: v down from the top), tris (m, 3) and the
    material's images (name -> RGB(A) uint8 array) of a GLB's first mesh primitive."""
    b = Path(path).read_bytes()
    jl = struct.unpack("<I", b[12:16])[0]
    js = json.loads(b[20:20 + jl])
    off = 20 + jl
    binc = b[off + 8: off + 8 + struct.unpack("<I", b[off:off + 4])[0]]

    def acc(i):
        a = js["accessors"][i]
        bv = js["bufferViews"][a["bufferView"]]
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[a["componentType"]]
        arr = np.frombuffer(binc, dt, a["count"] * n, bv.get("byteOffset", 0) + a.get("byteOffset", 0))
        return arr.reshape(a["count"], n) if n > 1 else arr

    def image(tex_index):
        img = js["images"][js["textures"][tex_index]["source"]]
        bv = js["bufferViews"][img["bufferView"]]
        data = binc[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]]
        return np.asarray(Image.open(io.BytesIO(data)))

    pr = js["meshes"][0]["primitives"][0]
    mat = js["materials"][pr.get("material", 0)]
    out = {"positions": acc(pr["attributes"]["POSITION"]).astype(np.float64),
           "normals": acc(pr["attributes"]["NORMAL"]).astype(np.float64),
           "uvs": acc(pr["attributes"]["TEXCOORD_0"]).astype(np.float64),
           "tris": acc(pr["indices"]).reshape(-1, 3).astype(np.int64), "images": {}}
    pbr = mat.get("pbrMetallicRoughness", {})
    for name, ref in (("color", pbr.get("baseColorTexture")), ("metal_rough", pbr.get("metallicRoughnessTexture")),
                      ("normal", mat.get("normalTexture"))):
        if ref:
            out["images"][name] = image(ref["index"])
    return out


def place(m: dict, height: float, forward=(0.0, 0.0, 1.0), align: bool = True, tip: float = 0.0, span: float | None = None,
          stand: bool = False) -> dict:
    """The mesh in Bedrock model units: facing -z, feet at y 0, centred on x and z, `height` tall.
    `forward` is roughly where the animal faces in the GLB (Pixal3D multiview: +z; its single-image
    workflow lays the cow along +x, turned ~10 degrees): the body's long axis (the main direction of
    its points seen from above) is turned onto that way, so legs and head split cleanly. `align`
    off: `forward` itself is turned onto it, for a body with no long axis (the upright creeper). `tip`:
    degrees to turn it about x first, for a body pictured from above (the phantom). `span`: scale it to
    this length or wingspan instead of `height`, for a mob that isn't measured by its height (a fish).
    `stand`: first turn the mesh's long axis straight up (a squid the 3D step left leaning)."""
    pos, nrm = m["positions"], m["normals"]
    if stand:
        cen = pos - pos.mean(0)
        v = np.linalg.eigh(cen.T @ cen)[1][:, -1]
        v = v if v[1] > 0 else -v
        up = np.array([0.0, 1.0, 0.0])
        w, cos = np.cross(v, up), float(v @ up)  # Rodrigues: v onto up
        k = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        turn = np.eye(3) + k + k @ k / (1 + cos)
        pos, nrm = pos @ turn.T, nrm @ turn.T
    if tip:
        c, s = np.cos(np.radians(tip)), np.sin(np.radians(tip))
        turn = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        pos, nrm = pos @ turn.T, nrm @ turn.T
    xz = pos[:, [0, 2]] - pos[:, [0, 2]].mean(0)
    axis = np.linalg.eigh(xz.T @ xz)[1][:, -1] if align else np.asarray(forward, float)[[0, 2]]
    if axis @ np.asarray(forward, float)[[0, 2]] < 0:
        axis = -axis
    a = np.arctan2(axis[0], axis[1])  # turn by -a about y: the long axis onto +z
    rot = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    p = (pos @ rot.T) * np.array([1.0, 1.0, -1.0])
    n = (nrm @ rot.T) * np.array([1.0, 1.0, -1.0])
    return fit_size({**m, "positions": p, "normals": n}, height, span)


def fit_size(m: dict, height: float, span: float | None = None) -> dict:
    """Scaled to `height` (or `span`: its longest side seen from above), feet at y 0, centred on x and z."""
    p = m["positions"]
    q = p[np.unique(m["tris"])] if "tris" in m else p
    s = span / max(np.ptp(q[:, 0]), np.ptp(q[:, 2])) if span else height / np.ptp(q[:, 1])
    return {**m, "positions": (p - np.array([(q[:, 0].min() + q[:, 0].max()) / 2, q[:, 1].min(), (q[:, 2].min() + q[:, 2].max()) / 2])) * s}


def clipped(m: dict, height: float, span: float | None, z: tuple) -> dict:
    """The mesh without what lies outside fractions `z` of its length (0 at the snout), resized: thin
    feelers and tail bristles the 3D step makes into long sticks (the silverfish's took half its length)."""
    p, t = m["positions"], m["tris"]
    lo, hi = float(p[:, 2].min()), float(p[:, 2].max())
    c = p[t].mean(1)[:, 2]
    return fit_size({**m, "tris": t[(c >= lo + z[0] * (hi - lo)) & (c <= lo + z[1] * (hi - lo))]}, height, span)


KNEE = 0.45
"""Knee (front legs) or hock (hind legs) height as a share of the belly's: the lower leg is its own
bone, so a step folds the leg instead of swinging it as one stiff stick (2026-10-26: the legs "are
cropped at the bottom and doesnt look quite realistic")."""
BONES = ["body", "head", "leg0", "leg1", "leg2", "leg3", "leg4", "leg5", "leg0_lower", "leg1_lower", "leg2_lower", "leg3_lower"]


def parent_of(name: str) -> str:
    return name.removesuffix("_lower") if name.endswith("_lower") else "root"


def belly_height(p: np.ndarray) -> float:
    """How high the underside of the body is: the lowest point on the centre line, along the middle
    third of the animal's length (between its front and hind legs)."""
    half_w = np.ptp(p[:, 0]) / 2
    z0, z1 = np.quantile(p[:, 2], [0.4, 0.7])
    mid = (np.abs(p[:, 0]) < 0.15 * half_w) & (p[:, 2] > z0) & (p[:, 2] < z1)
    return float(np.median([p[mid & (p[:, 2] >= a) & (p[:, 2] < a + (z1 - z0) / 4), 1].min()
                            for a in np.linspace(z0, z1, 5)[:-1] if (mid & (p[:, 2] >= a) & (p[:, 2] < a + (z1 - z0) / 4)).any()]))


def cut_plane(m: dict, axis: int, value: float, where=None) -> dict:
    """`m` with every triangle that crosses the plane coordinate[`axis`] = `value` split along it
    (points, normals and UVs interpolated; triangles kept in order and winding), so parts cut there
    end in a clean loop instead of a zigzag of whole triangles (2026-10-27: the teeth of a cut by
    triangle centres stuck out when a head turned). `where(points)`: only triangles it accepts."""
    p, n, uv, t = m["positions"], m["normals"], m["uvs"], m["tris"]
    eps = 1e-6 * max(float(np.ptp(p)), 1.0)
    d = p[:, axis] - value
    d = np.where(np.abs(d) < eps, 0.0, d)
    P, N, U, made = [p], [n], [uv], {}
    count = [len(p)]

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in made:
            k = d[a] / (d[a] - d[b])
            nn = n[a] + k * (n[b] - n[a])
            P.append((p[a] + k * (p[b] - p[a]))[None])
            N.append((nn / max(np.linalg.norm(nn), 1e-9))[None])
            U.append((uv[a] + k * (uv[b] - uv[a]))[None])
            made[key] = count[0]
            count[0] += 1
        return made[key]

    out, src = [], []
    for k, tri in enumerate(t):
        ds = d[tri]
        if not (ds.min() < 0 < ds.max()) or (where is not None and not where(p[tri])):
            out.append(tri)
            src.append(k)
            continue
        for side in (1.0, -1.0):  # Sutherland-Hodgman: the part on each side, a fan
            poly = []
            for i in range(3):
                cur, nxt = tri[i], tri[(i + 1) % 3]
                if side * d[cur] >= 0:
                    poly.append(cur)
                if d[cur] * d[nxt] < 0:
                    poly.append(mid(cur, nxt))
            out += [[poly[0], poly[j], poly[j + 1]] for j in range(1, len(poly) - 1)]
            src += [k] * (len(poly) - 2)
    return {**m, "positions": np.concatenate(P), "normals": np.concatenate(N), "uvs": np.concatenate(U),
            "tris": np.asarray(out, dtype=t.dtype), "from_tri": np.asarray(src)}


def split_quadruped(m: dict, head_from: float = 0.22, collar: float = 0.12, knees: bool = True,
                    ears: bool = True, top: float | None = None, z_cut: float | None = None,
                    legs: int = 4) -> tuple[np.ndarray, dict, dict]:
    """Bone name per triangle and each bone's pivot, for a four-legged animal facing -z: below the
    belly, the four biggest pieces that reach the ground are legs (leg0 back right .. leg3 front
    left, as vanilla's quadrupeds; pivot at their top), anything else down there (udder, tail tip)
    stays with the body; the head (with the neck) is what lies in front of the front legs by more
    than `head_from` of their distance to the snout and above the neck's underside (the chest stays
    with the body), its pivot where it is cut. Bedrock moves each bone rigidly, so a turned head or
    a swinging leg opens its cut (2026-10-26: "it tears the model and I can see through it"): the
    third result gives each moving bone a sleeve, (direction into the body, length `collar` of its
    height): its own surface continued from the cut into the body, hidden at rest, filling the gap
    when it turns (copies of the body's skin turned with it instead stuck out as a fringe). `ears`:
    what sticks out sideways just behind the cut goes with the head (a cow's ears and horns); off
    for a neck as wide as the head (the polar bear's shoulder fur turned out like a flap, 2026-10-27).
    `top`, `z_cut`: where the legs and the head are cut, when already chosen (split_clean). `legs`:
    6 for the sniffer: the middle pair is leg4 (right) and leg5 (left)."""
    p, t = m["positions"], m["tris"]
    c = p[t].mean(1)
    h = np.ptp(p[:, 1])
    top = belly_height(p) - 0.02 * h if top is None else top
    low = c[:, 1] < top
    _, weld = np.unique(np.round(p, 3), axis=0, return_inverse=True)  # the unwrap splits points along UV seams
    weld = weld.ravel()
    tw = weld[t]
    parent = np.arange(weld.max() + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, d in tw[low]:
        for u, v in ((a, b), (b, d)):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    pieces: dict[int, list[int]] = {}
    for k in np.nonzero(low)[0]:
        pieces.setdefault(find(tw[k, 0]), []).append(k)
    half_w = np.ptp(p[:, 0]) / 2
    split = []
    for v in pieces.values():  # front legs that touch under the chest come as one piece
        v = np.array(v)
        if c[v, 0].min() < -0.1 * half_w and c[v, 0].max() > 0.1 * half_w:
            split += [v[c[v, 0] < -0.06 * half_w], v[c[v, 0] > 0.06 * half_w]]  # the middle stays body
        else:
            split.append(v)
    grounded = [v for v in split if len(v) and p[t[v]].reshape(-1, 3)[:, 1].min() < 0.06 * h]
    picked = sorted(grounded, key=len, reverse=True)[:legs]
    bone = np.array(["body"] * len(t), dtype=object)
    pivots = {}
    middle = []
    if legs == 6:  # the middle one on each side, by length along the body
        for side in (False, True):
            row = sorted((v for v in picked if bool(c[v, 0].mean() > 0) == side), key=lambda v: c[v, 2].mean())
            middle += [id(row[len(row) // 2])] if len(row) == 3 else []
    for v in picked:
        x, z = c[v, 0].mean(), c[v, 2].mean()
        name = {(False, False): "leg0", (True, False): "leg1", (False, True): "leg2", (True, True): "leg3"}[(bool(x > 0), bool(z < 0))]
        if id(v) in middle:
            name = "leg5" if x > 0 else "leg4"
        bone[v] = name
        near = c[v][c[v, 1] > top - 0.08 * h]
        near = near if len(near) else c[v]
        pivots[name] = [float(near[:, 0].mean()), float(top), float(near[:, 2].mean())]
    for name in [k for k in pivots if k.startswith("leg") and knees]:  # knee (front) or hock (hind): the lower leg bends
        knee = KNEE * top
        lower = (bone == name) & (c[:, 1] < knee)
        bone[lower] = f"{name}_lower"
        ring = c[(bone == f"{name}_lower") & (c[:, 1] > knee - 0.06 * h)]
        ring = ring if len(ring) else c[lower]
        pivots[f"{name}_lower"] = [float(ring[:, 0].mean()), float(knee), float(ring[:, 2].mean())]
    front = [pivots[k][2] for k in ("leg2", "leg3") if k in pivots]
    z_front = min(front) if front else float(np.quantile(p[:, 2], 0.3))
    z_tip = float(p[:, 2].min())
    z_cut = z_front - head_from * (z_front - z_tip) if z_cut is None else z_cut
    head = (c[:, 2] < z_cut) & (c[:, 1] > top + 0.18 * h)  # above the neck's underside: the chest stays
    bone[head] = "head"
    # ears (and horns) stick out sideways just behind a cut close to the skull: wider than the neck there
    neck = c[(c[:, 2] >= z_cut) & (c[:, 2] < z_cut + 0.04 * h) & (c[:, 1] > top + 0.18 * h)]
    if len(neck) and ears:
        half = np.abs(neck[:, 0]).max()
        y_low = np.quantile(neck[:, 1], 0.5)
        ears = (bone == "body") & (c[:, 2] >= z_cut) & (c[:, 2] < z_cut + 0.15 * h) & (np.abs(c[:, 0]) > 1.05 * half * (
            0.75 + 0.25 * (c[:, 2] - z_cut) / (0.15 * h))) & (c[:, 1] > y_low)
        bone[ears] = "head"
    ring = p[t[head]].reshape(-1, 3)
    ring = ring[ring[:, 2] > z_cut - 0.04 * h]
    pivots["head"] = [0.0, float(ring[:, 1].mean() if len(ring) else 0.6 * h), float(z_cut)]
    pivots["body"] = [0.0, float(0.75 * h), float((p[:, 2].min() + p[:, 2].max()) / 2)]
    sleeves = {name: ((0.0, 1.0, 0.0) if name.startswith("leg") else (0.0, 0.0, 1.0),
                      (0.4 if name.endswith("_lower") else 1.0) * collar * h)
               for name in pivots if name != "body"}
    return bone, pivots, sleeves


def _components(tris: np.ndarray, weld: np.ndarray) -> np.ndarray:
    """A label per triangle: its connected piece (through shared welded points)."""
    parent = np.arange(weld.max() + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in weld[tris]:
        for u, v in ((a, b), (b, c)):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    return np.array([find(w) for w in weld[tris[:, 0]]])


def split_clean(m: dict, head_from: float = 0.45, collar: float = 0.12, ears: bool = True, legs: int = 4) -> tuple[dict, np.ndarray, dict, dict]:
    """split_quadruped on a mesh cut cleanly where the parts meet (cut_plane): the legs at the plane
    just under the belly, the head at a plane across the neck at the skull. Each part then ends in a
    loop on its plane around its pivot (head: the loop's middle), where `dome` closes it. Bits the
    cut leaves on the body that touch only a moving part go with it (2026-10-27: the goat's horns
    sweep back past the cut and tore; ear tips, beards). `ears` is not used: the plane plus that rule
    replaces it. Returns the cut mesh too."""
    h = np.ptp(m["positions"][:, 1])
    top = belly_height(m["positions"]) - 0.02 * h
    m = cut_plane(m, 1, top)
    _, pivots, _ = split_quadruped(m, head_from=head_from, collar=collar, knees=False, ears=False, top=top, legs=legs)
    z_cut = pivots["head"][2]
    throat = top + 0.18 * h
    m = cut_plane(m, 2, z_cut, where=lambda q: q[:, 1].max() > throat)
    bone, pivots, extras = split_quadruped(m, head_from=head_from, collar=collar, knees=False, ears=False, top=top, z_cut=z_cut, legs=legs)
    p, t = m["positions"], m["tris"]
    _, weld = np.unique(np.round(p, 3), axis=0, return_inverse=True)
    weld = weld.ravel()
    body = np.nonzero(bone == "body")[0]
    label = _components(t[body], weld)
    pieces, sizes = np.unique(label, return_counts=True)
    torso = pieces[sizes.argmax()]
    for piece in pieces[pieces != torso]:
        tri = body[label == piece]
        pts = set(weld[t[tri]].ravel())
        best, most = None, 0
        for name in set(bone) - {"body"}:
            shared = len(pts & set(weld[t[bone == name]].ravel()))
            if shared > most:
                best, most = name, shared
        if best:
            bone[tri] = best
    ring = p[np.unique(t[bone == "head"])]
    ring = ring[np.abs(ring[:, 2] - z_cut) < 1e-4 * h]
    if len(ring):
        pivots["head"] = [float(ring[:, 0].mean()), float(ring[:, 1].mean()), float(z_cut)]
    return m, bone, pivots, extras


def frac_to_model(p: np.ndarray, axis: int, f: float) -> float:
    """A model coordinate from a fraction of the mesh's extent: x from -1 (its -x side) to 1, y from 0
    (the feet) to 1, z from 0 (the snout, -z) to 1."""
    lo, hi = float(p[:, axis].min()), float(p[:, axis].max())
    return (lo + hi) / 2 + f * (hi - lo) / 2 if axis == 0 else lo + f * (hi - lo)


def card(picture: np.ndarray, size: float, cells: int = 48, level: float = 0.5) -> dict:
    """A flat two-sided mesh in the outline of a cut-out picture (RGBA uint8), `size` units across its
    long side, in the x-z plane at y 0 with the picture's top toward -z (2026-10-31, for thin wings: the
    3D step makes them stiff lumps). Marching triangles over a grid of `cells` along the long side, so the
    outline is smooth at a few hundred triangles; the texture is the picture itself."""
    alpha = picture[..., 3].astype(np.float64) / 255
    h, w = alpha.shape
    step = max(w, h) / cells
    xs, ys = np.arange(0, w + step, step), np.arange(0, h + step, step)
    grid = alpha[np.clip(ys.astype(int), 0, h - 1)][:, np.clip(xs.astype(int), 0, w - 1)]
    verts: dict = {}
    tris = []

    def vid(x: float, y: float) -> int:
        return verts.setdefault((round(x, 3), round(y, 3)), len(verts))

    def emit(corners) -> None:  # one triangle of (x, y, alpha), clipped to alpha >= level
        out = []
        for i, a in enumerate(corners):
            b = corners[(i + 1) % 3]
            if a[2] >= level:
                out.append(a[:2])
            if (a[2] >= level) != (b[2] >= level):
                f = (level - a[2]) / (b[2] - a[2])
                out.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
        ids = [vid(*q) for q in out]
        tris.extend((ids[0], ids[k], ids[k + 1]) for k in range(1, len(ids) - 1))

    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            q = [(xs[c], ys[r], grid[r, c]), (xs[c + 1], ys[r], grid[r, c + 1]), (xs[c + 1], ys[r + 1], grid[r + 1, c + 1]), (xs[c], ys[r + 1], grid[r + 1, c])]
            emit([q[0], q[2], q[1]])
            emit([q[0], q[3], q[2]])
    xy = np.array(list(verts), float).reshape(-1, 2)
    t = np.array(tris, np.int64).reshape(-1, 3)
    used = np.unique(t)
    k = size / max(np.ptp(xy[used, 0]), np.ptp(xy[used, 1]))
    pos = np.stack([(xy[:, 0] - xy[used, 0].mean()) * k, np.zeros(len(xy)), (xy[:, 1] - xy[used, 1].mean()) * k], 1)
    n = len(pos)
    gap = np.array([0.0, 0.002 * size, 0.0])  # (a hair apart: the same plane twice flickers where a material draws both faces)
    return {"positions": np.concatenate([pos + gap, pos - gap]), "normals": np.concatenate([np.tile([0.0, 1.0, 0.0], (n, 1)), np.tile([0.0, -1.0, 0.0], (n, 1))]),
            "uvs": np.tile(xy / [w, h], (2, 1)), "tris": np.concatenate([t, t[:, ::-1] + n]), "images": {"color": picture[..., :3]}}


def rod(picture: np.ndarray, height: float, radius: float, sides: int = 8, rows: int = 6) -> dict:
    """A plain round rod with domed ends, at y 0..height on the y axis, its texture the brightest upright strip of `picture`
    (RGB uint8, a rod on black; 2026-10-31: thin things through the 3D step come back as sticks with fins)."""
    lum = picture.astype(np.float64).mean(-1)
    cols = np.nonzero(lum.mean(0) > 0.5 * lum.mean(0).max())[0]
    lines = np.nonzero(lum[:, cols].mean(1) > 0.5 * lum[:, cols].mean(1).max())[0]
    strip = picture[lines.min():lines.max() + 1, cols.min():cols.max() + 1]
    prof = [(0.0, 0.0)] + [(radius * r, height * y) for r, y in ((0.8, 0.0), (1.0, 0.06))] + [(radius, height * (0.06 + 0.88 * i / rows)) for i in range(1, rows + 1)] + [(radius * 0.8, height), (0.0, height)]
    pos, nrm, uv, tris = [], [], [], []
    for i, (r, y) in enumerate(prof):
        for k in range(sides + 1):
            a = 2 * np.pi * k / sides
            pos.append((r * np.cos(a), y, r * np.sin(a)))
            n = np.array([np.cos(a), 0.0, np.sin(a)]) if 0 < i < len(prof) - 1 else np.array([0.0, 1.0 if i else -1.0, 0.0])
            nrm.append(n)
            uv.append((abs(k / sides - 0.5) * 2 * 0.98 + 0.01, 1.0 - y / height * 0.98 - 0.01))
    w = sides + 1
    for i in range(len(prof) - 1):
        for k in range(sides):
            a, b, c, d = i * w + k, i * w + k + 1, (i + 1) * w + k + 1, (i + 1) * w + k
            tris += [(a, c, b), (a, d, c)]
    return {"positions": np.array(pos, float), "normals": np.array(nrm, float), "uvs": np.array(uv, float), "tris": np.array(tris, np.int64), "images": {"color": strip}}


def attach(m: dict, bone: np.ndarray, piece: dict, places: list, pivots: dict, parents: dict, share: float = 0.3) -> tuple[dict, np.ndarray]:
    """Mesh `piece` (a card, or a GLB of its own) added to `m` as bones of their own, its texture sharing
    m's (`share` of the width): a part the 3D step can't make in one go with the rest, like a wing, or one
    used many times, like a rod. `places`: (name, parent, pivot, transform, keep), transform a 4x4 matrix
    on the piece's points (a mirror flips its winding), keep an optional mask of the piece's triangles."""
    color, other = m["images"]["color"][..., :3], piece["images"]["color"][..., :3]
    H, W = color.shape[:2]
    left = round(W * (1 - share))
    atlas = np.zeros((H, W, 3), np.uint8)
    atlas[:, :left] = np.asarray(Image.fromarray(color).resize((left, H), Image.LANCZOS))
    atlas[:, left:] = np.asarray(Image.fromarray(other).resize((W - left, H), Image.LANCZOS))
    pos, nrm, uv, tris = [m["positions"]], [m["normals"]], [m["uvs"] * [left / W, 1.0]], [m["tris"]]
    names = [np.array(bone, dtype=object)]
    n = len(m["positions"])
    puv = piece["uvs"] * [(W - left) / W, 1.0] + [left / W, 0.0]
    for name, parent, pivot, matrix, *keep in places:
        M = np.asarray(matrix, float)
        rot = M[:3, :3]
        pos.append(piece["positions"] @ rot.T + M[:3, 3])
        normal = piece["normals"] @ np.linalg.inv(rot)
        nrm.append(normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9))
        uv.append(puv)
        t = piece["tris"][keep[0]] if keep and keep[0] is not None else piece["tris"]
        t = t[:, ::-1] if np.linalg.det(rot) < 0 else t
        tris.append(t + n)
        names.append(np.array([name] * len(t), dtype=object))
        n += len(piece["positions"])
        pivots[name] = [float(v) for v in pivot]
        parents[name] = parent
    out = {**m, "positions": np.concatenate(pos), "normals": np.concatenate(nrm), "uvs": np.concatenate(uv), "tris": np.concatenate(tris), "images": {**m["images"], "color": atlas}}
    return out, np.concatenate(names)


def carve(m: dict, bone: np.ndarray, pivots: dict, extras: dict, parents: dict, name: str, axis: str, at: float, side: str,
          box: dict | None = None, of: str = "body", parent: str | None = None, ranges=None) -> tuple[dict, np.ndarray]:
    """One more moving part (2026-10-30, for bodies that aren't split_clean's four legs and a head): what
    lies past a plane across `axis` ("x", "y", "z") at fraction `at` (frac_to_model), on `side` "+" or
    "-", and inside `box` ({"x": (lo, hi), ...} in the same fractions), taken from bone `of`, as bone
    `name`. The mesh is cut along the plane, so it ends in a clean loop that a ball joint closes;
    `ranges`: the fractions as measured on another mesh's extent (keep one frame across several carves)."""
    a = "xyz".index(axis)
    src = ranges if ranges is not None else m["positions"]
    value = frac_to_model(src, a, at)
    limits = {("xyz".index(k)): (frac_to_model(src, "xyz".index(k), v[0]), frac_to_model(src, "xyz".index(k), v[1])) for k, v in (box or {}).items()}
    h = float(np.ptp(src[:, 1]))

    def in_box(q):
        return all(q[:, i].max() >= lo and q[:, i].min() <= hi for i, (lo, hi) in limits.items())

    m2 = cut_plane(m, a, value, where=in_box)
    bone2 = bone[m2["from_tri"]] if "from_tri" in m2 else bone
    bone2 = np.array(bone2, dtype=object)
    p, t = m2["positions"], m2["tris"]
    c = p[t].mean(1)
    sign = 1.0 if side == "+" else -1.0
    take = (bone2 == of) & (sign * (c[:, a] - value) > 0)
    for i, (lo, hi) in limits.items():
        take &= (c[:, i] >= lo) & (c[:, i] <= hi)
    bone2[take] = name
    tol = 2e-3 * max(h, 1.0)
    used = np.unique(t[take]) if take.any() else np.zeros(0, int)
    ring = p[used]
    ring = ring[np.abs(ring[:, a] - value) < tol] if len(ring) else ring
    centre = ring.mean(0) if len(ring) else (p[used].mean(0) if len(used) else np.zeros(3))
    centre[a] = value
    pivots[name] = [float(v) for v in centre]
    d = np.zeros(3)
    d[a] = -sign
    extras[name] = (tuple(d), 0.1 * h, 0.7)
    parents[name] = parent or (of if of != "body" else "root")
    return m2, bone2


def pieces_past(m: dict, bone: np.ndarray, pivots: dict, extras: dict, parents: dict, axis: str, at: float, side: str, count: int, names,
                box: dict | None = None, of: str = "body", order: int = 2, parent: str | None = None) -> tuple[dict, np.ndarray]:
    """Like carve for several parts at once: the `count` biggest separate pieces past the plane, each its
    own bone, named by `names` in order along coordinate `order` (the spider's legs, a squid's tentacles;
    2026-10-30). A piece that is really two legs that touch stays one; look at the cuts on the sheet."""
    a = "xyz".index(axis)
    value = frac_to_model(m["positions"], a, at)
    h = float(np.ptp(m["positions"][:, 1]))
    limits = {("xyz".index(k)): (frac_to_model(m["positions"], "xyz".index(k), v[0]), frac_to_model(m["positions"], "xyz".index(k), v[1])) for k, v in (box or {}).items()}
    m2 = cut_plane(m, a, value, where=lambda q: all(q[:, i].max() >= lo and q[:, i].min() <= hi for i, (lo, hi) in limits.items()))
    bone2 = np.array(bone[m2["from_tri"]], dtype=object)
    p, t = m2["positions"], m2["tris"]
    c = p[t].mean(1)
    sign = 1.0 if side == "+" else -1.0
    far = (bone2 == of) & (sign * (c[:, a] - value) > 0)
    for i, (lo, hi) in limits.items():
        far &= (c[:, i] >= lo) & (c[:, i] <= hi)
    _, weld = np.unique(np.round(p, 3), axis=0, return_inverse=True)
    weld = weld.ravel()
    idx = np.nonzero(far)[0]
    if not len(idx):
        return m2, bone2
    label = _components(t[idx], weld)
    pieces = sorted((idx[label == lab] for lab in np.unique(label)), key=len, reverse=True)[:count]
    pieces = sorted(pieces, key=lambda v: c[v, order].mean())
    for i, v in enumerate(pieces):
        name = names(c[v].mean(0)) if callable(names) else names[i]
        bone2[v] = name
        used = np.unique(t[v])
        ring = p[used][np.abs(p[used][:, a] - value) < 2e-3 * max(h, 1.0)]
        centre = ring.mean(0) if len(ring) else p[used].mean(0)
        centre[a] = value
        pivots[name] = [float(x) for x in centre]
        d = np.zeros(3)
        d[a] = -sign
        extras[name] = (tuple(d), 0.1 * h, 0.7)
        parents[name] = parent or "root"
    return m2, bone2


def dome(m: dict, bone: np.ndarray, name: str, against: str, pivot, direction, steps: int = 3,
         depth: float = 0.5, cup: bool = False) -> np.ndarray:
    """Triangles (k, 3, 8 as in sleeve) closing bone `name`'s cut loop (its edges with `against` on
    the plane through `pivot` across `direction`) with a dome bulging `direction`-wards, into the
    body: a ball joint (2026-10-27). Turning about the pivot keeps a dome on itself, so the gap a
    turn opens shows a rounded surface with the loop's texels, not a hole, a flat cap or the teeth
    of a cut. `depth`: how far it bulges, in loop radii (a full half ball, 1, poked out of the
    goat's thin neck behind its skull). `cup`: the body's side instead, deeper than the part's dome
    so it stays behind it, wound like the body's own surface (it shows only through the opening)."""
    p, uv = m["positions"], m["uvs"]
    dv = np.asarray(direction, float)
    dv = dv / np.linalg.norm(dv)
    piv = np.asarray(pivot, float)
    tol = 1e-3 * max(float(np.ptp(p[:, 1])), 1.0)
    edges = [(a, b) for a, b in seam(m, bone, name, against)
             if abs((p[a] - piv) @ dv) < tol and abs((p[b] - piv) @ dv) < tol]
    if not edges:
        return np.zeros((0, 3, 8))
    _, weld = np.unique(np.round(p, 3), axis=0, return_inverse=True)  # the main loop only (not a horn's)
    weld = weld.ravel()
    label = _components(np.array([[a, b, b] for a, b in edges]), weld)
    keep = np.unique(label, return_counts=True)
    edges = [e for e, lab in zip(edges, label) if lab == keep[0][keep[1].argmax()]]
    flat = lambda q: (q - piv) - ((q - piv) @ dv) * dv
    rho = float(np.mean([np.linalg.norm(flat(p[i])) for e in edges for i in e]))

    def corner(i, j):
        a = np.pi / 2 * j / steps
        o = flat(p[i])
        r = np.linalg.norm(o)
        q = piv + o * np.cos(a) + dv * depth * (r + (rho - r) * np.sin(a)) * np.sin(a)
        nrm = m["normals"][i] if cup else q - piv  # a cup lit like the fur around it: facing in, it was a dark slash
        nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
        return np.concatenate([q, nrm, uv[i]])

    rows = []
    for a, b in edges:
        for j in range(steps):
            A0, B0, A1, B1 = corner(a, j), corner(b, j), corner(a, j + 1), corner(b, j + 1)
            last = j + 1 == steps  # the pole: A1 and B1 meet
            if cup:
                rows.append([A0, B0, B1])
                if not last:
                    rows.append([A0, B1, A1])
            else:
                rows.append([B0, A0, A1])
                if not last:
                    rows.append([B0, A1, B1])
    return np.array(rows)


def seam(m: dict, bone: np.ndarray, name: str, against: str) -> list[tuple[int, int]]:
    """The edges bone `name` shares with bone `against` (on welded points: the unwrap splits points
    along UV seams), directed as in `name`'s own triangles."""
    p, t = m["positions"], m["tris"]
    _, weld = np.unique(np.round(p, 3), axis=0, return_inverse=True)
    weld = weld.ravel()
    owner: dict = {}
    for k, tri in enumerate(t):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            owner.setdefault(frozenset((weld[a], weld[b])), set()).add(bone[k])
    edges = []
    for k in np.nonzero(bone == name)[0]:
        tri = t[k]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if against in owner[frozenset((weld[a], weld[b]))]:
                edges.append((a, b))
    return edges


def sleeve(m: dict, bone: np.ndarray, name: str, direction, length: float, shrink: float = 0.85,
           against: str = "body", cap: bool = False) -> np.ndarray:
    """Triangles (k, 3, corners as (position, normal, uv) rows: k x 3 x 8) continuing bone `name`'s
    surface from its cut edge `length` units along `direction`, drawn in toward its middle by
    `shrink` so it stays inside the body: each edge the bone shares with bone `against` (what it
    turns against: the body, or the upper leg for a lower leg) becomes a quad, wound like the bone's
    own triangle across that edge, with the edge's texels. `cap` closes its far end, so the part is
    a closed piece whose top turns inside the body."""
    p, n, uv = m["positions"], m["normals"], m["uvs"]
    edge_pts = seam(m, bone, name, against)
    if not edge_pts:
        return np.zeros((0, 3, 8))
    d = np.asarray(direction, float) * length
    ring = np.array([p[i] for e in edge_pts for i in e])
    mid = ring.mean(0)

    def corner(i, moved):
        q = p[i] + d if moved else p[i]
        if moved:
            off = q - (mid + d)
            off -= np.dot(off, d / np.linalg.norm(d)) * d / np.linalg.norm(d)
            q = q - (1 - shrink) * off
        return np.concatenate([q, n[i], uv[i]])

    rows = []
    for a, b in edge_pts:  # quad b, a, a', b' as two triangles
        A, B, A2, B2 = corner(a, False), corner(b, False), corner(a, True), corner(b, True)
        rows += [[B, A, A2], [B, A2, B2]]
        if cap:  # the far end: a fan to the end's middle, facing on along `direction`
            C = np.concatenate([mid + d, np.asarray(direction, float), uv[a]])
            rows.append([B2, A2, C])
    return np.array(rows)


def hole_cap(m: dict, bone: np.ndarray, name: str, against: str, facing) -> np.ndarray:
    """Triangles closing the hole bone `name` leaves in bone `against` when cut off: a fan from the
    seam to its middle, wound like `against`'s own surface and facing `facing` (out of the body), with
    the seam's texels (it shows only when the part turns)."""
    p, uv = m["positions"], m["uvs"]
    edges = seam(m, bone, name, against)
    if not edges:
        return np.zeros((0, 3, 8))
    mid = np.array([p[i] for e in edges for i in e]).mean(0)
    f = np.asarray(facing, float)
    row = lambda q, i: np.concatenate([q, f, uv[i]])
    return np.array([[row(p[a], a), row(p[b], b), row(mid, a)] for a, b in edges])


def geometry(m: dict, bone: np.ndarray, pivots: dict, identifier: str, tex_size: tuple[int, int],
             extras: dict | None = None, parents: dict | None = None, seal: bool = False, balls: bool = False) -> dict:
    """The Bedrock geometry JSON: a root bone and one bone per part, each with a poly_mesh of its
    triangles (as quads repeating the last corner) and its sleeve (`extras`: name -> (direction,
    length), see sleeve) in model space.
    Bedrock's normalised UVs count v up from the bottom (glTF's count down: as written, the cow
    came out a patchwork in game, 2026-10-26)."""
    p, n, uv, t = m["positions"], m["normals"], m["uvs"], m["tris"]
    uv = np.stack([uv[:, 0], 1.0 - uv[:, 1]], 1)
    extras = extras or {}
    parents = parents or {name: parent_of(name) for name in BONES}
    order, todo = [], ["root"]
    while todo:  # parents before children
        cur = todo.pop(0)
        kids = sorted(k for k, v in parents.items() if v == cur)
        order += kids
        todo += kids
    bones = [{"name": "root", "pivot": [0, 0, 0]}]
    for name in order:
        sel = t[bone == name]
        if not len(sel) and not any(v == name for v in parents.values() if v != name) and name not in pivots:
            continue  # no triangles and nothing hanging from it (no knees)
        if not len(sel):  # an empty link still carries its children's turn
            bones.append({"name": name, "parent": parents[name], "pivot": [round(v, 3) for v in pivots.get(name, [0, 0, 0])]})
            continue
        used, local = np.unique(sel, return_inverse=True)
        local = local.reshape(-1, 3)
        P, N, U = p[used], n[used], uv[used]
        add = []
        against = lambda k: parents[k] if parents.get(k, "root") in extras else "body"  # what a part turns against
        if name in extras and balls:  # a ball joint: its dome into the body
            add.append(dome({**m, "uvs": uv}, bone, name, against(name), pivots[name], extras[name][0]))
        elif name in extras:  # its sleeve into the body
            add.append(sleeve({**m, "uvs": uv}, bone, name, *extras[name],  # what it turns against: the body
                              against="body" if parents[name] == "root" else parents[name], cap=seal))  # under the root
        if seal and (name == "body" or name in extras):  # close the holes the moving parts leave
            for child, (direction, *_) in extras.items():
                if balls and against(child) == name:
                    add.append(dome({**m, "uvs": uv}, bone, child, name, pivots[child], direction, steps=2, depth=0.6, cup=True))
                elif not balls and name == "body":
                    add.append(hole_cap({**m, "uvs": uv}, bone, child, "body", -np.asarray(direction, float)))
        for sl in add:
            if len(sl):
                base = len(P)
                P = np.concatenate([P, sl[:, :, :3].reshape(-1, 3)])
                N = np.concatenate([N, sl[:, :, 3:6].reshape(-1, 3)])
                U = np.concatenate([U, sl[:, :, 6:8].reshape(-1, 2)])
                local = np.concatenate([local, base + np.arange(len(sl) * 3).reshape(-1, 3)])
        polys = [[[int(i), int(i), int(i)] for i in (a, b, c, c)] for a, b, c in local]
        bones.append({"name": name, "parent": parents[name], "pivot": [round(v, 3) for v in pivots.get(name, [0, 0, 0])],
                      "poly_mesh": {"normalized_uvs": True,
                                    "positions": np.round(P, 4).tolist(),
                                    "normals": np.round(N, 4).tolist(),
                                    "uvs": np.round(U, 5).tolist(),
                                    "polys": polys}})
    lo, hi = p.min(0), p.max(0)
    span = float(max(hi[0] - lo[0], hi[2] - lo[2])) / 16 + 1
    return {"format_version": "1.16.0", "minecraft:geometry": [{
        "description": {"identifier": identifier, "texture_width": tex_size[0], "texture_height": tex_size[1],
                        "visible_bounds_width": round(span, 2), "visible_bounds_height": round(float(hi[1]) / 16 + 1, 2),
                        "visible_bounds_offset": [0, round(float(hi[1]) / 32, 2), 0]},
        "bones": bones}]}


def make(glb: str | Path, identifier: str, texture: str, out_dir: Path, height: float = 23.0,
         forward=(0.0, 0.0, 1.0), jointed: bool = True, head_from: float | None = None, ears: bool = True,
         legs: int = 4, align: bool = True, plan=None, tip: float = 0.0, span: float | None = None, stand: bool = False, clip: tuple | None = None, log=print) -> tuple[Path, Path]:
    """Write <out_dir>/<identifier without "geometry.">.geo.json and <texture stem>.png from a GLB
    whose animal faces roughly `forward`. `jointed` (2026-10-26, "a model built for jointed
    animation ... that still fits the vanilla minecraft animations"): built like vanilla's own mobs
    for the game's walk and look-at, no knees: the body's holes sealed, each leg and the head (cut
    just behind the skull, the long neck staying with the body) a closed piece whose tapered top
    reaches a little into the body, so a swing turns it inside the body instead of opening it. Without it: knee bones and short open sleeves, for our own walk
    (animation_file, entity_file). `head_from`: where the head is cut (see split_quadruped); further
    forward for a short neck; `ears`, `legs` see split_quadruped; `align` see place. `plan`: for a body
    split_clean doesn't fit, a function (m, bone, pivots, extras, parents) -> (m, bone) that carves its
    parts out of one "body"; "quadruped+": a function run after split_clean, as ("+", function)."""
    m = place(load_glb(glb), height, forward, align, tip, span, stand)
    if clip:
        m = clipped(m, height, span, clip)
    head_from = head_from if head_from is not None else (0.45 if jointed else 0.22)
    parents = None
    if plan is not None and not isinstance(plan, tuple):
        bone, pivots, extras, parents = np.array(["body"] * len(m["tris"]), dtype=object), {}, {}, {}
        m, bone = plan(m, bone, pivots, extras, parents)
        pivots.setdefault("body", [0.0, float(0.6 * np.ptp(m["positions"][:, 1])), 0.0])
        parents = {"body": "root", **parents}
    elif jointed:  # clean cuts closed by ball joints (2026-10-27)
        m, bone, pivots, extras = split_clean(m, head_from=head_from, collar=0.12, ears=ears, legs=legs)
    else:
        bone, pivots, extras = split_quadruped(m, head_from=head_from, collar=0.12, knees=True, ears=ears)
    if jointed and parents is None:  # short, tapered ends: long ones swung out of the chest and rump (head turned 60 degrees)
        h = np.ptp(m["positions"][:, 1])
        extras = {k: (d, (0.10 if k == "head" else 0.15) * h, 0.7) for k, (d, _) in extras.items()}
        if isinstance(plan, tuple):  # more parts after the legs and head
            parents = {k: "root" for k in pivots}
            m, bone = plan[1](m, np.array(bone, dtype=object), pivots, extras, parents)
    color = m["images"]["color"]
    geo = geometry(m, bone, pivots, identifier, (color.shape[1], color.shape[0]), extras, parents=parents, seal=jointed, balls=jointed)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpath = out_dir / f"{identifier.removeprefix('geometry.')}.geo.json"
    gpath.write_text(json.dumps(geo, separators=(",", ":")))
    tpath = out_dir / f"{texture.rsplit('/', 1)[-1]}.png"
    rgba = np.concatenate([color[..., :3].astype(np.float32) / 255, np.ones(color.shape[:2] + (1,), np.float32)], -1)
    I.save_png(rgba, tpath)
    counts = {k: int((bone == k).sum()) for k in sorted(set(bone))}
    log(f"{gpath} ({len(m['tris'])} triangles: {counts}), pivots {pivots}; texture {tpath}")
    return gpath, tpath


HIP_SWING, KNEE_SWING, KNEE_FOLD = 10.0, 25.0, 40.0
"""Degrees of our walk. The hip, where the leg's wide top meets the body, turns little: a 30 degree
swing there opened the body ("the polygons break apart", 2026-10-26; spreading the bend over slices
of skin only cracked it in more places). The lower leg carries the stride from the narrow knee:
swinging with the hip and folding as the leg steps forward."""
WALK_GAIN = 2.5
"""The game weighs a walk by query.modified_move_speed, only ~0.3-0.5 for a strolling cow: ours is
weighed by that times WALK_GAIN (at most 1), so it is full at walking pace. Weighed by the speed
alone, 20 degrees came out at 6-10 and "the cow legs almost dont even move" (2026-10-26)."""


def walk_angles(t: float, speed: float = 1.0) -> dict:
    """Bone -> [x, 0, 0] of our walk at anim time `t` (distance moved), as the animation file computes
    it: diagonal pairs in step like vanilla's, the lower leg swinging from the knee and folding while
    its leg steps forward (front knees fold back, hocks the other way)."""
    c, s = np.cos(np.radians(t * 38.17)), np.sin(np.radians(t * 38.17))
    out = {}
    for leg, sign in (("leg0", 1), ("leg1", -1), ("leg2", -1), ("leg3", 1)):
        fold = max(0.0, sign * s) * KNEE_FOLD
        out[leg] = [sign * c * HIP_SWING * speed, 0, 0]
        out[f"{leg}_lower"] = [(sign * c * KNEE_SWING + (fold if leg in ("leg2", "leg3") else -0.7 * fold)) * speed, 0, 0]
    return out


def animation_file(prefix: str) -> dict:
    """animation.<prefix>.walk and .look_at in Bedrock's animation format (Molang, degrees)."""
    bones = {}
    for leg, sign in (("leg0", "1"), ("leg1", "-1"), ("leg2", "-1"), ("leg3", "1")):
        cos = f"math.cos(query.anim_time * 38.17)"
        fold = f"math.max(0, {sign} * math.sin(query.anim_time * 38.17)) * {KNEE_FOLD}"
        bones[leg] = {"rotation": [f"{sign} * {cos} * {HIP_SWING}", 0.0, 0.0]}
        bones[f"{leg}_lower"] = {"rotation": [
            f"{sign} * {cos} * {KNEE_SWING} + " + (fold if leg in ("leg2", "leg3") else f"-0.7 * {fold}"), 0.0, 0.0]}
    return {"format_version": "1.8.0", "animations": {
        f"animation.{prefix}.walk": {"loop": True, "anim_time_update": "query.modified_distance_moved", "bones": bones},
        f"animation.{prefix}.look_at": {"loop": True, "bones": {"head": {
            "relative_to": {"rotation": "entity"},
            "rotation": ["query.target_x_rotation * 0.6 - this", "query.target_y_rotation * 0.6 - this", 0.0]}}}}}


def entity_file(vanilla: dict, prefix: str, variant: str) -> dict:
    """Vanilla's client entity file with our walk and look-at for `variant` (a Molang condition, e.g.
    "v.index == 0 && !query.is_baby") and vanilla's for every other variant."""
    import copy

    out = copy.deepcopy(vanilla)
    desc = out["minecraft:client_entity"]["description"]
    desc["animations"]["mesh_walk"] = f"animation.{prefix}.walk"
    desc["animations"]["mesh_look_at"] = f"animation.{prefix}.look_at"
    animate = []
    for entry in desc["scripts"]["animate"]:
        name = entry if isinstance(entry, str) else next(iter(entry))
        cond = None if isinstance(entry, str) else entry[name]
        if name == "walk":
            animate += [{"walk": f"({variant}) ? 0.0 : ({cond})"},
                        {"mesh_walk": f"({variant}) ? math.min(1.0, ({cond}) * {WALK_GAIN}) : 0.0"}]
        elif name == "look_at_target":
            animate += [{"look_at_target": f"!({variant})"}, {"mesh_look_at": f"{variant}"}]
        else:
            animate.append(entry)
    desc["scripts"]["animate"] = animate
    return out


def fit_to_vanilla(geo: dict, vanilla: dict, rename: dict | None = None, links: bool = False) -> dict:
    """Our geometry made to answer the mob's own animations (2026-10-27): our leg0..leg3 take the
    names of `vanilla`'s (its vanilla geometry) leg bones, matched by quarter (left or right, front
    or back: the goat calls them right_back_leg..., the hoglin leg_back_right..., and the two
    disagree on which side is right), and each bone hangs from the parent vanilla gives it when we
    have one (the polar bear's head and legs from its body, so they rise with it when it rears up),
    and the body turns about vanilla's body pivot (kept inside ours): the bear's rears up about its
    hips, and about our body's middle its hindquarters went into the ground (2026-10-27). `rename`: names
    to use where the quarters don't settle it ({"leg4": "right_mid_leg"}), or for a part that isn't a
    leg (the horse's head and neck, cut as one, are its "Neck", which its animations turn)."""
    import copy

    out = copy.deepcopy(geo)
    bones = out["minecraft:geometry"][0]["bones"]
    names = dict(rename or {})
    ours = {b["name"]: b["pivot"] for b in bones if b["name"].startswith("leg") and "lower" not in b["name"]}
    theirs = {b["name"]: b["pivot"] for b in vanilla["bones"] if "leg" in b["name"].lower() and "pivot" in b}
    quarter = lambda piv: (piv[0] > 0, piv[2] < 0)
    for name, piv in ours.items():  # (quarters only settle four legs: a spider's eight would collide)
        match = [n for n, q in theirs.items() if quarter(q) == quarter(piv)]
        if len(match) == 1 and name not in names and len(theirs) == 4:
            names[name] = match[0]
    for b in bones:
        b["name"] = names.get(b["name"], b["name"])
        if "parent" in b:
            b["parent"] = names.get(b["parent"], b["parent"])
    if links:  # vanilla's empty bones between ours and their parents, which its animations also move
        vanilla_by = {b["name"].lower(): b for b in vanilla["bones"]}
        have = {b["name"].lower() for b in bones}
        todo = [b["name"].lower() for b in bones]
        while todo:
            up = str((vanilla_by.get(todo.pop()) or {}).get("parent")).lower()
            if up in vanilla_by and up not in have:
                v = vanilla_by[up]
                bones.insert(1, {"name": v["name"], "pivot": list(v.get("pivot") or [0, 0, 0]), "parent": "root", "link": True})
                have.add(up)
                todo.append(up)
    ours_by_lower = {b["name"].lower(): b["name"] for b in bones}  # the game matches bone names in any case
    vparent = {b["name"].lower(): str(b.get("parent")).lower() for b in vanilla["bones"]}  # (the horse's "Body")
    for b in bones:
        parent = vparent.get(b["name"].lower())
        if parent in ours_by_lower and parent != b["name"].lower():
            b["parent"] = ours_by_lower[parent]
    for b in bones:  # a link turns its children about where they are on ours, not on vanilla's smaller or larger model
        if b.pop("link", False):
            kids = [k["pivot"] for k in bones if k.get("parent") == b["name"]]
            b["pivot"] = [round(float(v), 4) for v in np.mean(kids, axis=0)] if kids else b["pivot"]
    vbody = next((b for b in vanilla["bones"] if b["name"].lower() == "body" and "pivot" in b), None)
    body = next((b for b in bones if b["name"] == "body" and b.get("poly_mesh")), None)
    if vbody and body:
        pts = np.array(body["poly_mesh"]["positions"], float)
        lo, hi = pts.min(0), pts.max(0)
        body["pivot"] = [0.0] + [round(float(np.clip(vbody["pivot"][i], lo[i], hi[i])), 3) for i in (1, 2)]
    return out


def shifted(geo: dict, offset) -> dict:
    """`geo` moved by `offset` (x, y, z), for a mob that doesn't stand on y 0 in vanilla's model (a
    squid hangs below its pivot, a fish lies around it)."""
    import copy

    out = copy.deepcopy(geo)
    d = np.asarray(offset, float)
    for b in out["minecraft:geometry"][0]["bones"]:
        if b["name"] == "root":
            continue
        b["pivot"] = [round(float(v), 4) for v in np.asarray(b.get("pivot", (0, 0, 0)), float) + d]
        if b.get("poly_mesh"):
            b["poly_mesh"]["positions"] = np.round(np.asarray(b["poly_mesh"]["positions"], float) + d, 4).tolist()
    return out


def keep_bones(geo: dict, vanilla: dict, names, scale=None) -> dict:
    """`geo` with vanilla's empty `names` added under the bones vanilla hangs them from, where the game
    looks for them (the vex's and allay's rightItem: what they hold), moved with their parent's offset
    between vanilla and ours. Call after fit_to_vanilla."""
    import copy

    out = copy.deepcopy(geo)
    bones = out["minecraft:geometry"][0]["bones"]
    ours = {b["name"].lower(): b for b in bones}
    van = {b["name"].lower(): b for b in vanilla["bones"]}
    for name in names:
        v = van.get(name.lower())
        if v is None or name.lower() in ours:
            continue
        up = ours.get(str(v.get("parent")).lower()) or next(b for b in bones if b["name"] != "root")
        vup = van.get(str(v.get("parent")).lower()) or {"pivot": up["pivot"]}
        d = np.asarray(v.get("pivot", (0, 0, 0)), float) - np.asarray(vup.get("pivot", (0, 0, 0)), float)
        bones.append({"name": v["name"], "parent": up["name"], "pivot": [round(float(x), 4) for x in np.asarray(up["pivot"], float) + d]})
    return out


def add_saddle(geo: dict, tex: np.ndarray, name: str = "Saddle", z: tuple = (0.34, 0.6), depth: float = 0.42, offset: float = 0.3,
               color=(0.30, 0.17, 0.09), trim=(0.62, 0.48, 0.22)) -> tuple[dict, np.ndarray]:
    """`geo` with a saddle that fits this animal's back and `tex` with its leather (2026-10-30): the
    body's own surface over the back and down the sides, raised a little, as bone `name` (which vanilla's
    render controller shows only when saddled). Vanilla's saddle is cubes made for its box horse, and an
    animal that turned back into boxes when saddled would be worse. `z`: where it sits along the body
    (fractions, 0 at the chest); `depth`: how far down the sides, in shares of the body's depth there.
    Its texels are painted into the largest part of the texture no triangle uses."""
    import copy

    out = copy.deepcopy(geo)
    g = out["minecraft:geometry"][0]
    body = next(b for b in g["bones"] if b["name"] == "body" and b.get("poly_mesh"))
    P = np.array(body["poly_mesh"]["positions"], float)
    N = np.array(body["poly_mesh"]["normals"], float)
    polys = body["poly_mesh"]["polys"]
    z0, z1 = np.quantile(P[:, 2], 0.0), np.quantile(P[:, 2], 1.0)
    za, zb = z0 + z[0] * (z1 - z0), z0 + z[1] * (z1 - z0)
    corners = np.array([[c[0] for c in poly[:3]] for poly in polys])
    cen = P[corners].mean(1)
    band = (cen[:, 2] >= za) & (cen[:, 2] <= zb)
    top, low = P[corners[band]][..., 1].max(), P[corners[band]][..., 1].min()
    keep = band & (cen[:, 1] > top - depth * (top - low))
    chosen = corners[keep]
    # the free square: where no triangle of any bone has texels
    h, w = tex.shape[:2]
    cells = 256
    used = np.zeros((cells, cells), bool)
    for b in g["bones"]:
        pm = b.get("poly_mesh")
        if pm:
            uv = np.clip(np.array(pm["uvs"], float), 0, 0.9999)
            for poly in pm["polys"]:
                q = uv[[c[2] for c in poly[:3]]]
                lo, hi = (q.min(0) * cells).astype(int), (q.max(0) * cells).astype(int)
                used[cells - 1 - hi[1]: cells - lo[1], lo[0]: hi[0] + 1] = True
    used[0, :] = used[-1, :] = True
    used[:, 0] = used[:, -1] = True
    dist = ndimage.distance_transform_cdt(~used, metric="chessboard")
    r, c = np.unravel_index(dist.argmax(), dist.shape)
    half = max(int(dist[r, c]) - 1, 1)
    x0, x1 = int((c - half) / cells * w), int((c + half) / cells * w)
    y0, y1 = int((r - half) / cells * h), int((r + half) / cells * h)
    out_tex = tex.copy()
    alpha = out_tex[y0:y1, x0:x1, 3:].copy() if out_tex.shape[2] > 3 else None
    noise = np.random.default_rng(7).normal(0, 0.02, (y1 - y0, x1 - x0, 1))
    patch = np.clip(np.array(color, np.float32) + noise, 0, 1)
    edge = max((x1 - x0) // 10, 1)
    patch[:edge] = patch[-edge:] = trim
    out_tex[y0:y1, x0:x1, :3] = patch
    if alpha is not None:
        out_tex[y0:y1, x0:x1, 3:] = alpha
    # the piece: its own points, raised along their normals, UVs laid flat into the patch
    idx, local = np.unique(chosen, return_inverse=True)
    local = local.reshape(-1, 3)
    Q = P[idx] + N[idx] * offset
    span = np.ptp(Q[:, [0, 2]], axis=0) + 1e-6
    lo = Q[:, [0, 2]].min(0)
    fx = (Q[:, 0] - lo[0]) / span[0]
    fz = (Q[:, 2] - lo[1]) / span[1]
    margin = 0.14
    u = (x0 + (x1 - x0) * (margin + (1 - 2 * margin) * fz)) / w
    v = 1 - (y0 + (y1 - y0) * (margin + (1 - 2 * margin) * fx)) / h  # v counts up
    g["bones"].append({"name": name, "parent": "body", "pivot": list(body["pivot"]), "poly_mesh": {
        "normalized_uvs": True, "positions": np.round(Q, 4).tolist(), "normals": np.round(N[idx], 4).tolist(),
        "uvs": np.round(np.stack([u, v], 1), 5).tolist(),
        "polys": [[[int(i), int(i), int(i)] for i in (a, b, c_, c_)] for a, b, c_ in local]}})
    return out, out_tex


def cancel_rest_moves(geo: dict, rest: dict) -> dict:
    """`geo` with each bone the game moves at rest (`rest`: render.setup_pose on it) hung from a new
    still "<bone>_anchor" bone that makes the move nothing. Vanilla's rest moves read "c - this",
    `this` being the bone's pivot less its parent's (its place in the Java model), so on vanilla's
    model they cancel; on ours, hung from a root at 0, the sheep's "-6 - this" put the head 28 px
    down and the polar bear's "-9 - this" its body 35 (2026-10-27). The anchor sits at the bone's
    pivot less c, so `this` becomes c."""
    import copy

    out = copy.deepcopy(geo)
    bones = out["minecraft:geometry"][0]["bones"]
    by = {b["name"]: b for b in bones}
    for b in list(bones):
        pos = (rest.get(b["name"].lower()) or {}).get("position")
        if not pos or max(abs(v) for v in pos) < 1e-6:
            continue
        parent = by.get(b.get("parent"))
        piv = np.array(b["pivot"], float)
        this = piv - (np.array(parent["pivot"], float) if parent else np.array([0.0, 24.0, 0.0]))
        c = np.array(pos, float) + this
        anchor = {"name": f"{b['name']}_anchor", "pivot": [round(float(v), 4) for v in piv - c]}
        if "parent" in b:
            anchor["parent"] = b["parent"]
        b["parent"] = anchor["name"]
        bones.insert(bones.index(b), anchor)
    return out


def bone_mask(geo: dict, size: tuple[int, int], bones: set[str], inside: float = 1.0, outside: float = 3 / 255) -> np.ndarray:
    """A texture-sized mask (rows, cols) from the model's own parts: `inside` where the UVs of
    `bones` lie, `outside` where the other parts' lie; texels no part covers take the nearest
    covered one's value, so mipmaps keep each island's value at its edges (islands side by side in
    the texture are rarely neighbours on the animal, so they aren't grown into each other). For the
    sheep's colour mask (2026-10-27): vanilla's sheep texture is opaque where the dye goes (wool)
    and alpha 3 on bare skin, so the body (fleece) is `inside`, the head and legs keep their colour."""
    from PIL import Image, ImageDraw
    from scipy import ndimage

    w, h = size
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for want, label in ((False, 2), (True, 1)):  # inside drawn last: it wins a shared edge
        for b in geo["minecraft:geometry"][0]["bones"]:
            mesh = b.get("poly_mesh")
            if not mesh or (b["name"] in bones) != want:
                continue
            uv = np.asarray(mesh["uvs"], float)
            for poly in mesh["polys"]:
                draw.polygon([(uv[c[2]][0] * w, (1 - uv[c[2]][1]) * h) for c in poly], fill=label)  # v counts up
    label = np.asarray(img)
    if label.any():
        near = ndimage.distance_transform_edt(label == 0, return_distances=False, return_indices=True)
        label = label[near[0], near[1]]
    return np.where(label == 2, outside, inside).astype(np.float32)


def position_map(geo: dict, size: tuple[int, int]) -> np.ndarray:
    """Each texel's point on the model (rows, cols, xyz), NaN where no triangle has it."""
    w, h = size
    out = np.full((h, w, 3), np.nan, np.float32)
    for b in geo["minecraft:geometry"][0]["bones"]:
        pm = b.get("poly_mesh")
        if not pm:
            continue
        P, U = np.asarray(pm["positions"], float), np.asarray(pm["uvs"], float)
        for poly in pm["polys"]:
            i = [v[0] for v in poly[:3]]
            x, y = U[i, 0] * w, (1 - U[i, 1]) * h
            x0, x1, y0, y1 = max(int(x.min()), 0), min(int(x.max()) + 2, w), max(int(y.min()), 0), min(int(y.max()) + 2, h)
            if x1 <= x0 or y1 <= y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            d = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            if abs(d) < 1e-9:
                continue
            a = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / d
            c = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / d
            e = 1 - a - c
            inside = (a >= -0.02) & (c >= -0.02) & (e >= -0.02)
            out[y0:y1, x0:x1][inside] = (a[..., None] * P[i[0]] + c[..., None] * P[i[1]] + e[..., None] * P[i[2]])[inside]
    return out


def texel_mask(geo: dict, size: tuple[int, int], where, bones=None) -> np.ndarray:
    """Where in the texture (rows, cols of bool) the triangles lie that `where(centre, normal)` picks, of `bones` or all."""
    from PIL import Image, ImageDraw

    w, h = size
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for b in geo["minecraft:geometry"][0]["bones"]:
        pm = b.get("poly_mesh")
        if not pm or (bones is not None and b["name"] not in bones):
            continue
        P, N, U = np.asarray(pm["positions"], float), np.asarray(pm["normals"], float), np.asarray(pm["uvs"], float)
        for poly in pm["polys"]:
            i = [v[0] for v in poly[:3]]
            if where(P[i].mean(0), N[i].mean(0)):
                draw.polygon([(U[k][0] * w, (1 - U[k][1]) * h) for k in i], fill=255)
    return np.asarray(img) > 0


def plain_side(geo: dict, tex: np.ndarray, where, bones=None, detail: float = 0.25, sigma: float = 40.0) -> np.ndarray:
    """`tex` with the features painted out where `where` picks (2026-10-31: the 3D step gave the ghast its
    sad face on the back too): the side's own median colour, with a little of its fine grain left."""
    from scipy import ndimage

    mask = texel_mask(geo, (tex.shape[1], tex.shape[0]), where, bones)
    if not mask.any():
        return tex
    out = tex.copy()
    rgb = tex[..., :3]
    blur = np.stack([ndimage.gaussian_filter(rgb[..., i], sigma) for i in range(3)], -1)
    flat = np.median(rgb[mask], axis=0) + detail * (rgb - blur)
    out[..., :3] = np.where(mask[..., None], np.clip(flat, 0, 1), rgb)
    return out


def variant_animation(anim: dict, variant: str, edit) -> dict:
    """A copy of a vanilla animation that plays edited for the variant drawn as our mesh and as
    vanilla for the rest (babies, other colours share the entity): each channel value e becomes
    "(variant) ? (edit(e, bone, channel, axis)) : (e)", keyframes and pre/post values included;
    `edit` returns None to leave a value as vanilla's. Vanilla's moves are made for boxes: a
    sheep's grazing and a goat's ram drop the head 9 and 8 px (on a real head, off the neck), a
    run swings legs 80 degrees, a look-at turns the head 75 (2026-10-27)."""
    import copy

    out = copy.deepcopy(anim)

    def wrap(vals, bone, ch):
        if isinstance(vals, list):
            res = []
            for i, e in enumerate(vals):
                new = edit(e, bone, ch, i)
                res.append(e if new is None else f"({variant}) ? ({new}) : ({e})")
            return res
        if isinstance(vals, dict):  # keyframes by time, or pre/post (a curve's lerp_mode goes: it needs constant keyframes)
            return {k: wrap(v, bone, ch) for k, v in vals.items() if k != "lerp_mode"}
        return vals

    for bone, chans in (out.get("bones") or {}).items():
        for ch in ("rotation", "position", "scale"):
            if ch in chans:
                vals = chans[ch] if isinstance(chans[ch], (list, dict)) else [chans[ch]] * 3  # (a scale can be one number)
                chans[ch] = wrap(vals, bone, ch)
    return out


def rest_pose(geo: dict, head: str = "head", kind: str = "sit", max_tilt: float = 40.0, upright: float | None = None) -> dict:
    """A four-legged mesh sitting or lying down, worked out from its own legs: bone -> {"rotation":
    [a, 0, 0], "position": [0, y, z]} (2026-10-30). Vanilla's sits place box parts by absolute numbers
    ("-18 - this"), which pull a realistic body apart (a wolf's head floated off, a cat's legs lay
    beside it). `sit`: the body tilts up about the hips until the rump nears the ground with the front
    legs still straight under the shoulders; the hind legs fold forward. `lie`: it sinks level, all legs
    folded forward. The head stays about level. Each bone's numbers are in its parent's frame, as the
    game reads them (legs hang from the body on some mobs, from the root on others). `upright`: sit at
    this tilt, front legs off the ground and held forward (a panda sits up)."""
    from . import render as R

    bones = {b["name"]: b for b in geo["minecraft:geometry"][0]["bones"]}
    body = bones["body"]
    pb = np.array(body["pivot"], float)
    legs = [b for b in bones.values() if "leg" in b["name"].lower() and b.get("poly_mesh")]
    mid = np.mean([b["pivot"][2] for b in legs])
    hind = [b for b in legs if b["pivot"][2] > mid]
    front = [b for b in legs if b["pivot"][2] <= mid]
    ph = np.mean([b["pivot"] for b in hind], axis=0)
    ps = np.mean([b["pivot"] for b in front], axis=0)
    top = float(ph[1])
    if kind == "sit":
        reach = float(ph[2] - ps[2])
        tilt = float(np.degrees(np.arcsin(np.clip(0.72 * top / max(reach, 1e-6), 0.0, np.sin(np.radians(max_tilt))))))
        drop = reach * np.sin(np.radians(tilt))
        if upright is not None:
            tilt, drop = upright, 0.72 * top
    else:
        tilt, drop = 0.0, 0.85 * top
    rb = R._rot((-tilt, 0, 0))
    tb = ph + np.array([0.0, -drop, 0.0]) - (rb @ (ph - pb) + pb)
    world = lambda q: rb @ (np.asarray(q, float) - pb) + pb + tb
    out = {"body": {"rotation": [round(-tilt, 3), 0.0, 0.0], "position": [round(float(v), 3) for v in tb]}}

    def place(b, angle):
        on_body = b.get("parent") == "body"  # else it hangs from the root (or a still anchor)
        moved = np.zeros(3) if on_body else world(b["pivot"]) - np.array(b["pivot"], float)
        out[b["name"]] = {"rotation": [round(angle + (tilt if on_body else 0.0), 3), 0.0, 0.0],
                          "position": [round(float(v), 3) for v in moved]}

    for b in hind:
        place(b, -80.0)
    for b in front:
        place(b, -80.0 if kind == "lie" else (-40.0 if upright is not None else 0.0))
    if head in bones:
        place(bones[head], 0.0)
    return out


def pose_animation(anim: dict, variant: str, pose: dict) -> dict:
    """A copy of vanilla's sit (or lie-down) that holds `pose` (rest_pose; {} to stand as at rest) for
    the variant drawn as our mesh and plays as vanilla for the rest (babies, other colours)."""
    import copy

    out = copy.deepcopy(anim)
    bones = out.setdefault("bones", {})
    pose = {k.lower(): v for k, v in pose.items()}

    def wrap(vals, new):
        if isinstance(vals, dict):  # keyframes by time, or pre/post (a curve's lerp_mode goes: it needs constant keyframes)
            return {k: wrap(v, new) for k, v in vals.items() if k != "lerp_mode"}
        vals = vals if isinstance(vals, list) else [vals] * 3
        return [f"({variant}) ? ({n}) : ({v})" for n, v in zip(new, vals)]

    for bone, chans in bones.items():  # vanilla's moves are vanilla's only
        for ch in ("rotation", "position"):
            if ch in chans:
                chans[ch] = wrap(chans[ch], (pose.get(bone.lower()) or {}).get(ch, [0.0, 0.0, 0.0]))
    have = {k.lower(): k for k in bones}
    for bone, chans in pose.items():  # what vanilla's doesn't move
        cur = bones.setdefault(have.get(bone, bone), {})
        for ch, vals in chans.items():
            cur.setdefault(ch, [f"({variant}) ? ({v}) : (0.0)" for v in vals])
    return out


def mesh_for_variant(entity: dict, controllers: dict, controller: str, geometry: str, variant: str,
                     texture: str | None = None, animations: dict | None = None, layers: dict | None = None,
                     suffix: str = "mesh", material: str | None = None) -> tuple[dict, dict]:
    """For a mob whose variants share one vanilla geometry (2026-10-27: the temperate, warm and cold
    pigs all draw geometry.pig.v3, only their textures differ, so replacing it would lay the warm and
    cold textures on our mesh): (the client entity file with our mesh added as `Geometry.mesh` and
    `controller` swapped for its copy "<controller>.mesh", that copy as a render controllers file).
    The copy draws the mesh where `variant` holds (a Molang condition, e.g. "v.index == 0 &&
    !query.is_baby") and vanilla's geometry elsewhere; other controllers (the saddle) are kept.
    `texture`: the mesh's own texture (a pack path without extension), for a vanilla texture that
    other variants also draw on vanilla's boxes (the sheep's, woolly and shorn); it replaces the
    first texture layer there (the llama's carpet layers stay, clear when it wears none).
    `animations`: the entity's animation names pointed at other ids ({"walk": "animation.sheep.mesh.walk"},
    see variant_animation); its animation controllers play them by name. `layers`: what other
    texture layers read on the mesh ({1: "Texture.markings_none"}: the horse's markings are painted
    for vanilla's UVs and would come out scrambled). `suffix`: of the copy's id, when two entities
    copy one controller (the llama's and the trader llama's). `material`: the game's material for
    the mesh when the mob's own would tint it (a tropical fish's multiplies its random colours over it)."""
    member = {"key": "mesh", "geometry": geometry, "variant": variant, "texture": texture, "layers": layers, "material": material}
    return meshes_for_variants(entity, controllers, controller, [member], animations, suffix)


def meshes_for_variants(entity: dict, controllers: dict, controller: str, members: list, animations: dict | None = None,
                        suffix: str = "mesh") -> tuple[dict, dict]:
    """mesh_for_variant for several meshes of one entity (2026-10-31: each coat of the cat its own model):
    `members`: dicts of key (the mesh's name in the entity file), geometry, variant, and optionally texture,
    layers and material, as there. The first whose variant holds is drawn, vanilla's where none does."""
    import copy

    out = copy.deepcopy(entity)
    desc = out["minecraft:client_entity"]["description"]
    if animations:
        desc.setdefault("animations", {}).update(animations)
    ours = f"{controller}.{suffix}"
    desc["render_controllers"] = [ours if rc == controller else {ours: rc[controller]} if isinstance(rc, dict) and controller in rc
                                  else rc for rc in desc["render_controllers"]]
    rc = copy.deepcopy(controllers["render_controllers"][controller])
    for m in reversed(members):
        key, variant = m["key"], m["variant"]
        desc["geometry"][key] = m["geometry"]
        rc["geometry"] = f"({variant}) ? Geometry.{key} : ({rc['geometry']})"
        if m.get("texture"):
            desc["textures"][key] = m["texture"]
            pick = f"Texture.{key}"
            if m.get("alt"):  # another texture in one state: (when, its path)
                desc["textures"][f"{key}_alt"] = m["alt"][1]
                pick = f"(({m['alt'][0]}) ? Texture.{key}_alt : Texture.{key})"
            rc["textures"][0] = f"({variant}) ? {pick} : ({rc['textures'][0]})"
        for i, layer in (m.get("layers") or {}).items():
            rc["textures"][i] = f"({variant}) ? {layer} : ({rc['textures'][i]})"
        if m.get("material"):
            desc["materials"][key] = m["material"]
            rc["materials"] = [{k: f"({variant}) ? Material.{key} : ({v})" for k, v in entry.items()} for entry in rc["materials"]]
    return out, {"format_version": controllers.get("format_version", "1.8.0"), "render_controllers": {ours: rc}}


def _without_always(text: str) -> str:
    text = text.replace("!(true)", "0.0")
    while "(true) ? " in text:
        start = text.index("(true) ? ")
        ours = start + len("(true) ? ")
        end, depth = ours, 0
        while depth or not text.startswith(" : ", end):
            depth += {"(": 1, ")": -1}.get(text[end], 0)
            end += 1
        text = text[:start] + text[ours:end] + text[_after_group(text, end + 3):]
    while "(true) || " in text:
        start = text.index("(true) || ")
        text = text[:start] + "1.0" + text[_after_group(text, start + len("(true) || ")):]
    return "1.0" if text == "true" else text


def _after_group(text: str, start: int) -> int:
    """The index just past the ")" closing the "(" at `start`."""
    if text[start] != "(":
        raise ValueError(f"no group at {start}: {text}")
    depth = 0
    for i in range(start, len(text)):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if not depth:
            return i + 1
    raise ValueError(f"unbalanced: {text}")


def plain_molang(data):
    """A copy of an entity, animation or render controller file with the condition of a mob that has one
    model everywhere (variant "true") taken out: "(true) ? Geometry.mesh : (Geometry.default)" becomes
    "Geometry.mesh", the form vanilla writes (2026-10-31). Vanilla's own strings are left alone."""
    if isinstance(data, str):
        return _without_always(data)
    if isinstance(data, list):
        return [plain_molang(v) for v in data]
    if isinstance(data, dict):
        return {k: plain_molang(v) for k, v in data.items()}
    return data


def entity_for_game(entity: dict, min_engine: str | None = None) -> dict:
    """A client entity file as the game accepts it (2026-10-31, from its content log). Mojang's sample files
    are every version of a mob merged, so a copy can break its own format's rules, and the game then keeps
    its own mob, the only sign a log line: format 1.8.0 takes no conditional render controllers (a file that
    needs them moves to 1.10.0, where controllers are run from `animations` and `scripts.animate`), 1.10.0 on
    no `animation_controllers`. A controller that never draws goes, one that always draws loses its condition.
    `min_engine`: the game's own file for this mob, when it says more than the sample: the higher number wins."""
    import copy

    out = copy.deepcopy(entity)
    desc = out["minecraft:client_entity"]["description"]
    if min_engine:
        desc["min_engine_version"] = min_engine
    kept = []
    for rc in desc.get("render_controllers", []):
        name, when = (rc, None) if isinstance(rc, str) else next(iter(rc.items()))
        if when != "0.0":
            kept.append(name if when in (None, "1.0") else rc)
    desc["render_controllers"] = kept
    if out["format_version"] != "1.8.0":
        desc.pop("animation_controllers", None)
    elif any(isinstance(rc, dict) for rc in kept):
        out["format_version"] = "1.10.0"
        animate = desc.setdefault("scripts", {}).setdefault("animate", [])
        for entry in desc.pop("animation_controllers", []):
            for short, controller in entry.items():
                desc.setdefault("animations", {})[f"{short}_controller"] = controller
                animate.append(f"{short}_controller")
    return out


def draw_mesh(geo: dict, tex: np.ndarray, yaw: float, pitch: float, px: int = 420, flip_v: bool = True,
              pose: dict | None = None, flat: bool = False, zbuf_out: list | None = None,
              frame: tuple | None = None) -> np.ndarray:
    """A Bedrock geometry of poly_meshes seen from (yaw, pitch) like render.draw: drawn at (-x, y, z),
    orthographic, lit from the upper left; UVs read as the game reads them, v up from the bottom
    (`flip_v=False` reads them top down); `flat`: the texture's colours unlit; `zbuf_out` gets the
    depth buffer. `pose`: bone -> [rx, ry, rz] degrees about its pivot (to see what turning the head
    does), or {"rotation": [...], "position": [...]} as render.setup_pose gives (the game's rest
    pose). `frame`: (centre, radius) in drawing space, to draw at another model's scale (render.bounds)."""
    from . import render as R

    g = geo["minecraft:geometry"][0]
    by_name = {b["name"]: b for b in g["bones"]}
    tris, uvs, norms = [], [], []
    for b in g["bones"]:
        pm = b.get("poly_mesh")
        if not pm:
            continue
        P = np.array(pm["positions"], float) * [-1, 1, 1]
        N = np.array(pm["normals"], float) * [-1, 1, 1]
        U = np.array(pm["uvs"], float)
        cur = b
        while cur is not None and pose:  # its own turn, then each parent's
            bp = pose.get(cur["name"], pose.get(cur["name"].lower()))
            if bp is not None:
                bp = bp if isinstance(bp, dict) else {"rotation": bp}
                piv = np.array(cur.get("pivot", (0, 0, 0)), float) * [-1, 1, 1]
                rm = R._rot(bp.get("rotation") or (0, 0, 0))
                P = (P - piv) @ rm.T + piv + np.array(bp.get("position") or (0, 0, 0), float) * [-1, 1, 1]
                N = N @ rm.T
            cur = by_name.get(cur.get("parent"))
        for poly in pm["polys"]:
            idx = [v for v in poly][:3]
            tris.append(P[[i[0] for i in idx]])
            uvs.append(U[[i[2] for i in idx]])
            norms.append(N[[i[1] for i in idx]])
    tris, uvs, norms = np.array(tris), np.array(uvs), np.array(norms)
    pts = tris.reshape(-1, 3)
    center = (pts.min(0) + pts.max(0)) / 2
    radius = float(np.linalg.norm(pts - center, axis=1).max())
    if frame is not None:
        center, radius = np.asarray(frame[0], float), float(frame[1])
    view, scale = R.view_of(yaw, pitch, px, radius)
    q = (tris - center) @ view.T
    sx, sy = px / 2 - scale * q[..., 0], px / 2 - scale * q[..., 1]
    nv = norms @ view.T  # per corner: smooth (Gouraud) shading, as the game shades vertex normals
    lit = 0.38 + 0.72 * np.clip((nv / np.maximum(np.linalg.norm(nv, axis=2, keepdims=True), 1e-9)) @ R.LIGHT, 0, None)
    if flat:
        lit = np.ones_like(lit)
    img = np.full((px, px, 3), 0.80, np.float32)
    zbuf = np.full((px, px), np.inf)
    th, tw = tex.shape[:2]
    for k in range(len(tris)):
        x, y, z = sx[k], sy[k], q[k, :, 2]
        x0, x1 = max(int(x.min()), 0), min(int(x.max()) + 2, px)
        y0, y1 = max(int(y.min()), 0), min(int(y.max()) + 2, px)
        if x1 <= x0 or y1 <= y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        d = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if abs(d) < 1e-9:
            continue
        l0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / d
        l1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / d
        l2 = 1 - l0 - l1
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        depth = l0 * z[0] + l1 * z[1] + l2 * z[2]
        win = zbuf[y0:y1, x0:x1]
        hit = inside & (depth < win)
        if not hit.any():
            continue
        u = l0 * uvs[k, 0, 0] + l1 * uvs[k, 1, 0] + l2 * uvs[k, 2, 0]
        v = l0 * uvs[k, 0, 1] + l1 * uvs[k, 1, 1] + l2 * uvs[k, 2, 1]
        v = 1 - v if flip_v else v
        col = tex[np.clip((v * th).astype(int), 0, th - 1), np.clip((u * tw).astype(int), 0, tw - 1), :3] / 255.0
        shade = l0 * lit[k, 0] + l1 * lit[k, 1] + l2 * lit[k, 2]
        img[y0:y1, x0:x1][hit] = np.clip(col[hit] * shade[hit][:, None], 0, 1)
        win[hit] = depth[hit]
    if zbuf_out is not None:
        zbuf_out.append(zbuf)
    return img


def view_frame(geo: dict, yaw: float, pitch: float, px: int) -> tuple[np.ndarray, np.ndarray, float]:
    """(centre, view rotation, px per unit) draw_mesh uses for `geo`, to project points the same way."""
    from . import render as R

    pts = np.concatenate([np.array(b["poly_mesh"]["positions"], float) * [-1, 1, 1]
                          for b in geo["minecraft:geometry"][0]["bones"] if b.get("poly_mesh")])
    center = (pts.min(0) + pts.max(0)) / 2
    view, scale = R.view_of(yaw, pitch, px, float(np.linalg.norm(pts - center, axis=1).max()))
    return center, view, scale


def retouch(geo: dict, tex: np.ndarray, picture: np.ndarray, mask: np.ndarray, yaw: float, pitch: float,
            facing: float = 0.3) -> np.ndarray:
    """`tex` with the texels that `picture` (a repaint of draw_mesh's flat view of `geo` from (yaw,
    pitch), its size) shows inside `mask` replaced by it: each texel of every triangle is taken to
    its 3D point, projected into the view, and kept if the view sees it (depth) squarely enough."""
    px = picture.shape[0]
    z: list = []
    draw_mesh(geo, tex, yaw, pitch, px=px, flat=True, zbuf_out=z)
    zbuf = z[0]
    center, view, scale = view_frame(geo, yaw, pitch, px)
    out = tex.copy()
    th, tw = tex.shape[:2]
    for b in geo["minecraft:geometry"][0]["bones"]:
        pm = b.get("poly_mesh")
        if not pm:
            continue
        P = np.array(pm["positions"], float) * [-1, 1, 1]
        N = np.array(pm["normals"], float) * [-1, 1, 1]
        U = np.array(pm["uvs"], float)
        for poly in pm["polys"]:
            i = [v[0] for v in poly[:3]]
            n = N[i].mean(0) @ view.T
            if -n[2] / (np.linalg.norm(n) or 1) < facing:
                continue
            q = (P[i] - center) @ view.T
            sx, sy = px / 2 - scale * q[:, 0], px / 2 - scale * q[:, 1]
            if not mask[np.clip(sy.astype(int), 0, px - 1), np.clip(sx.astype(int), 0, px - 1)].any() and \
                    not mask[np.clip(int(sy.mean()), 0, px - 1), np.clip(int(sx.mean()), 0, px - 1)]:
                continue
            cu, cr = U[i, 0] * tw, (1 - U[i, 1]) * th  # texel column, row
            c0, c1 = max(int(cu.min()), 0), min(int(cu.max()) + 2, tw)
            r0, r1 = max(int(cr.min()), 0), min(int(cr.max()) + 2, th)
            gx, gy = np.meshgrid(np.arange(c0, c1) + 0.5, np.arange(r0, r1) + 0.5)
            d = (cr[1] - cr[2]) * (cu[0] - cu[2]) + (cu[2] - cu[1]) * (cr[0] - cr[2])
            if abs(d) < 1e-9:
                continue
            l0 = ((cr[1] - cr[2]) * (gx - cu[2]) + (cu[2] - cu[1]) * (gy - cr[2])) / d
            l1 = ((cr[2] - cr[0]) * (gx - cu[2]) + (cu[0] - cu[2]) * (gy - cr[2])) / d
            l2 = 1 - l0 - l1
            inside = (l0 >= -0.02) & (l1 >= -0.02) & (l2 >= -0.02)
            vx = np.clip((l0 * sx[0] + l1 * sx[1] + l2 * sx[2]).astype(int), 0, px - 1)
            vy = np.clip((l0 * sy[0] + l1 * sy[1] + l2 * sy[2]).astype(int), 0, px - 1)
            depth = l0 * q[0, 2] + l1 * q[1, 2] + l2 * q[2, 2]
            ok = inside & mask[vy, vx] & (depth <= zbuf[vy, vx] + 0.05)
            region = out[r0:r1, c0:c1]
            region[ok] = (picture[vy[ok], vx[ok]] * 255).astype(np.uint8)
    return out
