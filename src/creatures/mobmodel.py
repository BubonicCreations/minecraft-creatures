"""Mobs with our own model (2026-10-26, the user: "Is there a way to change the actual shape of the
model?" after face pictures on vanilla's boxes kept looking wrong): a model written as boxes in code
(MODELS), saved as a Bedrock geometry under the vanilla id it replaces, keeping vanilla's bone names
and pivots so the game's animations still move it, with every face's UV packed into one atlas.

Its texture is painted on the model, not face by face: a base atlas coloured by part and position
(patches continue across boxes), rendered flat from several views, each view repainted by Qwen as a
photo (image to image), and the views projected back onto the atlas, each texel taking the views
that see it most squarely.

    mobmodel cow                 # art/models/cow.v2.geo.json + base atlas, renders
    mobmodel cow --paint 2801    # Qwen paints the views -> art/models/cow_v2.png
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage

from mcreskin import imaging as I
from . import render as R

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "art" / "models"
WORK = ROOT / "work" / "gen" / "mobmodels"
DENSITY = 24
"""Atlas pixels per model unit (a 64-texel vanilla texture gets 32 at the 512 setting): the cow's
48 boxes then fit 2048 x 2048; at 32 they needed 2048 x 4096."""
PAD = 4
"""Atlas pixels between faces (mipmaps and filtering bleed into neighbours)."""


def box(origin, size, part="coat", **kw) -> dict:
    """One cube: origin (min corner), size, its `part` (what the base atlas paints it as), and any
    Bedrock cube field (pivot, rotation, inflate)."""
    return {"origin": [float(v) for v in origin], "size": [float(v) for v in size], "part": part, **kw}


def _cow() -> dict:
    """A cow on vanilla's geometry.cow.v2 skeleton (root, head, body, leg0-3; the same pivots, so the
    walk and look-at animations still fit): a barrel with a back ridge, withers, hip bones and a
    rounded rump, a neck and dewlap, an udder, a tail with a tuft (static), a skull with a muzzle and
    wet nose, ears, short horns and eyes, and legs with thighs or forearms, knees, cannons and hooves."""
    def leg(name, cx, rear):
        cubes = ([box((cx - 2.25, 6.5, 3.5), (4.5, 6.5, 5), "thigh"), box((cx - 1.25, 1.5, 4.5), (2.5, 5.25, 2.5), "leg"),
                  box((cx - 1.5, 0, 4.25), (3, 1.5, 3), "hoof")] if rear else
                 [box((cx - 2, 6, -9), (4, 6.5, 4), "thigh"), box((cx - 1.5, 4.75, -8.5), (3, 1.5, 3), "leg"),
                  box((cx - 1.25, 1.5, -8.25), (2.5, 3.5, 2.5), "leg"), box((cx - 1.5, 0, -8.5), (3, 1.5, 3), "hoof")])
        return {"name": name, "parent": "root", "pivot": [cx, 12, 6 if rear else -7], "cubes": cubes}

    teats = [box((x, 8.3, z), (0.6, 1.2, 0.6), "udder") for x in (-1.6, 1.0) for z in (4.8, 7.4)]
    return {
        "id": "geometry.cow.v2", "file": "cow.v2.geo.json", "texture": "entity/cow/cow_v2",
        "visible_bounds": (5, 3.5, [0, 1.25, 0]),
        "bones": [
            {"name": "root", "pivot": [0, 0, -1]},
            {"name": "head", "parent": "root", "pivot": [0, 20, -9], "locators": {"lead": [0, 20, -9]}, "cubes": [
                box((-3.5, 17, -15), (7, 7, 6), "head"),                     # skull
                box((-2.5, 24, -13), (5, 0.75, 3), "head"),                   # poll
                box((-2.5, 15.5, -19), (5, 4.5, 4), "muzzle"),
                box((-2.25, 16, -19.4), (4.5, 3, 0.4), "nose"),
                box((-2, 15, -18.5), (4, 0.75, 4), "muzzle"),                 # jaw
                box((3.5, 21, -12.5), (3, 1.25, 1.5), "ear", pivot=[3.5, 21.6, -11.75], rotation=[0, 0, -15]),
                box((-6.5, 21, -12.5), (3, 1.25, 1.5), "ear", pivot=[-3.5, 21.6, -11.75], rotation=[0, 0, 15]),
                box((3, 23, -12.5), (1.75, 1, 1), "horn"), box((4.25, 23.5, -12.5), (0.75, 1.75, 0.75), "horn"),
                box((-4.75, 23, -12.5), (1.75, 1, 1), "horn"), box((-5, 23.5, -12.5), (0.75, 1.75, 0.75), "horn"),
                box((3.5, 20.5, -14), (0.25, 1, 1.25), "eye"), box((-3.75, 20.5, -14), (0.25, 1, 1.25), "eye"),
            ]},
            {"name": "body", "parent": "root", "pivot": [0, 19, 1], "locators": {"lead_hold": [0, 34, 0]}, "cubes": [
                box((-5.5, 12, -8), (11, 10, 18)),                            # barrel
                box((-4.5, 22, -7.5), (9, 1, 16.5)),                          # back
                box((-4.5, 11, -6), (9, 1, 13)),                              # belly
                box((5.5, 13, -7), (0.5, 8, 16)), box((-6, 13, -7), (0.5, 8, 16)),  # ribs
                box((-3.5, 23, -7.5), (7, 0.75, 5)),                          # withers
                box((3, 22.5, 6), (2.5, 1.25, 3)), box((-5.5, 22.5, 6), (2.5, 1.25, 3)),  # hip bones
                box((-1.5, 22.5, 9), (3, 1, 1.5)),                            # tail head
                box((-4.5, 13, 10), (9, 8.5, 0.75)),                          # rump
                box((-4.5, 14, -8.75), (9, 7, 0.75)),                         # shoulders
                box((-3.5, 11.5, -9), (7, 5, 1.5)),                           # brisket
                box((-3, 15.5, -11.5), (6, 6.5, 4)),                          # neck
                box((-1, 11.5, -11), (2, 4.5, 2.5)),                          # dewlap
                box((-2.5, 9.5, 4), (5, 2, 4.5), "udder"), *teats,
            ]},
            {"name": "tail", "parent": "body", "pivot": [0, 22.5, 10.25], "cubes": [
                box((-0.5, 10, 10.25), (1, 12.5, 1)), box((-1, 7, 10), (2, 3.5, 1.5), "tuft")]},
            leg("leg0", -4, True), leg("leg1", 4, True), leg("leg2", -4, False), leg("leg3", 4, False),
        ]}


MODELS = {"cow": _cow}
FACES = ("north", "east", "south", "west", "up", "down")


def _face_size(size, face: str) -> tuple[float, float]:
    sx, sy, sz = (abs(v) for v in size)
    return {"north": (sx, sy), "south": (sx, sy), "east": (sz, sy), "west": (sz, sy),
            "up": (sx, sz), "down": (sx, sz)}[face]


def geometry(model: dict, density: int = DENSITY) -> tuple[dict, dict]:
    """(the Bedrock geometry file's JSON, (bone, cube, face) -> part) with every face's UV packed on
    shelves into an atlas `density` px per unit wide 2048 (the height a power of two that fits)."""
    faces = []
    for bone in model["bones"]:
        for ci, cube in enumerate(bone.get("cubes", [])):
            for f in FACES:
                w, h = _face_size(cube["size"], f)
                faces.append((bone["name"], ci, f, max(1, round(w * density)), max(1, round(h * density))))
    width, x, y, shelf, place = 2048, PAD, PAD, 0, {}
    for name, ci, f, w, h in sorted(faces, key=lambda t: -t[4]):
        if x + w + PAD > width:
            x, y, shelf = PAD, y + shelf + PAD, 0
        place[(name, ci, f)] = (x, y, w, h)
        x, shelf = x + w + PAD, max(shelf, h)
    height = 1 << int(np.ceil(np.log2(y + shelf + PAD)))
    bones, parts = [], {}
    for bone in model["bones"]:
        out = {k: v for k, v in bone.items() if k != "cubes"}
        if bone.get("cubes"):
            out["cubes"] = []
            for ci, cube in enumerate(bone["cubes"]):
                c = {k: v for k, v in cube.items() if k != "part"}
                c["uv"] = {f: {"uv": list(place[(bone["name"], ci, f)][:2]), "uv_size": list(place[(bone["name"], ci, f)][2:])}
                           for f in FACES}
                out["cubes"].append(c)
                parts.update({(bone["name"], ci, f): cube["part"] for f in FACES})
        bones.append(out)
    vw, vh, vo = model["visible_bounds"]
    geo = {"format_version": "1.16.0", "minecraft:geometry": [{
        "description": {"identifier": model["id"], "texture_width": width, "texture_height": height,
                        "visible_bounds_width": vw, "visible_bounds_height": vh, "visible_bounds_offset": vo},
        "bones": bones}]}
    return geo, parts


def as_render_geometry(geo: dict) -> dict:
    """The file's JSON in the {"bones", "size"} shape vanilla.geometries gives the renderer."""
    g = geo["minecraft:geometry"][0]
    d = g["description"]
    return {"bones": g["bones"], "size": (d["texture_width"], d["texture_height"])}


def _noise3(p: np.ndarray, scale: float, seed: int) -> np.ndarray:
    """Smooth value noise 0..1 at points p (..., 3), features about `scale` units wide."""
    rng = np.random.default_rng(seed)
    grid = rng.random((24, 24, 24))
    grid = ndimage.gaussian_filter(grid, 1.0, mode="wrap")
    grid = (grid - grid.min()) / (np.ptp(grid) or 1)
    q = (p / scale + 8.0) % 24
    return ndimage.map_coordinates(grid, [q[..., 0].ravel(), q[..., 1].ravel(), q[..., 2].ravel()],
                                   order=1, mode="grid-wrap").reshape(p.shape[:-1])


COW_COLOURS = {"coat": (0.30, 0.20, 0.14), "white": (0.88, 0.86, 0.82), "hoof": (0.17, 0.15, 0.14),
               "horn": (0.84, 0.79, 0.67), "nose": (0.42, 0.31, 0.31), "eye": (0.05, 0.04, 0.04),
               "udder": (0.86, 0.62, 0.60), "tuft": (0.88, 0.86, 0.82)}


def cow_colour(part: str, p: np.ndarray) -> np.ndarray:
    """A brown cow with white patches, a white blaze, belly and socks (p: bedrock-space points)."""
    out = np.empty(p.shape[:-1] + (3,), np.float32)
    if part in ("hoof", "horn", "nose", "eye", "udder", "tuft"):
        out[:] = COW_COLOURS[part]
        if part == "horn":  # dark tips
            out[p[..., 1] > 24.6] = (0.35, 0.30, 0.25)
        return out
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    white = _noise3(p, 5.0, 7) > 0.62
    white |= (part == "leg") & (y < 4.5)                                            # socks
    white |= (y < 12.6) & (np.abs(z) < 9) & (part == "coat")                        # belly
    white |= (part in ("head", "muzzle")) & (np.abs(x) < 1.4 + (z < -15) * 0.8) & (z < -14.6)  # blaze
    out[:] = COW_COLOURS["coat"]
    out[white] = COW_COLOURS["white"]
    return out


def face_points(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    """Drawing-space points (h, w, 3) at the texel centres of a face whose corners are pts[:3]."""
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    return pts[0] + u[None, :, None] * (pts[1] - pts[0]) + v[:, None, None] * (pts[2] - pts[0])


def base_atlas(geo: dict, parts: dict, colour=cow_colour) -> np.ndarray:
    """The atlas coloured by part and position, faces in their rest pose."""
    g = as_render_geometry(geo)
    w, h = g["size"]
    atlas = np.zeros((h, w, 4), np.float32)
    for pts, (fu, fv, fw, fh), key in R.quads(g, with_ids=True):
        x0, y0, pw, ph = int(fu), int(fv), int(fw), int(fh)
        p = face_points(pts, pw, ph) * np.array([-1.0, 1.0, 1.0])  # drawing space -> bedrock
        atlas[y0:y0 + ph, x0:x0 + pw, :3] = colour(parts[key], p)
        atlas[y0:y0 + ph, x0:x0 + pw, 3] = 1.0
    return atlas


VIEWS = {"side_left": (90, 0), "side_right": (-90, 0), "front": (0, 0), "back": (180, 0), "top": (0, 89),
         "front_left": (40, 15), "front_right": (-40, 15)}
"""(yaw, pitch) of the views painted and projected back."""
VIEW_PX = 1024
GREEN = (0.15, 0.72, 0.25)
"""Views are painted on a green screen: Qwen's animal comes out a little smoother than the boxes, and
texels at the edges must not take its background (round one put grey fringes on the cow's back)."""


def subject_mask(img: np.ndarray, erode: int = 3) -> np.ndarray:
    """Where a painted view shows the animal, not the green screen (shrunk by `erode` pixels)."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    green = (g > r + 0.12) & (g > b + 0.08)
    return ndimage.binary_erosion(~green, iterations=erode)


def views(geo: dict, atlas: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (flat colour view, depth buffer) of the model with `atlas`, VIEW_PX square."""
    g = as_render_geometry(geo)
    qs = R.quads(g, with_ids=True)
    center, radius = R.bounds(qs)
    skin = R.Skin(atlas, 0.5)
    out = {}
    for name, (yaw, pitch) in VIEWS.items():
        z: list = []
        img = R.draw(qs, skin, g["size"], yaw, pitch, VIEW_PX, center, radius, background=GREEN,
                     supersample=1, flat=True, zbuf_out=z)
        out[name] = (img, z[0])
    return out


def project(geo: dict, base: np.ndarray, painted: dict[str, tuple], sharpness: float = 4.0) -> np.ndarray:
    """The atlas with every texel taking the painted views (name -> (picture, depth buffer, subject
    mask)) that see it (depth test, inside the mask), weighted by how squarely (cos^sharpness);
    texels no view sees keep `base`."""
    g = as_render_geometry(geo)
    qs = R.quads(g, with_ids=True)
    center, radius = R.bounds(qs)
    out = base.copy()
    for pts, (fu, fv, fw, fh), _ in qs:
        x0, y0, pw, ph = int(fu), int(fv), int(fw), int(fh)
        p = face_points(pts, pw, ph)
        normal = (pts[3] - pts[0]) / (np.linalg.norm(pts[3] - pts[0]) or 1.0)
        acc = np.zeros((ph, pw, 3), np.float32)
        wsum = np.zeros((ph, pw), np.float32)
        for name, (yaw, pitch) in VIEWS.items():
            img, zbuf, mask = painted[name]
            view, scale = R.view_of(yaw, pitch, VIEW_PX, radius)
            facing = -(view @ normal)[2]
            if facing <= 0.05:
                continue
            q = (p - center) @ view.T
            sx, sy = VIEW_PX / 2 - scale * q[..., 0], VIEW_PX / 2 - scale * q[..., 1]
            ix, iy = np.clip(sx.astype(int), 0, VIEW_PX - 1), np.clip(sy.astype(int), 0, VIEW_PX - 1)
            seen = ((sx >= 0) & (sx < VIEW_PX) & (sy >= 0) & (sy < VIEW_PX) & (q[..., 2] <= zbuf[iy, ix] + 0.08)
                    & mask[iy, ix])
            wgt = seen * facing ** sharpness
            acc += img[iy, ix] * wgt[..., None]
            wsum += wgt
        have = wsum > 1e-6
        region = out[y0:y0 + ph, x0:x0 + pw, :3]
        region[have] = acc[have] / wsum[have][..., None]
    return out


PROMPTS = {
    "side_left": "A real photograph of a dark brown cow with white patches standing, seen exactly from the side, the whole cow in frame",
    "side_right": "A real photograph of a dark brown cow with white patches standing, seen exactly from the side, the whole cow in frame",
    "front": "A real photograph of a dark brown cow with a white blaze standing, seen straight from the front, its face toward the camera",
    "back": "A real photograph of a dark brown cow with white patches standing, seen straight from behind",
    "top": "A real photograph of the back of a dark brown cow with white patches, seen from directly above",
    "front_left": "A real photograph of a dark brown cow with white patches standing, seen from the front and side",
    "front_right": "A real photograph of a dark brown cow with white patches standing, seen from the front and side",
}
STYLE = "detailed short coarse hair, natural soft daylight, DSLR photo, plain bright green screen background"


def make(name: str, paint: int | None = None, denoise: float = 0.6, log=print) -> Path:
    """Write art/models/<file> and its base atlas (work/gen/mobmodels/<name>/base.png, flat views);
    with `paint` (a seed), Qwen repaints the views and art/models/<texture stem>.png is projected."""
    model = MODELS[name]()
    geo, parts = geometry(model)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / model["file"]).write_text(json.dumps(geo, indent=1))
    work = WORK / name
    work.mkdir(parents=True, exist_ok=True)
    base = base_atlas(geo, parts)
    I.save_png(base, work / "base.png")
    flat = views(geo, base)
    for v, (img, _) in flat.items():
        I.save_png(img, work / f"view_{v}.png", mode="RGB")
    log(f"{name}: {ART / model['file']} ({len(parts) // 6} boxes), base atlas and views in {work}")
    if paint is None:
        return ART / model["file"]
    from mcreskin.comfy import Comfy
    from .mobfaces import repaint

    comfy, painted = Comfy(), {}
    for i, (v, (img, zbuf)) in enumerate(flat.items()):
        with tempfile.TemporaryDirectory() as tmp:
            pic = repaint(comfy, img, f"{PROMPTS[v]}. {STYLE}", paint + i, denoise, Path(tmp))
        I.save_png(pic, work / f"s{paint}_{v}.png", mode="RGB")
        painted[v] = (pic, zbuf, subject_mask(pic))
        log(f"  {v} painted")
    atlas = project(geo, base, painted)
    stem = model["texture"].rsplit("/", 1)[-1]
    I.save_png(atlas, ART / f"{stem}.png")
    log(f"{name}: texture {ART / f'{stem}.png'}")
    return ART / f"{stem}.png"


def preview(name: str, atlas: Path | None = None, px: int = 320) -> Path:
    """work/gen/mobmodels/<name>/preview.jpg: vanilla's mob three-quarter, then the new model with
    `atlas` (default: its base atlas) from render.VIEWS' angles, lit as `tools/oldmobs.py render` lights them."""
    from PIL import Image, ImageDraw

    from mcreskin.build import PackBuilder

    from .builders import _entity_geometry
    from mcreskin.preview import _font

    model = MODELS[name]()
    geo, parts = geometry(model)
    g = as_render_geometry(geo)
    work = WORK / name
    tex = I.load_rgba(atlas) if atlas else I.load_rgba(work / "base.png")
    qs = R.quads(g)
    center, radius = R.bounds(qs)
    row = Image.new("RGB", (px * len(R.VIEWS), px + 26), (40, 40, 40))
    d = ImageDraw.Draw(row)
    d.text((8, 4), f"{model['texture']} (own model: {len(parts) // 6} boxes)", fill=(235, 235, 235), font=_font(18))
    b = PackBuilder(root=ROOT)
    van, _ = R.skins(b, model["texture"], R.PACK)
    vgeo = b.van.geometry(_entity_geometry(b, model["texture"], van.color.shape[1], van.color.shape[0]))
    vq = R.quads(vgeo)
    vc, vr = R.bounds(vq)
    vsize = vgeo.get("size") or (van.color.shape[1], van.color.shape[0])
    for i, (label, yaw, pitch) in enumerate(R.VIEWS):
        if label == "vanilla":
            img = R.draw(vq, van, vsize, yaw, pitch, px, vc, vr)
        else:
            img = R.draw(qs, R.Skin(tex, 0.5), g["size"], yaw, pitch, px, center, radius)
        row.paste(Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)), (i * px, 26))
        d.text((i * px + 6, 30), label, fill=(60, 60, 60), font=_font(14))
    out = work / "preview.jpg"
    row.save(out, quality=90)
    return out
