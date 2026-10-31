"""Mobs drawn outside the game (2026-10-26, asked to look the reskinned mobs over without hunting
with spawn eggs): vanilla's entity model with a pack's texture, normal map and MER, from a few
angles with plain lighting, next to vanilla's look. One picture per mob in work/render/ and contact
sheets of several mobs. The model stands in its rest pose (no animation), so legacy models whose
setup animation lays the body down draw it upright.

Bedrock model space mirrors x against the screen (as Java's renderer's scale(-1, -1, 1) and
Blockbench's import do): a point is drawn at (-x, y, z) and a bone's rotation [x, y, z] (degrees,
about its pivot, applied x then y then z) turns by -x, -y, z there. Box UV unfolds as Java's
ModelPart.Cube: the strip left of the front ("east" in _uv_rects) is the -x side.
"""
from __future__ import annotations

import fnmatch
import re
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mcreskin import imaging as I
from mcreskin.vanilla import load_json_lenient

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "render"
PACK = ROOT / "build" / "BubonicCreaturesDEV"
VIEWS = (("vanilla", 35, 20), ("front", 0, 0), ("three-quarter", 35, 20), ("side", 90, 0),
         ("back", 215, 20), ("top", 0, 70))
"""(label, yaw, pitch) in degrees: yaw turns the mob's face toward the viewer's right, pitch looks down."""
LIGHT = np.array([0.45, 0.75, -0.5]) / np.linalg.norm([0.45, 0.75, -0.5])
"""Toward the light in view space (x to the viewer's left, y up, z away): upper left, in front."""
UNDER = {"entity/villager2/professions/": ("entity/villager2/villager", "entity/villager2/biomes/biome_plains"),
         "entity/villager2/biomes/": ("entity/villager2/villager",)}
"""Overlay textures the game draws over others on the same model: drawn over these (texel by texel)."""
SUPERSAMPLE = 2


def _rot(r) -> np.ndarray:
    """A Bedrock rotation [x, y, z] (degrees) as a matrix in drawing space."""
    ax, ay, az = np.radians([-float(r[0]), -float(r[1]), float(r[2])])
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _flip(p) -> np.ndarray:
    return np.array([-float(p[0]), float(p[1]), float(p[2])])


OUTWARD = {"up": (0, 1, 0), "down": (0, -1, 0), "east": (-1, 0, 0), "west": (1, 0, 0),
           "north": (0, 0, -1), "south": (0, 0, 1)}
"""Bedrock-space outward direction of each face (east and west swap on a mirrored cube)."""


def _face_corners(lo, hi, mirror: bool) -> dict[str, tuple]:
    """Bedrock-space corners (uv 0,0), (1,0), (0,1) of each face of the box lo..hi, as Java's
    ModelPart.Cube maps its texture (mirror swaps the x ends)."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    xa, xb = (x1, x0) if mirror else (x0, x1)
    return {"up": ((xa, y1, z1), (xb, y1, z1), (xa, y1, z0)),
            "down": ((xa, y0, z1), (xb, y0, z1), (xa, y0, z0)),
            "east": ((xa, y1, z1), (xa, y1, z0), (xa, y0, z1)),
            "north": ((xa, y1, z0), (xb, y1, z0), (xa, y0, z0)),
            "west": ((xb, y1, z0), (xb, y1, z1), (xb, y0, z0)),
            "south": ((xb, y1, z1), (xa, y1, z1), (xb, y0, z1))}


def quads(geometry: dict, pose: dict | None = None, hidden: set[str] = frozenset(), with_ids: bool = False) -> list[tuple]:
    """Every textured cube face of the model in its rest pose: (4 x 3 drawing-space points: the
    corners at uv (0,0), (1,0), (0,1) and the first plus the face's outward normal; its rectangle
    u, v, du, dv in model texels). `pose`: bone name (lower case) -> {"rotation": total, "position":
    offset} from the setup animations (setup_pose); `hidden`: bones whose cubes are not drawn;
    `with_ids` adds (bone name, cube index, face) to each."""
    bones = {b.get("name"): b for b in geometry["bones"]}
    pose = pose or {}

    def place(bone: dict | None, p: np.ndarray) -> np.ndarray:
        seen = 0
        while bone is not None and seen < 64:
            piv = _flip(bone.get("pivot", (0, 0, 0)))
            posed = pose.get(str(bone.get("name")).lower(), {})
            # bind_pose_rotation turns the bone's own cubes only: the panda's and sheep's legs hang
            # from their body bone but stand upright
            bind = (bone.get("bind_pose_rotation") or posed.get("bind")) if seen == 0 else None
            for r in (bind, posed.get("rotation", bone.get("rotation"))):
                if r:
                    p = (p - piv) @ _rot(r).T + piv
            if posed.get("position") is not None:
                p = p + _flip(posed["position"])
            bone, seen = bones.get(bone.get("parent")), seen + 1
        return p

    out = []
    for bone in geometry["bones"]:
        if bone.get("neverRender") or str(bone.get("name")).lower() in hidden:
            continue
        for ci, cube in enumerate(bone.get("cubes", [])):
            uv = cube.get("uv")
            if uv is None:
                continue
            size = np.array(cube.get("size", (0, 0, 0)), float)
            lo = np.array(cube.get("origin", (0, 0, 0)), float) - float(cube.get("inflate", 0.0))
            hi = lo + size + 2 * float(cube.get("inflate", 0.0))
            mirror = bool(cube.get("mirror", bone.get("mirror", False)))
            corners = _face_corners(lo, hi, mirror)
            if isinstance(uv, dict):
                rects = {f: (*fv["uv"], *fv.get("uv_size", (0, 0))) for f, fv in uv.items() if "uv" in fv}
            else:
                sx, sy, sz = np.abs(size)
                u, v = float(uv[0]), float(uv[1])
                rects = {"north": (u + sz, v + sz, sx, sy), "east": (u, v + sz, sz, sy),
                         "south": (u + 2 * sz + sx, v + sz, sx, sy), "west": (u + sz + sx, v + sz, sz, sy),
                         "up": (u + sz, v, sx, sz), "down": (u + sz + sx, v, sx, sz)}
            for face, rect in rects.items():
                if face not in corners or rect[2] == 0 or rect[3] == 0:
                    continue
                out_dir = np.array(OUTWARD[face], float) * (-1 if mirror and face in ("east", "west") else 1)
                pts = np.array([_flip(c) for c in (*corners[face], np.array(corners[face][0]) + out_dir)])
                if cube.get("rotation"):  # about the cube's centre unless it gives a pivot
                    piv = _flip(cube.get("pivot", (lo + hi) / 2))
                    pts = (pts - piv) @ _rot(cube["rotation"]).T + piv
                out.append((place(bone, pts), tuple(float(x) for x in rect)) + ((bone.get("name"), ci, face),) * with_ids)
    return out


RESTING_STATES = ("default", "walking", "idle")
"""Animation controller states that hold a mob standing still (the cat's controller starts lying down)."""
POSE_FIX = {"geometry.fox": {"body": {"bind": [90.0, 0.0, 0.0]}},
            "geometry.polarbear": {"body": {"bind": [90.0, 0.0, 0.0]}},
            "geometry.cat": {"body": {"bind": [90.0, 0.0, 0.0]}},
            "geometry.villager_v2": {"brim": {"rotation": [-90.0, 0.0, 0.0]}}}
"""Rest poses no data file gives (the game sets them in code): "bind" turns the bone's own cubes
like bind_pose_rotation, leaving the legs and head that hang from it upright; the villagers' hat
brim lies flat round the head."""
MOLANG_NAME = re.compile(r"\b(?:query|q|variable|v|temp|t|context|c)\.[A-Za-z_][\w.]*")


MOLANG_FUNCS = {"_sin": lambda d: np.sin(np.radians(d)), "_cos": lambda d: np.cos(np.radians(d)),
                "abs": abs, "min": min, "max": max, "clamp": lambda v, lo, hi: min(max(v, lo), hi)}


def molang(expr, this: float, unread: float = 0.0) -> float:
    """A Molang number at rest: `this` is the channel's rest value, queries and variables are 0
    (an adult, not moving), `a ? b : c` picks c. Anything else unreadable counts as `unread`."""
    if isinstance(expr, bool) or not isinstance(expr, (str, int, float)):
        return float(expr) if isinstance(expr, bool) else unread  # keyframes: not read
    if not isinstance(expr, str):
        return float(expr)
    e = MOLANG_NAME.sub("0", expr.lower()).replace("this", f"({this})")
    e = re.sub(r"\btrue\b", "1", re.sub(r"\bfalse\b", "0", e))
    e = re.sub(r"!(?!=)", " not ", e.replace("&&", " and ").replace("||", " or "))
    e = re.sub(r"math\.(sin|cos)\(", r"_\1(", e).replace("math.", "")
    if "?" in e:
        cond, _, rest = e.partition("?")
        yes, _, no = rest.partition(":")
        e = f"(({yes}) if ({cond}) else ({no}))"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "8 (1 + ...)" and such: unreadable, 0
            return float(eval(e, {"__builtins__": {}}, MOLANG_FUNCS))
    except Exception:
        return unread


def _defs(rp: Path, folder: str, key: str) -> dict:
    """Every definition under resource_pack/<folder>/*.json's top-level `key`, by id."""
    out = {}
    for f in sorted((rp / folder).glob("*.json")):
        try:
            out.update(load_json_lenient(f).get(key, {}))
        except Exception:
            continue
    return out


def _entity_data(b, cache: dict) -> dict:
    if "anims" not in cache:
        rp = b.van.rp
        cache["anims"] = _defs(rp, "animations", "animations")
        cache["controllers"] = _defs(rp, "animation_controllers", "animation_controllers")
        cache["render"] = _defs(rp, "render_controllers", "render_controllers")
        cache["entities"] = []
        for f in sorted((rp / "entity").glob("*.json")):
            try:
                cache["entities"].append(load_json_lenient(f)["minecraft:client_entity"]["description"])
            except Exception:
                continue
    return cache


def _entities_of(b, key: str, geometry_id: str, cache: dict) -> list[dict]:
    """The client entity descriptions that draw texture `key` on `geometry_id`."""
    out = []
    for desc in _entity_data(b, cache)["entities"]:
        texs = [t for t in (desc.get("textures") or {}).values() if isinstance(t, str)]
        geos = [g for g in (desc.get("geometry") or {}).values() if isinstance(g, str)]
        if f"textures/{key}" in texs and geometry_id in geos:
            out.append(desc)
    return out


def _on(entries) -> list[str]:
    """Names in an animate list or a controller state's animations that play at rest."""
    out = []
    for entry in entries or []:
        out += [entry] if isinstance(entry, str) else [k for k, c in entry.items() if molang(c, 0)]
    return out


def hidden_bones(b, key: str, geometry_id: str, cache: dict) -> set[str]:
    """Bones (lower case) that no render controller (draw pass) shows at rest: a horse's saddle
    bags and a llama's chests unless it carries them. Each pass has its own part_visibility."""
    data = _entity_data(b, cache)
    passes = []
    for desc in _entities_of(b, key, geometry_id, cache):
        for rc in desc.get("render_controllers") or []:
            names = [rc] if isinstance(rc, str) else [k for k, c in rc.items() if molang(c, 0, unread=1.0)]
            for name in names:
                rules = {}
                for entry in (data["render"].get(name) or {}).get("part_visibility") or []:
                    rules.update({g.lower(): bool(molang(c, 0, unread=1.0)) for g, c in entry.items()})
                passes.append(rules)
    out = set()
    for bone in b.van.geometry(geometry_id)["bones"]:
        n = str(bone.get("name")).lower()
        if passes and not any(_visible(n, rules) for rules in passes):
            out.add(n)
    return out


def _visible(bone: str, rules: dict[str, bool]) -> bool:
    vis = True
    for glob, v in rules.items():  # later rules win
        if fnmatch.fnmatchcase(bone, glob):
            vis = v
    return vis


def setup_pose(b, key: str, geometry_id: str, cache: dict, geometry: dict | None = None) -> dict:
    """The pose the game's "setup" and always-on animations, and the first state of its animation
    controllers, put `geometry_id` in at rest for texture `key` (a pig's or wolf's body laid down,
    guardian spikes, blaze rods): bone -> {"rotation": total degrees, "position": offset}. States
    the game switches to (sitting, angry) are left out. A rotation channel's `this` is the bone's rest rotation and the
    result is added to it. A position's `this` is the bone's place in the Java model these were
    ported from: its pivot less its parent's (a root bone's parent sits at y 24), so the wolf's
    "-14 - this" leaves its body where it is and the guardian's spikes move out to its sides.
    `geometry`: our own model (glbmodel) posed by `geometry_id`'s mob's animations, to see what the
    game does to it (2026-10-27: the sheep's "-6 - this" sends a head hung from a root at 0 underground)."""
    data = _entity_data(b, cache)
    names = []
    for desc in _entities_of(b, key, geometry_id, cache):
        anims = desc.get("animations") or {}
        on = [k for k in anims if "setup" in k or "setup" in str(anims[k])]
        on += _on((desc.get("scripts") or {}).get("animate"))
        on += [c for entry in desc.get("animation_controllers") or [] for c in entry.values()]
        for k in on:  # an animation, or a controller: what its resting state plays
            target = anims.get(k, k)
            ctl = data["controllers"].get(target)
            if ctl:
                states = ctl.get("states") or {}
                rest = next((n for n in RESTING_STATES if n in states), ctl.get("initial_state"))
                state = states.get(rest) or {}
                names += [anims.get(a, a) for a in _on(state.get("animations"))]
            else:
                names.append(target)
    bones = {str(bn.get("name")).lower(): bn for bn in (geometry or b.van.geometry(geometry_id))["bones"]}
    pose: dict = {}
    for name in dict.fromkeys(names):
        for bone, ch in ((data["anims"].get(name) or {}).get("bones") or {}).items():
            rest = bones.get(bone.lower())
            if rest is None or not isinstance(ch, dict):
                continue
            out = pose.setdefault(bone.lower(), {})
            if isinstance(ch.get("rotation"), list):
                r0 = out.get("rotation") or [float(x) for x in (rest.get("rotation") or (0, 0, 0))]
                out["rotation"] = [r0[i] + molang(ch["rotation"][i], r0[i]) for i in range(3)]
            if isinstance(ch.get("position"), list):
                parent = bones.get(str(rest.get("parent")).lower())
                piv = np.array(rest.get("pivot", (0, 0, 0)), float)
                piv -= np.array(parent.get("pivot", (0, 0, 0)), float) if parent else (0, 24, 0)
                p0 = out.get("position") or [0.0, 0.0, 0.0]
                this = piv + p0
                out["position"] = [p0[i] + molang(ch["position"][i], this[i]) for i in range(3)]
    for bone, fix in ({} if geometry else POSE_FIX.get(geometry_id, {})).items():
        pose.setdefault(bone, {}).update(fix)
    return pose


class Skin:
    """A texture as the renderer samples it: colour, the drawn mask, and optional normal and MER."""

    def __init__(self, color: np.ndarray, drawn_floor: float, normal: np.ndarray | None = None,
                 mer: np.ndarray | None = None):
        self.color, self.floor, self.normal, self.mer = color, drawn_floor, normal, mer

    def scaled(self, k: float) -> "Skin":
        """Shrunk by k (< 1) with area averaging, as the game's mipmaps do at a distance."""
        if k >= 1:
            return self
        h, w = self.color.shape[:2]
        size = (max(1, round(w * k)), max(1, round(h * k)))
        shrink = lambda a: None if a is None else I.resize(a, size, method="area", wrap=False)
        return Skin(shrink(self.color), self.floor, shrink(self.normal), shrink(self.mer))


def view_of(yaw: float, pitch: float, n: int, radius: float) -> tuple[np.ndarray, float]:
    """(rotation into view space, pixels per model unit) of an n-pixel view: a point p in drawing
    space lands at (n/2 - scale * q.x, n/2 - scale * q.y), depth q.z, q = view @ (p - centre)."""
    return _rot((pitch, -yaw, 0)), 0.46 * n / radius  # in drawing space: Rx(-pitch) @ Ry(yaw)


def draw(qs, skin: Skin, tsize: tuple[float, float], yaw: float, pitch: float, px: int,
         center: np.ndarray, radius: float, background=(0.80, 0.80, 0.78), supersample: int = SUPERSAMPLE,
         flat: bool = False, zbuf_out: list | None = None) -> np.ndarray:
    """The model seen from (yaw, pitch), orthographic, `px` square, RGB float. `flat`: the texture's
    colours unlit; `zbuf_out` gets the depth buffer (px * supersample square)."""
    n = px * supersample
    view, scale = view_of(yaw, pitch, n, radius)
    kx, ky = skin.color.shape[1] / tsize[0], skin.color.shape[0] / tsize[1]
    rgb = np.empty((n, n, 3), np.float32)
    rgb[:] = background
    zbuf = np.full((n, n), np.inf, np.float32)
    for pts, (fu, fv, fw, fh), *_ in qs:
        p = (pts - center) @ view.T
        s = np.stack([n / 2 - scale * p[:, 0], n / 2 - scale * p[:, 1]], 1)
        a, b = s[1] - s[0], s[2] - s[0]
        det = a[0] * b[1] - a[1] * b[0]
        if abs(det) < 1e-6:
            continue
        ends = np.array([s[0], s[1], s[2], s[1] + b])
        x0, y0 = np.maximum(np.floor(ends.min(0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(ends.max(0)).astype(int), n)
        if x1 <= x0 or y1 <= y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        dx, dy = gx - s[0, 0], gy - s[0, 1]
        u = (dx * b[1] - dy * b[0]) / det
        v = (a[0] * dy - a[1] * dx) / det
        inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        if not inside.any():
            continue
        tang, bitan = p[1] - p[0], p[2] - p[0]
        normal = (p[3] - p[0]) / (np.linalg.norm(p[3] - p[0]) or 1.0)
        facing = normal[2] < 0  # toward the viewer, who looks along +z
        depth = p[0, 2] + u * (p[1, 2] - p[0, 2]) + v * (p[2, 2] - p[0, 2]) + (0 if facing else 1e-3)
        tx = np.clip(((fu + u * fw) * kx).astype(int), int(min(fu, fu + fw) * kx), int(np.ceil(max(fu, fu + fw) * kx)) - 1)
        ty = np.clip(((fv + v * fh) * ky).astype(int), int(min(fv, fv + fh) * ky), int(np.ceil(max(fv, fv + fh) * ky)) - 1)
        tx = np.clip(tx, 0, skin.color.shape[1] - 1)
        ty = np.clip(ty, 0, skin.color.shape[0] - 1)
        texel = skin.color[ty, tx]
        win = zbuf[y0:y1, x0:x1]
        hit = inside & (texel[..., 3] > skin.floor) & (depth < win)
        if not hit.any():
            continue
        nrm = np.broadcast_to(normal, hit.shape + (3,))
        if skin.normal is not None:
            t = tang / (np.linalg.norm(tang) or 1.0)
            bt = bitan / (np.linalg.norm(bitan) or 1.0)
            nt = skin.normal[ty, tx, :3] * 2 - 1
            nrm = nt[..., :1] * t + nt[..., 1:2] * bt + nt[..., 2:3] * normal
            nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
        if not facing:
            nrm = -nrm
        lit = 1.0 if flat else 0.38 + 0.72 * np.clip(nrm @ LIGHT, 0, None)
        col = texel[..., :3] * np.asarray(lit)[..., None] if not flat else texel[..., :3]
        if skin.mer is not None and not flat:
            mer = skin.mer[ty, tx]
            rough = np.clip(mer[..., 2], 0.05, 1.0)
            half = LIGHT + np.array([0, 0, -1.0])
            half /= np.linalg.norm(half)
            spec = np.clip(nrm @ half, 0, None) ** (2.0 / rough ** 4) * (1 - rough) ** 2 * 0.8
            col = col + spec[..., None] + texel[..., :3] * mer[..., 1:2] * 0.8
        rgb[y0:y1, x0:x1][hit] = np.clip(col[hit], 0, 1)
        win[hit] = depth[hit]
    if zbuf_out is not None:
        zbuf_out.append(zbuf)
    return I.resize(rgb, px, method="area", wrap=False) if supersample > 1 else rgb


def _layers(pack_dir: Path, key: str) -> list[np.ndarray] | None:
    """The pack's colour, normal and MER of `key` (None where missing), or None without a colour."""
    base = pack_dir / "textures" / key
    color = next((p for p in (Path(f"{base}.png"), Path(f"{base}.tga")) if p.exists()), None)
    if color is None:
        return None
    load = lambda p: I.load_rgba(p) if p.exists() else None
    return [I.load_rgba(color), load(Path(f"{base}_normal.png")), load(Path(f"{base}_mer.png"))]


def _over(top: list, under: list, floor: float) -> list:
    """Overlay `top` drawn over `under` (colour, normal, MER each), texel by texel where it is drawn."""
    h, w = top[0].shape[:2]
    fit = lambda a: a if a is None or a.shape[:2] == (h, w) else I.resize(a, (w, h), method="nearest", wrap=False)
    under = [fit(a) for a in under]
    m = (top[0][..., 3] > floor)[..., None]
    flat = lambda like, c: np.broadcast_to(np.array(c, np.float32), like.shape[:2] + (4,))
    out = [np.where(m, top[0], under[0])]
    for i, c in ((1, (0.5, 0.5, 1.0, 1.0)), (2, (0.0, 0.0, 0.85, 1.0))):
        t, u = top[i], under[i]
        out.append(None if t is None and u is None else
                   np.where(m, t if t is not None else flat(top[0], c), u if u is not None else flat(top[0], c)))
    out[0][..., 3] = np.maximum(top[0][..., 3], under[0][..., 3])
    return out


def _candidate(b, key: str, seed: int) -> list | None:
    """Colour of `key` with the face pictures of `tools/oldmobs.py faces` seed `seed` pasted in (no normal
    map or MER yet), or None if that seed has none for it."""
    from .builders import paste_faces
    from .mobfaces import folder_name

    folder = ROOT / "work" / "gen" / "mobfaces" / folder_name(key) / f"s{seed}"
    if not any(folder.glob("*.png")):
        return None
    return [paste_faces(b, key, b.specs.get(key) or {}, 128, folder), None, None]


def skins(b, key: str, pack_dir: Path, candidates: int | None = None) -> tuple[Skin, Skin | None]:
    """(vanilla's texture, the pack's texture with its normal map and MER, or None if the pack has none),
    overlays (villager clothes) drawn over what the game draws under them. With `candidates`, a
    seed's face pictures (work/gen/mobfaces/) take the pack's place wherever that seed has some."""
    from mcreskin.build import _alpha_floor

    def van_and_pack(k: str):
        van = b.van.image(k)
        ours = (_candidate(b, k, candidates) if candidates is not None else None) or _layers(pack_dir, k)
        return van, max(_alpha_floor(van), 0.5 / 255), ours

    van, floor, ours = van_and_pack(key)
    van_l = [van, None, None]
    for prefix, under_keys in UNDER.items():
        if key.startswith(prefix):
            for uk in reversed(under_keys):  # the nearest layer first: plains clothes, then the body
                uvan, _, uours = van_and_pack(uk)
                van_l = _over(van_l, [uvan, None, None], floor)
                if ours is not None:
                    ours = _over(ours, uours or [uvan, None, None], floor)
    return Skin(van_l[0], floor), (Skin(*ours[:1], floor, *ours[1:]) if ours is not None else None)


def bounds(qs) -> tuple[np.ndarray, float]:
    """Centre and radius of the model's corners."""
    pts = np.concatenate([q[0][:3] for q in qs] + [(q[0][1] + q[0][2] - q[0][0])[None] for q in qs])
    center = (pts.min(0) + pts.max(0)) / 2
    return center, float(np.linalg.norm(pts - center, axis=1).max()) or 1.0


def render_mob(b, key: str, pack_dir: Path = PACK, px: int = 300, cache: dict | None = None,
               candidates: int | None = None) -> Image.Image | None:
    """One row: the mob's name over its VIEWS (vanilla's look first), or None if the model has no faces."""
    from .builders import _entity_geometry
    from mcreskin.preview import _font

    van, ours = skins(b, key, pack_dir, candidates)
    spec = b.specs.get(key) or {}
    geo_id = spec.get("geometry") or _entity_geometry(b, key, van.color.shape[1], van.color.shape[0])
    geo = b.van.geometry(geo_id)
    tsize = geo.get("size") or (van.color.shape[1], van.color.shape[0])
    cache = {} if cache is None else cache
    qs = quads(geo, setup_pose(b, key, geo_id, cache), hidden_bones(b, key, geo_id, cache))
    if not qs:
        return None
    center, radius = bounds(qs)
    per_unit = 0.46 * px * SUPERSAMPLE / radius
    shrink = lambda s: s.scaled(per_unit / (s.color.shape[1] / tsize[0]))
    van_s, ours_s = shrink(van), (shrink(ours) if ours is not None else None)
    label_h = 26
    row = Image.new("RGB", (px * len(VIEWS), px + label_h), (40, 40, 40))
    d = ImageDraw.Draw(row)
    d.text((8, 4), key.removeprefix("entity/") + ("" if ours is not None else "   (not in the pack: vanilla only)"),
           fill=(235, 235, 235), font=_font(18))
    for i, (label, yaw, pitch) in enumerate(VIEWS):
        skin = van_s if label == "vanilla" or ours_s is None else ours_s
        img = draw(qs, skin, tsize, yaw, pitch, px, center, radius)
        row.paste(Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)), (i * px, label_h))
        d.text((i * px + 6, label_h + 4), label, fill=(60, 60, 60), font=_font(14))
    return row


def render(keys: list[str] | None = None, pack_dir: Path = PACK, px: int = 300, per_sheet: int = 8,
           candidates: int | None = None, log=print) -> list[Path]:
    """A picture per mob in work/render/ and contact sheets of `per_sheet` mobs. `keys`: globs of
    pack.yaml entries (default: every mob_faces entry). `candidates`: draw that `tools/oldmobs.py faces`
    seed's pictures instead of the pack's, into work/render/s<seed>/."""
    from mcreskin.build import PackBuilder
    from .mobfaces import folder_name

    b = PackBuilder(root=ROOT)
    wanted = [k for k, s in b.specs.items() if k.startswith("entity/") and
              (any(fnmatch.fnmatchcase(k, g) for g in keys) if keys else (s or {}).get("builder") == "mob_faces")]
    if not wanted:
        raise ValueError(f"no pack.yaml entity entry matches {keys}")
    out = OUT / f"s{candidates}" if candidates is not None else OUT
    out.mkdir(parents=True, exist_ok=True)
    rows, cache = [], {}
    for k in wanted:
        row = render_mob(b, k, Path(pack_dir), px, cache, candidates)
        if row is None:
            log(f"{k}: its model has no textured faces, skipped")
            continue
        row.save(out / f"{folder_name(k)}.jpg", quality=90)
        rows.append(row)
    sheets = []
    for i in range(0, len(rows), per_sheet):
        part = rows[i:i + per_sheet]
        sheet = Image.new("RGB", (part[0].width, sum(r.height for r in part)), (40, 40, 40))
        y = 0
        for r in part:
            sheet.paste(r, (0, y))
            y += r.height
        sheets.append(out / f"sheet-{i // per_sheet + 1:02d}.jpg")
        sheet.save(sheets[-1], quality=88)
    log(f"{len(rows)} mobs in {out}, {len(sheets)} sheet(s): {sheets[0].name} .. {sheets[-1].name}" if sheets else "nothing drawn")
    return sheets
