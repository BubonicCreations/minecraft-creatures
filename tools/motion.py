"""Our mesh mobs played by their own vanilla animations at chosen moments (walking slow and at a run,
looking around within the game's limits, each mob's special move), as the game would: channels
summed per bone, a walk scaled by its blend weight, a position's `this` its pivot less its parent's.
Usage: motion.py <out_dir> <name>..."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mcreskin import glbmodel as G
from mcreskin import imaging as I
from mcreskin import render as R
from mcreskin.build import PackBuilder

sys.path.insert(0, str(Path(__file__).parent))
from review import ART, MOBS  # noqa: E402

QUAD = ("animation.quadruped.walk", "animation.common.look_at_target")
OURS = lambda n, extra=(): (f"animation.{n}.mesh.walk", f"animation.{n}.mesh.look_at_target", list(extra))
ANIMS = {  # walk, look-at, specials: (label, animation, time, env); ours: the shipped per-variant copies
    "cow": QUAD + ([],), "pig": QUAD + ([],),
    "llama": OURS("llama"), "mooshroom": OURS("mooshroom"),
    "sheep": OURS("sheep", [("grazing", "animation.sheep.mesh.grazing", 1.0, {})]),
    "polar_bear": OURS("polar_bear", [("rearing up", "animation.polar_bear.mesh.move", 0.0, {"query.standing_scale": 1.0})]),
    "goat": OURS("goat", [("ram", "animation.goat.mesh.ram_attack", 2.0, {}),
                          ("attack", "animation.goat.attack", 0.0, {"variable.attack_head_rot": -37.3})]),
    "hoglin": OURS("hoglin", [("attack", "animation.hoglin.attack", 0.0, {"variable.attack_head_rot": -37.3})]),
}


def value(expr, this, env):
    if not isinstance(expr, str):
        return float(expr)
    e = expr.lower()
    for k in sorted(env, key=len, reverse=True):
        e = e.replace(k, f"({env[k]})")
    return R.molang(e, this)


def channel(ch, t, this, env):
    """A channel's [x, y, z] at time t: three expressions, or keyframes (linear between them)."""
    if isinstance(ch, list):
        return [value(c, this[i], env) for i, c in enumerate(ch)]
    keys = sorted((float(k), v) for k, v in ch.items())
    pick = lambda v, side: v.get(side, v.get("post", v.get("pre"))) if isinstance(v, dict) else v
    if t <= keys[0][0]:
        return channel(pick(keys[0][1], "pre"), t, this, env)
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            a, b = channel(pick(v0, "post"), t, this, env), channel(pick(v1, "pre"), t, this, env)
            k = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            return [x + (y - x) * k for x, y in zip(a, b)]
    return channel(pick(keys[-1][1], "post"), t, this, env)


def play(anims: dict, geometry: dict, names: list, env: dict, t: float = 0.0, base: dict | None = None) -> dict:
    bones = {b["name"].lower(): b for b in geometry["bones"]}
    pose = json.loads(json.dumps(base or {}))
    for name, weight in names:
        for bone, ch in (anims.get(name) or {}).get("bones", {}).items():
            rest = bones.get(bone.lower())
            if rest is None:
                continue
            out = pose.setdefault(bone.lower(), {})
            if "rotation" in ch:
                r = channel(ch["rotation"], t, [0, 0, 0], env)
                out["rotation"] = [a + weight * b for a, b in zip(out.get("rotation") or [0, 0, 0], r)]
            if "position" in ch:
                parent = bones.get(str(rest.get("parent")).lower())
                this = np.array(rest["pivot"], float) - (np.array(parent["pivot"], float) if parent else np.array([0, 24, 0]))
                pz = channel(ch["position"], t, list(this), env)
                out["position"] = [a + weight * b for a, b in zip(out.get("position") or [0, 0, 0], pz)]
    return pose


def walk_env(speed):
    tcos = speed * 57.3  # at the top of the stride: cos(...) = 1
    return {"query.modified_move_speed": speed, "query.anim_time": 0.0, "query.modified_distance_moved": 0.0,
            "variable.gliding_speed_value": 1.0, "variable.tcos_right_side": tcos, "variable.tcos_left_side": -tcos}


def sheet(b, cache, anims, name, px=320):
    key, gid, geo_name, tex_name = MOBS[name]
    walk, look, specials = ANIMS[name]
    geo = json.loads((ART / f"{geo_name}.geo.json").read_text())
    g = geo["minecraft:geometry"][0]
    rest = R.setup_pose(b, key, gid, cache, geometry=g)
    tex = I.load_rgba(ART / f"{tex_name}.png") * 255
    weighted = walk == "animation.quadruped.walk" or (walk.endswith(".mesh.walk") and name not in ("goat", "hoglin"))
    shots = [("rest", rest, 35, 10)]
    for label, speed in (("walk", 0.4), ("run", 1.0)):
        shots.append((f"{label} {speed}", play(anims, g, [(walk, speed if weighted else 1.0)], walk_env(speed), base=rest), 90, 5))
    for label, env in (("look left 45", {"query.target_y_rotation": 45}), ("look right 45", {"query.target_y_rotation": -45}),
                       ("look down 30", {"query.target_x_rotation": 30}), ("look up 30", {"query.target_x_rotation": -30})):
        shots.append((label, play(anims, g, [(look, 1.0)], env, base=rest), 35, 10))
    for label, anim, t, env in specials:
        base = {} if name == "polar_bear" else rest
        shots.append((label, play(anims, g, [(anim, 1.0)], env, t, base=base), 70, 8))
    # one frame for all: the rest pose's, with room
    probe = G.draw_mesh  # frame from the rest pose
    pts = np.concatenate([np.array(bn["poly_mesh"]["positions"]) for bn in g["bones"] if bn.get("poly_mesh")]) * [-1, 1, 1]
    center = (pts.min(0) + pts.max(0)) / 2
    radius = float(np.linalg.norm(pts - center, axis=1).max()) * (1.45 if name == "polar_bear" else 1.15)
    tiles = []
    for label, pose, yaw, pitch in shots:
        img = Image.fromarray((np.clip(G.draw_mesh(geo, tex, yaw, pitch, px, pose=pose, frame=(center, radius)), 0, 1) * 255).astype(np.uint8))
        ImageDraw.Draw(img).text((6, 6), f"{name}: {label}", fill=(30, 30, 30))
        tiles.append(np.asarray(img))
    cols = 5
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 204))
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    return Image.fromarray(np.concatenate(rows, 0))


if __name__ == "__main__":
    out = Path(sys.argv[1])
    b, cache = PackBuilder(root=Path.cwd()), {}
    anims = dict(R._entity_data(b, cache)["anims"])
    for f in ART.glob("*.mesh.animation.json"):
        anims.update(json.loads(f.read_text())["animations"])
    for name in sys.argv[2:]:
        sheet(b, cache, anims, name).save(out / f"motion_{name}.jpg", quality=85)
        print(out / f"motion_{name}.jpg")
