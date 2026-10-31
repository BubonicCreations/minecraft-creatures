"""The render check after each conversion, one sheet per mob, drawn as the game draws it (the entity's
own rest pose on our bones): vanilla's model and ours at the same scale, ours from all round and from
below, its parts in flat colours, the widest step, the head turned and lowered, and the mob's own
special moves (the spec's `shots`, played from its vanilla animations with our edits).
Usage: check.py <out_dir> <name>..."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from creatures import glbmodel as G
from mcreskin import imaging as I
from creatures import render as R

sys.path.insert(0, str(Path(__file__).parent))
import convert as C  # noqa: E402
from motion import play  # noqa: E402

COLOURS = [(230, 60, 50), (60, 170, 80), (70, 110, 230), (240, 200, 40), (200, 80, 220), (60, 210, 220), (250, 140, 40),
           (150, 150, 150), (130, 70, 30), (240, 240, 240), (20, 20, 20), (120, 200, 120)]


def tinted(geo: dict) -> tuple[dict, np.ndarray]:
    """The model with each bone one flat colour, to see where it was cut."""
    out = json.loads(json.dumps(geo))
    bones = [b for b in out["minecraft:geometry"][0]["bones"] if b.get("poly_mesh")]
    tex = np.zeros((1, len(bones), 3), np.float32)
    for i, b in enumerate(bones):
        tex[0, i] = COLOURS[i % len(COLOURS)]
        n = len(b["poly_mesh"]["uvs"])
        b["poly_mesh"]["uvs"] = [[(i + 0.5) / len(bones), 0.5]] * n
    return out, tex


def add(pose: dict, extra: dict) -> dict:
    """`pose` (bones as the game's rest moves them) with more rotation on some bones."""
    out = json.loads(json.dumps(pose))
    for bone, rot in extra.items():
        cur = out.setdefault(bone.lower(), {})
        cur["rotation"] = [a + b for a, b in zip(cur.get("rotation") or [0, 0, 0], rot)]
    return out


ROW = ("ours side", "ours front 3/4", "ours back 3/4", "cuts")


def sheet(b, cache, anims, name: str, px: int = 300, row: bool = False) -> Image.Image:
    """`row`: one row of a few views, for a coat whose parent's moves were looked at already."""
    spec = C.fill(name)
    key, gid = spec["key"], spec["gid"]
    geo = json.loads((C.ART / f"{spec.get('geo_of', name)}.mesh.geo.json").read_text())
    g = geo["minecraft:geometry"][0]
    tex = I.load_rgba(C.ART / f"{spec['texture'].rsplit('/', 1)[-1]}.png")
    if spec.get("wool"):
        tex[..., 3] = 1.0
    tex = tex * 255
    van_geo = b.van.geometry(gid)
    qs = R.quads(van_geo, R.setup_pose(b, key, gid, cache), R.hidden_bones(b, key, gid, cache))
    center, radius = R.bounds(qs)
    pts = np.concatenate([np.array(bn["poly_mesh"]["positions"]) for bn in g["bones"] if bn.get("poly_mesh")]) * [-1, 1, 1]
    radius = max(radius, float(np.linalg.norm(pts - center, axis=1).max())) * 1.15
    van = R.Skin(b.van.image(key), 0.5 / 255)
    tsize = van_geo.get("size") or (van.color.shape[1], van.color.shape[0])
    van = van.scaled(0.46 * px * R.SUPERSAMPLE / radius / (van.color.shape[1] / tsize[0]))
    # the game plays our edited copies for this variant: pose it with them, not with vanilla's
    moves = [(source, edit) for source, edit in spec.get("anims", {}).values() if cache["anims"].get(source)]
    saved = {source: cache["anims"][source] for source, _ in moves}
    for source, edit in moves:
        cache["anims"][source] = C.edited(saved[source], "1.0", C.graze_edit(geo) if edit == "GRAZE" else edit, geo, spec)
    rest = R.setup_pose(b, key, gid, cache, geometry=g)
    legs = [bn for bn in g["bones"] if "leg" in bn["name"].lower() and bn.get("poly_mesh")]
    step = {bn["name"]: [35 if (bn["pivot"][0] > 0) == (bn["pivot"][2] > 0) else -35, 0, 0] for bn in legs}
    head = spec.get("rename", {}).get("head", "head")
    shots = [("vanilla side", None, 90, 5), ("ours side", rest, 90, 5), ("ours front 3/4", rest, 35, 12), ("ours back 3/4", rest, 215, 15),
             ("ours below", rest, 30, -50), ("cuts", "tint", 60, 12), ("step 35", add(rest, step), 90, 5),
             ("head left 35", add(rest, {head: [0, 35, 0]}), 30, 12), ("head right 35", add(rest, {head: [0, -35, 0]}), 330, 12),
             ("head down 30", add(rest, {head: [30, 0, 0]}), 60, 8)]
    for label, anim, t, env in spec.get("shots", []):
        shots.append((label, play(cache["anims"], g, [(anim, 1.0)], env, t, base=rest), 70, 8))
    for source, anim in saved.items():
        cache["anims"][source] = anim
    if row:  # four views and the first of its special moves
        special = [q for q in shots if q[0] not in {x[0] for x in shots[:10]}][:1]
        shots = [q for q in shots if q[0] in ROW] + special
    saddle = spec.get("saddle")
    bare = json.loads(json.dumps(geo))
    bare["minecraft:geometry"][0]["bones"] = [bn for bn in bare["minecraft:geometry"][0]["bones"] if bn["name"] != saddle]
    tint_geo, tint_tex = tinted(bare)
    if saddle and not row:
        shots.append(("saddled", rest, 110, 22))
    tiles = []
    for label, pose, yaw, pitch in shots:
        if pose is None:
            img = R.draw(qs, van, tsize, yaw, pitch, px, center, radius)
        elif isinstance(pose, str):
            img = G.draw_mesh(tint_geo, tint_tex, yaw, pitch, px, pose=rest, frame=(center, radius))
        else:
            img = G.draw_mesh(geo if label == "saddled" else bare, tex, yaw, pitch, px, pose=pose, frame=(center, radius))
        pic = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        ImageDraw.Draw(pic).text((5, 5), f"{name}: {label}", fill=(20, 20, 20))
        tiles.append(np.asarray(pic))
    while len(tiles) % 5:
        tiles.append(np.full_like(tiles[0], 204))
    return Image.fromarray(np.concatenate([np.concatenate(tiles[i:i + 5], 1) for i in range(0, len(tiles), 5)], 0))


if __name__ == "__main__":
    if sys.argv[1] == "--rows":  # check.py --rows out.jpg name...: a row per mob on one sheet, to save on pictures
        cache = {}
        anims = dict(R._entity_data(C.B, cache)["anims"])
        rows = []
        for name in sys.argv[3:]:
            try:
                rows.append(np.asarray(sheet(C.B, cache, anims, name, px=190, row=True)))
            except Exception as e:
                print(f"{name}: check FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
        width = max(r.shape[1] for r in rows)
        rows = [np.concatenate([r, np.full((r.shape[0], width - r.shape[1], 3), 204, np.uint8)], 1) if r.shape[1] < width else r for r in rows]
        Image.fromarray(np.concatenate(rows, 0)).save(sys.argv[2], quality=82)
        print(sys.argv[2])
        sys.exit(0)
    out = Path(sys.argv[1])
    cache = {}
    anims = dict(R._entity_data(C.B, cache)["anims"])
    for f in C.ART.glob("*.mesh.animation.json"):
        anims.update(json.loads(f.read_text())["animations"])
    for name in sys.argv[2:]:
        f = out / f"check_{name}.jpg"
        try:
            sheet(C.B, cache, anims, name).save(f, quality=84)
        except Exception as e:  # one mob's failure doesn't stop the rest
            print(f"{name}: check FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        print(f, flush=True)
