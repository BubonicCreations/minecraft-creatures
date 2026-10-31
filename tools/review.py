"""Our mesh mobs as the game poses them at rest (their own setup/always-on animations, render.setup_pose
on our bones) next to vanilla's model at the same scale. Usage: review.py <out_dir> <name>..."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mcreskin import glbmodel as G
from mcreskin import imaging as I
from mcreskin import render as R
from mcreskin.build import PackBuilder

ART = Path("art/models/mesh")
MOBS = {  # vanilla texture key (finds the entity), vanilla geometry, our geometry file, our texture
    "cow": ("entity/cow/cow_v2", "geometry.cow.v2", "cow.v2", "cow_v2"),
    "pig": ("entity/pig/pig_v3", "geometry.pig.v3", "pig.mesh", "pig_v3"),
    "sheep": ("entity/sheep/sheep", "geometry.sheep.v1.8", "sheep.mesh", "sheep_mesh"),
    "polar_bear": ("entity/polar_bear", "geometry.polarbear", "polar_bear.mesh", "polar_bear_mesh"),
    "llama": ("entity/llama/llama_creamy", "geometry.llama.v1.8", "llama.mesh", "llama_mesh"),
    "goat": ("entity/goat/goat", "geometry.goat", "goat.mesh", "goat_mesh"),
    "hoglin": ("entity/hoglin/hoglin", "geometry.hoglin", "hoglin.mesh", "hoglin_mesh"),
    "mooshroom": ("entity/cow/mooshroom_v2", "geometry.mooshroom.v2", "mooshroom.mesh", "mooshroom_mesh"),
}
VIEWS = [("side", 90, 5), ("front 3/4", 35, 12), ("front", 0, 5), ("back 3/4", 215, 15)]


def row(b, cache, name, px=380, ours_pose=None):
    key, gid, geo_name, tex_name = MOBS[name]
    van_geo = b.van.geometry(gid)
    qs = R.quads(van_geo, R.setup_pose(b, key, gid, cache), R.hidden_bones(b, key, gid, cache))
    center, radius = R.bounds(qs)
    radius *= 1.25
    van = R.Skin(b.van.image(key), 0.5 / 255)
    tsize = van_geo.get("size") or (van.color.shape[1], van.color.shape[0])
    van = van.scaled(0.46 * px * R.SUPERSAMPLE / radius / (van.color.shape[1] / tsize[0]))
    geo = json.loads((ART / f"{geo_name}.geo.json").read_text())
    pose = R.setup_pose(b, key, gid, cache, geometry=geo["minecraft:geometry"][0])
    pose = {**pose, **(ours_pose or {})}
    tex = I.load_rgba(ART / f"{tex_name}.png") * 255
    tiles = []
    for label, yaw, pitch in VIEWS:
        a = R.draw(qs, van, tsize, yaw, pitch, px, center, radius)
        o = G.draw_mesh(geo, tex, yaw, pitch, px, pose=pose, frame=(center, radius))
        tiles += [a, o]
    img = Image.fromarray((np.clip(np.concatenate(tiles, 1), 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for i, (label, *_) in enumerate(VIEWS):
        d.text((i * 2 * px + 6, 6), f"{name} vanilla {label}", fill=(40, 40, 40))
        d.text(((i * 2 + 1) * px + 6, 6), f"ours, game rest pose", fill=(40, 40, 40))
    moved = {k: v for k, v in pose.items() if any(abs(x) > 1e-6 for x in (v.get("position") or [0, 0, 0]) + list(v.get("rotation") or [0, 0, 0]))}
    print(name, "rest pose on ours:", json.dumps(moved))
    return img


if __name__ == "__main__":
    out = Path(sys.argv[1])
    b, cache = PackBuilder(root=Path.cwd()), {}
    for name in sys.argv[2:]:
        row(b, cache, name).save(out / f"review_{name}.jpg", quality=85)
