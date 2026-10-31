"""The ender dragon. Its vanilla file names no parents, its neck and tail exist only in the game's code (clones of
`neck`), and its children's boxes are in their parents' frames, so nothing here depends on how the game wires them:
geometry.dragon again, same bones and pivots, each part of the GLB scaled into the box of vanilla's bone it replaces.
Wings are cards (glbmodel.card). Usage: dragon.py <sheet.jpg>"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from mcreskin import glbmodel as G
from mcreskin import imaging as I
from mcreskin import render as R
from mcreskin.build import PackBuilder

sys.path.insert(0, str(Path(__file__).parent))
import plans as P  # noqa: E402

WORK = Path("work/models3d/mobs")
ART = Path("art/models/mesh")
B = PackBuilder(root=Path.cwd())
VAN = {b["name"]: b for b in B.van.geometry("geometry.dragon")["bones"]}
DROP = ("x_rearleg", "x_frontleg", "x_stub")  # the other side's legs (one side's are mirrored) and the cut tail's end


def box(name: str) -> tuple[np.ndarray, np.ndarray]:
    """The corners of vanilla's bone (all its cubes), as the file has them."""
    pts = []
    for c in VAN[name]["cubes"]:
        o, s = np.array(c["origin"], float), np.array(c["size"], float)
        pts += [o, o + s]
    return np.min(pts, 0), np.max(pts, 0)


def build() -> dict:
    m = G.place(G.load_glb(WORK / "ender_dragon_single_1500.glb"), 100.0, (1.0, 0.0, 0.0))
    ranges = m["positions"].copy()
    rules = [("head", "y", 0.76, "+", {"z": (-0.01, 0.26)}), ("jaw", "y", 0.815, "-", {"z": (-0.01, 0.2)}, {"of": "head", "parent": "head"}), ("neck", "y", 0.56, "+", {"z": (-0.01, 0.37)}),
             ("x_stub", "z", 0.9, "+"),
             ("rearleg", "y", 0.4, "-", {"x": (-1.01, 0.0), "z": (0.5, 0.9)}), ("rearlegtip", "y", 0.2, "-", None, {"of": "rearleg", "parent": "rearleg"}), ("rearfoot", "y", 0.06, "-", None, {"of": "rearlegtip", "parent": "rearlegtip"}),
             ("frontleg", "y", 0.36, "-", {"x": (-1.01, 0.0), "z": (0.2, 0.5)}), ("frontlegtip", "y", 0.18, "-", None, {"of": "frontleg", "parent": "frontleg"}), ("frontfoot", "y", 0.05, "-", None, {"of": "frontlegtip", "parent": "frontlegtip"}),
             ("x_rearleg", "y", 0.4, "-", {"x": (0.0, 1.01), "z": (0.5, 0.9)}), ("x_frontleg", "y", 0.36, "-", {"x": (0.0, 1.01), "z": (0.2, 0.5)})]
    bone = np.array(["body"] * len(m["tris"]), dtype=object)
    pivots, extras, parents = {}, {}, {}
    for name, axis, at, side, *rest in rules:
        kw = rest[1] if len(rest) > 1 else {}
        m, bone = G.carve(m, bone, pivots, extras, parents, name, axis, at, side, box=rest[0] if rest else None, ranges=ranges, **kw)
    pivots["body"], parents["body"] = [0.0, 50.0, 0.0], "root"
    wing = G.card(np.asarray(Image.open(WORK / "dragon_wing_cut_s82_0.png").convert("RGBA")), 100.0)
    p = wing["positions"]
    used = p[np.unique(wing["tris"])]
    lo, hi = used.min(0), used.max(0)
    # the whole wing in the world as vanilla's two pieces make it: x from the shoulder -12 out to -124, z 4 to 60, y 19
    M = np.diag([-112.0 / (hi[0] - lo[0]), 1.0, 56.0 / (hi[2] - lo[2]), 1.0])
    M[:3, 3] = [-12.0 + 112.0 * lo[0] / (hi[0] - lo[0]), 19.0, 4.0 - 56.0 * lo[2] / (hi[2] - lo[2])]
    cx = p[wing["tris"]].mean(1)[:, 0]
    middle = (lo[0] + hi[0]) / 2
    places = [("wing", "root", VAN["wing"]["pivot"], M, cx < middle), ("wingtip", "root", VAN["wingtip"]["pivot"], M, cx >= middle)]
    m, bone = G.attach(m, bone, wing, places, pivots, parents, share=0.3)
    keep = np.array([b not in DROP for b in bone])
    color = m["images"]["color"]
    return G.geometry(m, bone, pivots, "geometry.dragon", (color.shape[1], color.shape[0]), extras, parents=parents, seal=True, balls=True), color, keep


def fitted(geo: dict) -> dict:
    """Each part in the box of vanilla's bone, flat under the root as vanilla's file has it (the jaw on the head)."""
    src = {b["name"]: b for b in geo["minecraft:geometry"][0]["bones"]}
    out = [{"name": "root", "pivot": VAN["root"]["pivot"]}]

    def mesh(name, turn=None, target=None, mirror=False, shift=(0.0, 0.0, 0.0), core=0.0):
        pm = json.loads(json.dumps(src[name]["poly_mesh"]))
        q = np.array(pm["positions"], float)
        n = np.array(pm["normals"], float)
        if turn is not None:
            q, n = turn(q), turn(n)
        if target is not None:  # `core`: quantile of its points that fills the box (spikes stick out past it)
            a, b = np.quantile(q, core, axis=0), np.quantile(q, 1 - core, axis=0)
            lo, hi = target
            q = lo + (q - a) * (np.array(hi) - lo) / np.maximum(b - a, 1e-6)
        q = q + np.array(shift)
        if mirror:
            q, n = q * [-1, 1, 1], n * [-1, 1, 1]
            pm["polys"] = [poly[::-1][1:] + poly[::-1][:1] for poly in pm["polys"]]
        pm["positions"] = np.round(q, 3).tolist()
        pm["normals"] = np.round(n, 4).tolist()
        return pm

    def add(name, pm, parent=None):
        b = {"name": name, "pivot": VAN[name]["pivot"], "poly_mesh": pm}
        if parent:
            b["parent"] = parent
        out.append(b)

    lie = lambda v: np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)  # the neck stands up in the GLB; vanilla's lies along z, its back up
    add("head", mesh("head", target=box("head")))
    add("jaw", mesh("jaw", target=box("jaw")), "head")
    lo, hi = box("neck")
    add("neck", mesh("neck", lie, (np.array([lo[0], lo[1], -9.0]), np.array([hi[0], hi[1] - 4.0, 9.0])), core=0.08))  # (longer than the 10 between copies: they overlap)
    add("body", mesh("body", target=box("body")))
    # the wing's world numbers are its bones' own; the tip's frame starts at the wing's end (see the module's note)
    add("wing", mesh("wing"))
    add("wingtip", mesh("wingtip", shift=(12.0, 5.0, -2.0)))
    add("wing1", mesh("wing", shift=(24.0, 0.0, 0.0)))
    add("wingtip1", mesh("wingtip", shift=(12.0, 5.0, -2.0)))
    for root in ("rearleg", "frontleg"):
        add(root, mesh(root, target=box(root)))
        add(root + "1", mesh(root, target=box(root), mirror=True))
        for part in (root + "tip", root.replace("leg", "foot")):
            add(part, mesh(part, target=box(part)))
            add(part + "1", mesh(part, target=box(part + "1"), mirror=True))
    return {"format_version": "1.16.0", "minecraft:geometry": [{"description": {"identifier": "geometry.dragon", "visible_bounds_width": 14, "visible_bounds_height": 13, "visible_bounds_offset": [0, 2, 0],
                                                                                "texture_width": geo["minecraft:geometry"][0]["description"]["texture_width"], "texture_height": geo["minecraft:geometry"][0]["description"]["texture_height"]},
                                                                "bones": out}]}


def flying(g: dict, flap: float = 0.0, wing_sign: float = 1.0) -> dict:
    """Our guess of how the game assembles it, to look at: children in their parents' frames (offset = pivot less 24 up), wing1 mirrored, neck and tail copies in a row."""
    bones = {b["name"]: b for b in g["minecraft:geometry"][0]["bones"]}
    flat = []

    def place(name, world, new=None):
        b = bones[name]
        pm = json.loads(json.dumps(b["poly_mesh"]))
        pm["positions"] = np.round(world(np.array(pm["positions"], float)), 3).tolist()
        pm["normals"] = pm["normals"]
        flat.append({"name": new or name, "pivot": [0, 0, 0], "poly_mesh": pm})

    def about(pivot, rot, then=lambda q: q):
        pv = np.array(pivot, float)
        r = R._rot(rot)
        return lambda q: then((q - pv) @ r.T + pv)

    place("body", lambda q: q)
    for i in range(5):  # the neck forward of the body
        z = -8.0 - 10.0 * i
        place("neck", lambda q, z=z: q + [0.0, -20.0, z - 0.0], f"neck{i}")
    for i in range(12):
        z = 56.0 + 10.0 * i
        place("neck", lambda q, z=z, i=i: q * [1 - 0.04 * i, 1 - 0.04 * i, 1] + [0.0, -20.0 + 2.0 * (1 - 0.04 * i), z], f"tail{i}")
    place("head", lambda q: q + [0.0, -20.0, -56.0])
    hj = np.array(VAN["jaw"]["pivot"], float)
    place("jaw", about(hj, (12.0, 0, 0), lambda q: q + [0.0, -20.0, -56.0]))
    wing_r = (7.16, 0.0, 14.0)
    w0 = np.array(VAN["wing"]["pivot"], float)
    wmat = about(w0, wing_r)
    place("wing", wmat)
    tp = np.array(VAN["wingtip"]["pivot"], float)
    tip_world = lambda q: wmat(((q - tp) @ R._rot((0, 0, -20.0)).T) + w0 + tp - [0.0, 24.0, 0.0])
    place("wingtip", tip_world)
    mirror = np.array([-1.0, 1.0, 1.0])
    place("wing", lambda q: wmat(q) * mirror, "wing1")
    place("wingtip", lambda q: tip_world(q) * mirror, "wingtip1")
    for leg, tip, foot, a0, a1, a2 in (("rearleg", "rearlegtip", "rearfoot", 57.3, 28.65, 43.0), ("frontleg", "frontlegtip", "frontfoot", 74.5, -28.65, 43.0)):
        for side, pm in (("", np.array([1.0, 1.0, 1.0])), ("1", np.array([1.0, 1.0, 1.0]))):
            lp, tp2, fp = (np.array(VAN[n + side]["pivot"], float) for n in (leg, tip, foot))
            lmat = about(lp, (a0, 0, 0))
            tmat = lambda q, lp=lp, tp2=tp2, lmat=lmat, a1=a1: lmat((q - tp2) @ R._rot((a1, 0, 0)).T + lp + tp2 - [0.0, 24.0, 0.0])
            fmat = lambda q, tp2=tp2, fp=fp, tmat=tmat, a2=a2: tmat((q - fp) @ R._rot((a2, 0, 0)).T + tp2 + fp - [0.0, 24.0, 0.0])
            place(leg + side, lmat)
            place(tip + side, tmat)
            place(foot + side, fmat)
    return {"format_version": "1.16.0", "minecraft:geometry": [{"description": g["minecraft:geometry"][0]["description"], "bones": [{"name": "root", "pivot": [0, 0, 0]}] + [{**b, "parent": "root"} for b in flat]}]}


if __name__ == "__main__":
    geo, color, _ = build()
    geo["minecraft:geometry"][0]["bones"] = [b for b in geo["minecraft:geometry"][0]["bones"] if b["name"] not in DROP]
    out = fitted(geo)
    gpath = ART / "ender_dragon.mesh.geo.json"
    gpath.write_text(json.dumps(out, separators=(",", ":")))
    rgba = np.concatenate([color[..., :3].astype(np.float32) / 255, np.ones(color.shape[:2] + (1,), np.float32)], -1)
    tint = np.array([1.25, 1.25, 1.3])
    rgba[..., :3] = np.clip(rgba[..., :3] * tint, 0, 1)
    I.save_png(rgba, ART / "ender_dragon_mesh.png")
    (ART / "ender_dragon_single_1500.glb").write_bytes((WORK / "ender_dragon_single_1500.glb").read_bytes())
    print([(b["name"], len(b.get("poly_mesh", {}).get("polys", []))) for b in out["minecraft:geometry"][0]["bones"]])
    tex = rgba * 255
    view = flying(out)
    tiles = [np.asarray(G.draw_mesh(view, tex, y, p, px=420))[..., :3] for y, p in ((90, 5), (35, 15), (215, 20), (0, 60))]
    Image.fromarray((np.clip(np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0), 0, 1) * 255).astype(np.uint8)).save(sys.argv[1], quality=84)
