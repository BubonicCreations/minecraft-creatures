"""How bodies that aren't four legs and a head are cut into the bones the game's animations move
(glbmodel.carve, pieces_past), in fractions of each mesh's extent: x -1..1, y 0 (feet)..1, z 0 (snout)..1.
Each plan takes and returns what glbmodel.make's `plan` does; the check sheet's "cuts" tile shows the result."""
from mcreskin import glbmodel as G

QUARTERS = {"leg0": ((-1, 0), (0.5, 1)), "leg1": ((0, 1), (0.5, 1)), "leg2": ((-1, 0), (0, 0.5)), "leg3": ((0, 1), (0, 0.5))}  # x, z


def parts(*rules):
    """A plan from carve rules: (name, axis, at, side, box, keywords...), all in the first frame."""
    def plan(m, bone, pivots, extras, parents):
        ranges = m["positions"].copy()
        for rule in rules:
            name, axis, at, side, *rest = rule
            box = rest[0] if rest else None
            kw = rest[1] if len(rest) > 1 else {}
            m, bone = G.carve(m, bone, pivots, extras, parents, name, axis, at, side, box=box, ranges=ranges, **kw)
        return m, bone
    return plan


def legs_under(at: float, names=("leg0", "leg1", "leg2", "leg3")) -> list:
    """Four legs below height `at`, one per quarter."""
    return [(n, "y", at, "-", {"x": QUARTERS[q][0], "z": QUARTERS[q][1]}) for n, q in zip(names, QUARTERS)]


def quarter(c) -> str:
    """Which leg a piece centred at `c` is, as glbmodel's: leg0 back right (x < 0) .. leg3 front left."""
    return {(False, False): "leg0", (True, False): "leg1", (False, True): "leg2", (True, True): "leg3"}[(bool(c[0] > 0), bool(c[2] < 0))]


def creeper(m, bone, pivots, extras, parents):
    ranges = m["positions"].copy()
    m, bone = G.carve(m, bone, pivots, extras, parents, "head", "y", 0.72, "+", ranges=ranges)
    return G.pieces_past(m, bone, pivots, extras, parents, "y", 0.13, "-", 4, quarter)


CREEPER = creeper


def side_name(right: str, left: str):
    return lambda c: left if c[0] > 0 else right


def biped(legs_at: float, head_at: float, head_y: float, right="leg0", left="leg1"):
    """Two legs below a height and a head forward of a plane (chicken, parrot)."""
    def plan(m, bone, pivots, extras, parents):
        ranges = m["positions"].copy()
        m, bone = G.carve(m, bone, pivots, extras, parents, "head", "z", head_at, "-", box={"y": (head_y, 1.01)}, ranges=ranges)
        return G.pieces_past(m, bone, pivots, extras, parents, "y", legs_at, "-", 2, side_name(right, left))
    return plan


def chain(fronts, backs, of="body", axis="z"):
    """A body sliced along its length, each slice turning against the next one toward the middle
    (silverfish, endermite). fronts/backs: [(name, at)], from the middle out."""
    def plan(m, bone, pivots, extras, parents):
        ranges = m["positions"].copy()
        for seq, side in ((fronts, "-"), (backs, "+")):
            last = of
            for name, at in seq:
                m, bone = G.carve(m, bone, pivots, extras, parents, name, axis, at, side, of=last, parent=None if last == of else last, ranges=ranges)
                last = name
        return m, bone
    return plan


def spider(m, bone, pivots, extras, parents):
    """Eight legs past a plane on each side, the head forward of the body."""
    ranges = m["positions"].copy()
    m, bone = G.pieces_past(m, bone, pivots, extras, parents, "x", 0.5, "+", 4, ["leg7", "leg5", "leg3", "leg1"])
    m, bone = G.pieces_past(m, bone, pivots, extras, parents, "x", -0.5, "-", 4, ["leg6", "leg4", "leg2", "leg0"])
    return G.carve(m, bone, pivots, extras, parents, "head", "z", 0.3, "-", box={"x": (-0.5, 0.5)}, ranges=ranges)


def tail_after(at: float, name: str = "tail", y=(0.0, 1.01)):
    """After the legs and head: the tail behind a plane."""
    def plan(m, bone, pivots, extras, parents):
        return G.carve(m, bone, pivots, extras, parents, name, "z", at, "+", box={"y": y})
    return ("+", plan)


def parrot(m, bone, pivots, extras, parents):
    ranges = m["positions"].copy()
    for rule in (("head", "z", 0.42, "-", {"y": (0.55, 1.01)}), ("tail", "z", 0.68, "+", {"y": (0.0, 0.6)}),
                 ("wing0", "x", 0.45, "+", {"y": (0.3, 0.85)}), ("wing1", "x", -0.45, "-", {"y": (0.3, 0.85)})):
        m, bone = G.carve(m, bone, pivots, extras, parents, *rule[:4], box=rule[4], ranges=ranges)
    m, bone = G.carve(m, bone, pivots, extras, parents, "leg0", "y", 0.16, "-", box={"x": (0.0, 1.01)}, ranges=ranges)
    return G.carve(m, bone, pivots, extras, parents, "leg1", "y", 0.16, "-", box={"x": (-1.01, 0.0)}, ranges=ranges)


def bee(m, bone, pivots, extras, parents):
    ranges = m["positions"].copy()
    m, bone = G.carve(m, bone, pivots, extras, parents, "rightwing_bone", "y", 0.68, "+", box={"x": (-1.01, 0.0)}, ranges=ranges)
    return G.carve(m, bone, pivots, extras, parents, "leftwing_bone", "y", 0.68, "+", box={"x": (0.0, 1.01)}, ranges=ranges)


def fairy(right_arm: str, left_arm: str, right_wing: str, left_wing: str):
    """A small upright flier seen from the front: wings behind the body, arms at its sides, the head on top."""
    def plan(m, bone, pivots, extras, parents):
        ranges = m["positions"].copy()
        rules = (("head", "y", 0.58, "+", {"x": (-0.6, 0.6)}), (right_wing, "z", 0.62, "+", {"x": (-1.01, 0.0)}), (left_wing, "z", 0.62, "+", {"x": (0.0, 1.01)}),
                 (right_arm, "x", -0.42, "-", {"y": (0.15, 0.6)}), (left_arm, "x", 0.42, "+", {"y": (0.15, 0.6)}))
        for rule in rules:
            m, bone = G.carve(m, bone, pivots, extras, parents, *rule[:4], box=rule[4], ranges=ranges)
        return m, bone
    return plan


def squid(m, bone, pivots, extras, parents):
    """Eight arms below a height, named by where they hang round the body (as vanilla's, by angle)."""
    import numpy as np

    def name(c):
        a = np.degrees(np.arctan2(c[2], c[0])) % 360  # tentacle1 at +x, tentacle3 at +z, ...
        return f"tentacle{int(round(a / 45)) % 8 + 1}"

    centre = m["positions"].mean(0)
    def named(c):
        return name(c - centre)
    return G.pieces_past(m, bone, pivots, extras, parents, "y", 0.4, "-", 8, named)


def human(head: float = 0.84, arm: float = 0.4, legs: float = 0.47, names=("rightArm", "leftArm", "rightLeg", "leftLeg")):
    """Standing, facing the camera: the head above a height, the arms past planes at the shoulders, the legs below the hips."""
    return parts(("head", "y", head, "+", {"x": (-arm, arm)}),
                 (names[0], "x", -arm, "-", {"y": (legs - 0.1, head + 0.02)}), (names[1], "x", arm, "+", {"y": (legs - 0.1, head + 0.02)}),
                 (names[2], "y", legs, "-", {"x": (-1.01, 0.0)}), (names[3], "y", legs, "-", {"x": (0.0, 1.01)}))


def robed(head: float = 0.76, legs: float = 0.2, names=("leg0", "leg1")):
    """Standing in a long robe, arms folded on the chest (they stay with the body): the head above a height, the legs under the hem."""
    return parts(("head", "y", head, "+", {"x": (-0.6, 0.6)}), (names[0], "y", legs, "-", {"x": (-1.01, 0.0)}), (names[1], "y", legs, "-", {"x": (0.0, 1.01)}))


def only(name: str):
    """One bone: the whole mesh."""
    def plan(m, bone, pivots, extras, parents):
        return m, bone
    return plan


def blaze(rod_glb):
    """The GLB is the head alone; twelve copies of one rod (from a picture: a rod through the 3D step came back a stick with fins) where vanilla's rest before its animation sends them round."""
    import numpy as np

    def plan(m, bone, pivots, extras, parents):
        m = {**m, "positions": m["positions"] + [0.0, 20.0, 0.0]}
        bone = np.array(["head"] * len(m["tris"]), dtype=object)
        pivots["head"], parents["head"] = [0.0, 24.0, 0.0], "root"
        from PIL import Image
        rod = G.rod(np.asarray(Image.open(rod_glb).convert("RGB")), 8.0, 1.1)
        M = np.eye(4)
        M[:3, 3] = [1.0, 16.0, 1.0]
        places = [(f"upperBodyParts{i}", "root", (0.0, 24.0, 0.0), M) for i in range(12)]
        return G.attach(m, bone, rod, places, pivots, parents, share=0.2)
    return plan


def shulker(head_glb):
    """A closed shell cut across its seam, the lid a bone of its own; the head (a GLB of its own) sits inside."""
    import numpy as np

    def plan(m, bone, pivots, extras, parents):
        ranges = m["positions"].copy()
        m, bone = G.carve(m, bone, pivots, extras, parents, "lid", "y", 0.5, "+", ranges=ranges)
        head = G.place(G.load_glb(head_glb), 6.0, (0.3, 0.0, 1.0), False)
        M = np.eye(4)
        M[1, 3] = 6.0
        return G.attach(m, bone, head, [("head", "root", (0.0, 12.0, 0.0), M)], pivots, parents, share=0.25)
    return plan


def wings_on(m, bone, wing_png: str, size: float, right_at, left_at, tip_from: float, pivots, parents, names, tip_pivot):
    """A card of one wing (a cut-out with its root on the picture's left, its top edge up) on each side, as a
    flat piece upright across the body: names = (right, right tip, left, left tip)."""
    import numpy as np
    from PIL import Image

    card = G.card(np.asarray(Image.open(wing_png).convert("RGBA")), size)
    p = card["positions"]
    used = p[np.unique(card["tris"])]
    x0, z0 = used[:, 0].min(), used[:, 2].min()
    cx = p[card["tris"]].mean(1)[:, 0]
    inner, outer = cx < x0 + tip_from * size, cx >= x0 + tip_from * size

    def matrix(side, at):
        M = np.eye(4)
        M[:3, :3] = [[side, 0, 0], [0, 0, -1], [0, 1, 0]]
        M[:3, 3] = [at[0] - side * x0, at[1] + z0, at[2]]
        return M

    r, l = matrix(-1.0, right_at), matrix(1.0, left_at)
    tip_r = [right_at[0] - tip_from * size, right_at[1], right_at[2]]
    tip_l = [left_at[0] + tip_from * size, left_at[1], left_at[2]]
    places = [(names[0], "root", right_at, r, inner), (names[1], names[0], tip_r, r, outer), (names[2], "root", left_at, l, inner), (names[3], names[2], tip_l, l, outer)]
    return places, card


def bat(wing_png: str):
    """A wingless body standing upright facing the camera, with card wings."""
    import numpy as np

    def plan(m, bone, pivots, extras, parents):
        ranges = m["positions"].copy()
        m, bone = G.carve(m, bone, pivots, extras, parents, "head", "y", 0.58, "+", ranges=ranges)
        m, bone = G.carve(m, bone, pivots, extras, parents, "feet", "y", 0.14, "-", ranges=ranges)
        top = float(np.quantile(m["positions"][:, 1], 0.72))
        places, card = wings_on(m, bone, wing_png, 9.0, [-1.2, top, 0.4], [1.2, top, 0.4], 0.3, pivots, parents,
                                ("rightWing", "rightWingTip", "leftWing", "leftWingTip"), None)
        return G.attach(m, bone, card, places, pivots, parents, share=0.3)
    return plan


def ghast(rows=(0.33, 0.67), at: float = 0.42):
    """The body above a height, nine tentacles below, named by vanilla's three rows of three (front row 0-2, x from its right)."""
    def plan(m, bone, pivots, extras, parents):
        import numpy as np
        c0 = m["positions"].copy()
        lo, hi = c0.min(0), c0.max(0)

        def name(c):
            row = int(np.clip((c[2] - lo[2]) / (hi[2] - lo[2]) * 3, 0, 2))
            col = int(np.clip((c[0] - lo[0]) / (hi[0] - lo[0]) * 3, 0, 2))
            return f"tentacles_{row * 3 + col}"
        return G.pieces_past(m, bone, pivots, extras, parents, "y", at, "-", 9, name)
    return plan

