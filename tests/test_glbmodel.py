"""glbmodel: AI-made 3D mobs turned into Bedrock polygon models with their own walk."""
import numpy as np
import pytest

from creatures import glbmodel as G


def test_place_turns_the_long_axis_onto_bedrock_forward():
    """A body lying along +x (the single-image workflow) ends up facing -z, feet at 0, 23 tall."""
    pos = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.9, 0.5, 0.1], [-0.9, 0.4, -0.1]])
    m = {"positions": pos, "normals": np.zeros_like(pos)}
    out = G.place(m, 23.0, forward=(1, 0, 0))["positions"]
    assert np.isclose(out[:, 1].min(), 0) and np.isclose(np.ptp(out[:, 1]), 23)
    assert out[1, 2] < out[0, 2]          # the +x end (the head) is now toward -z
    assert np.ptp(out[:, 2]) > 3 * np.ptp(out[:, 0])


def test_walk_folds_knees_while_the_leg_swings_forward():
    for t in np.linspace(0, 360 / 38.17, 12):
        a = G.walk_angles(t)
        assert abs(a["leg0"][0]) <= G.HIP_SWING + 1e-9
        assert abs(a["leg2_lower"][0]) <= G.KNEE_SWING + G.KNEE_FOLD + 1e-9
    anims = G.animation_file("cow.mesh")["animations"]
    assert set(anims["animation.cow.mesh.walk"]["bones"]) >= {"leg0", "leg0_lower", "leg3_lower"}


def test_entity_file_keeps_vanilla_for_other_variants():
    vanilla = {"minecraft:client_entity": {"description": {
        "identifier": "minecraft:cow", "animations": {"walk": "animation.quadruped.walk", "look_at_target": "x"},
        "scripts": {"animate": ["setup", {"walk": "query.modified_move_speed"}, "look_at_target"]}}}}
    out = G.entity_file(vanilla, "cow.mesh", "v.index == 0")
    desc = out["minecraft:client_entity"]["description"]
    assert desc["animations"]["mesh_walk"] == "animation.cow.mesh.walk"
    animate = desc["scripts"]["animate"]
    assert animate[0] == "setup"
    assert {"walk": "(v.index == 0) ? 0.0 : (query.modified_move_speed)"} in animate
    assert {"mesh_look_at": "v.index == 0"} in animate
    assert {"mesh_walk": f"(v.index == 0) ? math.min(1.0, (query.modified_move_speed) * {G.WALK_GAIN}) : 0.0"} in animate
    assert vanilla["minecraft:client_entity"]["description"]["scripts"]["animate"][1] == {"walk": "query.modified_move_speed"}


def test_mesh_for_variant_keeps_vanilla_geometry_for_other_variants():
    """The pig's variants share geometry.pig.v3: only the temperate adult gets the mesh."""
    entity = {"minecraft:client_entity": {"description": {
        "identifier": "minecraft:pig", "geometry": {"default": "geometry.pig.v3", "baby": "geometry.pig.baby"},
        "render_controllers": ["controller.render.saddle", "controller.render.pig.v4"]}}}
    controllers = {"format_version": "1.8.0", "render_controllers": {"controller.render.pig.v4": {
        "geometry": "query.is_baby ? Geometry.baby : Geometry.default", "textures": ["Array.default[v.index]"]}}}
    out, rc = G.mesh_for_variant(entity, controllers, "controller.render.pig.v4", "geometry.pig.mesh",
                                 "v.index == 0 && !query.is_baby")
    desc = out["minecraft:client_entity"]["description"]
    assert desc["geometry"] == {"default": "geometry.pig.v3", "baby": "geometry.pig.baby", "mesh": "geometry.pig.mesh"}
    assert desc["render_controllers"] == ["controller.render.saddle", "controller.render.pig.v4.mesh"]
    ours = rc["render_controllers"]["controller.render.pig.v4.mesh"]
    assert ours["geometry"] == "(v.index == 0 && !query.is_baby) ? Geometry.mesh : (query.is_baby ? Geometry.baby : Geometry.default)"
    assert ours["textures"] == ["Array.default[v.index]"]
    assert "mesh" not in entity["minecraft:client_entity"]["description"]["geometry"]  # vanilla's left as it was


def test_mesh_for_variant_can_bring_its_own_texture():
    entity = {"minecraft:client_entity": {"description": {
        "geometry": {"default": "geometry.sheep.v1.8"}, "textures": {"default": "textures/entity/sheep/sheep"},
        "render_controllers": ["controller.render.sheep.v2"]}}}
    controllers = {"render_controllers": {"controller.render.sheep.v2": {
        "geometry": "Geometry.default", "textures": ["query.is_baby ? Texture.baby : Texture.default"]}}}
    out, rc = G.mesh_for_variant(entity, controllers, "controller.render.sheep.v2", "geometry.sheep.mesh",
                                 "!query.is_sheared", texture="textures/entity/sheep/sheep_mesh")
    assert out["minecraft:client_entity"]["description"]["textures"]["mesh"] == "textures/entity/sheep/sheep_mesh"
    assert rc["render_controllers"]["controller.render.sheep.v2.mesh"]["textures"] == [
        "(!query.is_sheared) ? Texture.mesh : (query.is_baby ? Texture.baby : Texture.default)"]


def test_fit_to_vanilla_renames_legs_by_quarter_and_takes_vanilla_parents():
    geo = {"minecraft:geometry": [{"bones": [
        {"name": "root", "pivot": [0, 0, 0]},
        {"name": "body", "parent": "root", "pivot": [0, 10, 0]}, {"name": "head", "parent": "root", "pivot": [0, 12, -8]},
        {"name": "leg0", "parent": "root", "pivot": [-3, 6, 6]}, {"name": "leg1", "parent": "root", "pivot": [3, 6, 6]},
        {"name": "leg2", "parent": "root", "pivot": [-3, 6, -6]}, {"name": "leg3", "parent": "root", "pivot": [3, 6, -6]}]}]}
    hoglin = {"bones": [{"name": "body"}, {"name": "head", "parent": "body"},
                        {"name": "leg_back_right", "pivot": [6, 8, 17]}, {"name": "leg_back_left", "pivot": [-6, 8, 17]},
                        {"name": "leg_front_right", "pivot": [-6, 12, -3]}, {"name": "leg_front_left", "pivot": [6, 12, -3]}]}
    out = G.fit_to_vanilla(geo, hoglin)
    bones = {b["name"]: b for b in out["minecraft:geometry"][0]["bones"]}
    assert bones["leg_back_left"]["pivot"] == [-3, 6, 6] and bones["leg_back_right"]["pivot"] == [3, 6, 6]
    assert bones["leg_front_right"]["pivot"] == [-3, 6, -6] and bones["leg_front_left"]["pivot"] == [3, 6, -6]
    assert bones["head"]["parent"] == "body" and bones["leg_back_left"]["parent"] == "root"
    assert geo["minecraft:geometry"][0]["bones"][2]["parent"] == "root"  # the input is left as it was


def test_cancel_rest_moves_makes_this_equal_the_constant():
    """The sheep's setup "-6 - this" on a head hung from the root at 0 (pivot y 21): -27 before, 0 after."""
    from creatures import render as R
    geo = {"minecraft:geometry": [{"bones": [{"name": "root", "pivot": [0, 0, 0]},
                                             {"name": "head", "parent": "root", "pivot": [0, 21, -8]}]}]}
    piv_y = 21 - 0
    rest = {"head": {"position": [0.0, R.molang("-6.0 - this", piv_y), 0.0]}}
    assert rest["head"]["position"][1] == -27
    out = G.cancel_rest_moves(geo, rest)
    bones = {b["name"]: b for b in out["minecraft:geometry"][0]["bones"]}
    assert bones["head"]["parent"] == "head_anchor" and bones["head_anchor"]["parent"] == "root"
    this = bones["head"]["pivot"][1] - bones["head_anchor"]["pivot"][1]
    assert R.molang("-6.0 - this", this) == 0


def test_bone_mask_marks_the_body_and_fills_the_gaps_from_the_nearest_part():
    tri = lambda u0, u1: {"uvs": [[u0, 0.0], [u1, 0.0], [u0, 1.0]], "polys": [[[0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 2]]]}
    geo = {"minecraft:geometry": [{"bones": [{"name": "body", "poly_mesh": tri(0.0, 0.4)},
                                             {"name": "head", "poly_mesh": tri(0.6, 1.0)}]}]}
    m = G.bone_mask(geo, (20, 10), {"body"})
    assert m.shape == (10, 20)
    assert m[5, 1] == 1.0 and np.isclose(m[5, 13], 3 / 255)
    assert m[9, 0] == 1.0 and np.isclose(m[9, 12], 3 / 255)  # the bottom row: v 0 is the texture's bottom
    assert set(np.unique(m)) <= {np.float32(1.0), np.float32(3 / 255)}


def test_variant_animation_edits_only_the_mesh_variant():
    from creatures import render as R
    grazing = {"animation_length": 2.0, "bones": {"head": {
        "position": {"0": [0.0, 0.0, 0.0], "0.2": [0.0, "query.is_baby ? -2.5 : -9.0", 0.0]},
        "rotation": {"0.2": {"pre": [36.0, 0.0, 0.0], "post": ["180.0 * 0.2", 0.0, 0.0]}}}}}
    out = G.variant_animation(grazing, "!query.is_sheared", lambda e, b, ch, i: "0.0" if ch == "position" else None)
    y = out["bones"]["head"]["position"]["0.2"][1]
    assert y == "(!query.is_sheared) ? (0.0) : (query.is_baby ? -2.5 : -9.0)"
    assert R.molang(y, 0.0) == 0.0  # read as a woolly adult: no drop
    assert out["bones"]["head"]["rotation"] == grazing["bones"]["head"]["rotation"]
    assert grazing["bones"]["head"]["position"]["0.2"][1] == "query.is_baby ? -2.5 : -9.0"


def test_cut_plane_splits_crossing_triangles_and_keeps_their_winding():
    m = {"positions": np.array([[0.0, 0, 0], [2.0, 0, 0], [0.0, 2, 0], [3.0, 3, 0]]), "normals": np.tile([0.0, 0, 1], (4, 1)),
         "uvs": np.array([[0.0, 0], [1.0, 0], [0.0, 1], [1.0, 1]]), "tris": np.array([[0, 1, 2], [1, 3, 2]])}
    out = G.cut_plane(m, 0, 1.0, where=lambda q: q[:, 1].max() < 2.5)  # only the first triangle
    t, p = out["tris"], out["positions"]
    assert len(t) == 1 + 3 and [1, 3, 2] in t.tolist()
    c = p[t].mean(1)
    assert all((c[:, 0] < 1) | (c[:, 0] > 1))  # none straddles x = 1 now
    area = lambda tri: np.cross(p[tri[1]] - p[tri[0]], p[tri[2]] - p[tri[0]])[2]
    assert all(area(tri) > 0 for tri in t)  # same winding as the input
    assert np.isclose(sum(area(tri) for tri in t if tri.tolist() != [1, 3, 2]), area([0, 1, 2]) if False else 4.0)
    k = out["uvs"][len(m["positions"]):]
    assert len(k) == 2 and np.allclose(sorted(k[:, 0]), [0.5, 0.5])


def _box(n=6):
    """A closed box 2 wide, 2 tall, 6 long (facing -z), with shared corners and a UV per point, its long
    sides split into `n` so a plane across it crosses triangles."""
    zs = np.linspace(-3, 3, n + 1)
    ring = [(-1.0, 0.0), (1.0, 0.0), (1.0, 2.0), (-1.0, 2.0)]
    pos = np.array([(x, y, z) for z in zs for x, y in ring], float)
    tris = []
    for i in range(n):
        for k in range(4):
            a, b, c, d = i * 4 + k, i * 4 + (k + 1) % 4, (i + 1) * 4 + (k + 1) % 4, (i + 1) * 4 + k
            tris += [[a, b, c], [a, c, d]]
    return {"positions": pos, "normals": np.tile([0.0, 1.0, 0.0], (len(pos), 1)), "uvs": np.zeros((len(pos), 2)),
            "tris": np.array(tris), "images": {}}


def test_carve_takes_what_lies_past_a_plane_and_pivots_on_its_cut():
    m = _box()
    bone, pivots, extras, parents = np.array(["body"] * len(m["tris"]), dtype=object), {}, {}, {}
    m2, b2 = G.carve(m, bone, pivots, extras, parents, "tail", "z", 0.75, "+")
    c = m2["positions"][m2["tris"]].mean(1)
    assert (c[b2 == "tail", 2] > 1.5 - 1e-6).all() and (c[b2 == "body", 2] < 1.5 + 1e-6).all()
    assert np.allclose(pivots["tail"], [0, 1, 1.5], atol=1e-6)
    assert extras["tail"][0] == (0.0, 0.0, -1.0)   # its joint bulges back into the body
    m3, b3 = G.carve(m2, b2, pivots, extras, parents, "tip", "z", 0.9, "+", of="tail")
    assert parents["tip"] == "tail" and (b3 == "tip").any() and (b3 == "tail").any()


def test_geometry_closes_a_chain_of_carved_parts_against_their_parents():
    m = _box()
    bone, pivots, extras, parents = np.array(["body"] * len(m["tris"]), dtype=object), {}, {}, {}
    m, bone = G.carve(m, bone, pivots, extras, parents, "tail", "z", 0.6, "+")
    m, bone = G.carve(m, bone, pivots, extras, parents, "tip", "z", 0.8, "+", of="tail")
    pivots["body"] = [0.0, 1.0, 0.0]
    geo = G.geometry(m, bone, pivots, "geometry.t", (4, 4), extras, parents={"body": "root", **parents}, seal=True, balls=True)
    bones = {b["name"]: b for b in geo["minecraft:geometry"][0]["bones"]}
    own = lambda name: int((bone == name).sum())
    assert len(bones["tail"]["poly_mesh"]["polys"]) > own("tail")   # its dome and the cup for the tip
    assert len(bones["tip"]["poly_mesh"]["polys"]) > own("tip")     # its dome
    assert bones["tip"]["parent"] == "tail"


def test_rest_pose_keeps_the_front_legs_on_the_ground_when_sitting():
    from creatures import render as R

    bones = [{"name": "root", "pivot": [0, 0, 0]}, {"name": "body", "parent": "root", "pivot": [0, 12, 0], "poly_mesh": {}},
             {"name": "head", "parent": "body", "pivot": [0, 14, -10], "poly_mesh": {}}]
    bones += [{"name": n, "parent": "root", "pivot": p, "poly_mesh": {"polys": [1]}} for n, p in
              (("leg0", [-2, 8, 8]), ("leg1", [2, 8, 8]), ("leg2", [-2, 8, -8]), ("leg3", [2, 8, -8]))]
    pose = G.rest_pose({"minecraft:geometry": [{"bones": bones}]})
    rb = R._rot(pose["body"]["rotation"])
    world = lambda q: rb @ (np.array(q, float) - [0, 12, 0]) + [0, 12, 0] + pose["body"]["position"]
    shoulder = world([2, 8, -8])
    assert np.isclose(shoulder[1], 8.0, atol=0.01)                 # the shoulder is still a leg's length up
    assert np.isclose(world([2, 8, 8])[1], 8 - 16 * np.sin(np.radians(-pose["body"]["rotation"][0])), atol=0.01)
    assert pose["leg3"]["rotation"][0] == 0.0 and pose["leg1"]["rotation"][0] == -80.0
    assert np.allclose(pose["leg3"]["position"], shoulder - [2, 8, -8], atol=0.01)   # a root leg goes where its shoulder went


def test_pose_animation_is_for_the_mesh_only_and_handles_keyframes():
    anim = {"bones": {"body": {"rotation": ["45 - this", 0, 0]}, "tail": {"rotation": {"0.0": [1, 2, 3], "1.0": {"pre": [4, 5, 6], "post": [7, 8, 9]}}}}}
    out = G.pose_animation(anim, "v", {"body": {"rotation": [-20.0, 0.0, 0.0]}, "head": {"rotation": [20.0, 0.0, 0.0]}})
    assert out["bones"]["body"]["rotation"][0] == "(v) ? (-20.0) : (45 - this)"
    assert out["bones"]["tail"]["rotation"]["1.0"]["post"][2] == "(v) ? (0.0) : (9)"
    assert out["bones"]["head"]["rotation"][0] == "(v) ? (20.0) : (0.0)"


def test_add_saddle_is_a_raised_piece_of_the_back_with_its_own_texels():
    m = _box()
    bone = np.array(["body"] * len(m["tris"]), dtype=object)
    m = {**m, "uvs": np.tile([0.02, 0.02], (len(m["positions"]), 1))}
    geo = G.geometry(m, bone, {"body": [0.0, 1.0, 0.0]}, "geometry.t", (64, 64))
    tex = np.zeros((64, 64, 4), np.float32)
    out, painted = G.add_saddle(geo, tex, "Saddle")
    bones = {b["name"]: b for b in out["minecraft:geometry"][0]["bones"]}
    pm = bones["Saddle"]["poly_mesh"]
    assert bones["Saddle"]["parent"] == "body" and pm["polys"]
    assert np.ptp(np.array(pm["positions"])[:, 2]) < 6.0    # only part of the body's length
    uv = np.array(pm["uvs"])
    t = painted[(1 - uv[:, 1]).clip(0, 0.999).__mul__(64).astype(int), (uv[:, 0].clip(0, 0.999) * 64).astype(int)]
    assert (t[:, :3].sum(1) > 0).all()                      # every corner reads the leather


def test_mesh_for_variant_can_override_layers_and_material():
    entity = {"minecraft:client_entity": {"description": {"geometry": {"default": "g"}, "textures": {"a": "t"}, "materials": {"default": "fish"},
                                                          "render_controllers": ["rc"]}}}
    rcs = {"render_controllers": {"rc": {"geometry": "Geometry.default", "textures": ["Texture.a", "Texture.b"], "materials": [{"*": "Material.default"}]}}}
    out, rc = G.mesh_for_variant(entity, rcs, "rc", "geometry.m", "v", texture="textures/m", layers={1: "Texture.none"}, suffix="fish_mesh", material="cod")
    copy = rc["render_controllers"]["rc.fish_mesh"]
    assert copy["textures"][1] == "(v) ? Texture.none : (Texture.b)"
    assert copy["materials"][0]["*"] == "(v) ? Material.mesh : (Material.default)"
    assert out["minecraft:client_entity"]["description"]["materials"]["mesh"] == "cod"


def test_keep_bones_hangs_items_from_our_arm_at_vanillas_offset():
    geo = {"minecraft:geometry": [{"bones": [{"name": "root", "pivot": [0, 0, 0]}, {"name": "rightArm", "parent": "root", "pivot": [-8, 20, 0]}]}]}
    vanilla = {"bones": [{"name": "rightArm", "pivot": [-5, 22, 0]}, {"name": "rightItem", "parent": "rightArm", "pivot": [-6, 15, 1]}]}
    bones = G.keep_bones(geo, vanilla, ["rightItem"])["minecraft:geometry"][0]["bones"]
    item = next(b for b in bones if b["name"] == "rightItem")
    assert item["parent"] == "rightArm" and item["pivot"] == [-9.0, 13.0, 1.0]


def test_fit_to_vanilla_leaves_more_than_four_legs_to_their_own_names():
    """Quarters can't name a spider's eight legs: renaming by them gave two bones one name."""
    legs = [("leg0", -4, 2), ("leg1", 4, 2), ("leg6", -4, -1), ("leg7", 4, -1), ("leg2", -4, 1), ("leg3", 4, 1), ("leg4", -4, 0), ("leg5", 4, 0)]
    geo = {"minecraft:geometry": [{"bones": [{"name": "root", "pivot": [0, 0, 0]}] + [
        {"name": n, "parent": "root", "pivot": [x, 9, z]} for n, x, z in legs]}]}
    vanilla = {"bones": [{"name": n, "pivot": [x, 9, z]} for n, x, z in legs]}
    bones = G.fit_to_vanilla(geo, vanilla)["minecraft:geometry"][0]["bones"]
    assert sorted(b["name"] for b in bones if b["name"] != "root") == sorted(n for n, *_ in legs)


def test_card_follows_the_cut_out_outline_on_both_sides():
    pic = np.zeros((64, 128, 4), np.uint8)
    pic[16:48, 32:96] = (200, 100, 50, 255)  # a 64 x 32 block in the middle
    m = G.card(pic, size=16.0, cells=32)
    used = m["positions"][np.unique(m["tris"])]
    assert np.ptp(used[:, 0]) == pytest.approx(16.0)
    assert np.ptp(used[:, 2]) == pytest.approx(8.0, abs=0.6)
    assert np.ptp(used[:, 1]) < 0.1  # flat
    assert len(m["tris"]) % 2 == 0  # each triangle twice, once per side
    uv = m["uvs"][np.unique(m["tris"])]
    assert 0.2 < uv[:, 0].min() and uv[:, 0].max() < 0.8


def test_attach_shares_the_texture_and_mirrors_the_winding():
    base = {"positions": np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), "normals": np.array([[0.0, 0, 1]] * 3), "uvs": np.array([[0.0, 0], [1, 0], [0, 1]]),
            "tris": np.array([[0, 1, 2]]), "images": {"color": np.full((8, 8, 3), 10, np.uint8)}}
    piece = {**base, "images": {"color": np.full((8, 8, 3), 200, np.uint8)}}
    mirror = np.diag([-1.0, 1.0, 1.0, 1.0])
    pivots, parents = {}, {}
    m, bone = G.attach(base, np.array(["body"], dtype=object), piece, [("a", "root", (0, 0, 0), np.eye(4)), ("b", "a", (1, 2, 3), mirror)], pivots, parents, share=0.5)
    assert list(bone) == ["body", "a", "b"]
    assert parents == {"a": "root", "b": "a"} and pivots["b"] == [1.0, 2.0, 3.0]
    assert m["images"]["color"].shape == (8, 8, 3)
    assert m["images"]["color"][4, 1, 0] == 10 and m["images"]["color"][4, 6, 0] == 200
    assert m["uvs"][:3, 0].max() == pytest.approx(0.5) and m["uvs"][3:, 0].min() == pytest.approx(0.5)
    assert list(m["tris"][2]) == [8, 7, 6]  # mirrored: wound the other way


def test_plain_molang_takes_out_the_always_condition():
    plain = G.plain_molang
    assert plain("(true) ? Geometry.mesh : (Geometry.default)") == "Geometry.mesh"
    assert plain("(true) ? ((q.a) ? Texture.mesh_alt : Texture.mesh) : (Texture.default)") == "((q.a) ? Texture.mesh_alt : Texture.mesh)"
    assert plain("(true) ? (math.clamp(0.0, -30, 30)) : (q.x * (2 + 1))") == "(math.clamp(0.0, -30, 30))"
    assert plain("(q.a) ? X : ((true) ? Y : (Z))") == "(q.a) ? X : (Y)"
    assert plain("(true) || (q.casting || q.celebrating)") == "1.0"
    assert plain("(q.a) && (!(true))") == "(q.a) && (0.0)"
    assert plain("!(true) || (q.a)") == "0.0 || (q.a)"
    assert plain("true") == "1.0"
    assert plain({"animate": [{"walk": "!(true)"}, "look"], "loop": True, "n": 3}) == {"animate": [{"walk": "0.0"}, "look"], "loop": True, "n": 3}
    vanilla = "v.use_dedicated_baby_armor = true; v.d = (!(v.last ?? true) && q.x) ? 2 : 0;"
    assert plain(vanilla) == vanilla


def _entity(fmt, **description):
    return {"format_version": fmt, "minecraft:client_entity": {"description": {"identifier": "minecraft:m", **description}}}


def test_entity_for_game_newer_formats_lose_animation_controllers():
    out = G.entity_for_game(_entity("1.10.0", animation_controllers=[{"move": "controller.animation.m.move"}], render_controllers=["a", {"b": "q.c"}]))
    desc = out["minecraft:client_entity"]["description"]
    assert "animation_controllers" not in desc and desc["render_controllers"] == ["a", {"b": "q.c"}] and out["format_version"] == "1.10.0"


def test_entity_for_game_old_format_with_constant_conditions_stays_old():
    out = G.entity_for_game(_entity("1.8.0", animation_controllers=[{"move": "c.move"}], render_controllers=[{"clothes": "0.0"}, "mesh", {"glow": "1.0"}]))
    desc = out["minecraft:client_entity"]["description"]
    assert out["format_version"] == "1.8.0" and desc["render_controllers"] == ["mesh", "glow"] and desc["animation_controllers"] == [{"move": "c.move"}]


def test_entity_for_game_old_format_with_a_real_condition_moves_to_the_newer_one():
    entity = _entity("1.8.0", animations={"walk": "animation.m.walk"}, animation_controllers=[{"move": "c.move"}, {"look": "c.look"}], render_controllers=["mesh", {"mask": "!(q.a)"}],
                     scripts={"pre_animation": ["v.x = 1;"]})
    out = G.entity_for_game(entity, min_engine="1.26.1")
    desc = out["minecraft:client_entity"]["description"]
    assert out["format_version"] == "1.10.0" and "animation_controllers" not in desc and desc["min_engine_version"] == "1.26.1"
    assert desc["animations"] == {"walk": "animation.m.walk", "move_controller": "c.move", "look_controller": "c.look"}
    assert desc["scripts"] == {"pre_animation": ["v.x = 1;"], "animate": ["move_controller", "look_controller"]}
    assert entity["format_version"] == "1.8.0"  # the input is left alone


def test_edited_animations_drop_a_curve_that_needs_constant_keyframes():
    anim = {"bones": {"body": {"rotation": {"0.5": {"post": [1.0, 0.0, 0.0], "lerp_mode": "catmullrom"}}}}}
    walked = G.variant_animation(anim, "q.v", lambda e, b, c, i: f"{e} * 2")["bones"]["body"]["rotation"]["0.5"]
    assert list(walked) == ["post"] and walked["post"][0] == "(q.v) ? (1.0 * 2) : (1.0)"
    posed = G.pose_animation(anim, "q.v", {})["bones"]["body"]["rotation"]["0.5"]
    assert list(posed) == ["post"]


def test_meshes_for_variants_draws_the_first_match_and_vanilla_elsewhere():
    entity = {"minecraft:client_entity": {"description": {"geometry": {"default": "geometry.cat"}, "textures": {"white": "t/white"}, "materials": {"default": "cat"},
                                                          "render_controllers": ["controller.render.cat"]}}}
    rc = {"format_version": "1.8.0", "render_controllers": {"controller.render.cat": {"geometry": "Geometry.default", "materials": [{"*": "Material.default"}], "textures": ["Array.skins[query.variant]"]}}}
    members = [{"key": "mesh", "geometry": "geometry.cat.mesh", "variant": "query.variant == 8", "texture": "t/tabby"},
               {"key": "mesh_black", "geometry": "geometry.cat_black.mesh", "variant": "query.variant == 1", "texture": "t/black", "material": "plain"}]
    out, copy = G.meshes_for_variants(entity, rc, "controller.render.cat", members, {"walk": "animation.cat.mesh.walk"})
    desc = out["minecraft:client_entity"]["description"]
    assert desc["render_controllers"] == ["controller.render.cat.mesh"]
    assert desc["geometry"]["mesh_black"] == "geometry.cat_black.mesh" and desc["textures"]["mesh"] == "t/tabby"
    assert desc["animations"]["walk"] == "animation.cat.mesh.walk"
    made = copy["render_controllers"]["controller.render.cat.mesh"]
    assert made["geometry"] == "(query.variant == 8) ? Geometry.mesh : ((query.variant == 1) ? Geometry.mesh_black : (Geometry.default))"
    assert made["textures"][0] == "(query.variant == 8) ? Texture.mesh : ((query.variant == 1) ? Texture.mesh_black : (Array.skins[query.variant]))"
    assert made["materials"][0]["*"] == "(query.variant == 1) ? Material.mesh_black : (Material.default)"
    assert entity["minecraft:client_entity"]["description"]["geometry"] == {"default": "geometry.cat"}  # the input is left alone


def test_plain_side_paints_out_one_side_only():
    geo = {"minecraft:geometry": [{"bones": [{"name": "body", "poly_mesh": {
        "positions": [[0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, -1], [1, 0, -1], [0, 1, -1]],
        "normals": [[0, 0, 1]] * 3 + [[0, 0, -1]] * 3,
        "uvs": [[0, 0], [0.5, 0], [0, 1], [0.5, 0], [1, 0], [0.5, 1]],
        "polys": [[[0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 2]], [[3, 3, 3], [4, 4, 4], [5, 5, 5], [5, 5, 5]]]}}]}]}
    tex = np.full((64, 64, 4), 0.5, np.float32)
    tex[45:50, 2:6, :3] = 0.0  # a dark eye on the +z triangle's texels (u < 0.5)
    tex[45:50, 34:38, :3] = 0.0  # and one on the -z triangle's
    out = G.plain_side(geo, tex, lambda c, n: n[2] > 0.3, sigma=2.0, detail=0.0)
    assert out[47, 3, 0] == pytest.approx(0.5)
    assert out[47, 35, 0] == 0.0


def test_rod_stands_on_the_y_axis_with_its_strip_for_a_texture():
    pic = np.zeros((100, 100, 3), np.uint8)
    pic[10:90, 45:55] = (220, 180, 40)
    r = G.rod(pic, 8.0, 1.0)
    assert r["positions"][:, 1].min() == 0.0 and r["positions"][:, 1].max() == 8.0
    assert np.abs(r["positions"][:, [0, 2]]).max() == pytest.approx(1.0)
    assert r["images"]["color"].shape[:2] == (80, 10)
    assert r["uvs"].min() >= 0.0 and r["uvs"].max() <= 1.0
    a, b, c = r["positions"][r["tris"][30]]  # wound outward
    n = np.cross(b - a, c - a)
    assert n @ np.array([(a + b + c)[0], 0.0, (a + b + c)[2]]) > 0


def test_variant_animation_wraps_a_scale_given_as_one_number():
    anim = {"bones": {"head": {"position": [0.0, 4.0, 4.0], "scale": 2.0}}}
    out = G.variant_animation(anim, "query.is_baby", lambda e, b, ch, i: "1.0" if ch == "scale" else "0.0")
    assert out["bones"]["head"]["scale"][0] == "(query.is_baby) ? (1.0) : (2.0)"
    assert out["bones"]["head"]["position"][1] == "(query.is_baby) ? (0.0) : (4.0)"


def test_position_map_finds_each_texels_point():
    geo = {"minecraft:geometry": [{"bones": [{"name": "body", "poly_mesh": {"positions": [[0, 0, 0], [8, 0, 0], [0, 8, 0]], "uvs": [[0, 1], [1, 1], [0, 0]], "polys": [[[0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 2]]]}}]}]}
    out = G.position_map(geo, (8, 8))
    assert out[0, 0, 0] == pytest.approx(0.5, abs=0.01) and out[0, 0, 1] == pytest.approx(0.5, abs=0.01)  # v counts up: the texture's top row is the point's y 0
    assert np.isnan(out[7, 7, 0])

