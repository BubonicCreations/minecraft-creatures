"""mcreskin render: mobs drawn outside the game, and the model facts it relies on."""
import json

import numpy as np

from creatures import render as R
from creatures.builders import _geometry_size
from mcreskin.vanilla import Vanilla


def _cube_model(**bone):
    return {"bones": [{"name": "body", "pivot": [0, 0, 0],
                       "cubes": [{"origin": [-1, 0, -1], "size": [2, 2, 2], "uv": [0, 0]}], **bone}], "size": (8, 8)}


def _skin(face_colours: dict[str, tuple]) -> R.Skin:
    """An 8 x 8 texture for a 2 x 2 x 2 box at uv (0, 0), each face's rectangle in its own colour."""
    tex = np.zeros((8, 8, 4), np.float32)
    rects = {"up": (2, 0), "down": (4, 0), "east": (0, 2), "north": (2, 2), "west": (4, 2), "south": (6, 2)}
    for face, (u, v) in rects.items():
        tex[v:v + 2, u:u + 2] = (*face_colours.get(face, (0.0, 0.0, 1.0)), 1.0)
    return R.Skin(tex, 0.5)


def _centre(geo, skin, yaw=0, pitch=0) -> np.ndarray:
    qs = R.quads(geo)
    img = R.draw(qs, skin, (8, 8), yaw, pitch, 32, np.array([0.0, 1.0, 0.0]), 2.0)
    return img[16, 16]


def test_render_front_view_shows_the_north_face():
    rgb = _centre(_cube_model(), _skin({"north": (1.0, 0.0, 0.0)}))
    assert rgb[0] > 0.3 and rgb[2] < 0.05


def test_render_top_view_shows_the_up_face():
    rgb = _centre(_cube_model(), _skin({"up": (0.0, 1.0, 0.0)}), pitch=89)
    assert rgb[1] > 0.3 and rgb[2] < 0.05


def test_render_bone_rotation_follows_bedrock_signs():
    """The cow's body: [90, 0, 0] about its pivot turns the box's north face to the belly."""
    geo = _cube_model(rotation=[90, 0, 0], pivot=[0, 1, 0])
    north = [q for q in R.quads(geo) if q[1][:2] == (2.0, 2.0)][0][0]
    normal = north[3] - north[0]
    assert normal[1] < -0.9  # faces down


def test_render_hidden_bones_are_not_drawn():
    assert R.quads(_cube_model(), hidden={"body"}) == []


def test_molang_reads_rest_values():
    assert R.molang("90 - this", 30) == 60
    assert R.molang("query.is_baby ? 0.0 : (-6.0 - this)", -6) == 0
    assert R.molang("math.cos(query.life_time * 20) * 9", 0) == 9
    assert R.molang("!query.is_sheared && true", 0) == 1
    assert R.molang("variable.thing(", 0, unread=1.0) == 1.0


def test_geometry_size_of_models_without_one():
    low = {"bones": [{"cubes": [{"uv": [0, 32], "size": [4, 12, 4]}]}]}
    high = {"bones": [{"cubes": [{"uv": [0, 0], "size": [4, 12, 4]}]}]}
    assert _geometry_size(low) == (64, 64)
    assert _geometry_size(high) == (64, 32)
    assert _geometry_size({"size": (32, 32), "bones": []}) == (32, 32)


def test_old_models_add_their_cubes_to_the_parents_bones(tmp_path):
    """geometry.sheep.v1.8:geometry.sheep.sheared.v1.8 is the shorn sheep plus wool, not wool alone."""
    models = tmp_path / "models" / "entity"
    models.mkdir(parents=True)
    leg = lambda y, h: {"origin": [0, y, 0], "size": [4, h, 4], "uv": [0, 16]}
    (models / "sheep.geo.json").write_text(json.dumps({
        "format_version": "1.8.0",
        "geometry.sheared": {"bones": [{"name": "leg0", "pivot": [0, 12, 0], "cubes": [leg(0, 12)]},
                                       {"name": "head", "cubes": [leg(16, 6)]}]},
        "geometry.wool:geometry.sheared": {"bones": [{"name": "leg0", "pivot": [0, 12, 0],
                                                      "cubes": [{**leg(6, 6), "inflate": 0.5}]}]}}))
    van = Vanilla.__new__(Vanilla)
    van.rp = tmp_path
    wool = {b["name"]: b for b in van.geometries["geometry.wool"]["bones"]}
    assert [c["size"][1] for c in wool["leg0"]["cubes"]] == [12, 6]
    assert "head" in wool
