"""The mob builders and the face-picture method."""
from pathlib import Path

import numpy as np
import pytest

import creatures  # noqa: F401  (registers the builders)
from mcreskin import build as BLD

REAL_ROOT = Path(__file__).resolve().parents[1]
HAS_VANILLA = (REAL_ROOT / "vanilla" / "bedrock-samples-v1.26.50.4").exists()


def test_uv_rects_unfold_a_box_the_way_bedrock_does():
    from creatures.builders import _uv_rects

    head = {"bones": [{"name": "head", "cubes": [{"size": [8, 8, 8], "uv": [0, 0]}]},
                      {"name": "leg", "cubes": [{"size": [4, 12, 4], "uv": [0, 16]}, {"size": [4, 12, 4], "uv": [0, 16]}]}],
            "size": (64, 32)}
    rects = {f: r for f, *r in _uv_rects(head, 64, 32)}
    assert rects["head.north"] == [8, 8, 16, 16]  # the face of a player skin
    assert rects["head.up"] == [8, 0, 16, 8] and rects["head.south"] == [24, 8, 32, 16]
    assert rects["leg.north"] == [4, 20, 8, 32]
    assert len(_uv_rects(head, 64, 32)) == 12  # the second leg shares the first's UVs
    assert {f: r for f, *r in _uv_rects(head, 128, 64)}["head.north"] == [16, 16, 32, 32]  # HD texture


def test_mobfaces_prompts_and_init_pictures():
    import numpy as np
    from creatures.mobfaces import init_image, prompt_for

    faces = {"head.north": {"prompt": "A face", "denoise": 0.85}, "head#1.*": None, "*": "Fur"}
    assert prompt_for(faces, "head.north", "Sharp.") == ("A face. Sharp.", 0.85)
    assert prompt_for(faces, "head#1.east", "Sharp.") is None  # no picture: vanilla's texels stay
    assert prompt_for(faces, "leg0.west", "") == ("Fur.", None)
    van = np.zeros((32, 64, 4), np.float32)
    van[..., 3] = 1.0
    img = init_image(van, (16, 16, 20, 28))  # a 4 x 12 leg face
    assert img.shape[0] == 1024 and img.shape[1] % 16 == 0 and img.shape[2] == 3


def test_mobfaces_keep_copies_good_faces_of_an_earlier_seed(tmp_path):
    from creatures.mobfaces import keep_faces

    old, new = tmp_path / "s1", tmp_path / "s2"
    old.mkdir(), new.mkdir()
    for face in ("head.north", "head.east", "body.north"):
        (old / f"{face}.png").write_bytes(b"old")
    (new / "head.north.png").write_bytes(b"new")  # already repainted: stays
    (old / "head.west.png").write_bytes(b"old")
    keep_faces(new, ["head.north", "head.east", "head.west", "body.north"],
               {"from": 1, "faces": ["head.*"], "except": ["*.west"]})
    assert not (new / "head.west.png").exists()
    assert (new / "head.north.png").read_bytes() == b"new"
    assert (new / "head.east.png").read_bytes() == b"old"
    assert not (new / "body.north.png").exists()  # left for Qwen to repaint


def test_mobfaces_keep_from_art_copies_the_picked_set(tmp_path, monkeypatch):
    from creatures import mobfaces

    monkeypatch.setattr(mobfaces, "ROOT", tmp_path)
    art = tmp_path / "art" / "mobfaces" / "entity__wolf__wolf"
    art.mkdir(parents=True)
    (art / "head.east.png").write_bytes(b"picked")
    out = tmp_path / "work" / "gen" / "mobfaces" / "entity__wolf__wolf" / "s9"
    out.mkdir(parents=True)
    mobfaces.keep_faces(out, ["head.east", "head.north"], {"from": "art", "faces": ["*"], "except": ["head.north"]})
    assert (out / "head.east.png").read_bytes() == b"picked"
    assert not (out / "head.north.png").exists()


def test_mobfaces_front_canvas_puts_the_snout_in_front_of_the_face():
    """The wolf's snout box draws over the middle of its head's front, as the game shows it."""
    import numpy as np
    from creatures.mobfaces import front_canvas

    van = np.zeros((16, 16, 4), np.float32)
    van[0:6, 0:6] = (1, 0, 0, 1)      # head.north: red
    van[8:10, 0:2] = (0, 0, 1, 1)     # snout north: blue
    van[8:10, 0, :3] = (0, 1, 0)      # its left column green, to see the mirror
    boxes = [("head.north", (0, 0, 6, 6), (-3, 3, 0, 6, -3), False),
             ("head#1.north", (0, 8, 2, 10), (-1, 1, 0, 2, -5), True)]
    img, placed = front_canvas(van, boxes)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = dict((n, bx) for n, bx, _ in placed)["head#1.north"]
    assert (x0 + x1) / 2 == w / 2 and y1 == h                # the snout at the bottom middle
    assert img[h // 8, w // 2, 0] > 0.8                      # the face around and above it
    assert img[(y0 + y1) // 2, x0 + (x1 - x0) // 4, 2] > 0.8  # mirrored: blue left,
    assert img[(y0 + y1) // 2, x1 - (x1 - x0) // 4, 1] > 0.8  # its green column on the right


def test_mobfaces_paints_a_face_once_when_variants_share_it(tmp_path, monkeypatch):
    import numpy as np
    import yaml
    from types import SimpleNamespace
    from creatures import builders, mobfaces
    from mcreskin import build

    van = np.ones((8, 16, 4), np.float32)
    angry = van.copy()
    angry[0:4, 0:4, :3] = 0.2  # only the head front differs
    images = {"entity/wolf/wolf_tame": van, "entity/wolf/wolf_angry": angry}
    fake = SimpleNamespace(van=SimpleNamespace(image=images.__getitem__), specs={})
    monkeypatch.setattr(build, "PackBuilder", lambda root=None: fake)
    monkeypatch.setattr(builders, "paste_faces", lambda *a: np.zeros((8, 16, 4), np.float32))
    monkeypatch.setattr(mobfaces, "ROOT", tmp_path)
    monkeypatch.setattr(mobfaces, "faces_of", lambda b, t: [("head.north", 0, 0, 4, 4), ("body.north", 4, 0, 12, 8)])
    calls = []
    monkeypatch.setattr(mobfaces, "repaint", lambda c, img, p, s, d, tmp: calls.append(p) or np.zeros((16, 16, 3), np.float32))
    recipe = tmp_path / "r.yaml"
    recipe.write_text(yaml.safe_dump({"seeds": [1], "textures": {t: {"faces": {"*": "Fur"}} for t in images}}))
    mobfaces.generate(recipe, comfy=object(), log=lambda *_: None)
    assert len(calls) == 3  # both bodies are the same vanilla texels: one picture
    for t in images:
        assert (tmp_path / "work/gen/mobfaces" / mobfaces.folder_name(t) / "s1" / "body.north.png").exists()


@pytest.mark.skipif(not HAS_VANILLA, reason="vanilla not fetched")
def test_mob_model_atlas_caps_the_picture_but_a_tier_can_go_lower(tmp_path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (2048, 2048), (90, 60, 30)).save(tmp_path / "pic.png")
    (tmp_path / "m.geo.json").write_text("{}")
    cfg = BLD.load_config(REAL_ROOT)
    cfg.pop("full_detail", None)
    cfg["textures"]["entity/cow/test_mesh"] = {"builder": "mob_model", "material": "generic", "atlas": 1024,
                                               "picture": str(tmp_path / "pic.png"), "model": str(tmp_path / "m.geo.json")}
    root = BLD.PackBuilder(REAL_ROOT, cfg=cfg, log=lambda *a: None)
    assert root.at(512).get("entity/cow/test_mesh").color.shape[:2] == (1024, 1024)  # 2048 at this tier without the cap
    assert root.at(128).get("entity/cow/test_mesh").color.shape[:2] == (512, 512)
