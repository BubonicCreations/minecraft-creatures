"""A picture per face of a mob's model (2026-10-26, asked after mob_skin's materials "are not great").

Each face of the model's UV net is cut from vanilla's texture, enlarged and repainted by Qwen's
inpaint graph over its whole area (image to image at `denoise`) with a prompt for that body part,
so eyes, patches and clothes stay where the model shows them and neighbouring faces start from the
same colours. Candidates go to work/gen/mobfaces/<texture>/s<seed>/<face>.png with a preview of the
assembled texture; `pick` copies one seed's faces to art/mobfaces/<texture>/, which the `mob_faces`
builder pastes into the texture.

Recipe (recipes/mobfaces-*.yaml):
    denoise: 0.75            # how far Qwen may move from vanilla's face
    seeds: [2401, 2402]      # one full set of faces per seed ({seed: denoise} to vary it too)
    style: ...               # appended to every prompt
    textures:
      entity/cow/cow_v2:
        faces:               # face-name globs (bone.face, bone#i.face), first match wins
          head.north: {prompt: the face of a brown and white cow seen straight on, denoise: 0.85}
          "*": close-up of short dense dark brown hair with white patches   # never name the
                                     # animal or a garment here: Qwen draws a whole one per face
        keep: {from: 2401, faces: [head.north]}   # copied from that seed's set instead of repainted
                                     # (except: [globs] leaves some of those out; from: art takes
                                     # the picked set in art/mobfaces/)
        front: {bone: head, prompt: the face of a grey wolf seen straight from the front, denoise: 0.85}
                                     # the front of that bone's cubes (and its child bones') painted
                                     # as one picture and cut back into their north faces, so a snout
                                     # or nose box carries the face's own nose (2026-10-26: the wolf
                                     # had a whole face behind a second, separate nose)
"""
from __future__ import annotations

import fnmatch
import shutil
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

from mcreskin import imaging as I
from mcreskin.comfy import Comfy, load_graph, patch

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "comfy" / "qwen_seam_inpaint.json"
LONG_SIDE = 1024
MIN_TEXELS = 2
"""Faces narrower than this (horns, 1-texel rims) keep vanilla's softened texels."""
MIN_DRAWN = 0.15
REUSE_TOL = 0.004
"""A face whose vanilla texels differ from an already painted one (same prompt) by less than this
mean takes that picture: tame and angry wolves and tame cats share most of their faces."""


def folder_name(texture: str) -> str:
    return texture.replace("/", "__")


def faces_of(b, texture: str) -> list[tuple[str, int, int, int, int]]:
    """The model faces of `texture` worth a picture: at least MIN_TEXELS each way, and at least
    MIN_DRAWN of their texels drawn (outfit overlays cover part of a face; glow alpha counts)."""
    from mcreskin.build import _alpha_floor
    from .builders import _entity_geometry, _uv_rects

    van = b.van.image(texture)
    h, w = van.shape[:2]
    spec = b.specs.get(texture) or {}
    geo = spec.get("geometry") or _entity_geometry(b, texture, w, h)
    drawn = van[..., 3] > max(_alpha_floor(van), 0.5 / 255)
    out = []
    for face, x0, y0, x1, y1 in _uv_rects(b.van.geometry(geo), w, h):
        if min(x1 - x0, y1 - y0) >= MIN_TEXELS and drawn[y0:y1, x0:x1].mean() >= MIN_DRAWN:
            out.append((face, x0, y0, x1, y1))
    return out


def init_image(van: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Vanilla's face enlarged to LONG_SIDE (sides multiples of 16), clear texels filled with the
    face's mean colour, the pixel steps softened so Qwen repaints a surface, not a grid."""
    x0, y0, x1, y1 = rect
    sub = van[y0:y1, x0:x1].astype(np.float32)
    solid = sub[..., 3] > 0.5 / 255
    mean = sub[..., :3][solid].mean(0) if solid.any() else np.full(3, 0.5, np.float32)
    rgb = np.where(solid[..., None], sub[..., :3], mean)
    th, tw = rgb.shape[:2]
    s = LONG_SIDE / max(th, tw)
    gw, gh = max(16, round(tw * s / 16) * 16), max(16, round(th * s / 16) * 16)
    big = I.resize(rgb, (gw, gh), method="nearest")
    cell = gw / tw
    soft = ndimage.gaussian_filter(big, (0.15 * cell, 0.15 * cell, 0), mode="nearest")
    grain = np.random.default_rng(0).normal(0.0, 0.03, soft.shape[:2])[..., None]  # detail to latch onto
    return np.clip(soft + grain, 0, 1).astype(np.float32)


def repaint(comfy: Comfy, img: np.ndarray, prompt: str, seed: int, denoise: float, tmp: Path) -> np.ndarray:
    """One run of the inpaint graph over the whole picture: image to image."""
    h, w = img.shape[:2]
    rgba = np.concatenate([img, np.ones((h, w, 1), np.float32)], axis=-1)
    I.save_png(rgba, tmp / "face.png", mode="RGB")
    Image.new("RGB", (w, h), (255, 255, 255)).save(tmp / "face_mask.png")
    g = load_graph(GRAPH)
    g = patch(g, "1", image=comfy.upload_image(tmp / "face.png"))
    g = patch(g, "2", image=comfy.upload_image(tmp / "face_mask.png"))
    g = patch(g, "5", text=prompt)
    g = patch(g, "7", seed=int(seed), denoise=float(denoise))
    out = I.load_rgba(comfy.run(g, tmp, "face_out", node_ids=["10"])[0])
    if out.shape[:2] != (h, w):
        out = I.resize(out, (w, h), method="lanczos")
    return out[..., :3]


def prompt_for(faces: dict, face: str, style: str) -> tuple[str, float | None] | None:
    """(prompt, the face's own denoise or None) of the first glob matching `face`."""
    for pattern, entry in faces.items():
        if fnmatch.fnmatchcase(face, pattern):
            text, dn = (entry.get("prompt"), entry.get("denoise")) if isinstance(entry, dict) else (entry, None)
            return (f"{text}. {style}".strip(), dn) if text else None
    return None


def keep_faces(out_dir: Path, faces: list[str], keep: dict) -> None:
    """Copy the faces matching keep["faces"] globs (but not keep["except"]) from the sibling seed
    folder s<keep["from"]> (good faces of an earlier run), or from the picked set in art/mobfaces/
    with `from: art`, so `generate` repaints only the rest."""
    src_dir = (ROOT / "art" / "mobfaces" / out_dir.parent.name if keep.get("from") == "art"
               else out_dir.parent / f"s{keep.get('from')}")
    for face in faces:
        src = src_dir / f"{face}.png"
        wanted = any(fnmatch.fnmatchcase(face, g) for g in keep.get("faces", [])) \
            and not any(fnmatch.fnmatchcase(face, g) for g in keep.get("except", []))
        if wanted and src.exists() and not (out_dir / f"{face}.png").exists():
            shutil.copy2(src, out_dir / f"{face}.png")


def front_boxes(b, texture: str, geometry_id: str, bone: str) -> list[tuple[str, tuple, tuple, bool]]:
    """(face name, its texel rect, its box (x0, x1, y0, y1, z0) in model units, mirrored) of the north
    face of every cube of `bone` and the bones under it that the game shows at rest, untilted and
    with some texel drawn, named as _uv_rects names them ("" for a rect another cube already has,
    like a wolf's second ear: it is drawn, the first is cut back)."""
    from mcreskin.build import _alpha_floor
    from .builders import _uv_rects
    from .render import hidden_bones, setup_pose

    geo = b.van.geometry(geometry_id)
    pose = setup_pose(b, texture, geometry_id, {})
    van = b.van.image(texture)
    h, w = van.shape[:2]
    tw, th = geo.get("size") or (w, h)
    fx, fy = w / tw, h / th
    names = {f: r for f, *r in _uv_rects(geo, w, h)}
    hidden = hidden_bones(b, texture, geometry_id, {})
    kids = {}
    for bn in geo["bones"]:
        kids.setdefault(str(bn.get("parent")).lower(), []).append(bn)
    todo, out = [bn for bn in geo["bones"] if str(bn.get("name")).lower() == bone.lower()], []
    while todo:
        bn = todo.pop(0)
        todo += kids.get(str(bn.get("name")).lower(), [])
        turned = pose.get(str(bn.get("name")).lower(), {}).get("rotation", bn.get("rotation")) or (0, 0, 0)
        if str(bn.get("name")).lower() in hidden or (out and any(abs(float(r)) > 1 for r in turned)):
            continue  # a tilted part (piglin ears, a hat brim) is not part of the flat front
        for ci, cube in enumerate(bn.get("cubes", [])):
            uv, size = cube.get("uv"), cube.get("size", (0, 0, 0))
            if uv is None or cube.get("rotation"):
                continue
            part = bn.get("name") if ci == 0 else f"{bn.get('name')}#{ci}"
            if isinstance(uv, dict):
                if "north" not in uv:
                    continue
                fu, fv, fw, fh = (*uv["north"]["uv"], *uv["north"].get("uv_size", (0, 0)))
            else:
                sz = abs(float(size[2]))
                fu, fv, fw, fh = float(uv[0]) + sz, float(uv[1]) + sz, abs(float(size[0])), abs(float(size[1]))
            x0, x1 = sorted((round(fu * fx), round((fu + fw) * fx)))
            y0, y1 = sorted((round(fv * fy), round((fv + fh) * fy)))
            if x1 <= x0 or y1 <= y0 or not (van[y0:y1, x0:x1, 3] > max(_alpha_floor(van), 0.5 / 255)).any():
                continue
            inf = float(cube.get("inflate", 0.0))
            o = [float(c) for c in cube.get("origin", (0, 0, 0))]
            box = (o[0] - inf, o[0] + abs(float(size[0])) + inf, o[1] - inf, o[1] + abs(float(size[1])) + inf, o[2] - inf)
            mirror = bool(cube.get("mirror", bn.get("mirror", False)))
            name = f"{part}.north"
            out.append((name if names.get(name) == [x0, y0, x1, y1] else "", (x0, y0, x1, y1), box, mirror))
    return out


def front_canvas(van: np.ndarray, boxes: list) -> tuple[np.ndarray, list[tuple[str, tuple, bool]]]:
    """The cubes' north faces as the model shows them from straight in front, far ones first: a
    picture LONG_SIDE on its long side (sides multiples of 16), each face cut from vanilla (clear
    texels show what is behind), softened like init_image; and (face, its pixel box, mirrored)."""
    from mcreskin.build import _alpha_floor

    floor = max(_alpha_floor(van), 0.5 / 255)
    xs = [v for _, _, bx, _ in boxes for v in bx[:2]]
    ys = [v for _, _, bx, _ in boxes for v in bx[2:4]]
    ux, uy = max(xs) - min(xs), max(ys) - min(ys)
    s = LONG_SIDE / max(ux, uy)
    gw, gh = max(16, round(ux * s / 16) * 16), max(16, round(uy * s / 16) * 16)
    sx, sy = gw / ux, gh / uy
    main = max(boxes, key=lambda bx: (bx[2][1] - bx[2][0]) * (bx[2][3] - bx[2][2]))
    x0, y0, x1, y1 = main[1]
    solid = van[y0:y1, x0:x1, 3] > floor
    canvas = np.empty((gh, gw, 3), np.float32)
    canvas[:] = van[y0:y1, x0:x1, :3][solid].mean(0) if solid.any() else 0.5
    placed, cell = [], 0.0
    for name, (tx0, ty0, tx1, ty1), (bx0, bx1, by0, by1, _), mirror in sorted(boxes, key=lambda bx: -bx[2][4]):
        px0, px1 = round((bx0 - min(xs)) * sx), round((bx1 - min(xs)) * sx)
        py0, py1 = round((max(ys) - by1) * sy), round((max(ys) - by0) * sy)
        if px1 <= px0 or py1 <= py0:
            continue
        crop = van[ty0:ty1, tx0:tx1]
        crop = crop[:, ::-1] if mirror else crop
        big = I.resize(crop, (px1 - px0, py1 - py0), method="nearest", wrap=False)
        m = big[..., 3] > floor
        canvas[py0:py1, px0:px1][m] = big[..., :3][m]
        placed.append((name, (px0, py0, px1, py1), mirror))
        cell = max(cell, (px1 - px0) / max(1, tx1 - tx0))
    soft = ndimage.gaussian_filter(canvas, (0.15 * cell, 0.15 * cell, 0), mode="nearest")
    grain = np.random.default_rng(0).normal(0.0, 0.03, soft.shape[:2])[..., None]
    return np.clip(soft + grain, 0, 1).astype(np.float32), placed


def paint_front(comfy: Comfy, b, texture: str, van: np.ndarray, front: dict, style: str, seed: int,
                dn: float, out_dir: Path, log=print) -> set[str]:
    """The `front` group of a recipe texture: one picture of the bone's front, cut back into the
    north faces it shows that are not in out_dir yet. Returns the group's face names."""
    from .builders import _entity_geometry

    spec = b.specs.get(texture) or {}
    geo = spec.get("geometry") or _entity_geometry(b, texture, van.shape[1], van.shape[0])
    boxes = front_boxes(b, texture, geo, str(front.get("bone", "head")))
    group = {name for name, *_ in boxes if name}
    if not boxes or all((out_dir / f"{n}.png").exists() for n in group):
        return group
    img, placed = front_canvas(van, boxes)
    prompt = f"{front['prompt']}. {style}".strip()
    with tempfile.TemporaryDirectory() as tmp:
        pic = repaint(comfy, img, prompt, seed, float(front.get("denoise", dn)), Path(tmp))
    I.save_png(np.concatenate([pic, np.ones(pic.shape[:2] + (1,), np.float32)], -1), out_dir.parent / f"front_s{seed}.png", mode="RGB")
    for name, (px0, py0, px1, py1), mirror in placed:
        dest = out_dir / f"{name}.png"
        if not name or dest.exists():
            continue
        crop = pic[py0:py1, px0:px1]
        crop = crop[:, ::-1] if mirror else crop
        I.save_png(np.concatenate([crop, np.ones(crop.shape[:2] + (1,), np.float32)], -1), dest, mode="RGB")
    log(f"{texture} s{seed}: front of {front.get('bone', 'head')} painted as one picture ({len(group)} faces)")
    return group


def generate(recipe: str | Path, only: list[str] | None = None, comfy: Comfy | None = None, log=print) -> None:
    from mcreskin.build import PackBuilder

    cfg = yaml.safe_load(Path(recipe).read_text())
    b = PackBuilder(root=ROOT)
    comfy = comfy or Comfy()
    denoise, style = float(cfg.get("denoise", 0.75)), str(cfg.get("style", "")).strip()
    painted: dict[tuple, list[tuple[np.ndarray, Path]]] = {}  # (prompt, denoise, seed, shape) -> (vanilla crop, picture)
    for texture, tcfg in cfg["textures"].items():
        if only and texture not in only:
            continue
        van = b.van.image(texture)
        faces = [(f, r) for f, *r in faces_of(b, texture)]
        if not faces:
            log(f"{texture}: no face is drawn enough for a picture, skipped")
            continue
        seeds = cfg.get("seeds", [1])
        for seed, dn in (seeds.items() if isinstance(seeds, dict) else ((s, denoise) for s in seeds)):
            out_dir = ROOT / "work" / "gen" / "mobfaces" / folder_name(texture) / f"s{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            keep_faces(out_dir, [f for f, _ in faces], tcfg.get("keep") or {})
            if tcfg.get("front"):
                paint_front(comfy, b, texture, van, tcfg["front"], style, seed, dn, out_dir, log)
            for i, (face, rect) in enumerate(faces):
                found = prompt_for(tcfg["faces"], face, style)
                dest = out_dir / f"{face}.png"
                if found is None:
                    continue
                prompt, face_dn = found
                x0, y0, x1, y1 = rect
                crop, dn_face = van[y0:y1, x0:x1], float(face_dn if face_dn is not None else dn)
                same = painted.setdefault((prompt, dn_face, seed, crop.shape), [])
                twin = next((p for c, p in same if np.abs(c - crop).mean() < REUSE_TOL), None)
                if dest.exists() or twin is not None:
                    if not dest.exists():
                        shutil.copy2(twin, dest)
                    same.append((crop, dest))
                    continue
                with tempfile.TemporaryDirectory() as tmp:
                    img = repaint(comfy, init_image(van, rect), prompt, seed + i, dn_face, Path(tmp))
                I.save_png(np.concatenate([img, np.ones(img.shape[:2] + (1,), np.float32)], -1), dest, mode="RGB")
                same.append((crop, dest))
            from .builders import paste_faces

            preview = paste_faces(b, texture, b.specs.get(texture) or {}, 128, out_dir)
            preview[..., 3] = np.maximum(preview[..., 3], 0.0)
            I.save_png(preview, out_dir.parent / f"s{seed}.png")
            log(f"{texture} s{seed}: {len(faces)} faces in {out_dir}, preview {out_dir.parent / f's{seed}.png'}")


def pick(texture: str, seed: int, log=print) -> Path:
    """Copy one seed's faces to art/mobfaces/<texture>/ (replacing what was there)."""
    src = ROOT / "work" / "gen" / "mobfaces" / folder_name(texture) / f"s{seed}"
    pngs = sorted(src.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"no faces in {src}; run `tools/oldmobs.py faces <recipe>` first")
    dest = ROOT / "art" / "mobfaces" / folder_name(texture)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for p in pngs:
        shutil.copy2(p, dest / p.name)
    log(f"{len(pngs)} faces of {texture} s{seed} -> {dest}")
    return dest
