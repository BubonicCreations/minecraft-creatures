"""The mob builders, registered with the reskin project's build tool when this package is imported:
`mob_model` (our own models, what the pack ships) and the older methods `mob_faces` and `mob_skin`."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy import ndimage

from mcreskin import imaging as I
from mcreskin.build import Made, PackBuilder, _alpha_floor, _hex_rgb, register_builder, tex_sub


def _uv_rects(geometry: dict, w: int, h: int) -> list[tuple[str, int, int, int, int]]:
    """(bone.face, x0, y0, x1, y1) texel rectangles of every cube face of an entity model in a w x h
    texture: box UV ([u, v] + size) unfolded the way Bedrock (and Blockbench) unfold a cube, per-face
    UV as given, scaled from the model's texture size. Faces are named bone.face for a bone's first
    cube and bone#i.face for its i-th (the cow's muzzle: head#1.north). A rectangle shared by several
    cubes (legs) appears once, under the first."""
    tw, th = geometry.get("size") or (w, h)
    fx, fy = w / tw, h / th
    out, seen = [], set()
    for bone in geometry["bones"]:
        for ci, cube in enumerate(bone.get("cubes", [])):
            uv = cube.get("uv")
            if uv is None:
                continue
            part = bone.get("name") if ci == 0 else f"{bone.get('name')}#{ci}"
            if isinstance(uv, dict):
                faces = {f: (*fv["uv"], *fv.get("uv_size", (0, 0))) for f, fv in uv.items() if "uv" in fv}
            else:
                sx, sy, sz = (abs(float(s)) for s in cube.get("size", (0, 0, 0)))
                u, v = float(uv[0]), float(uv[1])
                faces = {"north": (u + sz, v + sz, sx, sy), "east": (u, v + sz, sz, sy),
                         "south": (u + 2 * sz + sx, v + sz, sx, sy), "west": (u + sz + sx, v + sz, sz, sy),
                         "up": (u + sz, v, sx, sz), "down": (u + sz + sx, v, sx, sz)}
            for face, (fu, fv, fw, fh) in faces.items():
                x0, x1 = sorted((round(fu * fx), round((fu + fw) * fx)))
                y0, y1 = sorted((round(fv * fy), round((fv + fh) * fy)))
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
                if x1 > x0 and y1 > y0 and (x0, y0, x1, y1) not in seen:
                    seen.add((x0, y0, x1, y1))
                    out.append((f"{part}.{face}", x0, y0, x1, y1))
    return out


@register_builder("mob_skin", keys={"geometry", "keep", "keep_within", "features", "scale", "realism", "detail", "seed"},
                  art_keys={"materials"}, wrap=False)
def _build_mob_skin(b: PackBuilder, name: str, spec: dict, size: int) -> Made:
    """A mob's texture (`entity/` keys) in real materials, at size / 16 px per vanilla texel. Each
    texel goes to the `materials` art ({art: [vanilla colours]}, or one art name for the whole mob)
    of its nearest colour (OKLab); a material is its seamless photo (`scale` texels per tile,
    default 8; `detail` contrast, default 1) carrying vanilla's local colour of that material, so
    hide patches, snouts and shirts stay where vanilla has them, with ragged soft edges instead of
    pixel steps. Texels within `keep_within` (OKLab, default 0.12) of a `keep` colour (default
    black and white) inside the `features` faces (bone.face of the model, default head.north) stay
    vanilla's crisp pixels: eyes, nostrils, mouths; so do glowing eyes (colour at a low alpha, which
    the game reads as glow). The model is `geometry`, or the one vanilla's entities draw this texture
    on. All smoothing stays inside one face of the model's UV net, so nothing bleeds across a seam.
    `realism` 0..1 (default 0.85): how much of vanilla's pixel shading gives way to the material's."""
    van = b.van.image(name).astype(np.float32)
    H, W = van.shape[:2]
    k = max(1, round(size / 16))
    floor = _alpha_floor(van)
    out = I.resize(I.bleed_rgb(van, alpha_thr=floor), (W * k, H * k), method="nearest").copy()
    opaque = van[..., 3] >= 0.5
    glow = (van[..., 3] > floor) & ~opaque
    if glow.sum() > 0.25 * (opaque | glow).sum():
        glow[:] = False  # the glow squid: a glowing body, not glowing eyes
    solid = opaque | glow
    cfg = spec["materials"]
    if isinstance(cfg, str):  # one material: every texel
        cfg = {cfg: ["#%02x%02x%02x" % tuple(np.round(van[..., :3][solid].mean(0) * 255).astype(int))]}
    mats = {str(m): [_hex_rgb(c) for c in (cs if isinstance(cs, list) else [cs])] for m, cs in cfg.items()}
    names = list(mats) + ["keep"]
    ref_mat = np.array([i for i, m in enumerate(mats) for _ in mats[m]])
    lab = I.rgb_to_oklab(van[..., :3])
    d = ((lab[:, :, None, :] - I.rgb_to_oklab(np.stack([c for m in mats for c in mats[m]]))[None, None]) ** 2).sum(-1)
    real = ref_mat[d.argmin(-1)]
    keep_ref = I.rgb_to_oklab(np.stack([_hex_rgb(c) for c in spec.get("keep", ["#000000", "#ffffff"])]))
    d_keep = ((lab[:, :, None, :] - keep_ref[None, None]) ** 2).sum(-1).min(-1)
    geo = spec.get("geometry") or _entity_geometry(b, name, W, H)
    rects = _uv_rects(b.van.geometry(geo), W, H)
    in_feature = np.zeros((H, W), bool)
    for face, x0, y0, x1, y1 in rects:
        if re.sub(r"#\d+", "", face) in spec.get("features", ["head.north"]):  # every cube of the bone
            in_feature[y0:y1, x0:x1] = True
    keep_t = (in_feature & (d_keep < d.min(-1)) & (d_keep < float(spec.get("keep_within", 0.12)) ** 2)) | glow
    label = np.where(keep_t, len(mats), real)
    means = [van[..., :3][(label == i) & solid].mean(0) if ((label == i) & solid).any() else mats[m][0]
             for i, m in enumerate(mats)]
    scale, detail = int(spec.get("scale", 8)), float(spec.get("detail", 1.0))
    tile_px = max(8, scale * k)

    def relief(m: str) -> np.ndarray:  # the photo's variation around its local mean, ~1.0
        art = I.resize(b._prepared_art(name, {"art": m}, size), tile_px, method="area")[..., :3]
        rel = art / np.maximum(ndimage.gaussian_filter(art, (tile_px / 6, tile_px / 6, 0), mode="wrap"), 1e-3)
        return np.clip(1.0 + detail * (rel - 1.0), 0.2, 3.0)

    reliefs = {m: b.shared(("mob_relief", m, tile_px, detail), lambda m=m: relief(m)) for m in mats}
    rng = np.random.default_rng(int(spec.get("seed", 0)) + 1)
    realism = float(spec.get("realism", 0.85))
    for _, x0, y0, x1, y1 in rects:
        sub, lab_sub, solid_sub = van[y0:y1, x0:x1, :3], label[y0:y1, x0:x1], solid[y0:y1, x0:x1]
        th, tw = lab_sub.shape
        rh, rw = th * k, tw * k
        yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
        jit = [ndimage.gaussian_filter(rng.standard_normal((rh, rw)).astype(np.float32), 0.6 * k) for _ in range(2)]
        jit = [j * (0.45 * k / (j.std() + 1e-6)) for j in jit]
        sharp = lab_sub[(yy // k).astype(int), (xx // k).astype(int)]
        ragged = lab_sub[np.clip((yy + 0.5 + jit[0]) // k, 0, th - 1).astype(int),
                         np.clip((xx + 0.5 + jit[1]) // k, 0, tw - 1).astype(int)]
        keep = len(mats)
        lab_up = np.where((sharp == keep) | (ragged == keep), sharp, ragged)
        soft = I.resize(sub, (rw, rh), method="nearest")
        acc, wsum = np.zeros((rh, rw, 3), np.float32), np.zeros((rh, rw), np.float32)
        for i in np.unique(lab_up):
            wgt = ndimage.gaussian_filter((lab_up == i).astype(np.float32), (0.12 if i == keep else 0.25) * k, mode="nearest")
            if i == keep:
                col = ndimage.gaussian_filter(soft, (0.12 * k, 0.12 * k, 0), mode="nearest")
            else:
                own = ((lab_sub == i) & solid_sub).astype(np.float32)
                num = ndimage.gaussian_filter(sub * own[..., None], (1.0, 1.0, 0), mode="nearest")
                den = ndimage.gaussian_filter(own, 1.0, mode="nearest")[..., None]
                local = np.where(den > 0.05, num / np.maximum(den, 1e-3), means[i])
                local = ndimage.gaussian_filter(I.resize(local, (rw, rh), method="nearest"), (0.5 * k, 0.5 * k, 0), mode="nearest")
                base = realism * local + (1 - realism) * ndimage.gaussian_filter(soft, (0.3 * k, 0.3 * k, 0), mode="nearest")
                rel = reliefs[names[i]]
                oy, ox = rng.integers(0, rel.shape[0]), rng.integers(0, rel.shape[1])
                rel = np.roll(rel, (oy, ox), axis=(0, 1))
                rel = np.tile(rel, (-(-rh // rel.shape[0]), -(-rw // rel.shape[1]), 1))[:rh, :rw]
                col = base * rel
            acc += wgt[..., None] * col
            wsum += wgt
        out[y0 * k:y1 * k, x0 * k:x1 * k, :3] = np.clip(acc / np.maximum(wsum, 1e-6)[..., None], 0.0, 1.0)
    return Made(I.bleed_rgb(b._grade(out, spec), alpha_thr=floor))


MOB_FACES_MAX_SIDE = 2048
"""mob_faces textures stay within this side (the dragon's 256 px texture at 32x would be 8192)."""


def paste_faces(b: PackBuilder, name: str, spec: dict, size: int, folder: Path) -> np.ndarray:
    """Vanilla's texture at size / 16 px per texel, its texels softened, with every picture
    <folder>/<face>.png scaled into its face of the model; alpha (and glow alpha) is vanilla's.
    `harmonize` (0..1, default 0.8) pulls each picture's broad colour (over ~a texel) to vanilla's
    softened face: faces painted one by one get one lighting, and vanilla's patches stay put."""
    van = b.van.image(name).astype(np.float32)
    H, W = van.shape[:2]
    k = max(1, min(round(size / 16), MOB_FACES_MAX_SIDE // max(H, W)))
    out = I.resize(I.bleed_rgb(van, alpha_thr=_alpha_floor(van)), (W * k, H * k), method="nearest").copy()
    out[..., :3] = ndimage.gaussian_filter(out[..., :3], (0.3 * k, 0.3 * k, 0), mode="nearest")
    geo = spec.get("geometry") or _entity_geometry(b, name, W, H)
    n = 0
    for face, x0, y0, x1, y1 in _uv_rects(b.van.geometry(geo), W, H):
        path = folder / f"{face}.png"
        if path.exists():
            pic = I.resize(I.load_rgba(path)[..., :3], ((x1 - x0) * k, (y1 - y0) * k), method="area")
            h = float(spec.get("harmonize", 0.8))
            if h > 0:
                sig = (k, k, 0)
                lo_v = ndimage.gaussian_filter(out[y0 * k:y1 * k, x0 * k:x1 * k, :3], sig, mode="nearest")
                lo_p = ndimage.gaussian_filter(pic, sig, mode="nearest")
                pic = pic * np.clip(lo_v / np.maximum(lo_p, 1e-3), 0.25, 4.0) ** h
            out[y0 * k:y1 * k, x0 * k:x1 * k, :3] = np.clip(pic, 0.0, 1.0)
            n += 1
    if n == 0:
        raise FileNotFoundError(f"{name}: no face pictures in {folder}; run `tools/oldmobs.py faces <recipe>` and "
                                f"`tools/oldmobs.py pick {name} <seed>`")
    return out


@register_builder("mob_faces", keys={"geometry", "faces", "harmonize"}, wrap=False)
def _build_mob_faces(b: PackBuilder, name: str, spec: dict, size: int) -> Made:
    """A mob's texture from a picture per face of its model (art/mobfaces/<texture>/<face>.png, or
    the folder `faces`, made by `tools/oldmobs.py faces`: Qwen repainting each of vanilla's faces), each
    scaled into its face at size / 16 px per vanilla texel; faces without one (horns, 1-texel rims)
    keep vanilla's texels, softened."""
    from .mobfaces import folder_name

    van = b.van.image(name)
    out = paste_faces(b, name, spec, size, b.root / "art" / "mobfaces" / str(spec.get("faces", folder_name(name))))
    return Made(I.bleed_rgb(b._grade(out, spec), alpha_thr=_alpha_floor(van)))


@register_builder("mob_model", keys={"model", "picture", "files", "color_mask", "atlas"}, wrap=False)
def _build_mob_model(b: PackBuilder, name: str, spec: dict, size: int) -> Made:
    """A mob on our own model (2026-10-26, mobmodel.py: the user wanted the shapes changed, not
    just the textures): `picture` is its texture atlas, made for the 512 setting and scaled by
    size / 512 below it; `model` the Bedrock geometry file the build ships to models/entity/, under
    the vanilla id it replaces (the game's animations still move its bones); `files` more files to
    ship ({pack path: source}): the mob's own animations and the client entity file that uses them.
    `color_mask`: keep the picture's alpha, which the mob's material reads as where its colour goes
    (the sheep: opaque wool takes the dye, near-clear skin keeps its own colour); else opaque. `atlas`: its
    longest side at most this many pixels, whatever the tier (with its normal map and MER a 2048 mob is about 64 MB of
    graphics memory and 7 MB of pack: 2026-10-31, 190 models made a 1.5 GB pack)."""
    pic = I.load_rgba(b.root / spec["picture"])
    model = b.root / spec["model"]
    if not model.is_file():
        raise FileNotFoundError(f"{name}: model {model} not found; see mobmodel.py")
    k = min(1.0, size / 512, int(spec.get("atlas", 1 << 16)) / max(pic.shape[:2]))
    if k < 1:
        pic = I.resize(pic, (max(16, round(pic.shape[1] * k)), max(16, round(pic.shape[0] * k))), method="area", wrap=False)
    if not spec.get("color_mask"):
        pic[..., 3] = 1.0
    return Made(b._grade(pic, spec))


def _entity_geometry(b: PackBuilder, name: str, w: int, h: int) -> str:
    """The model vanilla's entities draw texture `name` on: the first of its candidates whose texture
    size has the texture's shape (a 64x64 cow texture is not the old 64x32 cow's), else the first
    (llama spit's model declares another size; UVs scale by it, so the game stretches the texture)."""
    cands = [g for g in b.van.texture_geometries.get("textures/" + tex_sub(name), []) if g in b.van.geometries]
    if not cands:
        raise ValueError(f"{name}: no vanilla entity draws it on a model; give `geometry:`")
    fits = [g for g in cands if (s := _geometry_size(b.van.geometries[g]))[0] * h == s[1] * w]
    return (fits or cands)[0]


def _geometry_size(geometry: dict) -> tuple[float, float]:
    """The texture size a model's UVs count in: its own, else 64 x 32 (old models), or 64 x 64 when
    its box UVs reach below row 32 (the adult villager and sheep models: taking them for 64 x 32
    picked the baby's 32 x 32 model for their 64 x 64 textures, so faces went to the baby's layout)."""
    if geometry.get("size"):
        return tuple(geometry["size"])
    low = max((float(c["uv"][1]) + abs(float(c["size"][2])) + abs(float(c["size"][1]))
               for bone in geometry["bones"] for c in bone.get("cubes", [])
               if isinstance(c.get("uv"), list) and "size" in c), default=0.0)
    return (64, 64) if low > 32 else (64, 32)
