"""The pack.yaml entries for the finished mesh mobs, inserted at the end of the textures section (or
replaced there, between the markers). Usage: pack_entries.py <name>..."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import convert as C  # noqa: E402

BEGIN = "  # Mesh mobs of the 2026-10-30 round (work/models3d/mobs; glbmodel.py). One line per mob.\n"
END = "  # End of the 2026-10-30 mesh mobs.\n"
SMOOTH = {"cod": 0.5, "salmon": 0.45, "pufferfish": 0.55, "tropicalfish": 0.45, "dolphin": 0.4, "squid": 0.4, "glow_squid": 0.4, "guardian": 0.5,
          "elder_guardian": 0.5, "tadpole": 0.45, "nautilus": 0.4, "frog": 0.5, "axolotl": 0.45, "turtle": 0.55, "iron_golem": 0.55,
          "silverfish": 0.5, "endermite": 0.5, "zombie": 0.85, "skeleton": 0.8, "snow_golem": 0.9, "sheep": 0.9, "pig": 0.8, "slime": 0.25, "shulker": 0.5, "copper_golem": 0.45, "blaze": 0.6,
          "ghast": 0.6, "happy_ghast": 0.6, "breeze": 0.6, "wither": 0.8, "warden": 0.6}


HUGE = {"ghast", "happy_ghast", "wither", "warden", "ravager", "iron_golem", "elder_guardian"}  # (the rest are at 1024: see the mob_model builder's `atlas`)


def entry(name: str) -> str:
    spec = C.fill(name)
    if spec.get("baby_from"):  # its model ships with the family's lead; its texture is its parent's
        return ""
    family = spec.get("family", name)
    entity = C.fill(family).get("entity", family)
    tail = ""
    if family == name:  # (a family's other models only add their geometry and texture)
        files = {f"entity/{entity}.entity.json": f"art/models/mesh/{entity}.entity.json",
                 f"render_controllers/{name}.mesh.render_controllers.json": f"art/models/mesh/{name}.mesh.render_controllers.json"}
        if (C.ART / f"{name}.mesh.animation.json").exists() and any(C.fill(n).get("anims") for n in C.family_of(name)):
            files[f"animations/{name}.mesh.animation.json"] = f"art/models/mesh/{name}.mesh.animation.json"
        for n in C.family_of(name):
            if C.fill(n).get("baby_from") and not C.fill(n).get("geo_of"):
                files[f"models/entity/{n}.mesh.geo.json"] = f"art/models/mesh/{n}.mesh.geo.json"
        body = ", ".join(f"{k}: {v}" for k, v in files.items())
        tail = f", files: {{{body}}}"
    if spec.get("no_dye") or spec.get("wool"):
        tail = ", color_mask: true" + tail
    line = lambda tex, pic, more: (f"  {tex}: {{builder: mob_model, model: art/models/mesh/{spec.get('geo_of', name)}.mesh.geo.json, picture: art/models/mesh/{pic}.png, "
                                   f"material: generic, roughness: {SMOOTH.get(name, SMOOTH.get(family, 0.85))}, max_size: 512, atlas: {2048 if family in HUGE else 1024}{more}}}\n")
    out = line(spec["texture"], spec["texture"].rsplit("/", 1)[-1], tail)
    if spec.get("glow"):
        out += line(spec["texture"].replace("_mesh", "_alt_mesh"), f"{name}_alt_mesh", "")
    return out


def older(s: str, names) -> str:
    """Without the hand-written entries of mobs now listed in the block (they start on one line and end at their closing braces)."""
    for name in names:
        m = re.search(rf"^  [^\n]+: \{{builder: mob_model, model: art/models/mesh/{name}\.mesh\.geo\.json.*?\n(?=  [a-z#])", s, re.S | re.M)
        if m:
            s = s[:m.start()] + s[m.end():]
    return re.sub(r"(?m)^  entity/dragon/dragon: \{builder: model_skin[^\n]*\n", "", s)


def main(names: list[str]) -> None:
    p = Path("pack.yaml")
    s = p.read_text(encoding="utf-8")
    names = names or [n for n in C.MOBS if (C.ART / f"{C.MOBS[n].get('geo_of', n)}.mesh.geo.json").exists()]
    lines = "".join(entry(n) for n in names)
    lines += ("  entity/dragon/dragon: {builder: mob_model, model: art/models/mesh/ender_dragon.mesh.geo.json, picture: art/models/mesh/ender_dragon_mesh.png, "
              "material: generic, roughness: 0.7, max_size: 512}\n")
    block = BEGIN + lines + END
    s = older(s, ("pig", "mooshroom", "goat", "hoglin", "llama", "polar_bear", "sheep"))
    if BEGIN in s:
        s = s[:s.index(BEGIN)] + block + s[s.index(END) + len(END):]
    else:
        i = s.index("\nblocks:")
        s = s[:i + 1] + block + s[i + 1:]
    p.write_text(s, encoding="utf-8")
    print(f"{len(names)} entries in pack.yaml")


if __name__ == "__main__":
    main(sys.argv[1:])
