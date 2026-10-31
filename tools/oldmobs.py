"""The older mob methods, kept for reference: a picture per face of vanilla's model, and vanilla's model drawn outside the game.
Usage: oldmobs.py faces <recipe> [--only TEXTURES] | oldmobs.py pick <texture> <seed> | oldmobs.py render [--only GLOBS] [--pack DIR] [--size 300] [--per-sheet 8] [--candidates SEED]"""
import argparse
from pathlib import Path

from creatures import mobfaces as M
from creatures import render as R

ap = argparse.ArgumentParser(description=__doc__)
sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("faces")
s.add_argument("recipe")
s.add_argument("--only")
s = sub.add_parser("pick")
s.add_argument("texture")
s.add_argument("seed", type=int)
s = sub.add_parser("render")
s.add_argument("--only")
s.add_argument("--pack")
s.add_argument("--size", type=int, default=300)
s.add_argument("--per-sheet", type=int, default=8)
s.add_argument("--candidates", type=int)
a = ap.parse_args()
if a.cmd == "faces":
    M.generate(a.recipe, only=a.only.split(",") if a.only else None)
elif a.cmd == "pick":
    M.pick(a.texture, a.seed)
else:
    R.render(a.only.split(",") if a.only else None, Path(a.pack) if a.pack else R.PACK, px=a.size, per_sheet=a.per_sheet, candidates=a.candidates)
