"""One mosaic of the drafts: a column per mob, a row per seed. Usage: contact.py out.jpg seed,seed name..."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path("work/models3d/mobs")
out, seeds, names = sys.argv[1], [int(s) for s in sys.argv[2].split(",")], sys.argv[3:]
px = 250
sheet = Image.new("RGB", (px * len(names), px * len(seeds)), (60, 60, 60))
d = ImageDraw.Draw(sheet)
for c, name in enumerate(names):
    for r, seed in enumerate(seeds):
        f = OUT / f"{name}_s{seed}.png"
        if f.exists():
            sheet.paste(Image.open(f).convert("RGB").resize((px, px)), (c * px, r * px))
        d.text((c * px + 4, r * px + 4), f"{name} {seed}", fill=(255, 255, 0))
sheet.save(out, quality=82)
print(out)
