"""Build the pack with the reskin project's tool. build.py: build/BubonicCreaturesDEV; --deploy also copies it into Minecraft's
development_resource_packs (that one folder only); --release: dist/BubonicCreatures_<version>.mcpack for other computers,
bumping pack.yaml's version (Minecraft won't re-import a version it has); --only GLOBS: just those textures, to stack over a full build;
--fetch-vanilla: get Mojang's reference pack, which the build needs (a link to the reskin project's copy if it has one)."""
import argparse
import sys
from pathlib import Path

import creatures  # noqa: F401  (registers the mob builders)
from mcreskin.build import PackBuilder, deploy
from mcreskin.vanilla import fetch_vanilla

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--deploy", action="store_true")
ap.add_argument("--release", action="store_true")
ap.add_argument("--only", help="comma-separated pack.yaml entries")
ap.add_argument("--fetch-vanilla", action="store_true")
a = ap.parse_args()
if a.fetch_vanilla:
    copy = ROOT.parent / "minecraft-reskin-project" / "vanilla"
    if (ROOT / "vanilla").exists():
        print("vanilla: already here")
    elif copy.is_dir():
        (ROOT / "vanilla").symlink_to(copy)
        print(f"vanilla: linked to {copy}")
    else:
        print(f"vanilla: {fetch_vanilla(ROOT)}")
    sys.exit(0)
if a.deploy and a.release:
    sys.exit("--deploy is for DEV builds; a release makes the .mcpack")
b = PackBuilder(root=ROOT)
res = b.build(variant="release" if a.release else "dev", release=a.release, previews=False, only=a.only.split(",") if a.only else None)
errors = [i for i in res.issues if i.level == "error"]
for i in errors[:20]:
    print(f"error: {i}")
print(f"pack folder: {res.pack_dir}")
if errors and not a.only:
    sys.exit(f"{len(errors)} lint error(s)")
if res.mcpack:
    print(f"mcpack: {res.mcpack}  (version {'.'.join(map(str, res.version))})")
if a.deploy:
    print(f"deployed to {deploy(res)}")
