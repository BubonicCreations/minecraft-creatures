"""Checks the mod's entity files against the Minecraft actually installed, whose files can differ from Mojang's
samples: the rules of each file's format, and min_engine_version (a higher number in the game's file beats ours,
with no error). Run after a game update; the game's own word is its content log (Settings > Creator).
Usage: gamecheck.py [pack_folder]"""
import glob
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
ASK = "(Get-AppxPackage Microsoft.MinecraftUWP).InstallLocation"  # (WindowsApps can't be listed, only entered)
HOME = subprocess.run([POWERSHELL, "-NoProfile", "-Command", ASK], capture_output=True, text=True, check=True).stdout.strip()
GAME = subprocess.run(["wslpath", "-u", HOME], capture_output=True, text=True, check=True).stdout.strip() + "/data/resource_packs"
PACK = Path(sys.argv[1] if len(sys.argv) > 1 else "art/models/mesh")


def archive(path: str) -> dict:
    """A .brarchive: a 16 byte header, then 256 bytes per entry (name length, name, at 248 its offset and length)."""
    b = Path(path).read_bytes()
    count = struct.unpack_from("<I", b, 8)[0]
    out = {}
    for i in range(count):
        at = 16 + i * 256
        off, size = struct.unpack_from("<II", b, at + 248)
        start = 16 + count * 256 + off
        out[b[at + 1:at + 1 + b[at]].decode()] = b[start:start + size]
    return out


def version(v) -> tuple:
    return tuple(int(x) for x in str(v or "0").split("."))


best: dict = {}
for path in glob.glob(f"{GAME}/vanilla*/__brarchive/entity.brarchive"):
    for name, data in archive(path).items():
        if name.endswith(".entity.json"):
            text = re.sub(r"^\s*//.*$", "", data.decode("utf-8-sig"), flags=re.M)
            d = json.loads(text)["minecraft:client_entity"]["description"]
            best[d["identifier"]] = max(best.get(d["identifier"], (0,)), version(d.get("min_engine_version")))

files = [p for p in sorted(PACK.rglob("*.entity.json"))]
problems = 0
for p in files:
    data = json.loads(p.read_text(encoding="utf-8"))
    d = data["minecraft:client_entity"]["description"]
    found = []
    if data["format_version"] == "1.8.0" and any(isinstance(c, dict) for c in d.get("render_controllers", [])):
        found.append("format 1.8.0 with a conditional render controller")
    if data["format_version"] != "1.8.0" and "animation_controllers" in d:
        found.append(f"format {data['format_version']} with animation_controllers")
    if version(d.get("min_engine_version")) < best.get(d["identifier"], (0,)):
        found.append(f"min_engine_version {d.get('min_engine_version')} is below the game's {'.'.join(map(str, best[d['identifier']]))}")
    if d["identifier"] not in best:
        found.append("the game has no such mob")
    problems += len(found)
    for f in found:
        print(f"{p.name}: {f}")
print(f"{len(files)} entity files against {Path(GAME).parent.parent.name}: {problems} problems")
