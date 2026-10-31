"""Each chosen picture through the user's single-image 3D workflow (Pixal3D + TRELLIS.2), one at a
time, at the cow's settings (1,500 faces, 2048 texture); the GLBs land in work/models3d/mobs/."""
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

H = "http://172.26.96.1:8188"
OUT = Path("work/models3d/mobs")
WORKFLOW = "comfy/3d_pixal3d_trellis2_image_to_model.json"


def call(path, data=None, ctype="application/json", timeout=60):
    req = urllib.request.Request(H + path, data=data, headers={"Content-Type": ctype} if data else {})
    return urllib.request.urlopen(req, timeout=timeout).read()


def upload(png: Path) -> str:
    b = uuid.uuid4().hex
    body = b"".join([
        f"--{b}\r\nContent-Disposition: form-data; name=\"subfolder\"\r\n\r\nmcreskin\r\n".encode(),
        f"--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
        f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{png.name}\"\r\nContent-Type: image/png\r\n\r\n".encode(),
        png.read_bytes(), f"\r\n--{b}--\r\n".encode()])
    r = json.loads(call("/upload/image", body, f"multipart/form-data; boundary={b}"))
    return f"{r['subfolder']}/{r['name']}"


def run(name: str, png: Path, reseed: int = 0) -> Path:
    """`reseed`: added to the workflow's own fixed sampler seeds (nodes 3, 12, 18, 23), for another try
    at the same picture (the 2nd and 3rd hoglins came back as lumps without legs)."""
    w = json.load(open(WORKFLOW))
    for k in ("3", "12", "18", "23"):
        w[k]["inputs"]["seed"] += reseed
    w["122"]["inputs"]["image"] = upload(png)
    w["186"]["inputs"]["target_face_count"] = 1500
    w["288"]["inputs"]["value"] = 2048
    w["322"]["inputs"]["filename_prefix"] = f"3d/{name}_single_1500"
    pid = json.loads(call("/prompt", json.dumps({"prompt": w, "client_id": str(uuid.uuid4())}).encode()))["prompt_id"]
    t0 = time.time()
    while True:
        time.sleep(15)
        h = json.loads(call(f"/history/{pid}", timeout=20)).get(pid)
        if h and h["status"].get("status_str"):
            break
    if h["status"]["status_str"] != "success":
        errs = [m[1].get("exception_message", "")[:300] for m in h["status"].get("messages", []) if m[0] == "execution_error"]
        raise RuntimeError(f"{name}: {errs}")
    f = h["outputs"]["322"]["3d"][0]
    dest = OUT / f"{name}_single_1500{f'_r{reseed}' if reseed else ''}.glb"
    dest.write_bytes(call(f"/view?filename={f['filename']}&subfolder={f['subfolder']}&type=output", timeout=300))
    print(f"{name}: {dest} ({dest.stat().st_size // 1024} KB, {time.time() - t0:.0f} s)", flush=True)
    return dest


if __name__ == "__main__":
    call("/free", json.dumps({"unload_models": True, "free_memory": True}).encode())
    for arg in sys.argv[1:]:  # name:seed or name:seed:reseed
        name, seed, *re = arg.split(":")
        try:
            run(name, OUT / f"{name}_black_s{seed}.png", int(re[0]) if re else 0)
        except Exception as e:  # one failure doesn't stop the rest
            print(f"{name}: FAILED {e}", flush=True)
