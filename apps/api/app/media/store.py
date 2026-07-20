"""Asset storage + kit packaging for generated media.

Images (real bytes from Gemini, or mock SVG) are written under data/assets/<run_id>/
and served by the API at /assets/<run_id>/<name>. A run's kit can be zipped for
download. ffmpeg is used opportunistically to stitch storyboard frames into a slideshow
MP4; if ffmpeg isn't present the video step is skipped gracefully."""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

_ASSETS = Path(os.environ.get("CORTEX_DATA_DIR", Path(__file__).resolve().parents[4] / "data")) / "assets"
_ASSETS.mkdir(parents=True, exist_ok=True)

_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/svg+xml": "svg"}


def run_dir(run_id: str) -> Path:
    d = _ASSETS / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_image(run_id: str, name: str, data: bytes, mime: str) -> dict:
    """Write image bytes and return an asset descriptor with a served URL path."""
    ext = _EXT.get(mime, "png")
    fname = f"{name}.{ext}"
    (run_dir(run_id) / fname).write_bytes(data)
    return {"name": name, "file": fname, "mime": mime, "url": f"/api/assets/{run_id}/{fname}"}


def asset_path(run_id: str, fname: str) -> Path | None:
    p = (run_dir(run_id) / fname).resolve()
    # path-traversal guard: must stay inside the run dir
    if _ASSETS.resolve() in p.parents and p.exists():
        return p
    return None


def slideshow(run_id: str, frame_files: list[str], seconds: float = 1.6) -> dict | None:
    """Stitch storyboard frames into an MP4 slideshow if ffmpeg + raster frames exist.
    SVG (mock) frames are skipped since ffmpeg can't read them; returns None then."""
    if not shutil.which("ffmpeg"):
        return None
    rasters = [f for f in frame_files if not f.endswith(".svg")]
    if len(rasters) < 2:
        return None
    d = run_dir(run_id)
    listfile = d / "frames.txt"
    listfile.write_text("".join(f"file '{d/f}'\nduration {seconds}\n" for f in rasters))
    out = d / "storyboard.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
             "-vf", "scale=1080:-2,format=yuv420p", "-r", "30", str(out)],
            check=True, capture_output=True, timeout=120,
        )
    except Exception:
        return None
    return {"name": "storyboard_video", "file": "storyboard.mp4", "mime": "video/mp4",
            "url": f"/api/assets/{run_id}/storyboard.mp4"}


def zip_kit(run_id: str, manifest_json: str, manifest_md: str) -> bytes:
    """Bundle all of a run's assets + manifests into an in-memory zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("campaign.json", manifest_json)
        z.writestr("campaign.md", manifest_md)
        d = run_dir(run_id)
        for f in sorted(d.iterdir()):
            if f.is_file() and f.name not in {"frames.txt"}:
                z.write(f, f"assets/{f.name}")
    return buf.getvalue()
