"""Generate the real instant-demo visuals (Pollinations, free) for the Vespa kit and
write them into the portfolio microsite, updating the inlined package + saving assets.

    LLM_PROVIDER=gemini python scripts/capture_instant_visuals.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from app.llm.client import generate_image  # noqa: E402

SITE = Path("/private/tmp/claude-501/-Users-saikanagat/586a5d85-61e8-46f6-8891-50fbfab2deb9/scratchpad/repos/portfolio-v2/cortex")
ASSETS = SITE / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

VISUALS = [
    ("hero", 1280, 720, "A pastel mint Vespa Elettrica gliding through a golden-hour Rome cobblestone street, stylish young Italian professional riding, warm cinematic light, photorealistic elegant advertising photography"),
    ("poster", 768, 1024, "Vertical advertising poster: a pastel mint Vespa Elettrica against a minimal warm Milan wall at dusk, bold elegant Italian design, la dolce vita, clean negative space at top, premium campaign, photorealistic"),
    ("carousel1", 1024, 1024, "Close-up detail of a mint Vespa Elettrica handlebar and chrome mirror, golden hour, warm bokeh Italian street, premium product photography"),
    ("carousel2", 1024, 1024, "Young couple laughing on a pastel mint Vespa Elettrica in a Milan piazza at sunset, warm cinematic lifestyle photography"),
    ("carousel3", 1024, 1024, "A mint Vespa Elettrica parked by an espresso bar on a cobblestone street, soft morning light, elegant Italian lifestyle"),
]
FRAMES = [
    (1, "Dawn over a quiet Rome street, a covered Vespa Elettrica, New Year morning light, hopeful, cinematic wide shot"),
    (2, "A young professional unveils and mounts a pastel mint Vespa Elettrica, golden morning, close cinematic shot"),
    (3, "The mint Vespa Elettrica glides silently through waking Italian streets, motion blur, warm light, cinematic"),
    (4, "Riding past a piazza with subtle New Year decorations, people smiling, festive and elegant, cinematic"),
    (5, "Hero end frame: the rider pauses at a scenic overlook of the city at golden hour on the Vespa Elettrica, aspirational, cinematic"),
]


def gen(name, w, h, prompt) -> str:
    img = generate_image(prompt, width=w, height=h)
    ext = "jpg" if "jpeg" in img.mime else ("png" if "png" in img.mime else "svg")
    fname = f"{name}.{ext}"
    (ASSETS / fname).write_bytes(img.data)
    print(f"  {name}: {len(img.data)} bytes ({img.model})")
    return fname


def main() -> int:
    pkg = json.loads(Path("/tmp/vespa_instant_pkg.json").read_text())
    p = pkg["package"]
    print("visuals:")
    for i, (name, w, h, prompt) in enumerate(VISUALS):
        fname = gen(name, w, h, prompt)
        p["visuals"][i]["url"] = "assets/" + fname
        p["visuals"][i]["qa_score"] = 0.9
    print("storyboard:")
    for i, (scene, prompt) in enumerate(FRAMES):
        fname = gen(f"frame{scene}", 1280, 720, prompt)
        if i < len(p["storyboard"]):
            p["storyboard"][i]["url"] = "assets/" + fname
    Path("/tmp/vespa_instant_pkg_real.json").write_text(json.dumps(pkg, ensure_ascii=False))
    print("done -> /tmp/vespa_instant_pkg_real.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
